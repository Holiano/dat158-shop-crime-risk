"""Tren, vel og evaluer modellar for butikkrisiko. Køyr: python src/train.py

Tidsdeling (ingen blanding av framtid inn i fortid):
  train : måle-sesongar  <= 2023  (sesong 2023/24 og tidlegare)
  val   : 2024  (2024/25)   -> modellval og hyperparametrar
  test  : 2025  (2025/26)   -> blir berre brukt éin gong til slutt
Den endelege modellen blir trent på alle sesongar til og med 2025/26 og
predikerer 2026/27.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_panel, labelled, load_raw  # noqa: E402
from wrappers import DeltaRegressor, Log1pRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
SEED = 42

TRAIN_END, VAL_SEASON, TEST_SEASON = 2023, 2024, 2025


# ---------------------------------------------------------------- modellar
def make_models():
    impute_scale = lambda est: make_pipeline(  # noqa: E731
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est
    )
    return {
        "Ridge (log)": Log1pRegressor(impute_scale(Ridge(alpha=10.0))),
        "Random forest (log)": Log1pRegressor(
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED
                ),
            )
        ),
        "Gradient boosting (log)": Log1pRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.0, random_state=SEED,
            )
        ),
        "Gradient boosting (delta)": DeltaRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.0, random_state=SEED,
            )
        ),
    }


# ---------------------------------------------------------------- metrikkar
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_log": float(mean_absolute_error(np.log1p(y_true), np.log1p(y_pred))),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def evaluate(models, tr, te, label):
    rows = []
    # Baselinjar
    base = {
        "Baseline: same som i fjor": te["business_crime_lag1"].to_numpy(),
        "Baseline: snitt siste 3": te["business_crime_mean3"].fillna(te["business_crime_lag1"]).to_numpy(),
    }
    for name, pred in base.items():
        rows.append({"model": name, **metrics(te["business_crime"], pred)})
    preds = {}
    for name, m in models.items():
        m.fit(tr[FEATURES], tr["business_crime"].to_numpy())
        p = m.predict(te[FEATURES])
        preds[name] = p
        rows.append({"model": name, **metrics(te["business_crime"], p)})
    res = pd.DataFrame(rows).set_index("model").round(3)
    print(f"\n=== {label} (n={len(te)}) ===")
    print(res.to_string())
    return res, preds


def main():
    raw = load_raw()
    panel = build_panel(raw)
    data = labelled(panel)
    print(f"Merka eksempel: {len(data)}, stasjonar: {data.station.nunique()}")

    tr = data[data.season <= TRAIN_END]
    va = data[data.season == VAL_SEASON]
    te = data[data.season == TEST_SEASON]
    trva = data[data.season <= VAL_SEASON]
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    # ---- 1. modellval på valideringssettet
    val_res, _ = evaluate(make_models(), tr, va, "VALIDERING (2024/25)")
    candidates = val_res.drop(index=[i for i in val_res.index if i.startswith("Baseline")])
    best_name = candidates["MAE"].idxmin()
    print(f"\nBeste modell på validering (lågast MAE): {best_name}")

    # ---- 2. éin endeleg test: tren på train+val, test på 2025/26
    test_res, test_preds = evaluate(make_models(), trva, te, "TEST (2025/26) – berre ein gong")

    # ---- 3. feature-viktigheit (permutasjon) for beste modell, på testsett
    from sklearn.inspection import permutation_importance

    best = make_models()[best_name]
    best.fit(trva[FEATURES], trva["business_crime"].to_numpy())

    class _Sk:  # liten sklearn-adapter for permutation_importance
        def __init__(self, m): self.m = m
        def fit(self, *a): return self
        def predict(self, X): return np.log1p(self.m.predict(X))
        def score(self, X, y): return -mean_absolute_error(np.log1p(y), self.predict(X))

    imp = permutation_importance(
        _Sk(best), te[FEATURES], te["business_crime"].to_numpy(),
        n_repeats=10, random_state=SEED, scoring=lambda e, X, y: e.score(X, y),
    )
    imp_df = (
        pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
    )
    print("\nFeature-viktigheit (auke i MAE_log når feature blir stokka):")
    print(imp_df.head(8).round(4).to_string(index=False))

    # ---- 4. kvantilmodellar for usikkerheit (10 % og 90 %), tren på train+val
    def quantile_model(q):
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, learning_rate=0.05, max_iter=300,
            max_leaf_nodes=15, min_samples_leaf=30, random_state=SEED,
        )

    y_trva_log = np.log1p(trva["business_crime"].to_numpy())
    lo = quantile_model(0.1).fit(trva[FEATURES], y_trva_log)
    hi = quantile_model(0.9).fit(trva[FEATURES], y_trva_log)
    lo_te = np.expm1(lo.predict(te[FEATURES])).clip(min=0)
    hi_te = np.expm1(hi.predict(te[FEATURES]))
    y_te = te["business_crime"].to_numpy()
    coverage = float(np.mean((y_te >= lo_te) & (y_te <= hi_te)))
    print(f"\nDekning for 80 %-intervall på test: {coverage:.1%} (mål: 80 %)")

    # ---- 5. figurar
    fig, ax = plt.subplots(figsize=(5.5, 5))
    p = test_preds[best_name]
    ax.scatter(y_te + 1, p + 1, s=8, alpha=0.4)
    lim = [1, max(y_te.max(), p.max()) + 1]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="Faktisk (+1)", ylabel="Predikert (+1)",
           title=f"{best_name}\nTest 2025/26")
    fig.tight_layout(); fig.savefig(REPORTS / "fig_pred_vs_actual.png", dpi=150); plt.close(fig)

    yearly = raw.groupby("season")["business_crime"].sum()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(yearly.index, yearly.values, marker="o")
    ax.set(xlabel="Sesong (startår)", ylabel="Totalt antall",
           title="Innbrot + ran mot ikkje-bustad, heile Sør-Afrika")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORTS / "fig_national_trend.png", dpi=150); plt.close(fig)

    top = imp_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top["feature"], top["importance"])
    ax.set(xlabel="Auke i MAE (log) ved stokking", title="Feature-viktigheit")
    fig.tight_layout(); fig.savefig(REPORTS / "fig_feature_importance.png", dpi=150); plt.close(fig)

    # ---- 6. endeleg modell: alle merka data (til og med 2025/26) -> predikér 2026/27
    final = make_models()[best_name].fit(data[FEATURES], data["business_crime"].to_numpy())
    y_all_log = np.log1p(data["business_crime"].to_numpy())
    lo_f = quantile_model(0.1).fit(data[FEATURES], y_all_log)
    hi_f = quantile_model(0.9).fit(data[FEATURES], y_all_log)

    nxt_season = int(panel["season"].max())
    nxt = panel[panel["season"] == nxt_season].copy()
    nxt["pred"] = final.predict(nxt[FEATURES])
    nxt["pred_lo"] = np.expm1(lo_f.predict(nxt[FEATURES])).clip(min=0)
    nxt["pred_hi"] = np.expm1(hi_f.predict(nxt[FEATURES]))
    nxt["pred_lo"] = np.minimum(nxt["pred_lo"], nxt["pred"])
    nxt["pred_hi"] = np.maximum(nxt["pred_hi"], nxt["pred"])

    # Risikonivå = rangering av predikert tal blant alle stasjonar.
    cuts = {
        "medium_from": float(nxt["pred"].quantile(0.50)),
        "high_from": float(nxt["pred"].quantile(0.85)),
    }
    out_cols = ["station", "latitude", "longitude", "business_crime_lag1", "pred", "pred_lo", "pred_hi"]
    nxt[out_cols].round(2).to_csv(MODELS / "predictions_next_season.csv", index=False)
    # Feature-rader for neste sesong: appen køyrer modellen live på desse.
    nxt[["station"] + FEATURES].to_csv(MODELS / "features_next_season.csv", index=False)
    joblib.dump({"lo": lo_f, "hi": hi_f}, MODELS / "interval_models.joblib")

    # Historikk per stasjon til grafen i appen.
    hist = raw[["station", "season", "business_crime", "burglary_nonres", "robbery_nonres"]]
    hist.to_csv(MODELS / "history.csv", index=False)

    joblib.dump(final, MODELS / "model.joblib")
    meta = {
        "best_model": best_name,
        "target": "burglary_nonres + robbery_nonres",
        "features": FEATURES,
        "predicts_season": f"{nxt_season}/{nxt_season + 1}",
        "last_observed_season": f"{nxt_season - 1}/{nxt_season}",
        "risk_cuts": cuts,
        "validation": val_res.to_dict(orient="index"),
        "test": test_res.to_dict(orient="index"),
        "interval_coverage_test_80pct": coverage,
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    test_res.to_csv(REPORTS / "results_test.csv")
    val_res.to_csv(REPORTS / "results_validation.csv")
    imp_df.to_csv(REPORTS / "feature_importance.csv", index=False)
    print(f"\nLagra modell og prediksjonar for {meta['predicts_season']} i {MODELS}")


if __name__ == "__main__":
    main()
