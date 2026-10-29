"""Gradio-app: butikkrisiko per politistasjon i Sør-Afrika.

Køyr lokalt:  python app.py
Modellen blir køyrt live på feature-radene for neste sesong (2026/27).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import wrappers  # noqa: E402,F401  (må vere importert for at joblib skal laste modellen)

MODELS = ROOT / "models"
meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
model = joblib.load(MODELS / "model.joblib")
interval = joblib.load(MODELS / "interval_models.joblib")
FEATURES = meta["features"]
feats = pd.read_csv(MODELS / "features_next_season.csv").set_index("station")
preds_all = pd.read_csv(MODELS / "predictions_next_season.csv").set_index("station")
hist = pd.read_csv(MODELS / "history.csv")
CUTS = meta["risk_cuts"]
NEXT = meta["predicts_season"]
LAST = meta["last_observed_season"]

STATIONS = sorted(feats.index)
LABELS = {s: s.title() for s in STATIONS}
CHOICES = [(LABELS[s], s) for s in STATIONS]
DEFAULT = "khayelitsha" if "khayelitsha" in feats.index else STATIONS[0]


def run_model(row: pd.DataFrame):
    """Returnerer (prediksjon, nedre, øvre) for ei feature-rad."""
    X = row[FEATURES]
    p = float(model.predict(X)[0])
    lo = float(np.expm1(interval["lo"].predict(X))[0])
    hi = float(np.expm1(interval["hi"].predict(X))[0])
    return p, max(0.0, min(lo, p)), max(hi, p)


def risk_level(pred: float) -> str:
    if pred >= CUTS["high_from"]:
        return "🔴 Høgt"
    if pred >= CUTS["medium_from"]:
        return "🟠 Middels"
    return "🟢 Lågt"


def rank_of(pred: float) -> int:
    """Plassering blant alle stasjonar (1 = høgast predikert)."""
    return int((preds_all["pred"] > pred).sum()) + 1


def make_plot(station, pred, lo, hi):
    h = hist[hist.station == station].sort_values("season")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(h.season, h.business_crime, marker="o", label="Observert")
    nxt = int(h.season.max()) + 1
    ax.errorbar([nxt], [pred], yerr=[[pred - lo], [hi - pred]], fmt="s", color="crimson",
                capsize=5, label="Prediksjon (80 %-intervall)")
    ax.plot([h.season.max(), nxt], [h.business_crime.iloc[-1], pred], "--", color="crimson", alpha=0.5)
    ax.set(xlabel="Sesong (startår, t.d. 2025 = 2025/26)",
           ylabel="Innbrot + ran mot ikkje-bustad")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def describe(station, row, pred, lo, hi):
    last = float(row["business_crime_lag1"].iloc[0])
    change = (pred - last) / last * 100 if last > 0 else float("nan")
    arrow = "↑" if pred > last else "↓"
    return (
        f"### {LABELS[station]}\n"
        f"**Risikonivå for butikkar {NEXT}: {risk_level(pred)}**  \n"
        f"Predikert tal på innbrot og ran mot forretningslokale: **{pred:.0f}** "
        f"(80 %-intervall: {lo:.0f}–{hi:.0f})  \n"
        f"Siste observerte sesong ({LAST}): **{last:.0f}** → endring {arrow} {abs(change):.0f} %  \n"
        f"Rangering: **{rank_of(pred)}** av {len(preds_all)} stasjonar (1 = høgast).\n"
    )


def lookup(station):
    row = feats.loc[[station]].reset_index(drop=True)
    pred, lo, hi = run_model(row)
    return describe(station, row, pred, lo, hi), make_plot(station, pred, lo, hi)


def what_if(station, burglary, robbery):
    """Overstyr fjorårstala for innbrot/ran og sjå korleis prediksjonen endrar seg."""
    row = feats.loc[[station]].reset_index(drop=True).copy()
    total = burglary + robbery
    row["burglary_nonres_lag1"] = burglary
    row["robbery_nonres_lag1"] = robbery
    row["business_crime_lag1"] = total
    lags = row[["business_crime_lag1", "business_crime_lag2", "business_crime_lag3"]]
    row["business_crime_mean3"] = lags.mean(axis=1)
    row["business_crime_trend"] = np.log1p(total) - np.log1p(row["business_crime_lag3"])
    pred, lo, hi = run_model(row)
    return describe(station, row, pred, lo, hi), make_plot(station, pred, lo, hi)


def defaults_for(station):
    r = feats.loc[station]
    return int(r["burglary_nonres_lag1"]), int(r["robbery_nonres_lag1"])


def top_table(n):
    t = preds_all.sort_values("pred", ascending=False).head(int(n)).reset_index()
    t["station"] = t["station"].str.title()
    t = t.rename(columns={
        "station": "Stasjon", "business_crime_lag1": f"Observert {LAST}",
        "pred": f"Predikert {NEXT}", "pred_lo": "Nedre (10 %)", "pred_hi": "Øvre (90 %)",
    })
    return t[["Stasjon", f"Observert {LAST}", f"Predikert {NEXT}", "Nedre (10 %)", "Øvre (90 %)"]].round(0)


def metrics_table():
    t = pd.DataFrame(meta["test"]).T.reset_index().rename(columns={"index": "Modell"})
    return t.round(3)


ABOUT = f"""
## Om modellen
- **Mål:** førebu butikkeigarar på kor mange innbrot og ran mot forretningslokale ein politistasjon
  vil få neste sesong ({NEXT}).
- **Data:** SAPS årlege kriminalitetsstatistikk per politistasjon, 2005/06–2025/26
  (DataFirst, University of Cape Town).
- **Modell:** {meta["best_model"]}, trent på historikk frå stasjonen (fjorårstal, trend, andre
  kriminalitetstypar, plassering). Evaluert på heile sesongen 2025/26 som modellen ikkje har sett.
- **Risikonivå** er relativt: *Høgt* = blant dei 15 % stasjonane med flest predikerte tilfelle,
  *Middels* = nest øvre halvdel, *Lågt* = nedre halvdel.

## Avgrensingar
- Tala er **antal**, ikkje rate per innbyggjar, så store stasjonar får høgare risiko.
- Berre **registrerte** saker; underrapportering varierer mellom område.
- Kriminaliteten har falle kraftig dei to siste sesongane. Modellen kan ikkje vite om dette held fram.
- Dette er statistikk på stasjonsnivå, ikkje ei åtvaring om ein spesifikk butikk eller gate.
"""

with gr.Blocks(title="Butikkrisiko – Sør-Afrika") as demo:
    gr.Markdown(
        "# Butikkrisiko per politistasjon\n"
        f"Velg ein politistasjon og sjå kva ML-modellen predikerer for **{NEXT}**."
    )
    with gr.Tab("Stasjon"):
        dd = gr.Dropdown(CHOICES, value=DEFAULT, label="Politistasjon", filterable=True)
        out_md = gr.Markdown()
        out_plot = gr.Plot()
        dd.change(lookup, dd, [out_md, out_plot])
        demo.load(lookup, dd, [out_md, out_plot])
    with gr.Tab("Kva om?"):
        gr.Markdown(
            "Endre fjorårstala og sjå korleis prediksjonen endrar seg. "
            "Nyttig for å forstå kva modellen legg vekt på."
        )
        dd2 = gr.Dropdown(CHOICES, value=DEFAULT, label="Politistasjon", filterable=True)
        b0, r0 = defaults_for(DEFAULT)
        sl_b = gr.Slider(0, 1000, value=b0, step=1, label=f"Innbrot i ikkje-bustad, {LAST}")
        sl_r = gr.Slider(0, 400, value=r0, step=1, label=f"Ran mot ikkje-bustad, {LAST}")
        btn = gr.Button("Køyr modellen")
        w_md = gr.Markdown()
        w_plot = gr.Plot()
        dd2.change(defaults_for, dd2, [sl_b, sl_r])
        btn.click(what_if, [dd2, sl_b, sl_r], [w_md, w_plot])
    with gr.Tab("Høgast risiko"):
        n = gr.Slider(5, 50, value=15, step=5, label="Antal stasjonar")
        tbl = gr.Dataframe(value=top_table(15), interactive=False)
        n.change(top_table, n, tbl)
    with gr.Tab("Om modellen"):
        gr.Markdown(ABOUT)
        gr.Markdown("### Resultat på testsett (2025/26)")
        gr.Dataframe(value=metrics_table(), interactive=False)

if __name__ == "__main__":
    demo.launch()
