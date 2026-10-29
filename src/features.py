"""Datalasting og feature-bygging for butikkrisiko-modellen.

Eitt eksempel = ein politistasjon S og ein sesong T.
  - Målvariabel: butikkrelatert kriminalitet i sesong T
        = innbrot i ikkje-bustad + ran mot ikkje-bustad
  - Features: berre informasjon frå sesongar FØR T (ingen lekkasje).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sapacr-2005-2026-v1.4.csv"

TARGET_PARTS = ["burglary_nonres", "robbery_nonres"]

# Kategoriar vi brukar som historiske signal (lag-features).
LAG_COLS = [
    "business_crime",     # målvariabelen i tidlegare sesongar
    "burglary_nonres",
    "robbery_nonres",
    "shoplifting",
    "burglary_res",
    "aggr_robbery",
    "common_robbery",
    "malicious_damage",
    "drug_crime",
    "theft_from_vehicle",
    "murder",
]

FEATURES = (
    [f"{c}_lag1" for c in LAG_COLS]
    + ["business_crime_lag2", "business_crime_lag3", "business_crime_mean3",
       "business_crime_trend", "years_of_history", "latitude", "longitude"]
)


def season_start(season: str) -> int:
    """'2024/2025' -> 2024"""
    return int(season.split("/")[0])


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["season"] = df["year"].map(season_start)
    # Dei få negative tala (rapporteringsfeil) gir ingen meining som teljing.
    num = df.select_dtypes("number").columns.difference(["latitude", "longitude", "season"])
    df[num] = df[num].clip(lower=0)
    df["business_crime"] = df[TARGET_PARTS].sum(axis=1)
    return df


def _station_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Bygg lag-features for ein stasjon. Radene er éin per sesong."""
    g = g.sort_values("season").set_index("season")
    # Sørg for at manglande sesongar blir NaN i staden for å bli hoppa over.
    full = pd.RangeIndex(g.index.min(), g.index.max() + 2)  # +1 for neste sesong
    g = g.reindex(full)
    out = pd.DataFrame(index=g.index)
    out["business_crime"] = g["business_crime"]
    for c in LAG_COLS:
        out[f"{c}_lag1"] = g[c].shift(1)
    out["business_crime_lag2"] = g["business_crime"].shift(2)
    out["business_crime_lag3"] = g["business_crime"].shift(3)
    out["business_crime_mean3"] = out[
        ["business_crime_lag1", "business_crime_lag2", "business_crime_lag3"]
    ].mean(axis=1)
    # Relativ endring siste to sesongar (log-forhold), robust for små tal.
    out["business_crime_trend"] = np.log1p(out["business_crime_lag1"]) - np.log1p(
        out["business_crime_lag3"]
    )
    out["years_of_history"] = g["business_crime"].notna().cumsum().shift(1)
    out["latitude"] = g["latitude"].ffill().bfill()
    out["longitude"] = g["longitude"].ffill().bfill()
    out["station"] = g["station"].ffill().bfill()
    return out.reset_index(names="season")


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Éi rad per (stasjon, sesong), inkludert ei 'framtidsrad' for sesongen etter siste."""
    parts = [_station_frame(g) for _, g in df.groupby("station")]
    panel = pd.concat(parts, ignore_index=True)
    # Stasjonar som er lagde ned har ikkje data i siste sesong: behald berre
    # 'framtidsrader' for stasjonar som finst i den siste observerte sesongen.
    last = df["season"].max()
    active = set(df.loc[df["season"] == last, "station"])
    future = panel["season"] == last + 1
    panel = panel[~future | panel["station"].isin(active)]
    return panel.reset_index(drop=True)


def labelled(panel: pd.DataFrame) -> pd.DataFrame:
    """Rader med kjend mål og minst éi tidlegare sesong (lag1 finst)."""
    return panel[panel["business_crime"].notna() & panel["business_crime_lag1"].notna()].copy()
