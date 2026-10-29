---
title: Butikkrisiko Sør-Afrika
emoji: 🏪
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Butikkrisiko per politistasjon (DAT158 ML assignment 2)

Ein ML-basert nettstad som predikerer kor mange **innbrot og ran mot forretningslokale** ein sørafrikansk
politistasjon får neste sesong (2026/27), med risikonivå (lågt / middels / høgt) og eit 80 %-intervall.

- Rapport: [`reports/report.md`](reports/report.md)
- Demo-video av nettstaden: [`demo/demo.mp4`](demo/demo.mp4) (ca. 1 min 15 s, viser alle fire fanene)
- Nettstaden er ikkje publisert på internett (Gradio-hosting på Hugging Face krev betalt PRO-abonnement). Køyr han lokalt, sjå under.

## Køyr lokalt

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py              # trenar modellane, lagar models/ og reports/
python app.py                    # startar Gradio på http://127.0.0.1:7860
```

`models/` er ferdig generert i repoet, så `python app.py` fungerer òg utan å trene på nytt.

## Struktur

| Sti | Innhald |
|---|---|
| `data/sapacr-2005-2026-v1.4.csv` | SAPS-kriminalitetsdata per politistasjon, 2005/06–2025/26 |
| `src/features.py` | Lasting, reinsing og lag-features (utan lekkasje frå framtida) |
| `src/train.py` | Tidsdeling, modellsamanlikning, test, kvantilmodellar, figurar |
| `src/wrappers.py` | Log- og delta-wrappar rundt scikit-learn-modellar |
| `app.py` | Gradio-appen (også inngangen for Hugging Face Spaces) |
| `demo/` | Demo-video og skriptet som tok han opp (`record_demo.py`, Playwright) |
| `models/` | Lagra modell, prediksjonar, historikk og metadata |
| `reports/` | Rapport, figurar og resultattabellar |

## Kjelder

- SAPS annual crime records (sapacr), DataFirst, University of Cape Town. Sjå rapporten for detaljar.
- AI-verktøy: Claude Code (Anthropic) vart brukt til koding og rapportutkast. Sjå rapporten.
