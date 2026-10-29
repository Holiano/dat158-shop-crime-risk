# Butikkrisiko per politistasjon i Sør-Afrika

**Emne:** DAT158 Maskinlæring, ML assignment 2 (haust 2026)
**Forfattar:** Julian Hjartholm Bosdal (individuelt prosjekt)

## 1. Introduksjon og problemstilling

Små butikkeigarar i Sør-Afrika har lite førehandsinformasjon om kor utsett dei er for innbrot og ran.
Målet med prosjektet er ein nettstad der brukaren vel ein politistasjon og får ein ML-basert prediksjon av
**kor mange innbrot og ran mot forretningslokale (ikkje-bustad) stasjonen får neste sesong**, saman med eit
risikonivå (lågt, middels, høgt) og eit usikkerheitsintervall.

Dette er eit **regresjonsproblem på tabelldata** (tidsrekkjer per stasjon). Prosjektet er inspirert av ein
gründeridé om tryggleik for småbutikkar, men er avgrensa til offentleg statistikk.

## 2. Data

**Kjelde.** SAPS (South African Police Service) årlege kriminalitetsstatistikk per politistasjon, i
samla CSV-versjon `sapacr-2005-2026-v1.4.csv`. Datasettet blir publisert av DataFirst, University of Cape Town
(katalog: <https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/1012>).
*Merk: eg har ikkje kunna stadfeste lisensvilkåra for v1.4; sjå DataFirst for vilkår.*

**Innhald.** 24 206 rader, 1 205 politistasjonar, 21 sesongar (2005/06–2025/26; sesongen går april–mars),
koordinatar og ca. 30 kriminalitetskategoriar (antal registrerte saker).

**Kvalitet og reinsing.**
- Ingen duplikat på (stasjon, sesong).
- 107 rader manglar koordinatar. Desse blir fylte frå andre sesongar for same stasjon der det går.
- Fleire seksuallovbrotskolonnar har negative verdiar (opptil 443 rader) og `sexual_offences` manglar i
  3 423 rader. Desse kolonnane trengst ikkje til oppgåva og er ikkje brukte som features. Alle teljingar
  blir klipte til minst 0.
- Talet på stasjonar veks litt over tid (1 141 i 2005 til 1 174 i 2025), så nye stasjonar har kort historikk.

**Utforsking.** Fordelinga er svært skeiv (i siste sesong er medianen 32 og maks 432 per stasjon). Difor blir målet modellert i
log-rom. Nasjonalt auka innbrot og ran mot forretningslokale frå ca. 58 600 (2005/06) til ca. 96 400 (2016/17),
og har sidan falle til ca. 55 200 i 2025/26. **Dei to siste sesongane har eit kraftig fall** (–16 % og –14 %),
noko som gjer prediksjon vanskelegare (sjå 6). Sjå `fig_national_trend.png`.

## 3. Målvariabel og forsøksoppsett

- **Mål:** `business_crime = burglary_nonres + robbery_nonres` i sesong T for stasjon S.
- **Eksempel:** eitt per (stasjon, sesong) der det finst minst éin tidlegare sesong: 22 997 eksempel,
  1 199 stasjonar.
- **Tidsdeling (ingen tilfeldig blanding):**
  - Trening: mål-sesongar til og med 2023/24 (20 657 eksempel)
  - Validering: 2024/25 (1 172), brukt til modellval
  - Test: 2025/26 (1 168), brukt éin gong til slutt, etter at modellen var vald og trent på trening + validering
- Ein tilfeldig deling hadde latt modellen «sjå» framtida og gitt for gode tal, så tidsdeling gir eit meir
  realistisk estimat.

## 4. Features (berre informasjon frå sesongar før T)

| Gruppe | Features |
|---|---|
| Historikk for målet | fjorårstal, tal frå 2 og 3 år sidan, snitt siste 3, trend (log-forhold lag1/lag3) |
| Andre kriminalitetstypar (fjorår) | innbrot og ran ikkje-bustad, butikktjuveri, innbrot bustad, grovt ran, ran, skadeverk, narkotika, tjuveri frå bil, drap |
| Stad og historikk | breiddegrad, lengdegrad, tal sesongar med data |

Manglande verdiar (nye stasjonar) blir handterte av `HistGradientBoostingRegressor` direkte, og av
median-imputering (+ indikator) for Ridge og Random forest.

## 5. Modellar og evaluering

**Baselinjar:** (a) same tal som i fjor, (b) snitt av siste tre sesongar.
**Modellar:** Ridge, Random forest, Gradient boosting (log-mål), og Gradient boosting som lærer
*endringa frå fjoråret* i log-rom («delta»).
Hyperparametrar er sette moderat og haldne konstante (ingen omfattande tuning, på grunn av tidsfrist).
**Måltal:** MAE (antal), RMSE, MAE på log-skalaen og Spearman-rangkorrelasjon (rangering av stasjonar).

### Validering (2024/25)

| Modell | MAE | RMSE | MAE (log) | Spearman |
|---|---|---|---|---|
| Baselinje: same som i fjor | 15,82 | 27,89 | 0,338 | 0,958 |
| Baselinje: snitt siste 3 | 18,08 | 33,96 | 0,363 | 0,956 |
| Ridge (log) | 39,79 | 205,28 | 0,612 | 0,906 |
| Random forest (log) | 13,93 | 23,83 | 0,316 | 0,962 |
| Gradient boosting (log) | 13,06 | 22,13 | 0,307 | 0,963 |
| **Gradient boosting (delta)** | **12,85** | **22,04** | **0,302** | 0,963 |

Gradient boosting (delta) hadde lågast MAE og vart vald.

### Test (2025/26, éin gong)

| Modell | MAE | RMSE | MAE (log) | Spearman |
|---|---|---|---|---|
| Baselinje: same som i fjor | 13,50 | 22,83 | 0,344 | 0,951 |
| Baselinje: snitt siste 3 | 19,34 | 34,54 | 0,407 | 0,948 |
| Ridge (log) | 27,45 | 129,85 | 0,593 | 0,902 |
| Random forest (log) | 11,13 | 17,90 | 0,315 | 0,954 |
| Gradient boosting (log) | 10,31 | 15,96 | 0,304 | 0,955 |
| **Gradient boosting (delta)** (vald) | **10,61** | **16,73** | **0,307** | 0,955 |

Den valde modellen reduserer MAE frå 13,5 til 10,6 (ca. **21 % betre** enn «same som i fjor») og RMSE frå
22,8 til 16,7. Sjå `fig_pred_vs_actual.png`.

**Usikkerheit.** To kvantilmodellar (10 % og 90 %) gir eit 80 %-intervall. På testsettet dekkjer
intervallet **74,7 %** av dei faktiske verdiane, altså litt under målet på 80 %.

**Feature-viktigheit** (permutasjon på testsettet): fjorårstalet for målet dominerer, deretter
innbrot ikkje-bustad (fjorår), trend, skadeverk og innbrot bustad. Sjå `fig_feature_importance.png`.

## 6. Diskusjon og avgrensingar

- **Gevinsten er moderat.** Rangeringa av stasjonar (Spearman ≈ 0,95) er nesten lik som med baselinja:
  stasjonar med mykje kriminalitet i fjor har òg mykje i år. Modellens verdi ligg mest i betre **estimat av
  nivået**, ikkje i å finne heilt nye «hotspots».
- **Modellval på tvers av validering og test.** Gradient boosting (log) var ein anelse betre enn (delta) på
  testsettet (10,31 mot 10,61). Eg heldt meg til valet frå valideringssettet for å unngå å velje modell på
  testdata. Forskjellen er liten og kan vere tilfeldig.
- **Fordelingsskifte.** Kriminaliteten fall kraftig i 2024/25 og 2025/26. På testsettet fanga modellen mesteparten
  av fallet: summert over alle stasjonar predikerte han 56 484 mot 54 575 faktiske (ca. 3,5 % for høgt),
  medan «same som i fjor» ville gitt 63 618 (ca. 17 % for høgt). Modellen overestimerer likevel litt
  (gjennomsnittleg feil +1,6 per stasjon), og dette kan vere ein del av grunnen til at intervalldekninga er
  under 80 %. Eg kan ikkje vite om nedgangen held fram.
- **Ridge fungerer dårleg** (store RMSE-utslag), sannsynlegvis fordi relasjonane er ikkje-lineære og ein
  lineær modell ekstrapolerer ustabilt for store stasjonar. Eg la ikkje meir arbeid i å fikse han.
- **Teljingar, ikkje rater.** Datasettet har ikkje folketal per stasjon, så store stasjonar får høgare «risiko».
  Risikonivå er relative (topp 15 % = høgt, øvre halvdel = middels, resten = lågt), ikkje ein absolutt tryggleikspåstand.
- **Berre registrerte saker.** Underrapportering varierer mellom område.
- **Etisk.** Ein «risikoscore» for eit område kan stigmatisere. Nettstaden viser difor statistikk for
  politistasjonar, ikkje enkeltbutikkar eller gater, og seier tydeleg kva den ikkje kan svare på.
- **Vidare arbeid:** folketal og næringsdata (rater), meir omfattande hyperparametertuning, kryssvalidering over
  fleire tidsvindauge (rullerande opphav), og bruk av nyhende- og signaldata.

## 7. Nettstaden (deployment)

Appen er laga i **Gradio** (`app.py`) og køyrer modellen live. Fire faner:
1. **Stasjon:** vel politistasjon → prediksjon for 2026/27, intervall, risikonivå, rangering og historikkgraf.
2. **Kva om?:** overstyr fjorårets innbrots- og rantal og sjå korleis prediksjonen endrar seg.
3. **Høgast risiko:** tabell over stasjonane med høgast predikert tal.
4. **Om modellen:** resultat og avgrensingar.

**Visning:** Nettstaden er ikkje publisert offentleg, fordi Gradio-hosting på Hugging Face krev betalt PRO-abonnement.
I staden følgjer ei demo-video som viser alle fanene: `demo/demo.mp4` (opptaket er laga automatisk med
Playwright, `demo/record_demo.py`, og viser appen som køyrer lokalt).
Reproduksjon: sjå `README.md` (`python src/train.py`, deretter `python app.py`).

## 8. Kjelder og bruk av AI

- SAPS annual crime records (sapacr), DataFirst, University of Cape Town.
- Bibliotek: scikit-learn, pandas, NumPy, SciPy, Matplotlib, Gradio.
- **AI-verktøy:** Claude Code (Anthropic, modell Claude Sonnet 5.5) vart brukt til å skrive kode
  (feature-pipeline, treningsskript, Gradio-app) og eit utkast til denne rapporten. Eg har gått gjennom
  resultata og tolkingane sjølv.
