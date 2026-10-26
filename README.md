# The Alpine Acoustic Archive (Open-Source Cultural MIR & Benchmark)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10884920.svg)](https://doi.org/10.5281/zenodo.10884920)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-live%20app-FF4B4B.svg)](https://alpine-acoustic-archive.streamlit.app)
[![Basic Pitch](https://img.shields.io/badge/basic--pitch-Spotify%20MIR-1DB954.svg)](https://github.com/spotify/basic-pitch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**The Alpine Acoustic Archive** is an open-access research corpus and Music Information Retrieval (MIR) benchmark designed to preserve, transcribe, and analyze traditional Austrian folk accordion repertoire (*Volksmusik*). 

The project evaluates why commercial Western automated transcription models experience severe **algorithmic friction** (misinterpreting non-isochronous Alpine rubato as clock drift, stripping bellows volume envelopes, and deleting oral grace-notes).

---

## 📋 Common App Activity Summary

* **Activity Type:** Computer / Technology
* **Position / Leadership:** Founder & Lead Audio ML Archiver *(32 chars)*
* **Organization Name:** The Alpine Acoustic Archive (Open-Source Cultural MIR Project) *(62 chars)*
* **Description:** *Curated 40-track archival Austrian folk dataset; built open MIR benchmark evaluating why standard audio AI fails on Alpine rubato & bellows dynamics.* *(149 chars)*

---

## 📖 Mission & Ethnomusicological Context

Traditional Alpine folk accordion music—centered on the diatonic **Steirische Harmonika**, the Viennese **Schrammelharmonika**, and regional button accordions—has historically survived through oral and aural transmission (*nach dem Gehör*). 

Unlike Western classical scores or popular music, Alpine folk music relies on three acoustic principles:
1. **Dynamic Bellows Articulation (*Balgführung*):** Sound amplitude is governed by physical bellows compression (*Druck*) and expansion (*Zug*). Nuance occurs *after* the note attack, defying standard static MIDI velocity models ($0–127$).
2. **Non-Isochronous Micro-Timing (Rubato):** Alpine 3/4 Ländlers and Waltzes feature systemic elongation of the second beat to synchronize with circular regional dance steps. Traditional Zwiefachers alternate irregularly between 3/4 and 2/4 measures.
3. **Oral Micro-Ornamentation:** Master accordionists employ rapid appoggiaturas, mordents, and bellow-shakes (*Balg-Tremolo*) that transmit regional lineage.

---

## 📊 The 40-Track Benchmark Corpus (`data/metadata_corpus.csv`)

The archive curates **40 standardized audio recordings** (22.05 kHz 16-bit `.wav`) sampled across 4 distinct Austrian regional traditions:

| Regional Tradition | Primary Forms | Acoustic & Performance Profile | Sample Track Identifier |
| :--- | :--- | :--- | :--- |
| **Styria (*Steiermark*)** | Boarischer, Ländler, Schleuniger, Jodler | Booming Helikon contrabass drones, sharp syncopated bellows accents (*Balgakzente*). | `Styria_Boarischer_StachlArchive.wav` |
| **Lower Austria (*Niederösterreich*)** | Schrammelwalzer, Polka, Dialektwalzer | Lyrical Danube tavern phrasing, chromatic chord voicings, lilting second-beat emphasis. | `LowerAustria_Schrammelwalzer_HeurigerTrio.wav` |
| **Upper Austria (*Oberösterreich*)** | Salzkammergut Ländler, Zwiefacher, Schleuniger | Metric shifts (alternating 3/4 and 2/4), heavy *Schwebung* (musette tremolo beating). | `UpperAustria_Zwiefacher_HallstattEnsemble.wav` |
| **Carinthia (*Kärnten*)** | Gailtaler Rundtanz, Liedwalzer, Weis | Vocal-style grace notes, rapid bellow-shakes (*Balg-Tremolo*), expressive rubato phrasing. | `Carinthia_Rundtanz_GailtalerQuartett.wav` |

### Corpus Feature Metrics:
* **`bellows_dynamic_range_db`:** Root-Mean-Square energy peak-to-floor dynamic range in decibels.
* **`rubato_variance_score`:** Inter-Onset Interval (IOI) deviation score measuring departures from an isochronous metronomic beat grid.
* **`ornamentation_density_per_min`:** Rate of rapid grace notes (<160ms) and transient shakes per minute.
* **`algorithmic_friction_pct`:** Percentage of note events distorted or deleted by commercial transcription assumptions.

---

## 🔬 "Algorithmic Friction": Why Standard Audio AI Fails

```
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│  Standard Commercial Baseline (MIDI) │  vs   │   Alpine Acoustic Archive Pipeline   │
├──────────────────────────────────────┤       ├──────────────────────────────────────┤
│ ❌ Rigid 16th-note quantization      │       │ ✅ Micro-timing preservation         │
│    (Snaps rubato to artificial grid) │       │    (Accurate elastic onset timing)   │
│ ❌ Noise-gate thresholds (<120ms)    │       │ ✅ Multi-resolution grace note track │
│    (Deletes Balg-Tremolo & grace)    │       │    (Preserves oral ornamentation)    │
│ ❌ Static single-strike velocity     │       │ ✅ Continuous RMS bellows curves     │
│    (Strips post-attack air swells)   │       │    (Captures physical Balgführung)   │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
```

The interactive dashboard includes an **Algorithmic Friction Benchmark** tab with side-by-side split-view piano rolls and a quantitative scorecard demonstrating how commercial models mischaracterize regional folk heritages.

---

## 🚀 Running Locally

Get the application running in 3 simple terminal steps:

### 1. Clone repository & initialize virtual environment
```bash
git clone https://github.com/michellemingertl-create/alpine-acoustic-archive.git
cd alpine-acoustic-archive
python3.11 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch interactive dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏛️ Digital Preservation & Zenodo DOI Minting

To ensure permanent, citeable open-science preservation, this repository is linked with **Zenodo** (operated by CERN):

1. **GitHub Release:** Go to Releases → Draft a new release (e.g., `v1.0.0`).
2. **Zenodo Integration:** In Zenodo, enable the `michellemingertl-create/alpine-acoustic-archive` repository under GitHub Settings.
3. **Automated DOI Generation:** Publishing a release on GitHub triggers Zenodo to automatically mint a permanent Digital Object Identifier (DOI) and archive all 40 `.wav` files, `.mid` transcripts, and `metadata_corpus.csv`.

### Citation (BibTeX):
```bibtex
@dataset{ertl2026alpine,
  author       = {Michelle Ming Ertl},
  title        = {The Alpine Acoustic Archive: An Open MIR Corpus and Benchmark for Austrian Folk Accordion Traditions},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.10884920},
  url          = {https://doi.org/10.5281/zenodo.10884920}
}
```

---

## 📜 License

Distributed under the **MIT License**. Open-access data and code for ethnomusicologists, researchers, and folk musicians worldwide.
