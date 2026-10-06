# 🏔️ The Alpine Acoustic Archive (Alpine MIR & Heritage Preservation)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Basic Pitch](https://img.shields.io/badge/basic--pitch-Spotify%20MIR-1DB954.svg)](https://github.com/spotify/basic-pitch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-access web application and Music Information Retrieval (MIR) processing pipeline designed to preserve, transcribe, and analyze traditional Austrian folk accordion repertoire (*Volksmusik*).

---

## 📖 Mission & Cultural Background

Traditional Alpine accordion music—centered on the diatonic **Steirische Harmonika**, the Viennese **Schrammelharmonika**, and regional button accordions—has historically been transmitted through **oral and aural tradition** (*nach dem Gehör*). 

Unlike classical western art music, Alpine folk music relies extensively on:
* **Bellows Dynamics (*Balgführung*):** Expressive volume swells (*Zug und Druck*) that articulate phrasing, rhythmic bounce (*Balgstoß*), and dance steps.
* **Musette Reed Detuning (*Schwebung*):** Intentional acoustic beating produced by dual- and triple-reed setups in the treble register (*Chöre*).
* **Rubato & Metric Elasticity:** Nuanced tempo fluctuations in 3/4 Ländlers, syncopated 4/4 Boarischers, and erratic alternating metres in Zwiefachers (3/4 and 2/4 shifts).

Standard commercial automated transcription software often struggles with these acoustic nuances. The **Alpine Acoustic Archive** provides a dedicated, lightweight, open-source pipeline that models physical bellows pressure curves, analyzes reed spectral brightness, transcribes polyphonic note events into standard MIDI, and presents results in an interactive web dashboard.

---

## 🔬 Computational Acoustic Methodology

The pipeline combines two complementary MIR methodologies:

```
┌─────────────────────────────────┐
│     Audio Input (.wav, .mp3)    │
└────────────────┬────────────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
┌───────────────┐     ┌───────────────────────────────────┐
│ librosa (MIR) │     │ basic-pitch (Spotify Neural Net)  │
└───────┬───────┘     └─────────────────┬─────────────────┘
        │                               │
        ├─ RMS Energy Envelope          ├─ Polyphonic Pitch Prediction
        │  (Bellows Air Pressure)       ├─ Note Onset / Offset Timing
        ├─ Spectral Centroid            ├─ Velocity & Confidence Scoring
        │  (Reed Timbre / Brightness)   │
        ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│              Interactive Streamlit Dashboard            │
│  - Synchronized Bellows Curve & Attack Overlay (Plotly) │
│  - Polyphonic Piano Roll & Tonal Distributions          │
│  - Downloadable Standard MIDI (.mid) & Dynamics (.csv)  │
└─────────────────────────────────────────────────────────┘
```

1. **Bellows Pressure Dynamics (RMS Energy):**
   - Root-Mean-Square (RMS) energy is computed over sliding frames ($N=2048$, $\text{hop}=512$) at $22,050\text{ Hz}$.
   - Normalized between 0.0 and 1.0 to trace physical bellows air compression (*Balgführung*).
2. **Reed Timbre & Register Brightness (Spectral Centroid):**
   - Tracks the barycenter of the frequency spectrum to reflect multi-reed register changes (*Chor-Umschaltung*) and dynamic brightness shifts.
3. **Automated Polyphonic Transcription (`basic-pitch`):**
   - Leverages Spotify's open-source Convolutional Neural Network (CNN) model to detect polyphonic notes without requiring pitch-bend quantization artifacts.
   - Extracts note arrays: `[start_time, end_time, duration, pitch_midi, pitch_name, velocity, confidence]`.
   - Generates standardized MIDI files compatible with MuseScore, Sibelius, Dorico, and digital audio workstations.

---

## 📂 Curated Repertoire Catalog

The archive includes 5 standardized, curated reference recordings in `data/raw_audio/`:

| Piece | Region | Meter / Tempo | Key | Acoustic & Bellows Character |
| :--- | :--- | :--- | :--- | :--- |
| **Kitzbüheler Ländler** | Tyrol (Kitzbühel Alps) | Slow 3/4 (84 BPM) | G Major | Broad bellows swells, musette beating (*Schwebung*), and rubato phrasing. |
| **Zillertaler Polka** | Tyrol (Zillertal) | Brisk 2/4 (124 BPM) | C Major | Percussive staccato bellows bounces (*Balgstoß*) and crisp bass-chord leaps. |
| **Wachauer Dialektwalzer** | Lower Austria (Danube) | Flowing 3/4 (138 BPM) | F Major | Lyrical Danube waltz with expressive second-beat emphasis and vocal ornamentation. |
| **Steirischer Boarischer** | Styria (*Steiermark*) | Syncopated 4/4 (102 BPM) | D Major | Characteristic syncopated bellows accents (*Balgakzente*) and deep Helikon bass. |
| **Salzkammergut Zwiefacher** | Upper Austria | Alternating 3/4 & 2/4 (144 BPM) | B♭ Major | Traditional 10-beat cycle (3+3+2+2) with sudden directional bellows pressure shifts. |

*In addition to the curated tracks, users can upload custom field recordings directly in the dashboard.*

---

## 🚀 Running Locally

Get the application running in 3 simple terminal steps:

### 1. Clone the repository & create environment
```bash
git clone https://github.com/your-username/alpine-acoustic-archive.git
cd alpine-acoustic-archive
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Layout

```text
alpine-acoustic-archive/
├── data/
│   ├── raw_audio/          # 5 curated standardized Austrian .wav recordings
│   └── transcripts/        # Generated .mid transcripts and feature .csv files
├── src/
│   ├── __init__.py
│   ├── audio_engine.py     # MIR feature extraction & basic-pitch transcription pipeline
│   └── audio_synth.py      # Alpine repertoire acoustic synthesizer
├── app.py                  # Streamlit web dashboard with Plotly visualizers
├── requirements.txt        # Locked Python dependencies
├── .gitignore              # Ignored caches and local virtual environment
└── README.md               # Archival documentation & methodology
```

---

## ☁️ Deployment to Streamlit Community Cloud

This project is structured for 1-click hosting on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repository to GitHub.
2. Log into Streamlit Cloud and click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy!**

---

## 📜 License

Distributed under the **MIT License**. Free for ethnomusicologists, researchers, and folk musicians worldwide.
