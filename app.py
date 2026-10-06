"""
The Alpine Acoustic Archive
Computational Musicology & Digital Preservation of Austrian Accordion Traditions
"""

import os
import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.audio_engine import (
    TARGET_SR,
    REPERTOIRE_CATALOG,
    load_audio,
    extract_bellows_dynamics,
    transcribe_audio,
    compute_track_analysis,
    midi_to_note_name
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="The Alpine Acoustic Archive",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# High-Contrast Bright Styling with Alpine Forest Green Accents
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Clean base page styling */
    .stApp {
        background-color: #fbfcfb;
    }
    
    .archive-header {
        border-bottom: 3px solid #1e4d38;
        padding-bottom: 1.2rem;
        margin-bottom: 1.6rem;
    }
    
    .archive-title {
        font-family: "Georgia", "Times New Roman", serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #133927;
        letter-spacing: -0.01em;
        margin: 0 0 0.3rem 0;
    }
    
    .archive-subtitle {
        font-size: 1.05rem;
        color: #334d3d;
        margin: 0 0 0.6rem 0;
        line-height: 1.4;
    }

    .provenance-card {
        background-color: #f0f7f3;
        border-left: 4px solid #2d6a4f;
        border-radius: 4px;
        padding: 0.85rem 1.1rem;
        font-size: 0.88rem;
        color: #1f3d2b;
        margin-top: 0.8rem;
        line-height: 1.45;
    }
    
    .meta-tag {
        display: inline-block;
        background: #e8f5ec;
        color: #1b4332;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #b7dfc6;
    }

    /* High-contrast bright metric cards */
    .metric-card {
        background: #ffffff;
        border: 1.5px solid #2d6a4f;
        border-top: 4px solid #1e4d38;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        text-align: left;
        box-shadow: 0 1px 3px rgba(30, 77, 56, 0.08);
    }

    .metric-card-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #133927;
        line-height: 1.2;
    }

    .metric-card-lbl {
        font-size: 0.74rem;
        font-weight: 600;
        color: #4b6354;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    .history-card {
        background-color: #ffffff;
        border: 1px solid #d1e2d7;
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        color: #1f2937;
    }

    .history-header {
        color: #1e4d38;
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: "Georgia", serif;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Audio Analysis Pipeline
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_analyze_file(filepath: str) -> dict:
    return compute_track_analysis(filepath)


# ---------------------------------------------------------
# Sidebar: Repertoire Selection & Collection Controls
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("Collection Catalog")

    input_source = st.radio(
        "Source:",
        ["Curated Alpine Repertoire", "Upload Field Recording"],
        index=0
    )

    audio_path = None
    selected_meta = None
    track_title = ""

    if input_source == "Curated Alpine Repertoire":
        curated_files = sorted(list(REPERTOIRE_CATALOG.keys()))
        track_labels = [f"{REPERTOIRE_CATALOG[f]['title']} — {REPERTOIRE_CATALOG[f]['region']}" for f in curated_files]

        selected_label = st.selectbox("Select Piece:", track_labels)
        selected_filename = curated_files[track_labels.index(selected_label)]
        audio_path = os.path.join("data", "raw_audio", selected_filename)
        selected_meta = REPERTOIRE_CATALOG[selected_filename]
        track_title = selected_meta["title"]
    else:
        uploaded_file = st.file_uploader(
            "Upload Audio File (.wav, .mp3, .ogg):",
            type=["wav", "mp3", "ogg", "flac"]
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.read())
            tfile.flush()
            audio_path = tfile.name
            track_title = os.path.splitext(uploaded_file.name)[0].replace("_", " ").title()
            selected_meta = {
                "title": track_title,
                "region": "User Contribution / Field Recording",
                "genre": "User Recording",
                "tempo": "Variable",
                "key": "Acoustic Detection",
                "instrument": "Accordion",
                "description": "User-provided recording for automated acoustic analysis.",
                "cultural_notes": "Preserving uncataloged regional performance practices."
            }

    st.divider()
    st.markdown("#### Ethnomusicology Initiative")
    st.write(
        "The **Alpine Acoustic Archive** is an open-access research prototype designed to study "
        "and preserve unwritten Austrian accordion music (*Volksmusik*). Using physical acoustic "
        "modeling and convolutional neural transcription, it bridges oral heritage with digital musicology."
    )
    st.caption("Open source. Released under the MIT License.")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("""
<div class="archive-header">
    <div class="archive-title">The Alpine Acoustic Archive</div>
    <div class="archive-subtitle">
        Computational Musicology and Digital Preservation of Austrian Accordion Traditions
    </div>
    <div class="provenance-card">
        <strong>Data Provenance Notice:</strong> The 5 reference audio tracks below are computational acoustic models 
        synthesized to reflect traditional Austrian regional dance forms (Ländler, Polka, Waltz, Boarischer, Zwiefacher) 
        without copyright encumbrance. Field researchers and musicians are invited to analyze real authenticated 
        recordings via the <em>Upload Field Recording</em> option in the sidebar.
    </div>
</div>
""", unsafe_allow_html=True)

if not audio_path or not os.path.exists(audio_path):
    st.info("Select a piece or upload an audio recording from the sidebar to inspect analysis.")
    st.stop()

# ---------------------------------------------------------
# Processing
# ---------------------------------------------------------
with st.spinner("Processing audio features..."):
    analysis_data = load_and_analyze_file(audio_path)

df_dynamics = analysis_data["dynamics_df"]
df_notes = analysis_data["notes_df"]
duration = analysis_data["duration"]
total_notes = analysis_data["total_notes"]
pitch_range = analysis_data["pitch_range"]
avg_dur = analysis_data["avg_duration"]
dominant_pitch = analysis_data["dominant_pitch"]
midi_bytes = analysis_data["midi_bytes"]
csv_path = analysis_data["csv_path"]

# ---------------------------------------------------------
# Section 1: Track Details & Audio Playback
# ---------------------------------------------------------
col_meta, col_player = st.columns([3, 2])

with col_meta:
    st.markdown(f"### {selected_meta['title']}")
    st.markdown(f"""
        <span class="meta-tag">Region: {selected_meta['region']}</span>
        <span class="meta-tag">Genre: {selected_meta['genre']}</span>
        <span class="meta-tag">Tempo: {selected_meta['tempo']}</span>
        <span class="meta-tag">Key: {selected_meta['key']}</span>
        <span class="meta-tag">Instrument: {selected_meta['instrument']}</span>
    """, unsafe_allow_html=True)
    st.write(selected_meta["description"])

with col_player:
    st.markdown("#### Audio Playback")
    with open(audio_path, "rb") as f_aud:
        audio_bytes = f_aud.read()
    st.audio(audio_bytes, format="audio/wav")
    st.caption(f"Duration: **{duration:.2f} s** | Sampling Rate: **{TARGET_SR:,} Hz** | Format: **PCM 16-bit Mono**")

st.divider()

# ---------------------------------------------------------
# Section 2: High-Contrast Acoustic Summary Metrics
# ---------------------------------------------------------
st.markdown("#### Acoustic Summary")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-card-val">{total_notes}</div><div class="metric-card-lbl">Notes Transcribed</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-card-val">{pitch_range}</div><div class="metric-card-lbl">Pitch Range</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-card-val">{dominant_pitch}</div><div class="metric-card-lbl">Dominant Pitch</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-card-val">{avg_dur} s</div><div class="metric-card-lbl">Mean Note Duration</div></div>', unsafe_allow_html=True)
with m5:
    peak_val = f"{df_dynamics['rms_normalized'].max():.2f}"
    st.markdown(f'<div class="metric-card"><div class="metric-card-val">{peak_val}</div><div class="metric-card-lbl">Peak Bellows Energy</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 3: Bellows Dynamics & Timbre Visualizer (Plotly)
# ---------------------------------------------------------
st.markdown("#### Bellows Dynamics and Timbre")
st.write(
    "Acoustic bellows pressure is computed via Root-Mean-Square (RMS) energy envelopes. "
    "The spectral centroid traces shifts in reed brightness (*Chor-Klangfarbe*) across phrasing."
)

fig_dynamics = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(
        "Bellows Pressure Envelope (Normalized RMS) & Detected Note Onset Attacks",
        "Spectral Centroid (Hz) — Treble Reed Register Brightness"
    )
)

# 1. Bellows RMS curve (High contrast deep Alpine green)
fig_dynamics.add_trace(
    go.Scatter(
        x=df_dynamics["timestamp"],
        y=df_dynamics["rms_normalized"],
        mode="lines",
        name="Bellows Pressure (RMS)",
        line=dict(color="#1e4d38", width=2.4),
        fill="tozeroy",
        fillcolor="rgba(30, 77, 56, 0.12)",
        hovertemplate="Time: %{x:.2f}s<br>Bellows Pressure: %{y:.2f}<extra></extra>"
    ),
    row=1, col=1
)

# 2. Note Attacks (Onsets) overlay (distinct rust markers)
onset_times = analysis_data["onset_times"]
if len(onset_times) > 0:
    onset_rms = np.interp(onset_times, df_dynamics["timestamp"], df_dynamics["rms_normalized"])
    fig_dynamics.add_trace(
        go.Scatter(
            x=onset_times,
            y=onset_rms,
            mode="markers",
            name="Detected Note Onset",
            marker=dict(color="#c0392b", size=6, symbol="circle", line=dict(color="#ffffff", width=1)),
            hovertemplate="Onset Attack: %{x:.2f}s<extra></extra>"
        ),
        row=1, col=1
    )

# 3. Spectral Centroid Curve (Warm amber tone)
fig_dynamics.add_trace(
    go.Scatter(
        x=df_dynamics["timestamp"],
        y=df_dynamics["spectral_centroid"],
        mode="lines",
        name="Spectral Centroid",
        line=dict(color="#b45309", width=2.0),
        hovertemplate="Time: %{x:.2f}s<br>Centroid: %{y:.0f} Hz<extra></extra>"
    ),
    row=2, col=1
)

fig_dynamics.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
fig_dynamics.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", row=1, col=1)
fig_dynamics.update_yaxes(title_text="Normalized Pressure", row=1, col=1, range=[0, 1.08], gridcolor="#e2e8f0")
fig_dynamics.update_yaxes(title_text="Frequency (Hz)", row=2, col=1, gridcolor="#e2e8f0")

fig_dynamics.update_layout(
    height=450,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    hovermode="x unified",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafbfa"
)
st.plotly_chart(fig_dynamics, use_container_width=True)


# ---------------------------------------------------------
# Section 4: Transcription & Piano Roll
# ---------------------------------------------------------
st.markdown("#### Automated Polyphonic Transcription")
st.write(
    "Polyphonic note events detected via Spotify's Basic Pitch convolutional neural model. "
    "Each note records exact start time, duration, pitch class, and dynamic velocity."
)

tab_pianoroll, tab_pitch_dist, tab_notes_table = st.tabs([
    "Piano Roll Timeline",
    "Pitch Class Distribution",
    "Transcribed Notes Table"
])

with tab_pianoroll:
    if not df_notes.empty:
        fig_pr = go.Figure()

        for _, note in df_notes.iterrows():
            fig_pr.add_trace(
                go.Scatter(
                    x=[note["start_time"], note["end_time"]],
                    y=[note["pitch_midi"], note["pitch_midi"]],
                    mode="lines",
                    line=dict(color="#2d6a4f", width=6),
                    showlegend=False,
                    hovertemplate=(
                        f"Note: <b>{note['pitch_name']}</b> (MIDI {note['pitch_midi']})<br>"
                        f"Start: {note['start_time']}s | Duration: {note['duration']}s<br>"
                        f"Velocity: {note['velocity']}<extra></extra>"
                    )
                )
            )

        unique_pitches = sorted(df_notes["pitch_midi"].unique())
        tick_pitches = unique_pitches[::max(1, len(unique_pitches) // 12)]
        tick_labels = [midi_to_note_name(p) for p in tick_pitches]

        fig_pr.update_layout(
            title="Transcribed Piano Roll (Pitch vs. Time)",
            xaxis_title="Time (seconds)",
            yaxis=dict(
                title="Pitch Class",
                tickmode="array",
                tickvals=tick_pitches,
                ticktext=tick_labels,
                gridcolor="#e2e8f0"
            ),
            xaxis=dict(gridcolor="#e2e8f0", range=[0, duration]),
            height=400,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fafbfa",
            margin=dict(l=50, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info("No note events detected in this recording.")

with tab_pitch_dist:
    if not df_notes.empty:
        pitch_counts = df_notes["pitch_name"].value_counts().reset_index()
        pitch_counts.columns = ["pitch_name", "count"]

        pitch_counts["midi_num"] = pitch_counts["pitch_name"].apply(
            lambda name: [p for p in df_notes["pitch_midi"] if midi_to_note_name(p) == name][0]
        )
        pitch_counts.sort_values(by="midi_num", inplace=True)

        fig_dist = go.Figure(
            go.Bar(
                x=pitch_counts["pitch_name"],
                y=pitch_counts["count"],
                marker_color="#1e4d38",
                hovertemplate="Pitch: %{x}<br>Count: %{y}<extra></extra>"
            )
        )
        fig_dist.update_layout(
            title="Frequency Distribution of Detected Pitches",
            xaxis_title="Pitch Class",
            yaxis_title="Total Note Count",
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fafbfa",
            height=360,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

with tab_notes_table:
    if not df_notes.empty:
        st.dataframe(
            df_notes,
            use_container_width=True,
            column_config={
                "start_time": st.column_config.NumberColumn("Start (s)", format="%.3f"),
                "end_time": st.column_config.NumberColumn("End (s)", format="%.3f"),
                "duration": st.column_config.NumberColumn("Duration (s)", format="%.3f"),
                "pitch_midi": st.column_config.NumberColumn("MIDI Num"),
                "pitch_name": st.column_config.TextColumn("Note Name"),
                "velocity": st.column_config.ProgressColumn("Velocity", min_value=0, max_value=127, format="%d"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f")
            },
            hide_index=True
        )


# ---------------------------------------------------------
# Section 5: Rich Historical Context & Ethnomusicology Dossier
# ---------------------------------------------------------
st.divider()
st.markdown("#### Historical Context & Ethnomusicological Dossier")

col_hist1, col_hist2 = st.columns(2)

with col_hist1:
    st.markdown("""
    <div class="history-card">
        <div class="history-header">The Evolution of the Steirische Harmonika</div>
        <p>
            In 1829, Cyrill Demian registered the first accordion patent in Vienna. Over subsequent decades, 
            regional makers in Styria and Bohemia modified the diatonic button accordion into the distinct 
            <strong>Steirische Harmonika</strong> (pioneered by Josef Hlaváček and Peter Stachl in Graz).
        </p>
        <p>
            Key innovations set the instrument apart from standard western accordions:
        </p>
        <ul>
            <li><strong>Helikon Contrabass Reeds:</strong> Massive, wide-profile zinc or brass reeds housed in oversized resonance chambers, inspired by military brass <em>Helicon</em> tubas. They produce the deep, subterranean bass response fundamental to Alpine dance rhythm.</li>
            <li><strong>The Gleichton (Unison Button):</strong> Located on the second and third button rows, this feature sounds the exact same pitch on both push (<em>Druck</em>) and pull (<em>Zug</em>), allowing players to sustain continuous harmonic drones without interrupting bellows direction.</li>
            <li><strong>Schwebung (Musette Tremolo):</strong> Treble reed pairs tuned 2 to 5 Hz apart, generating acoustic interference beats that project across outdoor alpine valleys and dance floors.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_hist2:
    st.markdown("""
    <div class="history-card">
        <div class="history-header">Oral Transmission vs. Griffschrift Tablature</div>
        <p>
            Unlike classical European art music, Alpine <em>Volksmusik</em> was transmitted almost exclusively 
            <strong>by ear (nach dem Gehör)</strong> across generations of village musicians, timber workers, 
            and alpine herdsmen (<em>Sennerinnen und Senner</em>).
        </p>
        <p>
            Because diatonic accordions produce different notes depending on whether the bellows is being pushed 
            or pulled, traditional players found standard five-line staff notation impractical. In 1916, 
            innovator Max Rosenzopf and early pedagogues standardized <strong>Griffschrift</strong>: a four-line 
            button-position tablature showing <em>which button to press and whether to push or pull</em>, 
            rather than actual concert pitches.
        </p>
        <p>
            <strong>The Preservation Gap:</strong> When master players pass away, their unnotated repertoire and 
            subtle rhythmic rubato risk being lost. This archive applies neural transcription to reverse-engineer 
            standard musical scores from raw acoustic recordings, enabling cross-cultural study and permanent preservation.
        </p>
    </div>
    """, unsafe_allow_html=True)

with st.expander(f"Regional Context for {selected_meta['title']}"):
    st.write(selected_meta["cultural_notes"])
    st.markdown(f"""
    * **Regional Origin:** {selected_meta['region']}
    * **Musical Meter & Form:** {selected_meta['genre']} ({selected_meta['tempo']})
    * **Traditional Instrumentation:** {selected_meta['instrument']}
    """)


# ---------------------------------------------------------
# Section 6: Data & MIDI Export
# ---------------------------------------------------------
st.divider()
st.markdown("#### Archival Export")
st.write(
    "Export transcription data for analysis in music notation software (MuseScore, Sibelius, Dorico) "
    "or computational musicology environments."
)

col_exp_midi, col_exp_csv = st.columns(2)

with col_exp_midi:
    st.download_button(
        label="Download Transcribed MIDI (.mid)",
        data=midi_bytes,
        file_name=f"{analysis_data['basename']}_transcription.mid",
        mime="audio/midi",
        use_container_width=True,
        help="Export standard MIDI file for notation software."
    )
    st.caption("Standard polyphonic MIDI track with timing, velocity, and pitch data.")

with col_exp_csv:
    csv_bytes = df_dynamics.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dynamics Time Series (.csv)",
        data=csv_bytes,
        file_name=f"{analysis_data['basename']}_dynamics.csv",
        mime="text/csv",
        use_container_width=True,
        help="Export timestamped RMS energy and spectral centroid values."
    )
    st.caption("CSV dataset containing timestamps, normalized RMS energy, and centroid.")

# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 0.5rem 0 1.5rem 0;">
        The Alpine Acoustic Archive &mdash; Open-Source MIR & Ethnomusicology Tool
        <br>
        Built with Python, Librosa, Basic Pitch, Plotly, and Streamlit. Released under the MIT License.
    </div>
    """,
    unsafe_allow_html=True
)
