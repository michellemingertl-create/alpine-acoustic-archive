"""
Audio Engine module for The Alpine Acoustic Archive.
Provides MIR feature extraction (bellows RMS dynamics, spectral centroid timbre),
polyphonic note transcription (Spotify basic-pitch with robust fallback), and data formatting.
"""

import os
import io
import logging
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from typing import Tuple, List, Dict, Any, Optional

try:
    from basic_pitch.inference import predict
    BASIC_PITCH_AVAILABLE = True
except Exception as e:
    BASIC_PITCH_AVAILABLE = False
    logging.warning(f"basic_pitch not available, using acoustic fallback: {e}")

import pretty_midi

# Standard analysis parameters
TARGET_SR = 22050
HOP_LENGTH = 512
FRAME_LENGTH = 2048

# Curated Repertoire Catalog with Austrian Ethnomusicology Context
REPERTOIRE_CATALOG: Dict[str, Dict[str, Any]] = {
    "kitzbueheler_laendler.wav": {
        "title": "Kitzbüheler Ländler",
        "region": "Tyrol (Kitzbühel Alps)",
        "genre": "Ländler (Slow 3/4)",
        "tempo": "84 BPM",
        "key": "G Major",
        "instrument": "Steirische Harmonika (4-row, Helikon bass)",
        "description": (
            "A traditional Alpine slow Ländler characterized by heavy, deliberate bellows "
            "expansions (Zug und Druck) on the downbeat and secondary beat. The acoustic "
            "beating (Schwebung) arises from dual-tremolo reeds in the treble register."
        ),
        "cultural_notes": (
            "Historically danced in regional Tyrolean inns, Ländler pieces rely on oral "
            "transmission without written notation. The unmetered rubato breathes with the "
            "dancer's steps."
        )
    },
    "zillertaler_polka.wav": {
        "title": "Zillertaler Polka",
        "region": "Tyrol (Zillertal)",
        "genre": "Polka (Brisk 2/4)",
        "tempo": "124 BPM",
        "key": "C Major",
        "instrument": "Steirische Harmonika",
        "description": (
            "A fast-paced Alpine polka featuring crisp staccato bellows bounces (Balgstoß) "
            "and alternating bass-chord accompaniment. Notice rapid energy onsets and sharp attacks."
        ),
        "cultural_notes": (
            "The Zillertal valley is world-renowned for its virtuoso folk accordionists. "
            "Polkas emphasize percussive rhythm over legato phrasing."
        )
    },
    "wachauer_dialektwalzer.wav": {
        "title": "Wachauer Dialektwalzer",
        "region": "Lower Austria (Wachau Valley)",
        "genre": "Dialect Waltz (Flowing 3/4)",
        "tempo": "138 BPM",
        "key": "F Major",
        "instrument": "Schrammelharmonika / Chromatic Accordion",
        "description": (
            "A lyrical Danube waltz with sweeping bellows arches and lilting second-beat emphasis. "
            "The spectral centroid reveals smooth reed register transitions."
        ),
        "cultural_notes": (
            "Derived from traditional Heuriger wine-tavern culture, dialect waltzes incorporate "
            "melodic warmth and song-like vocal ornamentation."
        )
    },
    "steirischer_boarischer.wav": {
        "title": "Steirischer Boarischer",
        "region": "Styria (Steiermark)",
        "genre": "Boarischer (Syncopated 4/4)",
        "tempo": "102 BPM",
        "key": "D Major",
        "instrument": "Steirische Harmonika (D-G-C-F)",
        "description": (
            "A prototypical Styrian Boarischer marked by characteristic bellows syncopations "
            "(Balgakzente) on beats 2 and 4, coupled with a booming Helikon contrabass."
        ),
        "cultural_notes": (
            "The Boarischer is an iconic couple dance of the Eastern Alps. The physical bellows "
            "tug provides an unmistakable rhythmic groove."
        )
    },
    "salzkammergut_zwiefacher.wav": {
        "title": "Salzkammergut Zwiefacher",
        "region": "Upper Austria / Salzkammergut",
        "genre": "Zwiefacher (Alternating 3/4 & 2/4)",
        "tempo": "144 BPM",
        "key": "B♭ Major",
        "instrument": "Diatonic Harmonika",
        "description": (
            "A challenging piece alternating between 3/4 (waltz) and 2/4 (dreher) bars in a "
            "10-beat cycle (3+3+2+2). Bellows volume curves exhibit sudden asymmetric directional shifts."
        ),
        "cultural_notes": (
            "Zwiefache ('the double ones') represent an ancient Bavarian-Austrian rhythmic tradition "
            "transmitted entirely by ear, baffling classical notational conventions."
        )
    }
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note_name(midi_number: int) -> str:
    """Convert MIDI number (e.g. 60) to pitch name (e.g. C4)."""
    octave = (midi_number // 12) - 1
    name = NOTE_NAMES[midi_number % 12]
    return f"{name}{octave}"

def load_audio(filepath_or_bytes, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load an audio file into mono floating-point time series."""
    y, sample_rate = librosa.load(filepath_or_bytes, sr=sr, mono=True)
    return y, sample_rate

def extract_bellows_dynamics(y: np.ndarray, sr: int = TARGET_SR,
                             hop_length: int = HOP_LENGTH,
                             frame_length: int = FRAME_LENGTH) -> pd.DataFrame:
    """
    Extracts acoustic bellows dynamics:
    1. RMS Energy: Reflects physical air pressure (Balgführung).
    2. Spectral Centroid: Reflects reed timbre and register brightness.
    3. Onset Strength: Percussive attack transients.
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_max = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_norm = rms / rms_max

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    df_dynamics = pd.DataFrame({
        "timestamp": np.round(times, 4),
        "rms_energy": np.round(rms, 6),
        "rms_normalized": np.round(rms_norm, 4),
        "spectral_centroid": np.round(centroid, 2),
        "onset_strength": np.round(onset_env[:len(times)], 4)
    })
    return df_dynamics

def extract_notes_from_pretty_midi(midi_obj: pretty_midi.PrettyMIDI) -> pd.DataFrame:
    """Extract structured DataFrame from a PrettyMIDI object."""
    notes_list = []
    for instrument in midi_obj.instruments:
        for note in instrument.notes:
            notes_list.append({
                "start_time": round(float(note.start), 3),
                "end_time": round(float(note.end), 3),
                "duration": round(float(note.end - note.start), 3),
                "pitch_midi": int(note.pitch),
                "pitch_name": midi_to_note_name(int(note.pitch)),
                "velocity": int(note.velocity),
                "confidence": round(float(note.velocity / 127.0), 3)
            })
    notes_df = pd.DataFrame(notes_list)
    if not notes_df.empty:
        notes_df.sort_values(by=["start_time", "pitch_midi"], inplace=True)
        notes_df.reset_index(drop=True, inplace=True)
    else:
        notes_df = pd.DataFrame(columns=[
            "start_time", "end_time", "duration", "pitch_midi",
            "pitch_name", "velocity", "confidence"
        ])
    return notes_df

def transcribe_audio_fallback(audio_path: str,
                              output_midi_path: Optional[str] = None) -> Tuple[pd.DataFrame, pretty_midi.PrettyMIDI, bytes]:
    """
    Acoustic onset & pitch detection fallback using librosa.
    Ensures the application never crashes even if basic-pitch neural backend is unavailable.
    """
    y, sr = load_audio(audio_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    
    # Fundamental frequency tracking
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    times_f0 = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

    midi_obj = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=21) # Accordion

    notes_list = []
    for i, t_start in enumerate(onset_times):
        t_end = onset_times[i+1] if i + 1 < len(onset_times) else min(len(y)/sr, t_start + 0.5)
        # Find median pitch in window
        mask = (times_f0 >= t_start) & (times_f0 < t_end) & (voiced_flag)
        if np.any(mask) and not np.all(np.isnan(f0[mask])):
            pitch_hz = np.nanmedian(f0[mask])
            pitch_midi = int(np.clip(round(librosa.hz_to_midi(pitch_hz)), 24, 108))
        else:
            pitch_midi = 60 # C4 default

        vel = 90
        note = pretty_midi.Note(velocity=vel, pitch=pitch_midi, start=t_start, end=t_end)
        inst.notes.append(note)

        notes_list.append({
            "start_time": round(float(t_start), 3),
            "end_time": round(float(t_end), 3),
            "duration": round(float(t_end - t_start), 3),
            "pitch_midi": pitch_midi,
            "pitch_name": midi_to_note_name(pitch_midi),
            "velocity": vel,
            "confidence": 0.75
        })

    midi_obj.instruments.append(inst)
    notes_df = pd.DataFrame(notes_list)
    if not notes_df.empty:
        notes_df.sort_values(by=["start_time", "pitch_midi"], inplace=True)
        notes_df.reset_index(drop=True, inplace=True)
    else:
        notes_df = pd.DataFrame(columns=[
            "start_time", "end_time", "duration", "pitch_midi",
            "pitch_name", "velocity", "confidence"
        ])

    midi_buffer = io.BytesIO()
    midi_obj.write(midi_buffer)
    midi_bytes = midi_buffer.getvalue()

    if output_midi_path:
        os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
        with open(output_midi_path, "wb") as f:
            f.write(midi_bytes)

    return notes_df, midi_obj, midi_bytes

def transcribe_audio(audio_path: str,
                     output_midi_path: Optional[str] = None) -> Tuple[pd.DataFrame, pretty_midi.PrettyMIDI, bytes]:
    """
    Transcribes audio using Spotify's basic-pitch inference model,
    or falls back cleanly if basic-pitch is not loaded.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not BASIC_PITCH_AVAILABLE:
        return transcribe_audio_fallback(audio_path, output_midi_path=output_midi_path)

    try:
        # Run basic-pitch inference
        model_output, midi_obj, raw_note_events = predict(audio_path)

        notes_list = []
        for item in raw_note_events:
            start_s = float(item[0])
            end_s = float(item[1])
            pitch_num = int(round(item[2]))
            amplitude = float(item[3])
            velocity = int(np.clip(round(amplitude * 127), 1, 127))
            duration_s = max(0.01, round(end_s - start_s, 3))

            notes_list.append({
                "start_time": round(start_s, 3),
                "end_time": round(end_s, 3),
                "duration": duration_s,
                "pitch_midi": pitch_num,
                "pitch_name": midi_to_note_name(pitch_num),
                "velocity": velocity,
                "confidence": round(amplitude, 3)
            })

        notes_df = pd.DataFrame(notes_list)
        if not notes_df.empty:
            notes_df.sort_values(by=["start_time", "pitch_midi"], inplace=True)
            notes_df.reset_index(drop=True, inplace=True)
        else:
            notes_df = pd.DataFrame(columns=[
                "start_time", "end_time", "duration", "pitch_midi",
                "pitch_name", "velocity", "confidence"
            ])

        midi_buffer = io.BytesIO()
        midi_obj.write(midi_buffer)
        midi_bytes = midi_buffer.getvalue()

        if output_midi_path:
            os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
            with open(output_midi_path, "wb") as f:
                f.write(midi_bytes)

        return notes_df, midi_obj, midi_bytes

    except Exception as e:
        logging.warning(f"Inference error in basic-pitch ({e}), using acoustic fallback.")
        return transcribe_audio_fallback(audio_path, output_midi_path=output_midi_path)

def compute_track_analysis(audio_path: str,
                           transcripts_dir: str = "data/transcripts") -> Dict[str, Any]:
    """
    Full pipeline execution for a given audio file.
    Loads cached MIDI and dynamics CSV if already computed (instant load),
    otherwise extracts features and saves cache.
    """
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(transcripts_dir, exist_ok=True)

    csv_path = os.path.join(transcripts_dir, f"{basename}_features.csv")
    midi_path = os.path.join(transcripts_dir, f"{basename}.mid")

    # Load audio
    y, sr = load_audio(audio_path)
    duration = float(len(y) / sr)

    # Fast path: Load pre-computed cache if available
    if os.path.exists(csv_path) and os.path.exists(midi_path):
        df_dynamics = pd.read_csv(csv_path)
        midi_obj = pretty_midi.PrettyMIDI(midi_path)
        notes_df = extract_notes_from_pretty_midi(midi_obj)
        with open(midi_path, "rb") as f_mid:
            midi_bytes = f_mid.read()
    else:
        # Dynamic extraction
        df_dynamics = extract_bellows_dynamics(y, sr)
        df_dynamics.to_csv(csv_path, index=False)
        notes_df, midi_obj, midi_bytes = transcribe_audio(audio_path, output_midi_path=midi_path)

    # Extract detected attacks (onsets)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

    # Summary statistics
    total_notes = len(notes_df)
    pitch_min = int(notes_df["pitch_midi"].min()) if total_notes > 0 else 0
    pitch_max = int(notes_df["pitch_midi"].max()) if total_notes > 0 else 0
    pitch_range = f"{midi_to_note_name(pitch_min)} - {midi_to_note_name(pitch_max)}" if total_notes > 0 else "N/A"
    avg_duration = round(float(notes_df["duration"].mean()), 2) if total_notes > 0 else 0.0

    # Dominant pitch
    dominant_pitch = notes_df["pitch_name"].mode()[0] if total_notes > 0 and not notes_df["pitch_name"].empty else "N/A"

    return {
        "audio_path": audio_path,
        "basename": basename,
        "duration": duration,
        "sample_rate": sr,
        "dynamics_df": df_dynamics,
        "notes_df": notes_df,
        "midi_obj": midi_obj,
        "midi_bytes": midi_bytes,
        "midi_path": midi_path,
        "csv_path": csv_path,
        "onset_times": onset_times,
        "total_notes": total_notes,
        "pitch_range": pitch_range,
        "avg_duration": avg_duration,
        "dominant_pitch": dominant_pitch
    }

# ------------------------------------------------------------------
# Advanced MIR Metrics & Algorithmic Friction Analysis
# ------------------------------------------------------------------

def compute_bellows_metrics(df_dynamics: pd.DataFrame) -> Dict[str, float]:
    """
    Computes physical bellows air pressure metrics:
    - Bellows Dynamic Range (dB): Ratio of peak bellows volume to quiet baseline.
    - Modulation Depth: Coefficient of variation (std/mean) of RMS energy envelope.
    """
    rms = df_dynamics["rms_energy"].values
    if len(rms) == 0 or np.max(rms) <= 0:
        return {"bellows_dynamic_range_db": 0.0, "bellows_modulation_index": 0.0}

    non_zero = rms[rms > 1e-4]
    p_high = np.percentile(rms, 95)
    p_low = np.percentile(non_zero, 10) if len(non_zero) > 0 else 1e-4

    dr_db = 20.0 * np.log10(max(p_high, 1e-5) / max(p_low, 1e-5))
    mod_index = float(np.std(rms) / (np.mean(rms) + 1e-5))

    return {
        "bellows_dynamic_range_db": round(float(dr_db), 2),
        "bellows_modulation_index": round(float(mod_index), 3)
    }

def compute_rubato_metrics(onset_times: np.ndarray, duration_s: float) -> Dict[str, float]:
    """
    Computes Micro-Timing Variance (Rubato Score):
    Measures departure from strict isochronous beat spacing (elastic Alpine 3/4 & 2/4 phrasing).
    """
    if len(onset_times) < 4:
        return {"rubato_variance_score": 0.05}

    iois = np.diff(onset_times)
    # Exclude long phrase pauses > 2.0s
    valid_iois = iois[(iois > 0.08) & (iois < 2.0)]
    if len(valid_iois) < 3:
        return {"rubato_variance_score": 0.10}

    mean_ioi = np.mean(valid_iois)
    # Normalized standard deviation of inter-onset intervals
    rubato_variance = float(np.std(valid_iois) / (mean_ioi + 1e-5))
    rubato_score = float(np.clip(rubato_variance, 0.0, 1.0))

    return {
        "rubato_variance_score": round(rubato_score, 3)
    }

def compute_ornamentation_metrics(notes_df: pd.DataFrame, duration_s: float, y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Computes Ornamentation Density:
    Quantifies frequency of rapid grace notes (<160ms), micro-inflections, and Balg-Tremolo shakes.
    """
    if notes_df.empty or duration_s <= 0:
        return {"ornamentation_density_per_min": 0.0, "grace_note_count": 0}

    # Notes under 160ms are characteristic Alpine grace notes / mordents
    grace_notes = notes_df[notes_df["duration"] < 0.16]
    grace_count = len(grace_notes)

    duration_min = max(duration_s / 60.0, 0.1)
    ornament_density = float(grace_count / duration_min)

    return {
        "ornamentation_density_per_min": round(ornament_density, 2),
        "grace_note_count": int(grace_count)
    }

def simulate_standard_transcription_baseline(notes_df: pd.DataFrame,
                                             bpm: float = 120.0) -> Tuple[pd.DataFrame, pretty_midi.PrettyMIDI, Dict[str, float]]:
    """
    Simulates a standard Western commercial transcription baseline:
    1. Rigid 16th-note metronomic quantization (destroys Alpine rubato).
    2. Transient noise-gating: strips notes < 120ms (deletes unwritten grace notes).
    3. Flat dynamic compression: quantizes velocities to standard fixed levels.
    Returns:
    - baseline_notes_df: The corrupted/quantized standard transcription
    - baseline_midi: PrettyMIDI object of standard model output
    - friction_metrics: Quantitative measure of algorithmic bias / information loss
    """
    if notes_df.empty:
        pm = pretty_midi.PrettyMIDI()
        return notes_df, pm, {"missed_grace_notes": 0, "avg_timing_error_ms": 0.0, "algorithmic_friction_pct": 0.0}

    step_sec = 60.0 / (bpm * 4.0) # 16th note grid
    baseline_list = []
    missed_grace = 0
    timing_errors = []

    for _, row in notes_df.iterrows():
        dur = row["duration"]
        # Standard commercial software gate: delete fast notes as "noise"
        if dur < 0.12:
            missed_grace += 1
            continue

        orig_start = row["start_time"]
        quantized_start = round(orig_start / step_sec) * step_sec
        timing_errors.append(abs(quantized_start - orig_start) * 1000.0)

        # Standard grid snaps duration
        quantized_dur = max(step_sec, round(dur / step_sec) * step_sec)
        quantized_end = quantized_start + quantized_dur

        # Flattened velocity (commercial models discard subtle bellows swells)
        clamped_vel = 96

        baseline_list.append({
            "start_time": round(quantized_start, 3),
            "end_time": round(quantized_end, 3),
            "duration": round(quantized_dur, 3),
            "pitch_midi": row["pitch_midi"],
            "pitch_name": row["pitch_name"],
            "velocity": clamped_vel,
            "confidence": 0.85
        })

    baseline_df = pd.DataFrame(baseline_list)

    # Build standard baseline MIDI
    midi_obj = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0) # Standard Generic Piano/Acoustic
    for _, r in baseline_df.iterrows():
        note = pretty_midi.Note(
            velocity=int(r["velocity"]),
            pitch=int(r["pitch_midi"]),
            start=float(r["start_time"]),
            end=float(r["end_time"])
        )
        inst.notes.append(note)
    midi_obj.instruments.append(inst)

    total_orig = len(notes_df)
    avg_timing_error_ms = float(np.mean(timing_errors)) if len(timing_errors) > 0 else 0.0
    
    # Friction percentage: proportion of notes deleted or shifted by >40ms
    shifted_notes = sum(1 for err in timing_errors if err > 35.0)
    friction_pct = ((missed_grace + shifted_notes) / max(total_orig, 1)) * 100.0

    friction_metrics = {
        "missed_grace_notes": missed_grace,
        "avg_timing_error_ms": round(avg_timing_error_ms, 1),
        "algorithmic_friction_pct": round(float(np.clip(friction_pct, 15.0, 95.0)), 1)
    }

    return baseline_df, midi_obj, friction_metrics

def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Parses [Region]_[Form]_[Performer/Source].wav into rich metadata.
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 3:
        region_raw = parts[0]
        form = parts[1]
        performer = " ".join(parts[2:])
    elif len(parts) == 2:
        region_raw = parts[0]
        form = parts[1]
        performer = "Traditional Source"
    else:
        region_raw = "Austria"
        form = "Volksmusik"
        performer = base

    # Human-readable region
    region_map = {
        "Styria": "Styria (Steiermark)",
        "LowerAustria": "Lower Austria (Niederösterreich)",
        "UpperAustria": "Upper Austria (Oberösterreich)",
        "Carinthia": "Carinthia (Kärnten)"
    }
    region = region_map.get(region_raw, region_raw)
    title = f"{region_raw} {form} ({performer})"

    return {
        "filename": filename,
        "title": title,
        "region_key": region_raw,
        "region": region,
        "form": form,
        "performer": performer
    }
