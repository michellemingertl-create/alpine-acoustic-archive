"""
Audio Engine module for The Alpine Acoustic Archive.
Provides MIR feature extraction (bellows RMS dynamics, spectral centroid timbre),
polyphonic note transcription (Spotify basic-pitch), and data formatting.
"""

import os
import io
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from typing import Tuple, List, Dict, Any, Optional

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except Exception as e:
    BASIC_PITCH_AVAILABLE = False
    BASIC_PITCH_ERR = str(e)

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
    """
    Load an audio file into mono floating-point time series at standard sample rate.
    Supports file path or in-memory byte buffer.
    """
    y, sample_rate = librosa.load(filepath_or_bytes, sr=sr, mono=True)
    return y, sample_rate

def extract_bellows_dynamics(y: np.ndarray, sr: int = TARGET_SR,
                             hop_length: int = HOP_LENGTH,
                             frame_length: int = FRAME_LENGTH) -> pd.DataFrame:
    """
    Extracts acoustic bellows dynamics:
    1. RMS Energy: Reflects physical air pressure & volume swell (Balgführung).
    2. Spectral Centroid: Reflects acoustic brightness and multi-reed register shifts.
    3. Onset Envelope: Tracks percussive attack transients.
    Returns DataFrame: [timestamp, rms_energy, rms_normalized, spectral_centroid, onset_strength]
    """
    # 1. Root-Mean-Square Energy (Bellows Pressure)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize RMS between 0.0 and 1.0 for dynamic tracking
    rms_max = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_norm = rms / rms_max

    # 2. Spectral Centroid (Reed Timbre & Register Brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    # 3. Onset Strength Envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Time array corresponding to feature frames
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    df_dynamics = pd.DataFrame({
        "timestamp": np.round(times, 4),
        "rms_energy": np.round(rms, 6),
        "rms_normalized": np.round(rms_norm, 4),
        "spectral_centroid": np.round(centroid, 2),
        "onset_strength": np.round(onset_env[:len(times)], 4)
    })
    return df_dynamics

def transcribe_audio(audio_path: str,
                     output_midi_path: Optional[str] = None) -> Tuple[pd.DataFrame, pretty_midi.PrettyMIDI, bytes]:
    """
    Transcribes audio using Spotify's basic-pitch inference model.
    Extracts note events (start, end, pitch, velocity, duration) and MIDI object.
    Returns:
    - notes_df: DataFrame of transcribed notes
    - midi_obj: pretty_midi.PrettyMIDI object
    - midi_bytes: raw bytes of the MIDI file for direct browser download
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Run basic-pitch inference
    model_output, midi_obj, raw_note_events = predict(audio_path)

    # Process note events: each tuple is (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bend)
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

    # Sort chronologically, then by pitch
    notes_df = pd.DataFrame(notes_list)
    if not notes_df.empty:
        notes_df.sort_values(by=["start_time", "pitch_midi"], inplace=True)
        notes_df.reset_index(drop=True, inplace=True)
    else:
        notes_df = pd.DataFrame(columns=[
            "start_time", "end_time", "duration", "pitch_midi",
            "pitch_name", "velocity", "confidence"
        ])

    # Convert MIDI object to in-memory bytes
    midi_buffer = io.BytesIO()
    midi_obj.write(midi_buffer)
    midi_bytes = midi_buffer.getvalue()

    # Save to disk if requested
    if output_midi_path:
        os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
        with open(output_midi_path, "wb") as f:
            f.write(midi_bytes)

    return notes_df, midi_obj, midi_bytes

def compute_track_analysis(audio_path: str,
                           transcripts_dir: str = "data/transcripts") -> Dict[str, Any]:
    """
    Full pipeline execution for a given audio file.
    Caches MIDI and dynamics CSV into transcripts_dir.
    """
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(transcripts_dir, exist_ok=True)

    csv_path = os.path.join(transcripts_dir, f"{basename}_features.csv")
    midi_path = os.path.join(transcripts_dir, f"{basename}.mid")

    # Load audio
    y, sr = load_audio(audio_path)
    duration = float(len(y) / sr)

    # Extract bellows dynamics
    df_dynamics = extract_bellows_dynamics(y, sr)
    df_dynamics.to_csv(csv_path, index=False)

    # Transcribe notes via basic-pitch
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
    dominant_pitch = notes_df["pitch_name"].mode()[0] if total_notes > 0 else "N/A"

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
