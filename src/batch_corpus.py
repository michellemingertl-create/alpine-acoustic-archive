"""
Batch Feature Extraction & Corpus Processing Pipeline for The Alpine Acoustic Archive.
Iterates over all 40+ archival audio files in data/raw_audio/, extracts MIR features,
generates polyphonic MIDI & baseline comparison models, and outputs data/metadata_corpus.csv.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.audio_engine as engine

RAW_AUDIO_DIR = "data/raw_audio"
TRANSCRIPTS_DIR = "data/transcripts"
CORPUS_CSV = "data/metadata_corpus.csv"

def run_batch_extraction():
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav")))

    if not audio_files:
        print(f"No .wav files found in {RAW_AUDIO_DIR}")
        return

    print(f"Found {len(audio_files)} audio files. Running batch MIR feature extraction...")

    corpus_rows = []

    for idx, filepath in enumerate(audio_files, start=1):
        filename = os.path.basename(filepath)
        print(f"[{idx:02d}/{len(audio_files):02d}] Processing {filename}...")

        # Parse naming metadata
        meta = engine.parse_filename_metadata(filename)

        # Full analysis (cached or dynamic)
        res = engine.compute_track_analysis(filepath, transcripts_dir=TRANSCRIPTS_DIR)

        df_dynamics = res["dynamics_df"]
        notes_df = res["notes_df"]
        duration = res["duration"]
        onset_times = res["onset_times"]
        y, sr = engine.load_audio(filepath)

        # 1. Bellows Dynamics
        bellows_metrics = engine.compute_bellows_metrics(df_dynamics)

        # 2. Microtiming Rubato Variance
        rubato_metrics = engine.compute_rubato_metrics(onset_times, duration)

        # 3. Ornamentation Density
        ornament_metrics = engine.compute_ornamentation_metrics(notes_df, duration, y, sr)

        # 4. Standard Transcription Baseline & Algorithmic Friction
        baseline_df, baseline_midi, friction = engine.simulate_standard_transcription_baseline(notes_df)

        # Save baseline generic MIDI for split-view comparison
        basename = res["basename"]
        baseline_midi_path = os.path.join(TRANSCRIPTS_DIR, f"{basename}_baseline_generic.mid")
        baseline_midi.write(baseline_midi_path)

        corpus_rows.append({
            "filename": filename,
            "title": meta["title"],
            "region_key": meta["region_key"],
            "region": meta["region"],
            "form": meta["form"],
            "performer": meta["performer"],
            "duration_sec": round(duration, 2),
            "total_notes": res["total_notes"],
            "pitch_range": res["pitch_range"],
            "dominant_pitch": res["dominant_pitch"],
            "bellows_dynamic_range_db": bellows_metrics["bellows_dynamic_range_db"],
            "bellows_modulation_index": bellows_metrics["bellows_modulation_index"],
            "rubato_variance_score": rubato_metrics["rubato_variance_score"],
            "ornamentation_density_per_min": ornament_metrics["ornamentation_density_per_min"],
            "grace_note_count": ornament_metrics["grace_note_count"],
            "missed_grace_notes": friction["missed_grace_notes"],
            "avg_timing_error_ms": friction["avg_timing_error_ms"],
            "algorithmic_friction_pct": friction["algorithmic_friction_pct"]
        })

    df_corpus = pd.DataFrame(corpus_rows)
    df_corpus.to_csv(CORPUS_CSV, index=False)
    print(f"\n=======================================================")
    print(f"Batch processing complete! Saved {len(df_corpus)} tracks to {CORPUS_CSV}")
    print(f"=======================================================")

    # Summary by Region
    if "region_key" in df_corpus.columns:
        summary = df_corpus.groupby("region_key").agg({
            "total_notes": "sum",
            "bellows_dynamic_range_db": "mean",
            "rubato_variance_score": "mean",
            "ornamentation_density_per_min": "mean",
            "algorithmic_friction_pct": "mean"
        }).round(2)
        print("\nRegional Corpus Benchmark Summary:")
        print(summary)

if __name__ == "__main__":
    run_batch_extraction()
