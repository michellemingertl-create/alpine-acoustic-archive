"""
Synthesizer module for the 40-track Alpine Acoustic Archive corpus.
Generates 40 standardized Austrian folk accordion recordings across 4 regional traditions:
1. Styria (10 tracks) - Steirische Harmonika, Helikon contrabass, syncopated Boarischer & Ländler
2. Lower Austria (10 tracks) - Schrammel & tavern folk, flowing dialect waltzes, Danube phrasing
3. Upper Austria (10 tracks) - Salzkammergut Ländler, metric-shifting Zwiefacher, Schleuniger
4. Carinthia (10 tracks) - Vocal dialect tunes, Gailtaler Rundtanz, Balg-Tremolo ornamentation

Naming convention: [Region]_[Form]_[Performer/Source].wav
"""

import os
import numpy as np
import soundfile as sf
from typing import List, Tuple, Callable

SR = 22050

def midi_to_freq(midi_num: int) -> float:
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))

def generate_accordion_tone(freq: float, duration: float, sr: int = SR,
                            detune_hz: float = 1.8, bellows_env: np.ndarray = None,
                            is_bass: bool = False, tremolo_rate: float = 0.0) -> np.ndarray:
    """
    Synthesize an acoustic accordion reed tone using additive harmonic synthesis,
    dual-reed musette detune (Schwebung), bellows air pressure envelope, and optional Balg-Tremolo.
    """
    num_samples = int(duration * sr)
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.linspace(0, duration, num_samples, endpoint=False)

    if is_bass:
        harmonic_weights = [1.0, 0.85, 0.65, 0.45, 0.25, 0.15, 0.08]
    else:
        harmonic_weights = [1.0, 0.70, 0.50, 0.35, 0.20, 0.12, 0.08, 0.04]

    # Primary reed
    tone1 = np.zeros(num_samples, dtype=np.float32)
    for h_idx, weight in enumerate(harmonic_weights, start=1):
        h_freq = freq * h_idx
        if h_freq < sr / 2:
            tone1 += weight * np.sin(2 * np.pi * h_freq * t + 0.1 * h_idx)

    # Secondary detuned reed (Tremolo / Schwebung)
    if not is_bass and detune_hz > 0:
        detuned_freq = freq + detune_hz
        tone2 = np.zeros(num_samples, dtype=np.float32)
        for h_idx, weight in enumerate(harmonic_weights, start=1):
            h_freq = detuned_freq * h_idx
            if h_freq < sr / 2:
                tone2 += weight * 0.85 * np.sin(2 * np.pi * h_freq * t + 0.3 * h_idx)
        tone = 0.55 * tone1 + 0.45 * tone2
    else:
        tone = tone1

    # Subtle reed air turbulence / hiss
    noise = (np.random.rand(num_samples) - 0.5) * 0.012
    tone += noise

    # Key attack and release envelope (chiff / reed initiation)
    attack_len = min(int(0.02 * sr), num_samples // 4)
    release_len = min(int(0.03 * sr), num_samples // 4)
    reed_envelope = np.ones(num_samples, dtype=np.float32)
    if attack_len > 0:
        reed_envelope[:attack_len] = np.sin(np.linspace(0, np.pi / 2, attack_len))
    if release_len > 0:
        reed_envelope[-release_len:] = np.cos(np.linspace(0, np.pi / 2, release_len))

    # Apply Balg-Tremolo (bellow-shake micro-modulation) if requested
    if tremolo_rate > 0:
        tremolo_mod = 1.0 + 0.15 * np.sin(2 * np.pi * tremolo_rate * t)
        reed_envelope = reed_envelope * tremolo_mod

    # Apply external bellows pressure modulation if provided
    if bellows_env is not None and len(bellows_env) == num_samples:
        envelope = reed_envelope * bellows_env
    else:
        envelope = reed_envelope

    return (tone * envelope).astype(np.float32)

def synthesize_piece(notes_score: List[Tuple[float, float, int, int]],
                     chords_score: List[Tuple[float, float, List[int], int, bool]],
                     bpm: float,
                     bellows_pattern_func: Callable[[np.ndarray, float], np.ndarray],
                     sr: int = SR,
                     rubato_factor: float = 0.0,
                     tremolo_rate: float = 0.0) -> np.ndarray:
    """
    Renders melody and left-hand accompaniment with global bellows dynamics and microtiming rubato.
    notes_score: list of (start_beat, duration_beats, midi_pitch, velocity)
    chords_score: list of (start_beat, duration_beats, [midi_pitches], velocity, is_bass)
    """
    beat_sec = 60.0 / bpm
    total_beats = max(n[0] + n[1] for n in notes_score + chords_score) + 1.0
    total_samples = int(total_beats * beat_sec * sr)
    buffer = np.zeros(total_samples, dtype=np.float32)

    # Global time array for bellows pressure envelope
    t_global = np.linspace(0, total_beats * beat_sec, total_samples, endpoint=False)
    global_bellows = bellows_pattern_func(t_global, beat_sec)

    # Render melody (Right Hand - Diskant)
    for start_beat, dur_beats, midi_pitch, vel in notes_score:
        # Microtiming rubato: elastic timing deviations on weak or elongated beats
        beat_offset = 0.0
        if rubato_factor > 0:
            beat_pos = start_beat % 3.0
            if 0.9 < beat_pos < 2.1:
                beat_offset = rubato_factor * 0.06 * np.sin(start_beat)

        actual_start_sec = max(0.0, (start_beat * beat_sec) + beat_offset)
        start_samp = int(actual_start_sec * sr)
        dur_sec = dur_beats * beat_sec
        dur_samp = int(dur_sec * sr)

        if start_samp + dur_samp > total_samples:
            dur_samp = total_samples - start_samp
        if dur_samp <= 0:
            continue

        local_bellows = global_bellows[start_samp:start_samp + dur_samp]
        freq = midi_to_freq(midi_pitch)
        note_audio = generate_accordion_tone(freq, dur_sec, sr=sr,
                                             detune_hz=2.2, bellows_env=local_bellows,
                                             is_bass=False, tremolo_rate=tremolo_rate)
        buffer[start_samp:start_samp + len(note_audio)] += note_audio * (vel / 127.0) * 0.72

    # Render Left Hand (Bass & Chords - Begleitung)
    for start_beat, dur_beats, chord_pitches, vel, is_bass in chords_score:
        actual_start_sec = start_beat * beat_sec
        start_samp = int(actual_start_sec * sr)
        dur_sec = dur_beats * beat_sec
        dur_samp = int(dur_sec * sr)

        if start_samp + dur_samp > total_samples:
            dur_samp = total_samples - start_samp
        if dur_samp <= 0:
            continue

        local_bellows = global_bellows[start_samp:start_samp + dur_samp]
        for p in chord_pitches:
            freq = midi_to_freq(p)
            chord_audio = generate_accordion_tone(freq, dur_sec, sr=sr,
                                                  detune_hz=0.5 if not is_bass else 0.0,
                                                  bellows_env=local_bellows,
                                                  is_bass=is_bass)
            gain = 0.55 if is_bass else 0.28
            buffer[start_samp:start_samp + len(chord_audio)] += chord_audio * (vel / 127.0) * gain

    # Normalize audio buffer to -1.0 dBFS
    max_val = np.max(np.abs(buffer))
    if max_val > 0:
        buffer = (buffer / max_val) * 0.89

    return buffer


# ------------------------------------------------------------------
# Standard Bellows Dynamics Profiles
# ------------------------------------------------------------------
def bellows_laendler(t, beat_sec):
    """Deep 3-beat cyclic swell, emphasizing beat 2 with breathing rubato."""
    phase = (t / (3.0 * beat_sec)) * 2 * np.pi
    return 0.60 + 0.40 * np.sin(phase - np.pi / 4)

def bellows_polka(t, beat_sec):
    """Crisp staccato bellows bounces (Balgstoß) on 2/4 meter."""
    phase = (t / (2.0 * beat_sec)) * 2 * np.pi
    return 0.68 + 0.32 * np.abs(np.sin(phase))

def bellows_boarischer(t, beat_sec):
    """Sharp syncopated accents on beats 2 and 4 (characteristic Balgakzente)."""
    bar_t = (t % (4.0 * beat_sec)) / beat_sec
    accent = 0.28 * np.exp(-((bar_t - 1.0)**2) / 0.15) + 0.32 * np.exp(-((bar_t - 3.0)**2) / 0.15)
    return 0.62 + accent

def bellows_waltz(t, beat_sec):
    """Lyrical flowing arches with second-beat lilting emphasis."""
    phase = (t / (3.0 * beat_sec)) * 2 * np.pi
    return 0.62 + 0.38 * (0.5 * (1 + np.sin(phase - 0.4)))

def bellows_zwiefacher(t, beat_sec):
    """Asymmetric pressure shifts corresponding to alternating 3/4 and 2/4 bars."""
    cycle_t = (t % (10.0 * beat_sec)) / beat_sec
    phase_34 = (cycle_t / 3.0) * 2 * np.pi
    phase_24 = ((cycle_t - 6.0) / 2.0) * 2 * np.pi
    val_34 = 0.65 + 0.35 * np.sin(phase_34)
    val_24 = 0.78 + 0.22 * np.cos(phase_24)
    return np.where(cycle_t < 6.0, val_34, val_24)

def bellows_weis(t, beat_sec):
    """Slow expressive rubato mountain air with long sustained swells."""
    phase = (t / (6.0 * beat_sec)) * 2 * np.pi
    return 0.55 + 0.45 * (0.5 * (1 + np.sin(phase)))


# ------------------------------------------------------------------
# Musical Template Builders (Melody + Accompaniment Generators)
# ------------------------------------------------------------------
def build_34_piece(root_midi: int, melody_steps: List[Tuple[float, float, int, int]],
                   bpm: float, bellows_func, rubato: float = 0.0, tremolo: float = 0.0):
    """Constructs 3/4 Ländler or Waltz with standard bass-chord-chord accompaniment."""
    notes = [(sb, dur, root_midi + offset, vel) for sb, dur, offset, vel in melody_steps]
    total_bars = int(np.ceil(max(n[0] + n[1] for n in notes) / 3.0))

    chords = []
    # I - IV - V - I harmony cycling
    harmonies = [
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]), # I
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]), # I
        (root_midi - 19, [root_midi - 12, root_midi - 7, root_midi - 3]), # IV
        (root_midi - 22, [root_midi - 10, root_midi - 7, root_midi - 3]), # V7
        (root_midi - 22, [root_midi - 10, root_midi - 7, root_midi - 3]), # V7
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]), # I
    ]
    for b in range(total_bars):
        bass_pitch, chord_pitches = harmonies[b % len(harmonies)]
        bar_start = b * 3.0
        chords.append((bar_start, 0.75, [bass_pitch], 110, True))
        chords.append((bar_start + 1.0, 0.55, chord_pitches, 88, False))
        chords.append((bar_start + 2.0, 0.55, chord_pitches, 82, False))

    return synthesize_piece(notes, chords, bpm, bellows_func, rubato_factor=rubato, tremolo_rate=tremolo)

def build_24_piece(root_midi: int, melody_steps: List[Tuple[float, float, int, int]],
                   bpm: float, bellows_func, tremolo: float = 0.0):
    """Constructs 2/4 Polka or Rundtanz with crisp bass-chord accompaniment."""
    notes = [(sb, dur, root_midi + offset, vel) for sb, dur, offset, vel in melody_steps]
    total_bars = int(np.ceil(max(n[0] + n[1] for n in notes) / 2.0))

    chords = []
    harmonies = [
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]), # I
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]), # I
        (root_midi - 22, [root_midi - 10, root_midi - 7, root_midi - 3]), # V7
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]), # I
    ]
    for b in range(total_bars):
        bass_pitch, chord_pitches = harmonies[b % len(harmonies)]
        bar_start = b * 2.0
        chords.append((bar_start, 0.45, [bass_pitch], 115, True))
        chords.append((bar_start + 1.0, 0.40, chord_pitches, 95, False))

    return synthesize_piece(notes, chords, bpm, bellows_func, tremolo_rate=tremolo)

def build_44_piece(root_midi: int, melody_steps: List[Tuple[float, float, int, int]],
                   bpm: float, bellows_func):
    """Constructs 4/4 Boarischer with alternating bass and accented syncopated chords."""
    notes = [(sb, dur, root_midi + offset, vel) for sb, dur, offset, vel in melody_steps]
    total_bars = int(np.ceil(max(n[0] + n[1] for n in notes) / 4.0))

    chords = []
    harmonies = [
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]),
        (root_midi - 19, [root_midi - 12, root_midi - 7, root_midi - 3]),
        (root_midi - 22, [root_midi - 10, root_midi - 7, root_midi - 3]),
        (root_midi - 24, [root_midi - 12, root_midi - 8, root_midi - 5]),
    ]
    for b in range(total_bars):
        bass_pitch, chord_pitches = harmonies[b % len(harmonies)]
        bar_start = b * 4.0
        # Beat 1 Bass
        chords.append((bar_start, 0.45, [bass_pitch], 112, True))
        # Beat 2 Chord (accented)
        chords.append((bar_start + 1.0, 0.40, chord_pitches, 115, False))
        # Beat 3 Alternate Bass
        chords.append((bar_start + 2.0, 0.45, [bass_pitch + 7], 105, True))
        # Beat 4 Chord (accented)
        chords.append((bar_start + 3.0, 0.40, chord_pitches, 115, False))

    return synthesize_piece(notes, chords, bpm, bellows_func)

def build_zwiefacher_piece(root_midi: int, melody_steps: List[Tuple[float, float, int, int]], bpm: float):
    """Constructs alternating 3/4 (waltz) and 2/4 (dreher) Zwiefacher piece."""
    notes = [(sb, dur, root_midi + offset, vel) for sb, dur, offset, vel in melody_steps]
    chords = [
        # Bar 1 (3/4): I
        (0.0, 0.6, [root_midi - 24], 115, True), (1.0, 0.5, [root_midi-12, root_midi-8], 90, False), (2.0, 0.5, [root_midi-12, root_midi-8], 85, False),
        # Bar 2 (3/4): I
        (3.0, 0.6, [root_midi - 24], 115, True), (4.0, 0.5, [root_midi-12, root_midi-8], 90, False), (5.0, 0.5, [root_midi-12, root_midi-8], 85, False),
        # Bar 3 (2/4): V7
        (6.0, 0.5, [root_midi - 22], 115, True), (7.0, 0.45, [root_midi-10, root_midi-7], 95, False),
        # Bar 4 (2/4): I
        (8.0, 0.5, [root_midi - 24], 120, True), (9.0, 0.45, [root_midi-12, root_midi-8], 100, False),
        # Bar 5 (3/4): I
        (10.0, 0.6, [root_midi - 24], 115, True), (11.0, 0.5, [root_midi-12, root_midi-8], 90, False), (12.0, 0.5, [root_midi-12, root_midi-8], 85, False),
        # Bar 6 (3/4): V7
        (13.0, 0.6, [root_midi - 22], 115, True), (14.0, 0.5, [root_midi-10, root_midi-7], 90, False), (15.0, 0.5, [root_midi-10, root_midi-7], 85, False),
        # Bar 7 (2/4): V7
        (16.0, 0.5, [root_midi - 22], 115, True), (17.0, 0.45, [root_midi-10, root_midi-7], 95, False),
        # Bar 8 (2/4): I
        (18.0, 0.5, [root_midi - 24], 120, True), (19.0, 0.45, [root_midi-12, root_midi-8], 100, False),
    ]
    return synthesize_piece(notes, chords, bpm, bellows_zwiefacher)


# ------------------------------------------------------------------
# The 40 Corpus Track Definitions
# ------------------------------------------------------------------
def generate_all_40_tracks(target_dir: str = "data/raw_audio"):
    os.makedirs(target_dir, exist_ok=True)

    tracks = {}

    # Common melodic motives (relative pitch offsets in semitones)
    motive_laendler_1 = [
        (0.0, 1.0, 0, 100), (1.0, 1.0, 4, 110), (2.0, 1.0, 7, 115),
        (3.0, 1.5, 9, 115), (4.5, 0.5, 7, 105), (5.0, 1.0, 4, 95),
        (6.0, 1.0, 2, 100), (7.0, 1.0, 5, 110), (8.0, 1.0, 9, 115),
        (9.0, 2.8, 7, 120),
        (12.0, 1.0, 4, 105), (13.0, 1.0, 7, 115), (14.0, 1.0, 12, 125),
        (15.0, 1.5, 11, 115), (16.5, 0.5, 9, 105), (17.0, 1.0, 7, 95),
        (18.0, 1.0, 5, 100), (19.0, 1.0, 4, 95), (20.0, 1.0, 2, 90),
        (21.0, 2.9, 0, 110)
    ]

    motive_polka_1 = [
        (0.0, 0.5, 0, 110), (0.5, 0.5, 4, 100), (1.0, 0.5, 7, 115), (1.5, 0.5, 4, 95),
        (2.0, 0.5, -5, 105), (2.5, 0.5, 0, 100), (3.0, 0.5, 4, 110), (3.5, 0.5, 0, 90),
        (4.0, 0.5, 2, 115), (4.5, 0.5, 5, 105), (5.0, 0.5, 9, 120), (5.5, 0.5, 5, 100),
        (6.0, 0.7, 7, 125), (7.0, 0.7, 7, 125),
        (8.0, 0.5, 4, 115), (8.5, 0.5, 7, 105), (9.0, 0.5, 12, 125), (9.5, 0.5, 7, 100),
        (10.0, 0.5, 5, 115), (10.5, 0.5, 9, 105), (11.0, 0.5, 14, 125), (11.5, 0.5, 9, 100),
        (12.0, 0.5, 11, 110), (12.5, 0.5, 7, 100), (13.0, 0.5, 2, 95), (13.5, 0.5, -1, 90),
        (14.0, 1.8, 0, 120)
    ]

    motive_boarischer_1 = [
        (0.0, 0.5, 0, 100), (0.5, 0.5, 4, 105), (1.0, 1.0, 7, 120), (2.0, 0.5, 4, 100), (2.5, 0.5, 7, 105), (3.0, 1.0, 12, 125),
        (4.0, 0.5, 9, 110), (4.5, 0.5, 5, 100), (5.0, 1.0, 2, 115), (6.0, 1.0, -1, 105), (7.0, 1.0, 2, 110),
        (8.0, 0.5, 7, 105), (8.5, 0.5, 11, 110), (9.0, 1.0, 14, 125), (10.0, 0.5, 11, 105), (10.5, 0.5, 14, 110), (11.0, 1.0, 16, 125),
        (12.0, 1.0, 12, 125), (13.0, 0.5, 7, 105), (13.5, 0.5, 4, 100), (14.0, 1.8, 0, 120)
    ]

    motive_waltz_1 = [
        (0.0, 1.8, 4, 110), (2.0, 0.8, 5, 95),
        (3.0, 1.0, 7, 115), (4.0, 1.0, 4, 100), (5.0, 1.0, 0, 90),
        (6.0, 1.0, 2, 105), (7.0, 1.0, 5, 110), (8.0, 1.0, 9, 115),
        (9.0, 2.7, 7, 120),
        (12.0, 1.0, 12, 120), (13.0, 1.0, 11, 105), (14.0, 1.0, 9, 100),
        (15.0, 1.0, 7, 115), (16.0, 1.0, 4, 100), (17.0, 1.0, 0, 90),
        (18.0, 1.0, 2, 105), (19.0, 1.0, 7, 110), (20.0, 1.0, 11, 115),
        (21.0, 2.8, 0, 115)
    ]

    motive_zwiefacher_1 = [
        (0.0, 1.0, 0, 110), (1.0, 1.0, 4, 115), (2.0, 1.0, 7, 120),
        (3.0, 1.0, 9, 120), (4.0, 1.0, 7, 110), (5.0, 1.0, 4, 105),
        (6.0, 1.0, 5, 115), (7.0, 1.0, 2, 110),
        (8.0, 0.9, 0, 125), (9.0, 0.9, 0, 120),
        (10.0, 1.0, 4, 115), (11.0, 1.0, 7, 120), (12.0, 1.0, 12, 125),
        (13.0, 1.0, 11, 120), (14.0, 1.0, 7, 110), (15.0, 1.0, 2, 100),
        (16.0, 1.0, 4, 115), (17.0, 1.0, 2, 110),
        (18.0, 1.8, 0, 125)
    ]

    motive_weis_1 = [
        (0.0, 2.5, 0, 100), (2.5, 0.5, 2, 90), (3.0, 2.5, 4, 110), (5.5, 0.5, 5, 95),
        (6.0, 3.0, 7, 115), (9.0, 3.0, 12, 120),
        (12.0, 2.0, 9, 110), (14.0, 1.0, 7, 100), (15.0, 2.5, 4, 105), (17.5, 0.5, 2, 90),
        (18.0, 3.8, 0, 115)
    ]

    # Melodic variation helper
    def vary(motive, transpose=0, speed=1.0, add_grace=False):
        res = []
        for sb, dur, p, v in motive:
            p_new = p + transpose
            if add_grace and dur >= 1.0 and sb > 0:
                # Add rapid grace note inflection 0.1 beat prior
                res.append((sb * speed, 0.15, p_new - 1, 95))
                res.append(((sb + 0.15) * speed, (dur - 0.15) * speed, p_new, v))
            else:
                res.append((sb * speed, dur * speed, p_new, v))
        return res

    # ==================== REGION 1: STYRIA (10 Tracks) ====================
    tracks["Styria_Boarischer_StachlArchive.wav"] = build_44_piece(62, motive_boarischer_1, 102.0, bellows_boarischer)
    tracks["Styria_Laendler_MooserBuam.wav"] = build_34_piece(67, motive_laendler_1, 84.0, bellows_laendler, rubato=0.25)
    tracks["Styria_Polka_EnnstalTrio.wav"] = build_24_piece(60, motive_polka_1, 126.0, bellows_polka)
    tracks["Styria_Schleuniger_DachsteinBuam.wav"] = build_34_piece(69, vary(motive_laendler_1, 2, 0.8), 108.0, bellows_laendler)
    tracks["Styria_Walzer_AusseerMusi.wav"] = build_34_piece(65, vary(motive_waltz_1, 0), 136.0, bellows_waltz, rubato=0.15)
    tracks["Styria_Boarischer_GrazerDuo.wav"] = build_44_piece(67, vary(motive_boarischer_1, 5), 104.0, bellows_boarischer)
    tracks["Styria_Jodler_ErzherzogJohann.wav"] = build_34_piece(62, motive_weis_1, 68.0, bellows_weis, rubato=0.40)
    tracks["Styria_Tanz_SulmtalHarmonika.wav"] = build_24_piece(60, vary(motive_polka_1, 0, 0.95), 120.0, bellows_polka)
    tracks["Styria_Marsch_Hochsteirer.wav"] = build_44_piece(58, vary(motive_boarischer_1, -4), 112.0, bellows_boarischer)
    tracks["Styria_Polka_GleichenbergArchive.wav"] = build_24_piece(65, vary(motive_polka_1, 5), 130.0, bellows_polka)

    # ==================== REGION 2: LOWER AUSTRIA (10 Tracks) ====================
    tracks["LowerAustria_Schrammelwalzer_HeurigerTrio.wav"] = build_34_piece(65, motive_waltz_1, 138.0, bellows_waltz, rubato=0.20)
    tracks["LowerAustria_Polka_WachauKapelle.wav"] = build_24_piece(60, motive_polka_1, 122.0, bellows_polka)
    tracks["LowerAustria_Dialektwalzer_KremstalTrio.wav"] = build_34_piece(67, vary(motive_waltz_1, 2), 140.0, bellows_waltz, rubato=0.18)
    tracks["LowerAustria_Tanzlied_Wienerwald.wav"] = build_34_piece(62, vary(motive_laendler_1, -5), 92.0, bellows_laendler)
    tracks["LowerAustria_Marsch_MarchfeldTradition.wav"] = build_44_piece(58, vary(motive_boarischer_1, -4), 114.0, bellows_boarischer)
    tracks["LowerAustria_Laendler_WeinviertelDuo.wav"] = build_34_piece(65, vary(motive_laendler_1, -2), 86.0, bellows_laendler, rubato=0.22)
    tracks["LowerAustria_Walzer_DonautalEnsemble.wav"] = build_34_piece(60, vary(motive_waltz_1, -5), 142.0, bellows_waltz)
    tracks["LowerAustria_Polka_TullnerfeldBuam.wav"] = build_24_piece(67, vary(motive_polka_1, 7), 125.0, bellows_polka)
    tracks["LowerAustria_Schrammeltanz_GrinzingArchiv.wav"] = build_34_piece(62, vary(motive_waltz_1, -3), 134.0, bellows_waltz, rubato=0.25)
    tracks["LowerAustria_WeinstockWeis_SpitzArchive.wav"] = build_34_piece(65, motive_weis_1, 72.0, bellows_weis, rubato=0.35)

    # ==================== REGION 3: UPPER AUSTRIA (10 Tracks) ====================
    tracks["UpperAustria_Landler_SalzkammergutTrio.wav"] = build_34_piece(67, motive_laendler_1, 84.0, bellows_laendler, rubato=0.35)
    tracks["UpperAustria_Zwiefacher_HallstattEnsemble.wav"] = build_zwiefacher_piece(58, motive_zwiefacher_1, 144.0)
    tracks["UpperAustria_Schleuniger_Traunviertel.wav"] = build_34_piece(62, vary(motive_laendler_1, -5, 0.85), 110.0, bellows_laendler)
    tracks["UpperAustria_Almtanz_DachsteinEcho.wav"] = build_24_piece(60, vary(motive_polka_1, 0), 128.0, bellows_polka)
    tracks["UpperAustria_Landler_InnviertelMusi.wav"] = build_34_piece(65, vary(motive_laendler_1, -2), 82.0, bellows_laendler, rubato=0.38)
    tracks["UpperAustria_Zwiefacher_IschlerArchiv.wav"] = build_zwiefacher_piece(67, vary(motive_zwiefacher_1, 9), 148.0)
    tracks["UpperAustria_Walzer_AtterseeQuartett.wav"] = build_34_piece(62, vary(motive_waltz_1, -3), 136.0, bellows_waltz)
    tracks["UpperAustria_Polka_MuehlviertelDuo.wav"] = build_24_piece(60, vary(motive_polka_1, 0, 1.05), 124.0, bellows_polka)
    tracks["UpperAustria_Schleuniger_GosauBuam.wav"] = build_34_piece(69, vary(motive_laendler_1, 2, 0.8), 112.0, bellows_laendler)
    tracks["UpperAustria_Jodlertanz_Krippenstein.wav"] = build_34_piece(65, motive_weis_1, 70.0, bellows_weis, rubato=0.42)

    # ==================== REGION 4: CARINTHIA (10 Tracks) ====================
    tracks["Carinthia_Rundtanz_GailtalerQuartett.wav"] = build_24_piece(62, vary(motive_polka_1, 2, add_grace=True), 120.0, bellows_polka, tremolo=5.5)
    tracks["Carinthia_Liedwalzer_WoertherseeTrio.wav"] = build_34_piece(67, vary(motive_waltz_1, 2, add_grace=True), 132.0, bellows_waltz, rubato=0.28, tremolo=4.8)
    tracks["Carinthia_Polka_RosentalMusi.wav"] = build_24_piece(60, vary(motive_polka_1, 0, add_grace=True), 124.0, bellows_polka, tremolo=5.0)
    tracks["Carinthia_Weis_MoelltalArchiv.wav"] = build_34_piece(65, vary(motive_weis_1, 0, add_grace=True), 66.0, bellows_weis, rubato=0.45, tremolo=4.2)
    tracks["Carinthia_Tanz_DrautalEnsemble.wav"] = build_44_piece(58, vary(motive_boarischer_1, -4, add_grace=True), 100.0, bellows_boarischer)
    tracks["Carinthia_Liedwalzer_KlagenfurtTrio.wav"] = build_34_piece(62, vary(motive_waltz_1, -3, add_grace=True), 130.0, bellows_waltz, rubato=0.26)
    tracks["Carinthia_Polka_LavamuendDuo.wav"] = build_24_piece(67, vary(motive_polka_1, 7, add_grace=True), 128.0, bellows_polka, tremolo=5.2)
    tracks["Carinthia_Almweise_GlocknerEcho.wav"] = build_34_piece(60, vary(motive_weis_1, -5, add_grace=True), 64.0, bellows_weis, rubato=0.48)
    tracks["Carinthia_Rundtanz_GurktalKapelle.wav"] = build_24_piece(65, vary(motive_polka_1, 5, add_grace=True), 122.0, bellows_polka, tremolo=5.5)
    tracks["Carinthia_Jodlerwalzer_Karawanken.wav"] = build_34_piece(62, vary(motive_laendler_1, -5, add_grace=True), 80.0, bellows_laendler, rubato=0.32, tremolo=4.5)

    print(f"Synthesizing {len(tracks)} standardized archival tracks...")
    for filename, audio in tracks.items():
        filepath = os.path.join(target_dir, filename)
        sf.write(filepath, audio, SR, subtype='PCM_16')
        dur = len(audio) / SR
        print(f"Generated {filename} ({dur:.1f}s)")

    print(f"Successfully synthesized all {len(tracks)} corpus tracks into {target_dir}/")

if __name__ == "__main__":
    generate_all_40_tracks()
