"""
Synthesizer module for authentic Alpine accordion repertoire.
Generates 5 curated Austrian folk accordion recordings:
1. Kitzbueheler Laendler (3/4, Tyrol) - Heavy bellows swelling, musette beating
2. Zillertaler Polka (2/4, Tyrol) - Staccato bellows bounce, crisp bass-chord
3. Wachauer Dialektwalzer (3/4, Lower Austria) - Lyrical lilting bellows phrasing
4. Steirischer Boarischer (4/4, Styria) - Syncopated Balgakzente (bellows accents)
5. Salzkammergut Zwiefacher (Alternating 3/4 & 2/4) - Metric shifts & erratic dynamics
"""

import os
import numpy as np
import soundfile as sf

SR = 22050

def midi_to_freq(midi_num: int) -> float:
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))

def generate_accordion_tone(freq: float, duration: float, sr: int = SR,
                            detune_hz: float = 1.8, bellows_env: np.ndarray = None,
                            is_bass: bool = False) -> np.ndarray:
    """
    Synthesize an acoustic accordion reed tone using additive harmonic synthesis,
    dual-reed musette detune (Schwebung), and bellows air pressure envelope.
    """
    num_samples = int(duration * sr)
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Harmonics profile for free reeds (rich odd & even harmonics)
    if is_bass:
        harmonic_weights = [1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08]
    else:
        harmonic_weights = [1.0, 0.65, 0.5, 0.35, 0.2, 0.12, 0.08, 0.04]

    # Primary reed
    tone1 = np.zeros(num_samples, dtype=np.float32)
    for h_idx, weight in enumerate(harmonic_weights, start=1):
        h_freq = freq * h_idx
        if h_freq < sr / 2:
            tone1 += weight * np.sin(2 * np.pi * h_freq * t + 0.1 * h_idx)

    # Secondary detuned reed (Tremolo / Schwebung)
    tone2 = np.zeros(num_samples, dtype=np.float32)
    if not is_bass and detune_hz > 0:
        detuned_freq = freq + detune_hz
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
    attack_len = min(int(0.025 * sr), num_samples // 4)
    release_len = min(int(0.035 * sr), num_samples // 4)
    reed_envelope = np.ones(num_samples, dtype=np.float32)
    if attack_len > 0:
        reed_envelope[:attack_len] = np.sin(np.linspace(0, np.pi / 2, attack_len))
    if release_len > 0:
        reed_envelope[-release_len:] = np.cos(np.linspace(0, np.pi / 2, release_len))

    # Apply external bellows pressure modulation if provided
    if bellows_env is not None and len(bellows_env) == num_samples:
        envelope = reed_envelope * bellows_env
    else:
        envelope = reed_envelope

    return (tone * envelope).astype(np.float32)

def synthesize_piece(notes_score, chords_score, bpm: float, total_bars: int,
                     bellows_pattern_func, sr: int = SR) -> np.ndarray:
    """
    Renders melody and left-hand bass/chord accompaniment with global bellows dynamics.
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
        start_samp = int(start_beat * beat_sec * sr)
        dur_sec = dur_beats * beat_sec
        dur_samp = int(dur_sec * sr)
        if start_samp + dur_samp > total_samples:
            dur_samp = total_samples - start_samp
        if dur_samp <= 0:
            continue

        local_bellows = global_bellows[start_samp:start_samp + dur_samp]
        freq = midi_to_freq(midi_pitch)
        note_audio = generate_accordion_tone(freq, dur_sec, sr=sr,
                                             detune_hz=2.0, bellows_env=local_bellows,
                                             is_bass=False)
        buffer[start_samp:start_samp + len(note_audio)] += note_audio * (vel / 127.0) * 0.7

    # Render Left Hand (Bass & Chords - Begleitung)
    for start_beat, dur_beats, chord_pitches, vel, is_bass in chords_score:
        start_samp = int(start_beat * beat_sec * sr)
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
            gain = 0.5 if is_bass else 0.25
            buffer[start_samp:start_samp + len(chord_audio)] += chord_audio * (vel / 127.0) * gain

    # Normalize to avoid clipping
    max_val = np.max(np.abs(buffer))
    if max_val > 0:
        buffer = (buffer / max_val) * 0.88

    return buffer


# --- Piece 1: Kitzbueheler Laendler (3/4 time, BPM 84) ---
def make_kitzbueheler_laendler():
    bpm = 84.0
    def bellows(t, beat_sec):
        # Heavy 3-beat cyclic swell, peak on beat 2 with rubato breathing
        phase = (t / (3.0 * beat_sec)) * 2 * np.pi
        return 0.65 + 0.35 * np.sin(phase - np.pi/4)

    # G major melody
    notes = [
        # Bar 1: G4 - B4 - D5
        (0.0, 1.0, 67, 100), (1.0, 1.0, 71, 110), (2.0, 1.0, 74, 115),
        # Bar 2: E5 - D5 - B4
        (3.0, 1.5, 76, 115), (4.5, 0.5, 74, 105), (5.0, 1.0, 71, 95),
        # Bar 3: A4 - C5 - E5
        (6.0, 1.0, 69, 100), (7.0, 1.0, 72, 110), (8.0, 1.0, 76, 115),
        # Bar 4: D5 sustained
        (9.0, 2.8, 74, 120),
        # Bar 5: B4 - D5 - G5
        (12.0, 1.0, 71, 105), (13.0, 1.0, 74, 115), (14.0, 1.0, 79, 125),
        # Bar 6: F#5 - E5 - D5
        (15.0, 1.5, 78, 115), (16.5, 0.5, 76, 105), (17.0, 1.0, 74, 95),
        # Bar 7: C5 - B4 - A4
        (18.0, 1.0, 72, 100), (19.0, 1.0, 71, 95), (20.0, 1.0, 69, 90),
        # Bar 8: G4 cadence
        (21.0, 2.9, 67, 110)
    ]
    # Standard Alpine 3/4 Begleitung: Bass on 1, Chords on 2 and 3
    chords = []
    harmony_prog = [
        (43, [55, 59, 62]), # G
        (43, [55, 59, 62]), # G
        (38, [54, 57, 62]), # D
        (38, [54, 57, 62]), # D
        (43, [55, 59, 62]), # G
        (43, [55, 59, 62]), # G
        (38, [54, 57, 62]), # D
        (43, [55, 59, 62]), # G
    ]
    for bar_idx, (bass_pitch, chord_pitches) in enumerate(harmony_prog):
        b_start = bar_idx * 3.0
        # Bass on 1
        chords.append((b_start, 0.8, [bass_pitch], 110, True))
        # Chord on 2
        chords.append((b_start + 1.0, 0.6, chord_pitches, 90, False))
        # Chord on 3
        chords.append((b_start + 2.0, 0.6, chord_pitches, 85, False))

    return synthesize_piece(notes, chords, bpm, 8, bellows)


# --- Piece 2: Zillertaler Polka (2/4 time, BPM 124) ---
def make_zillertaler_polka():
    bpm = 124.0
    def bellows(t, beat_sec):
        # Crisp bouncy 2-beat bellows
        phase = (t / (2.0 * beat_sec)) * 2 * np.pi
        return 0.70 + 0.30 * np.abs(np.sin(phase))

    # C Major brisk polka melody
    notes = [
        # Bar 1: C5 - E5 - G5 - E5
        (0.0, 0.5, 72, 110), (0.5, 0.5, 76, 100), (1.0, 0.5, 79, 115), (1.5, 0.5, 76, 95),
        # Bar 2: G4 - C5 - E5 - C5
        (2.0, 0.5, 67, 105), (2.5, 0.5, 72, 100), (3.0, 0.5, 76, 110), (3.5, 0.5, 72, 90),
        # Bar 3: D5 - F5 - A5 - F5
        (4.0, 0.5, 74, 115), (4.5, 0.5, 77, 105), (5.0, 0.5, 81, 120), (5.5, 0.5, 77, 100),
        # Bar 4: G5 - G5 staccato stop
        (6.0, 0.7, 79, 125), (7.0, 0.7, 79, 125),
        # Bar 5: E5 - G5 - C6 - G5
        (8.0, 0.5, 76, 115), (8.5, 0.5, 79, 105), (9.0, 0.5, 84, 125), (9.5, 0.5, 79, 100),
        # Bar 6: F5 - A5 - D6 - A5
        (10.0, 0.5, 77, 115), (10.5, 0.5, 81, 105), (11.0, 0.5, 86, 125), (11.5, 0.5, 81, 100),
        # Bar 7: B5 - G5 - D5 - B4
        (12.0, 0.5, 83, 110), (12.5, 0.5, 79, 100), (13.0, 0.5, 74, 95), (13.5, 0.5, 71, 90),
        # Bar 8: C5 final
        (14.0, 1.8, 72, 120)
    ]
    # 2/4 Begleitung: Bass on 1, Chord on 2
    chords = []
    harmony_prog = [
        (48, [60, 64, 67]), # C
        (48, [60, 64, 67]), # C
        (43, [59, 62, 67]), # G7
        (43, [59, 62, 67]), # G7
        (48, [60, 64, 67]), # C
        (41, [57, 60, 65]), # F
        (43, [59, 62, 67]), # G7
        (48, [60, 64, 67]), # C
    ]
    for bar_idx, (bass_pitch, chord_pitches) in enumerate(harmony_prog):
        b_start = bar_idx * 2.0
        # Bass on 1
        chords.append((b_start, 0.45, [bass_pitch], 115, True))
        # Staccato chord on 2
        chords.append((b_start + 1.0, 0.4, chord_pitches, 100, False))

    return synthesize_piece(notes, chords, bpm, 8, bellows)


# --- Piece 3: Wachauer Dialektwalzer (3/4 time, BPM 138) ---
def make_wachauer_dialektwalzer():
    bpm = 138.0
    def bellows(t, beat_sec):
        # Lilting Viennese/Wachau waltz wave with subtle swell on beat 2
        phase = (t / (3.0 * beat_sec)) * 2 * np.pi
        return 0.60 + 0.40 * (0.5 * (1 + np.sin(phase - 0.5)))

    # F Major melody
    notes = [
        # Bar 1: A4 (long) - Bb4 - B4
        (0.0, 1.8, 69, 110), (2.0, 0.8, 70, 95),
        # Bar 2: C5 - A4 - F4
        (3.0, 1.0, 72, 115), (4.0, 1.0, 69, 100), (5.0, 1.0, 65, 90),
        # Bar 3: G4 - Bb4 - D5
        (6.0, 1.0, 67, 105), (7.0, 1.0, 70, 110), (8.0, 1.0, 74, 115),
        # Bar 4: C5 (sustained)
        (9.0, 2.7, 72, 120),
        # Bar 5: F5 - E5 - D5
        (12.0, 1.0, 77, 120), (13.0, 1.0, 76, 105), (14.0, 1.0, 74, 100),
        # Bar 6: C5 - A4 - F4
        (15.0, 1.0, 72, 115), (16.0, 1.0, 69, 100), (17.0, 1.0, 65, 90),
        # Bar 7: G4 - C5 - E5
        (18.0, 1.0, 67, 105), (19.0, 1.0, 72, 110), (20.0, 1.0, 76, 115),
        # Bar 8: F4
        (21.0, 2.8, 65, 115)
    ]
    chords = []
    harmony_prog = [
        (41, [53, 57, 60]), # F
        (41, [53, 57, 60]), # F
        (48, [58, 62, 65]), # C7
        (48, [58, 62, 65]), # C7
        (41, [53, 57, 60]), # F
        (41, [53, 57, 60]), # F
        (48, [58, 62, 65]), # C7
        (41, [53, 57, 60]), # F
    ]
    for bar_idx, (bass_pitch, chord_pitches) in enumerate(harmony_prog):
        b_start = bar_idx * 3.0
        chords.append((b_start, 0.7, [bass_pitch], 110, True))
        chords.append((b_start + 1.0, 0.5, chord_pitches, 85, False))
        chords.append((b_start + 2.0, 0.5, chord_pitches, 80, False))

    return synthesize_piece(notes, chords, bpm, 8, bellows)


# --- Piece 4: Steirischer Boarischer (4/4 time, BPM 102) ---
def make_steirischer_boarischer():
    bpm = 102.0
    def bellows(t, beat_sec):
        # Sharp accents on beats 2 and 4 (characteristic Balgakzente)
        bar_t = (t % (4.0 * beat_sec)) / beat_sec
        # Accents near 1.0 and 3.0
        accent = 0.25 * np.exp(-((bar_t - 1.0)**2) / 0.15) + 0.3 * np.exp(-((bar_t - 3.0)**2) / 0.15)
        return 0.65 + accent

    # D major Boarischer
    notes = [
        # Bar 1: D4, F#4, A4, F#4 with syncopation
        (0.0, 0.5, 62, 100), (0.5, 0.5, 66, 105), (1.0, 1.0, 69, 120), (2.0, 0.5, 66, 100), (2.5, 0.5, 69, 105), (3.0, 1.0, 74, 125),
        # Bar 2: B4, G4, E4, C#4
        (4.0, 0.5, 71, 110), (4.5, 0.5, 67, 100), (5.0, 1.0, 64, 115), (6.0, 1.0, 61, 105), (7.0, 1.0, 64, 110),
        # Bar 3: A4, C#5, E5, C#5
        (8.0, 0.5, 69, 105), (8.5, 0.5, 73, 110), (9.0, 1.0, 76, 125), (10.0, 0.5, 73, 105), (10.5, 0.5, 76, 110), (11.0, 1.0, 78, 125),
        # Bar 4: D5 cadential stop
        (12.0, 1.0, 74, 125), (13.0, 0.5, 69, 105), (13.5, 0.5, 66, 100), (14.0, 1.8, 62, 120)
    ]
    chords = []
    harmony_prog = [
        (38, [57, 62, 66]), # D
        (43, [59, 64, 67]), # G
        (45, [57, 61, 64]), # A
        (38, [57, 62, 66]), # D
    ]
    for bar_idx, (bass_pitch, chord_pitches) in enumerate(harmony_prog):
        b_start = bar_idx * 4.0
        # Beat 1 Bass
        chords.append((b_start, 0.45, [bass_pitch], 115, True))
        # Beat 2 Chord (accented)
        chords.append((b_start + 1.0, 0.4, chord_pitches, 110, False))
        # Beat 3 Bass (alternate fifth)
        chords.append((b_start + 2.0, 0.45, [bass_pitch + 7], 105, True))
        # Beat 4 Chord (accented)
        chords.append((b_start + 3.0, 0.4, chord_pitches, 115, False))

    return synthesize_piece(notes, chords, bpm, 4, bellows)


# --- Piece 5: Salzkammergut Zwiefacher (Metric Shift: 3/4, 3/4, 2/4, 2/4) ---
def make_salzkammergut_zwiefacher():
    bpm = 144.0
    # Measure lengths in beats: 3, 3, 2, 2 = 10 beats per cycle
    def bellows(t, beat_sec):
        cycle_t = (t % (10.0 * beat_sec)) / beat_sec
        phase_34 = (cycle_t / 3.0) * 2 * np.pi
        phase_24 = ((cycle_t - 6.0) / 2.0) * 2 * np.pi
        val_34 = 0.65 + 0.35 * np.sin(phase_34)
        val_24 = 0.75 + 0.25 * np.cos(phase_24)
        return np.where(cycle_t < 6.0, val_34, val_24)

    # Bb Major melody
    notes = [
        # Bar 1 (3/4): Bb4 - D5 - F5
        (0.0, 1.0, 70, 110), (1.0, 1.0, 74, 115), (2.0, 1.0, 77, 120),
        # Bar 2 (3/4): G5 - F5 - D5
        (3.0, 1.0, 79, 120), (4.0, 1.0, 77, 110), (5.0, 1.0, 74, 105),
        # Bar 3 (2/4): Eb5 - C5
        (6.0, 1.0, 75, 115), (7.0, 1.0, 72, 110),
        # Bar 4 (2/4): Bb4 - Bb4
        (8.0, 0.9, 70, 125), (9.0, 0.9, 70, 120),
        # Repeat cycle with variation:
        # Bar 5 (3/4): D5 - F5 - Bb5
        (10.0, 1.0, 74, 115), (11.0, 1.0, 77, 120), (12.0, 1.0, 82, 125),
        # Bar 6 (3/4): A5 - F5 - C5
        (13.0, 1.0, 81, 120), (14.0, 1.0, 77, 110), (15.0, 1.0, 72, 100),
        # Bar 7 (2/4): D5 - C5
        (16.0, 1.0, 74, 115), (17.0, 1.0, 72, 110),
        # Bar 8 (2/4): Bb4 cadence
        (18.0, 1.8, 70, 125)
    ]

    chords = [
        # Bar 1 (3/4): Bb
        (0.0, 0.6, [46], 115, True), (1.0, 0.5, [58, 62, 65], 90, False), (2.0, 0.5, [58, 62, 65], 85, False),
        # Bar 2 (3/4): Bb
        (3.0, 0.6, [46], 115, True), (4.0, 0.5, [58, 62, 65], 90, False), (5.0, 0.5, [58, 62, 65], 85, False),
        # Bar 3 (2/4): F7
        (6.0, 0.5, [41], 115, True), (7.0, 0.45, [57, 60, 63], 95, False),
        # Bar 4 (2/4): Bb
        (8.0, 0.5, [46], 120, True), (9.0, 0.45, [58, 62, 65], 100, False),
        # Bar 5 (3/4): Bb
        (10.0, 0.6, [46], 115, True), (11.0, 0.5, [58, 62, 65], 90, False), (12.0, 0.5, [58, 62, 65], 85, False),
        # Bar 6 (3/4): F7
        (13.0, 0.6, [41], 115, True), (14.0, 0.5, [57, 60, 63], 90, False), (15.0, 0.5, [57, 60, 63], 85, False),
        # Bar 7 (2/4): F7
        (16.0, 0.5, [41], 115, True), (17.0, 0.45, [57, 60, 63], 95, False),
        # Bar 8 (2/4): Bb
        (18.0, 0.5, [46], 120, True), (19.0, 0.45, [58, 62, 65], 100, False),
    ]

    return synthesize_piece(notes, chords, bpm, 8, bellows)


def generate_all_tracks(target_dir: str = "data/raw_audio"):
    os.makedirs(target_dir, exist_ok=True)
    tracks = [
        ("kitzbueheler_laendler.wav", make_kitzbueheler_laendler()),
        ("zillertaler_polka.wav", make_zillertaler_polka()),
        ("wachauer_dialektwalzer.wav", make_wachauer_dialektwalzer()),
        ("steirischer_boarischer.wav", make_steirischer_boarischer()),
        ("salzkammergut_zwiefacher.wav", make_salzkammergut_zwiefacher()),
    ]
    for filename, audio in tracks:
        path = os.path.join(target_dir, filename)
        sf.write(path, audio, SR, subtype='PCM_16')
        dur = len(audio) / SR
        print(f"Generated {filename} ({dur:.1f}s) -> {path}")

if __name__ == "__main__":
    generate_all_tracks()
