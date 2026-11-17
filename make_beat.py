"""
"Heavy Heart" - original love-drill style beat.
F minor, 142 BPM, 64 bars (~1:50).

Run:  python make_beat.py
Makes: output/heavy_heart_beat.wav + .mp3, output/stems/*.wav, output/midi/*.mid
"""
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
BPM = 142
STEP = 60 / BPM / 4          # one 16th note in seconds
BAR_STEPS = 16
PPQ = 480
TICKS_PER_STEP = PPQ // 4

ROOT = Path(__file__).parent
OUT = ROOT / "output"
STEMS = OUT / "stems"
MIDI = OUT / "midi"
TMP = OUT / "_render"
FLUIDSYNTH = r"C:\Users\osage\music-tools\fluidsynth\fluidsynth-v2.6.2-win10-x64-cpp11\bin\fluidsynth.exe"
SOUNDFONT = r"C:\Users\osage\music-tools\soundfonts\GeneralUser-GS.sf2"

rng = np.random.default_rng(7)

# mix levels (dB RMS in the hook) - turn these up/down to taste
# music and 808 are set by average loudness (RMS dB), drums by peak level (dB)
MIX = {"music": -17, "808": -14.5, "kick_peak": -10, "snare_peak": -8.5, "hats_peak": -13}
MASTER_RMS_DB = -10.5   # final loudness of the hook

# ----------------------------------------------------------------------------
# Song structure
# ----------------------------------------------------------------------------
SECTIONS = [("intro", 4), ("hook", 8), ("verse", 16), ("hook", 8),
            ("verse2", 16), ("hook", 8), ("outro", 4)]
BARS = []  # (section, bar_in_section, section_length)
for name, n in SECTIONS:
    for i in range(n):
        BARS.append((name, i, n))
TOTAL_BARS = len(BARS)
TAIL = 3.0
TOTAL_SEC = TOTAL_BARS * BAR_STEPS * STEP + TAIL
N = int(TOTAL_SEC * SR)

# ----------------------------------------------------------------------------
# Harmony: Fm9 -> Dbmaj9 -> Bbm(add9) -> C7b9   (one chord per bar)
# ----------------------------------------------------------------------------
KEYS = [[41, 53, 56, 60, 67],      # Fm9
        [37, 53, 56, 60, 63],      # Dbmaj9
        [46, 53, 56, 61, 72],      # Bbm add9
        [48, 52, 58, 61, 67]]      # C7b9
CHOIR = [[65, 68, 72], [65, 68, 72], [65, 68, 73], [64, 70, 73]]
ROOT_808 = [29, 37, 34, 36]        # F1, Db2, Bb1, C2

# melody: (step, midi, length) per chord bar; 3-3-4-2-4 drill rhythm
MELODY = [
    [(0, 72, 3), (3, 75, 3), (6, 77, 4), (10, 75, 2), (12, 72, 4)],
    [(0, 77, 3), (3, 80, 3), (6, 77, 4), (10, 75, 2), (12, 73, 4)],
    [(0, 73, 3), (3, 77, 3), (6, 80, 3), (9, 80, 3), (12, 77, 4)],
    [(0, 76, 3), (3, 79, 3), (6, 79, 4), (10, 76, 2), (12, 72, 4)],
]


def is_drop_bar(b):
    """Last bar before a hook: drums cut out on beats 3-4."""
    return b + 1 < TOTAL_BARS and BARS[b + 1][0] == "hook" and BARS[b + 1][1] == 0


# ----------------------------------------------------------------------------
# Note lists (shared by MIDI export and synth)
# ----------------------------------------------------------------------------
def build_parts():
    keys, melody, bells, choir = [], [], [], []
    kick, snare, ghost, hat, bass = [], [], [], [], []
    for b, (sec, i, n) in enumerate(BARS):
        s0 = b * BAR_STEPS
        c = b % 4
        is_hook = sec == "hook"
        is_verse = sec.startswith("verse")

        # --- keys ---
        if sec in ("intro", "outro") or is_hook:
            for k, note in enumerate(KEYS[c]):
                keys.append((s0, 14, note, 74 if k else 66))
            for note in KEYS[c][2:]:
                keys.append((s0 + 10, 6, note, 52))
        else:  # verse: softer chord + quiet arpeggio leaving room for vocals
            for k, note in enumerate(KEYS[c]):
                keys.append((s0, 16, note, 60 if k else 54))
            arp = KEYS[c][1:]
            for j, st in enumerate([3, 6, 10, 12]):
                keys.append((s0 + st, 3, arp[j % len(arp)] + 12, 40))

        # --- melody ---
        play_mel = is_hook or sec in ("intro", "outro") or (is_verse and (i // 4) % 2 == 1 and sec == "verse2")
        if play_mel:
            vel = 84 if is_hook else 64
            for st, note, ln in MELODY[c]:
                melody.append((s0 + st, ln, note, vel))
                if is_hook:
                    bells.append((s0 + st, ln, note + 12, 50))

        # --- choir pad ---
        if is_hook or sec in ("intro", "outro"):
            for note in CHOIR[c]:
                choir.append((s0, 16, note, 58 if is_hook else 48))

        # --- drums & 808 ---
        drums_on = is_hook or is_verse
        if sec == "intro" and i == n - 1:   # hat lead-in on last intro bar
            for st in range(8, 16, 2):
                hat.append((s0 + st, 0.35 + 0.08 * (st - 8) / 2))
        if not drums_on:
            if sec == "outro" and i == 0:
                bass.append((s0, 32, ROOT_808[c], False))
            continue

        drop = is_drop_bar(b)
        limit = 8 if drop else 16
        odd = b % 2 == 1
        busy = sec == "verse2" or is_hook

        kick_steps = [0, 10] if odd else [0]
        kick += [s0 + st for st in kick_steps if st < limit]
        if 8 < limit:
            snare.append(s0 + 8)
        if odd and limit == 16:
            ghost.append((s0 + 14, 0.45))
            if b % 4 == 3:
                ghost.append((s0 + 15, 0.35))

        # hats: 3-3-2 drill bounce, rolls on odd bars
        hat_a = [0, 3, 6, 8, 11, 14]
        hat_b = [0, 3, 6, 8, 10]
        for st in (hat_b if odd else hat_a):
            if st < limit:
                hat.append((s0 + st, 0.9 if st in (0, 8) else 0.65))
        if odd and limit == 16:
            if busy and b % 4 == 3:   # 32nd-note roll
                for k in range(8):
                    hat.append((s0 + 12 + k * 0.5, 0.4 + 0.06 * k))
            else:                     # triplet roll
                for k in range(6):
                    hat.append((s0 + 12 + k * 4 / 6, 0.45 + 0.07 * k))

        # 808 with slides (skip bar 8 of verses for a breather)
        if is_verse and i == 7:
            continue
        r = ROOT_808[c]
        nxt = ROOT_808[(c + 1) % 4]
        if not odd:
            pat = [(0, r, 6, False), (6, r, 4, False), (10, r + 12, 6, True)]
        else:
            pat = [(0, r, 3, False), (3, r, 4, False), (7, r, 5, False), (12, nxt, 4, True)]
        for st, note, ln, glide in pat:
            if st < limit:
                bass.append((s0 + st, min(ln, limit - st), note, glide))
    return dict(keys=keys, melody=melody, bells=bells, choir=choir,
                kick=kick, snare=snare, ghost=ghost, hat=hat, bass=bass)


# ----------------------------------------------------------------------------
# MIDI writing (no external libraries)
# ----------------------------------------------------------------------------
def vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def mtrk(events, name):
    """events: list of (tick, order, bytes)"""
    nm = name.encode()
    events = [(0, -1, b"\xff\x03" + vlq(len(nm)) + nm)] + events
    events.sort(key=lambda e: (e[0], e[1]))
    data, last = b"", 0
    for t, _, ev in events:
        data += vlq(t - last) + ev
        last = t
    data += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + data


def note_events(notes, ch, program=None):
    ev = []
    if program is not None:
        ev.append((0, -1, bytes([0xC0 | ch, program])))
    for st, ln, note, vel in notes:
        t0 = int(round(st * TICKS_PER_STEP))
        t1 = int(round((st + ln) * TICKS_PER_STEP)) - 1
        ev.append((t0, 1, bytes([0x90 | ch, note, vel])))
        ev.append((t1, 0, bytes([0x80 | ch, note, 0])))
    return ev


def tempo_track(end_tick):
    us = int(60_000_000 / BPM)
    ev = [(0, 0, b"\xff\x51\x03" + us.to_bytes(3, "big")),
          (0, 0, b"\xff\x58\x04\x04\x02\x18\x08"),
          (end_tick, 9, b"\xff\x01\x03end")]
    return mtrk(ev, "Tempo")


def write_midi(path, tracks):
    end_tick = int((TOTAL_BARS * BAR_STEPS * STEP + TAIL) / STEP * TICKS_PER_STEP)
    chunks = [tempo_track(end_tick)] + tracks
    hdr = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), PPQ)
    path.write_bytes(hdr + b"".join(chunks))


# ----------------------------------------------------------------------------
# Synth helpers
# ----------------------------------------------------------------------------
def t_axis(sec):
    return np.arange(int(sec * SR)) / SR


def bandpass(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def highpass(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x, axis=0)


def lowpass(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x, axis=0)


def make_kick():
    t = t_axis(0.35)
    f = 48 + 170 * np.exp(-t / 0.028)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t / 0.16)
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.002) * 0.4
    return np.tanh(1.6 * (x + click))


def make_snare():
    t = t_axis(0.35)
    tone = np.sin(2 * np.pi * 195 * t) * np.exp(-t / 0.045) * 0.6
    noise = bandpass(rng.standard_normal(len(t)), 1200, 9000) * np.exp(-t / 0.11)
    clap = np.zeros(len(t))
    burst = bandpass(rng.standard_normal(len(t)), 900, 3500)
    for d in (0.0, 0.011, 0.022):
        k = int(d * SR)
        env = np.zeros(len(t))
        env[k:] = np.exp(-(t[: len(t) - k]) / (0.009 if d < 0.02 else 0.13))
        clap += burst * env
    x = tone + 0.8 * noise + 0.9 * clap
    return x / np.max(np.abs(x))


def make_hat(decay=0.022):
    t = t_axis(0.12)
    x = highpass(rng.standard_normal(len(t)), 7500, 4) * np.exp(-t / decay)
    return x / np.max(np.abs(x))


def place(buf, sample, sec, gain):
    i = int(sec * SR)
    j = min(len(buf), i + len(sample))
    if i < len(buf):
        buf[i:j] += sample[: j - i] * gain


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def render_808(notes):
    """Monophonic 808 with legato pitch glides."""
    freq = np.zeros(N)
    amp = np.zeros(N)
    notes = sorted(notes)
    cur_f, cur_a = midi_hz(29), 0.0
    for idx, (st, ln, note, glide) in enumerate(notes):
        i0 = int(st * STEP * SR)
        end_step = st + ln
        if idx + 1 < len(notes):
            end_step = min(end_step, notes[idx + 1][0])
        i1 = min(N, int(end_step * STEP * SR))
        n = i1 - i0
        if n <= 0:
            continue
        t = np.arange(n) / SR
        target = midi_hz(note)
        if glide:
            g = np.clip(t / 0.085, 0, 1)
            f = cur_f * (target / cur_f) ** (g * g * (3 - 2 * g))
            a = cur_a * np.exp(-t / 1.4)
        else:
            f = np.full(n, target)
            a = np.exp(-t / 1.4)
            ramp = min(n, int(0.002 * SR))
            a[:ramp] = np.linspace(cur_a, 1.0, ramp)
        gap_follows = idx + 1 == len(notes) or notes[idx + 1][0] > st + ln
        rel = min(n, int(0.03 * SR))
        if gap_follows:
            a[-rel:] *= np.linspace(1, 0, rel)
        freq[i0:i1] = f
        amp[i0:i1] = a
        cur_f, cur_a = f[-1], a[-1]
    ph = 2 * np.pi * np.cumsum(np.where(freq > 0, freq, cur_f)) / SR
    clean = np.sin(ph) * amp
    dirty = np.tanh(3.0 * clean)                       # harmonics so it's heard on phones
    x = 0.65 * clean + 0.45 * dirty
    return lowpass(x, 900)


def riser(start_sec, dur):
    n = int(dur * SR)
    t = np.arange(n) / n
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    chunks = 16
    for k in range(chunks):
        a, b = k * n // chunks, (k + 1) * n // chunks
        f = 600 * (12 ** (k / chunks))
        out[a:b] = highpass(noise[a:b], min(f, 15000), 2)
    return int(start_sec * SR), out * t ** 2


# ----------------------------------------------------------------------------
# FluidSynth rendering for piano / bells / choir
# ----------------------------------------------------------------------------
def render_sf(name, notes_by_prog):
    tracks = []
    for ch, (prog, notes) in enumerate(notes_by_prog):
        tracks.append(mtrk(note_events(notes, ch, prog), f"{name}{ch}"))
    mid = TMP / f"{name}.mid"
    wav = TMP / f"{name}.wav"
    write_midi(mid, tracks)
    subprocess.run([FLUIDSYNTH, "-ni", "-q", "-g", "0.5", "-r", str(SR),
                    "-F", str(wav), SOUNDFONT, str(mid)], check=True)
    sr, data = wavfile.read(wav)
    assert sr == SR
    data = data.astype(np.float64) / 32768.0
    out = np.zeros((N, 2))
    m = min(N, len(data))
    out[:m] = data[:m]
    return out


def stereo(mono, pan=0.0):
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    return np.stack([mono * l * 1.414, mono * r * 1.414], axis=1)


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def limiter(x, ceiling, block=32, attack=0.002, release=0.08):
    """Look-ahead peak limiter: smooth gain that never lets a peak past the ceiling."""
    peak = np.max(np.abs(x), axis=1)
    nb = -(-len(peak) // block)
    pad = np.zeros(nb * block)
    pad[: len(peak)] = peak
    need = np.minimum(1.0, ceiling / np.maximum(pad.reshape(nb, block).max(1), 1e-9))
    rel = np.exp(-block / (release * SR))
    att = np.exp(-block / (attack * SR))
    g = need.copy()
    for i in range(1, nb):                       # slow release
        g[i] = min(need[i], 1 - (1 - g[i - 1]) * rel)
    for i in range(nb - 2, -1, -1):              # look-ahead attack
        g[i] = min(g[i], 1 - (1 - g[i + 1]) * att)
    gain = np.repeat(g, block)[: len(peak)]
    y = np.clip(x * gain[:, None], -ceiling, ceiling)
    return y, -20 * np.log10(g.min())


def save_wav(path, x):
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def main():
    for d in (OUT, STEMS, MIDI, TMP):
        d.mkdir(parents=True, exist_ok=True)
    p = build_parts()

    # ---- MIDI for your DAW (FL Studio / LMMS / BandLab) ----
    drum_notes = ([(s, 1, 36, 110) for s in p["kick"]] +
                  [(s, 1, 38, 115) for s in p["snare"]] +
                  [(s, 1, 38, int(127 * v)) for s, v in p["ghost"]] +
                  [(s, 0.5, 42, int(127 * min(1, v))) for s, v in p["hat"]])
    bass_notes = [(s, l, n + 12, 100) for s, l, n, g in p["bass"]]
    full = [mtrk(note_events(p["keys"], 0, 0), "Piano chords"),
            mtrk(note_events(p["melody"], 1, 0), "Piano melody"),
            mtrk(note_events(p["bells"], 2, 10), "Bells"),
            mtrk(note_events(p["choir"], 3, 52), "Choir"),
            mtrk(note_events(bass_notes, 4, 38), "808 (load your own 808)"),
            mtrk(note_events(drum_notes, 9), "Drums")]
    write_midi(MIDI / "heavy_heart_full.mid", full)
    for nm, notes, ch, prog in [("piano_chords", p["keys"], 0, 0), ("piano_melody", p["melody"], 0, 0),
                                ("choir", p["choir"], 0, 52), ("808", bass_notes, 0, 38),
                                ("drums", drum_notes, 9, None)]:
        write_midi(MIDI / f"{nm}.mid", [mtrk(note_events(notes, ch, prog), nm)])

    # ---- soundfont parts ----
    keys = render_sf("keys", [(0, p["keys"])])
    mel = render_sf("melody", [(0, p["melody"]), (10, p["bells"])])
    choir = render_sf("choir", [(52, p["choir"])])
    keys = highpass(keys, 95)
    choir = highpass(choir, 200)

    # ---- drums ----
    K, S, H, OH = make_kick(), make_snare(), make_hat(), make_hat(0.15)
    kick = np.zeros(N); snare = np.zeros(N); hats = np.zeros(N)
    for s in p["kick"]:
        place(kick, K, s * STEP, 1.0)
    for s in p["snare"]:
        place(snare, S, s * STEP, 1.0)
    for s, v in p["ghost"]:
        place(snare, S, s * STEP, v)
    for s, v in p["hat"]:
        jitter = rng.normal(0, 0.002) if s % 1 == 0 else 0
        place(hats, H, max(0, s * STEP + jitter), v)
    # snare reverb
    ir_t = t_axis(1.1)
    ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.3)
    ir = lowpass(ir, 6000) / np.sqrt(np.sum(ir ** 2))
    snare_verb = fftconvolve(snare, ir)[:N] * 0.25

    bass = render_808(p["bass"])

    # risers into each hook
    fx = np.zeros(N)
    for b in range(TOTAL_BARS):
        if is_drop_bar(b):
            i0, r = riser((b * BAR_STEPS + 4) * STEP, 12 * STEP)
            fx[i0:i0 + len(r)] += r[: N - i0]

    # ---- intro filter: muffled, opens up when the hook drops ----
    intro_end = int(4 * BAR_STEPS * STEP * SR)
    music = norm(keys, 0.62) + norm(mel, 0.42) + norm(choir, 0.26)
    muffled = lowpass(music, 700, 2)
    fade = np.ones(N)
    xf = int(0.05 * SR)
    fade[:intro_end - xf] = 0
    fade[intro_end - xf:intro_end] = np.linspace(0, 1, xf)
    music = muffled * (1 - fade[:, None]) + music * fade[:, None]

    # sidechain: duck music slightly under kick/808 hits
    env = np.zeros(N)
    duck = np.exp(-t_axis(0.25) / 0.08)
    for s in p["kick"]:
        place(env, duck, s * STEP, 1.0)
    env = np.clip(env, 0, 1)
    music *= (1 - 0.28 * env)[:, None]

    # balance each part by its loudness (RMS) in the first hook
    hook = slice(int(4 * BAR_STEPS * STEP * SR), int(12 * BAR_STEPS * STEP * SR))

    def fit(x, target_db):
        r = np.sqrt(np.mean(x[hook] ** 2))
        return x * (10 ** (target_db / 20) / r)

    def fit_peak(x, peak_db, drive):
        x = norm(x)
        x = np.tanh(x * drive) / np.tanh(drive)      # clip the transient a little
        return x * 10 ** (peak_db / 20)

    snare_bus = np.stack([snare + snare_verb * 0.9, snare + snare_verb * 1.1], 1)
    stems = {
        "piano_and_choir": fit(music, MIX["music"]),
        "808": fit(stereo(bass), MIX["808"]),
        "kick": fit_peak(stereo(kick), MIX["kick_peak"], drive=1.5),
        "snare_clap": fit_peak(snare_bus, MIX["snare_peak"], drive=2.5),
        "hihats": fit_peak(stereo(hats, 0.2), MIX["hats_peak"], drive=1.5),
        "fx_riser": stereo(norm(fx, 0.18), -0.1),
    }
    mix = sum(stems.values())
    stem_scale = 0.95 / np.max(np.abs(mix))

    # master: push to target loudness, gentle soft clip, then peak limiter at -1 dBFS
    pre = 10 ** (MASTER_RMS_DB / 20) / np.sqrt(np.mean(mix[hook] ** 2))
    m = mix * pre
    master = 0.95 * np.tanh(m / 0.95)          # only touches the biggest peaks
    master, gr = limiter(master, 10 ** (-1.2 / 20))
    print(f"limiter max gain reduction: {gr:.1f} dB")
    end_fade = int(2.5 * SR)
    master[-end_fade:] *= np.linspace(1, 0, end_fade)[:, None] ** 2
    master = norm(master, 10 ** (-1 / 20))

    for nm, st in stems.items():
        save_wav(STEMS / f"{nm}.wav", st * stem_scale)
    save_wav(OUT / "heavy_heart_beat.wav", master)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT / "heavy_heart_beat.wav"),
                    "-b:a", "320k", str(OUT / "heavy_heart_beat.mp3")], check=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"done: {TOTAL_SEC:.1f}s, bars={TOTAL_BARS}")


if __name__ == "__main__":
    main()
