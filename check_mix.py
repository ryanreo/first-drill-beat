"""Prints loudness per section/part and frequency balance of the rendered beat."""
import numpy as np
from scipy.io import wavfile
SR = 44100; STEP = 60 / 142 / 4; BAR = 16 * STEP
SECS = [("intro", 4), ("hook", 8), ("verse", 16), ("hook", 8), ("verse2", 16), ("hook", 8), ("outro", 4)]
def db(x): return 20 * np.log10(max(x, 1e-9))
def load(p): return wavfile.read(p)[1].astype(float) / 32768
m = load("output/heavy_heart_beat.wav"); mono = m.mean(1)
print("len %.1fs peak %.2f dBFS rms %.1f dBFS crest %.1f dB" % (len(m) / SR, db(np.abs(m).max()), db(np.sqrt((m ** 2).mean())), db(np.abs(m).max()) - db(np.sqrt((m ** 2).mean()))))
names = ["piano_and_choir", "808", "kick", "snare_clap", "hihats", "fx_riser"]
stems = {n: load(f"output/stems/{n}.wav").mean(1) for n in names}
b = 0
print("%-7s %6s | " % ("sect", "mix") + " ".join("%9s" % n[:9] for n in names))
for name, n in SECS:
    a, z = int(b * BAR * SR), int((b + n) * BAR * SR)
    print("%-7s %6.1f | " % (name, db(np.sqrt((mono[a:z] ** 2).mean()))) + " ".join("%9.1f" % db(np.sqrt((s[a:z] ** 2).mean())) for s in stems.values()))
    b += n
a, z = int(4 * BAR * SR), int(12 * BAR * SR)
X = np.abs(np.fft.rfft(mono[a:z])) ** 2; f = np.fft.rfftfreq(z - a, 1 / SR); tot = X.sum()
print("hook spectrum:", "  ".join(f"{lo}-{hi}:{10*np.log10(X[(f>=lo)&(f<hi)].sum()/tot):.0f}" for lo, hi in [(20,60),(60,120),(120,300),(300,1000),(1000,4000),(4000,10000),(10000,20000)]))
