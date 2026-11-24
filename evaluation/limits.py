"""Límites del método con señales construidas: fundamental débil, bajo en contrafase, hum con k > 4,
notas graves cortas y sub-bajo senoidal. Escribe results/limits.json."""
import json

import numpy as np

from _paths import RESULTS
from hpf_recommender import design, recommend
from signals import SR, bursts, harmonic

E1, E2 = 41.2034, 82.4069
out = {"weak_fundamental": [], "antiphase": [], "hum_k_above_4": [], "short_low_note": [], "sine_sub": []}


def loss(f, r):
    return round(float(design.chain_attenuation_db(f, r.cutoff_hz, r.order, r.base_hz, r.base_order)), 2)


for fd in (-20, -25, -30, -35, -40):
    x = np.concatenate([harmonic(E1, 1.5, fund_db=fd), harmonic(E2, 1.5, fund_db=fd), harmonic(110.0, 1.5, fund_db=fd)])
    r = recommend(x, SR)
    out["weak_fundamental"].append({"fund_db": fd, "re_h2_db": fd + 6, "f0": r.f0_min_hz, "fc": r.cutoff_hz, "loss_e1_db": loss(E1, r)})
for bass_db in (-12, -6, 0):
    hats = bursts(6.0, band=(2000.0, 12000.0))
    bass = 10 ** (bass_db / 20) * harmonic(E1, 6.0)
    m, s = recommend(hats + bass, SR), recommend(np.column_stack([hats + bass, hats - bass]), SR)
    out["antiphase"].append({"bass_db": bass_db, "mono_fc": m.cutoff_hz, "stereo_fc": s.cutoff_hz, "stereo_keep": s.keep_hz,
                             "stereo_loss_e1_db": loss(E1, s)})
t = np.arange(int(9 * SR)) / SR
for low, name in ((246.94, "B3 near 250 Hz (k=5)"), (349.23, "F4 near 350 Hz (k=7)")):
    line = np.concatenate([harmonic(low, 1.5)] + [harmonic(low * 2 ** (k / 12), 1.5) for k in (2, 4, 5, 7, 9)])
    hum = sum(0.002 / k * np.sin(2 * np.pi * 50 * k * t[:len(line)]) for k in range(1, 9))
    r = recommend(line + hum, SR)
    out["hum_k_above_4"].append({"case": name, "f0": r.f0_min_hz, "fc": r.cutoff_hz, "loss_db": loss(low, r)})
for d in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    x = np.concatenate([harmonic(E2, 3.0), harmonic(E1, d), harmonic(E2, 3.0)])
    r = recommend(x, SR)
    out["short_low_note"].append({"dur_ms": round(1000 * d), "f0": r.f0_min_hz, "loss_e1_db": loss(E1, r)})
ts = np.arange(8 * SR) / SR
env = np.exp(-(ts % 0.5) / 0.3)
for f_sub in (45.0, 55.0):
    for sub_db in (-18, -12, -6):
        sub = 0.3 * 10 ** (sub_db / 20) * np.sin(2 * np.pi * f_sub * ts) * env
        for name, top in (("hi-hat noise 2-12 kHz", bursts(8.0, band=(2000.0, 12000.0))),
                          ("pad 220-880 Hz", sum(harmonic(f, 8.0, n=4) for f in (220.0, 330.0, 440.0)) / 3)):
            r = recommend(top + sub, SR)
            out["sine_sub"].append({"sub_hz": f_sub, "sub_db": sub_db, "above": name, "source": r.source,
                                    "fc": r.cutoff_hz, "order": r.order, "loss_sub_db": loss(f_sub, r)})
for k, rows in out.items():
    print(k)
    for row in rows:
        print("  ", row)
(RESULTS / "limits.json").write_text(json.dumps(out, indent=1))
