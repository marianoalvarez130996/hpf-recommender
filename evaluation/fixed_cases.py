"""Casos sintéticos fijos con respuesta conocida. Escribe results/fixed_cases.json."""
import json

import numpy as np

from _paths import RESULTS
from hpf_recommender import design, recommend
from signals import SR, bursts, harmonic, only_harmonics, rumble

E1, B1, E2, B2 = 41.20, 61.74, 82.41, 123.47
t8 = np.arange(8 * SR) / SR
hum60 = sum(0.004 / k * np.sin(2 * np.pi * 60 * k * t8) for k in (1, 2, 3))


def guitar_with_bleed():
    # guitarra E2..A2 con pausas; sangrado continuo de bajo en B1 a -25 dB
    seg = [harmonic(f, 0.8) for f in (E2, 98.0, 110.0, 98.0)]
    gap = np.zeros(int(0.4 * SR))
    g = np.concatenate([np.concatenate([s, gap]) for s in seg] * 2)
    return g + harmonic(B1, len(g) / SR, amp=0.3 * 10 ** (-25 / 20))


cases = [
    # nombre, señal, nota más grave real (None si no es tonal), frecuencia que debe sobrevivir, rumble
    ("Bass E1 + rumble at 16 Hz", harmonic(E1, 6.0) + rumble(6.0, 16.0, 0.01), E1, E1, 16.0),
    ("Bass E1, fundamental -20 dB", harmonic(E1, 6.0, fund_db=-20) + rumble(6.0, 16.0, 0.01), E1, E1, 16.0),
    ("Chord E2 + B2", harmonic(E2, 6.0) + harmonic(B2, 6.0), E2, E2, None),
    ("Harmonics 2-8 of 55 Hz", only_harmonics(55.0, range(2, 9), 6.0), 110.0, 110.0, None),
    ("E1 for 8% of the time", np.concatenate([harmonic(E2, 5.5), harmonic(E1, 0.5)]), E1, E1, None),
    ("E1 for 4% of the time", np.concatenate([harmonic(E2, 6.0), harmonic(E1, 0.25)]), E1, E1, None),
    ("Guitar + continuous B1 bleed at -25 dB", guitar_with_bleed(), E2, E2, None),
    ("Noise bursts 200 Hz-8 kHz + rumble at 30 Hz", bursts(8.0) + rumble(8.0, 30.0, 0.02), None, 200.0, 30.0),
    ("Noise bursts + 60 Hz hum", bursts(8.0) + hum60, None, 200.0, None),
]

rows = []
for name, x, truth, protect, junk in cases:
    r = recommend(x, SR)
    row = {"case": name, "truth_hz": truth, "protect_hz": protect, "f0": r.f0_min_hz, "fc": r.cutoff_hz,
           "order": r.order,
           "loss_db": round(float(design.chain_attenuation_db(protect, r.cutoff_hz, r.order, r.base_hz, r.base_order)), 2),
           "source": r.source, "hum": r.hum and r.hum["base_hz"]}
    if junk:
        row["rej_db"] = round(float(design.chain_attenuation_db(junk, r.cutoff_hz, r.order, r.base_hz, r.base_order)), 1)
    rows.append(row)
    print(row, flush=True)
(RESULTS / "fixed_cases.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
