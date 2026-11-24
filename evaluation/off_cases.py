"""Con qué rumble deja el recomendador el filtro apagado: regenera los casos de la batería (generator.py,
mismas semillas) solo para leer el nivel y la frecuencia del rumble de cada uno, y los cruza con la
decisión guardada en results/baselines_<semilla>.json.

    python off_cases.py 31 40       -> results/off_cases.json (semillas 31 a 40)"""
import json
import sys

import numpy as np

from _paths import RESULTS
from generator import Battery

first, last = int(sys.argv[1]), int(sys.argv[2])
out = {"seeds": [first, last]}
rows = []
for seed in range(first, last + 1):
    cases = json.load(open(RESULTS / f"baselines_{seed}.json"))["cases"]
    battery = Battery(seed)
    for i, c in enumerate(cases):
        _, _, meta, _ = battery.case(i, 200)
        rows.append((meta["kind"], bool(c["active"]), meta["r_db"], meta["f_r"]))
for kind in ("tonal", "perc"):
    on = [r[2] for r in rows if r[0] == kind and r[1]]
    off = [r[2] for r in rows if r[0] == kind and not r[1]]
    out[kind] = {"n_on": len(on), "n_off": len(off),
                 "r_db_on": [float(np.min(on)), float(np.median(on)), float(np.max(on))] if on else None,
                 "r_db_off": [float(np.min(off)), float(np.median(off)), float(np.max(off))] if off else None,
                 "share_on_at_or_below_max_off": float(np.mean(np.array(on) <= max(off))) if on and off else None,
                 # apagados con rumble a menos de 30 dB de la señal: dónde está centrado ese rumble
                 "f_r_off_within_30db": sorted(float(r[3]) for r in rows if r[0] == kind and not r[1] and r[2] > -30.0)}
(RESULTS / "off_cases.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
