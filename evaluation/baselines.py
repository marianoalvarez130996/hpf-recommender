"""El recomendador frente a las alternativas de methods.py, sobre las mismas señales de la batería
(generator.py). En una línea tonal se protege su nota más grave con 1 dB; en una señal percusiva, el
borde útil de la fuente limpia (sin rumble ni hum) con 3 dB. Como bajo, para la regla_100, cuentan
las líneas cuya nota más grave está debajo de E2; la batería no tiene bombo.

    python baselines.py --seed 21            -> results/baselines_21.json"""
import argparse
import json
import time

from _paths import RESULTS
from generator import SR, Battery
from methods import EDGE_LOSS_DB, NOTE_LOSS_DB, base_of, clean_edge, corners, measure

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tonal", type=int, default=200)
ap.add_argument("--perc", type=int, default=100)
args = ap.parse_args()

E2 = 82.4069
cases = Battery(args.seed)
rows = []
t0 = time.time()
for i in range(args.tonal + args.perc):
    x, rumble, meta, clean = cases.case(i, args.tonal)
    tonal = meta["kind"] == "tonal"
    truth, budget = (meta["low"], NOTE_LOSS_DB) if tonal else (clean_edge(clean, SR), EDGE_LOSS_DB)
    methods, r, f_yin = corners(x, SR, is_bass=tonal and meta["low"] < E2)
    rows.append({"kind": meta["kind"], "low": meta["low"], "truth": truth, "budget": budget, "hum": meta["hum"],
                 "f0_yin": f_yin, "edge": r.spectral_edge_hz, "active": r.active, "source": r.source,
                 **{name: measure(fc, n, truth, budget, rumble, SR, base_of(name)) for name, (fc, n) in methods.items()}})
    if i % 50 == 0:
        print(i, round(time.time() - t0), flush=True)
(RESULTS / f"baselines_{args.seed}.json").write_text(json.dumps({"seed": args.seed, "cases": rows}, indent=1))
print("listo", round(time.time() - t0), "s")
