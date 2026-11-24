"""Ablación del tope de pendiente en el borde del espectro: dónde quedaría el corte si en una fuente sin
altura se permitieran hasta 48 dB/oct, como en una nota, y cuánto protegería.

La regla sin tope se reproduce con la interfaz pública. Con tope, recommend prueba los órdenes 1 a 4; si
ninguno deja el ruido grave en el objetivo, sin tope seguiría con 5 a 8 y tomaría el primero que cumple
(o el 8), igual que design.choose con todos los órdenes. Eso es lo que hace esta ablación, fijando cada
orden con order=. Solo cambia algo en los casos cuyo punto protegido es el borde del espectro.

    python ablation_slope.py --seed 21       -> results/ablation_slope_21.json"""
import argparse
import json
import time

from _paths import RESULTS
from generator import SR, Battery
from hpf_recommender import recommend
from methods import BASE, EDGE_LOSS_DB, NOTE_LOSS_DB, clean_edge, measure

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tonal", type=int, default=200)
ap.add_argument("--perc", type=int, default=100)
args = ap.parse_args()

cases = Battery(args.seed)
rows = []
t0 = time.time()
for i in range(args.tonal + args.perc):
    x, rumble, meta, clean = cases.case(i, args.tonal)
    r = recommend(x, SR)
    if r.source != "espectro":
        continue
    tonal = meta["kind"] == "tonal"
    truth, budget = (meta["low"], NOTE_LOSS_DB) if tonal else (clean_edge(clean, SR), EDGE_LOSS_DB)
    free = r
    if not r.rejection_met:
        for n in range(5, 9):
            free = recommend(x, SR, order=n)
            if free.rejection_met:
                break
    chosen = {name: (rec.cutoff_hz, rec.order) if rec.active else (None, 0) for name, rec in
              (("con_tope", r), ("sin_tope", free))}
    rows.append({"i": i, "kind": meta["kind"], "truth": truth, "budget": budget, "active": r.active,
                 **{name: measure(fc, n, truth, budget, rumble, SR, BASE) for name, (fc, n) in chosen.items()}})
(RESULTS / f"ablation_slope_{args.seed}.json").write_text(json.dumps({"seed": args.seed, "cases": rows}, indent=1))
print("listo", len(rows), "casos con borde,", round(time.time() - t0), "s")
