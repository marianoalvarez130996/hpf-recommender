"""Batería aleatoria con respuesta conocida.

    python battery.py --seed 11            -> results/battery_11.json

200 líneas tonales y 100 casos de percusión de generator.py. Cada caso se analiza dos veces: con el
método completo y sin la prueba por trama (la ablación).

Por caso se guarda la pérdida en la nota más grave (o en el borde de la percusión) y el rechazo del
rumble de dos formas: la atenuación del filtro analógico en la frecuencia central del rumble y la
energía del rumble que de verdad queda después del Butterworth recomendado."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from _paths import RESULTS
from generator import SR, Battery
from hpf_recommender import design, pitch, recommend

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--tonal", type=int, default=200)
ap.add_argument("--perc", type=int, default=100)
ap.add_argument("--out", type=str, default=None, help="por omisión, results/battery_<semilla>.json")
args = ap.parse_args()

FRAME_SNR_DB = pitch.FRAME_SNR_DB
cases = Battery(args.seed)


def without_frame_test(x):
    pitch.FRAME_SNR_DB = math.inf
    try:
        return recommend(x, SR)
    finally:
        pitch.FRAME_SNR_DB = FRAME_SNR_DB


def energy_db(r, y):
    return float(10 * np.log10(np.sum(r ** 2) / np.sum(np.asarray(y, dtype=np.float64) ** 2)))


rows = []
t0 = time.time()
for i in range(args.tonal + args.perc):
    x, rum, meta, _ = cases.case(i, args.tonal)
    r, a = recommend(x, SR), without_frame_test(x)
    rows.append(dict(meta, f0=r.f0_min_hz, fc=r.cutoff_hz, order=r.order, source=r.source, hum_flagged=bool(r.hum),
                     loss=float(design.chain_attenuation_db(meta['low'], r.cutoff_hz, r.order, r.base_hz, r.base_order)),
                     rej=float(design.chain_attenuation_db(meta['f_r'], r.cutoff_hz, r.order, r.base_hz, r.base_order)),
                     rej_energy=energy_db(rum, design.apply_hpf(design.apply_hpf(rum, SR, r.base_hz, r.base_order),
                                                                SR, r.cutoff_hz, r.order)),
                     ablation_f0=a.f0_min_hz,
                     ablation_loss=float(design.chain_attenuation_db(meta['low'], a.cutoff_hz, a.order, a.base_hz, a.base_order))))
    if i % 50 == 0:
        print(i, round(time.time() - t0), flush=True)
out = Path(args.out) if args.out else RESULTS / f"battery_{args.seed}.json"
out.write_text(json.dumps({"seed": args.seed, "cases": rows}, indent=1))
print("listo", round(time.time() - t0), "s")
