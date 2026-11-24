"""Energía grave que quitan los filtros recomendados en una sesión, y cuánto sobresale el bajo del
resto en las bandas graves antes y después.

Potencia por banda de 1/3 de octava (centros nominales de 20 a 250 Hz) de la densidad espectral de
Welch (Hann, 32768 muestras, 50 %). Cada pista se filtra con el corte y el orden que se le recomendó
(results/tracks), solo si el recomendador decidió filtrarla: una pista para la que recomienda no
filtrar queda igual. La relación bajo/resto es la potencia del bajo en la banda dividida por la suma de
las potencias de las pistas que no son de bajo, todas a ganancia unitaria y sin tener en cuenta la
fase: es una medida de solapamiento espectral de largo plazo, no un modelo perceptual de
enmascaramiento. Una pista con el mismo audio que otra (una copia exportada con otro nombre) se
marca con duplicate_of y no entra en las sumas.

    python session_energy.py "carpeta/con/wavs"      -> results/session_energy.json"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from _paths import RESULTS
from hpf_recommender import design

NOMINAL = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250]
TARGET = "BASS COUNTRYMAN_01"


def band_powers(x, sr):
    f, p = signal.welch(x, fs=sr, window="hann", nperseg=32768, noverlap=16384, axis=0)
    p = p.sum(axis=1) if p.ndim == 2 else p
    df = f[1] - f[0]
    bands = np.array([p[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * df for c in NOMINAL])
    return bands, f, p


tracks, first = {}, {}
for path in sorted(Path(sys.argv[1]).glob("*.wav")):
    rep = json.load(open(RESULTS / "tracks" / f"{path.stem}.json"))["report"]
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    first.setdefault(digest, path.stem)
    active = rep.get("active", True)
    # el filtro fijo va siempre; el recomendado, si la pista se filtra
    y = design.apply_hpf(x, sr, rep["base_hz"], rep["base_order"]) if rep.get("base_hz") else x
    y = design.apply_hpf(y, sr, rep["cutoff_hz"], rep["order"]) if active else y
    b0, f, p0 = band_powers(x, sr)
    b1, _, p1 = band_powers(y, sr)
    above, below = f >= rep["keep_hz"], (f < rep["keep_hz"]) & (f > 0)
    tracks[path.stem] = {"fc": rep["cutoff_hz"], "order": rep["order"], "keep": rep["keep_hz"], "active": active,
                         "before": b0.tolist(), "after": b1.tolist(),
                         "change_above_keep_db": float(10 * np.log10(p0[above].sum() / p1[above].sum())),
                         "removed_below_keep_db": float(10 * np.log10(p0[below].sum() / p1[below].sum()))}
    if first[digest] != path.stem:
        tracks[path.stem]["duplicate_of"] = first[digest]
    t = tracks[path.stem]
    print(path.stem, round(t["removed_below_keep_db"], 1), round(t["change_above_keep_db"], 2), flush=True)

names = [n for n in tracks if "duplicate_of" not in tracks[n]]
before = np.array([tracks[n]["before"] for n in names])
after = np.array([tracks[n]["after"] for n in names])
rest = [i for i, n in enumerate(names) if not n.startswith("BASS")]
t = names.index(TARGET)
ratio = {"before_db": (10 * np.log10(before[t] / before[rest].sum(axis=0))).tolist(),
         "after_db": (10 * np.log10(after[t] / after[rest].sum(axis=0))).tolist()}
for key, values in ratio.items():
    print(f"{TARGET} bajo/resto {key:9}", " ".join(f"{c}:{v:.1f}" for c, v in zip(NOMINAL, values)))
out = {"bands_hz": NOMINAL, "tracks": tracks, "target_to_rest": {TARGET: ratio}}
(RESULTS / "session_energy.json").write_text(json.dumps(out, indent=1))
