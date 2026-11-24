"""Estímulos para la prueba ABX del autor: 20 reactivos.

Se eligen 10 pistas distintas de una sesión (las copias idénticas de audio cuentan una vez): 5 con
altura y 5 sin altura, repartidas a lo largo de su frecuencia protegida. De cada una se corta el
fragmento de 15 s con más energía por debajo de 250 Hz y se escribe sin filtrar, con el corte que
calcula el recomendador (el más alto dentro de la pérdida permitida, con su pendiente) y, como control,
con ese corte una octava más arriba. El corte existe aunque el recomendador decida no filtrar porque no
hay nada medible que quitar; el manifiesto guarda esa decisión (active). Cada pista da dos reactivos
ABX: original contra el corte y original contra una octava arriba. abx.html los
presenta en orden aleatorio y suma los aciertos por condición.

Los estímulos salen de la sesión, que no se redistribuye: la carpeta de salida debe quedar fuera
del repositorio.

    python prepare_abx.py "carpeta/con/wavs" "carpeta/de/salida" [--per-type 5] [--seconds 15]"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from hpf_recommender import design, recommend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("out")
ap.add_argument("--per-type", type=int, default=5, help="pistas con altura y sin altura (5 + 5 = 20 reactivos)")
ap.add_argument("--seconds", type=float, default=15.0)
args = ap.parse_args()


def loudest_low_segment(x, sr, seconds):
    """Inicio del fragmento con más energía bajo 250 Hz, donde un pasa-altos se oiría primero."""
    low = signal.sosfilt(signal.butter(4, 250.0, fs=sr, output="sos"), x.mean(axis=1))
    n = int(seconds * sr)
    if len(low) <= n:
        return 0
    cumulative = np.concatenate(([0.0], np.cumsum(low ** 2)))
    starts = np.arange(0, len(low) - n + 1, int(sr))                 # un candidato por segundo
    return int(starts[np.argmax(cumulative[starts + n] - cumulative[starts])])


def fades(x, sr, ms=20):
    m = int(ms * sr / 1000)
    ramp = np.linspace(0.0, 1.0, m)[:, None]
    x = x.copy()
    x[:m] *= ramp
    x[-m:] *= ramp[::-1]
    return x


def spread(items, k):
    """k elementos repartidos a lo largo de la lista ordenada (incluye los extremos)."""
    if len(items) <= k:
        return items
    return [items[int(round(i * (len(items) - 1) / (k - 1)))] for i in range(k)]


# pistas distintas: el mismo audio con otro nombre cuenta una vez
tracks, seen = [], set()
for path in sorted(Path(args.source).glob("*.wav")):
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    if digest in seen:
        print("copia idéntica, se omite:", path.name)
        continue
    seen.add(digest)
    r = recommend(x, sr)
    if r.cutoff_hz is not None:
        tracks.append((path, r))

pitched = sorted([t for t in tracks if t[1].source == "f0"], key=lambda t: t[1].keep_hz)
unpitched = sorted([t for t in tracks if t[1].source != "f0"], key=lambda t: t[1].keep_hz)
chosen = spread(pitched, args.per_type) + spread(unpitched, args.per_type)
for kind, group in (("con altura", pitched), ("sin altura", unpitched)):
    if len(group) < args.per_type:
        print(f"Aviso: solo hay {len(group)} pistas {kind}; habrá menos de {4 * args.per_type} reactivos.")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
comparisons = []
for path, r in chosen:
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    info = sf.info(str(path))
    start = loudest_low_segment(x, sr, args.seconds)
    seg = x[start:start + int(args.seconds * sr)]
    octave_up = min(2.0 * r.cutoff_hz, 0.45 * sr)
    versions = {"original": seg, "corte": design.apply_hpf(seg, sr, r.cutoff_hz, r.order),
                "octava_arriba": design.apply_hpf(seg, sr, octave_up, r.order)}
    files = {}
    for name, y in versions.items():
        files[name] = f"{path.stem}__{name}.wav"
        sf.write(str(out / files[name]), fades(y, sr), sr, subtype=info.subtype)
    kind = "con altura" if r.source == "f0" else "sin altura"
    for condition, fc in (("corte", r.cutoff_hz), ("octava_arriba", octave_up)):
        comparisons.append({"track": path.stem, "type": kind, "condition": condition, "a": files["original"],
                            "b": files[condition], "cutoff_hz": fc, "order": r.order, "keep_hz": r.keep_hz,
                            "active": r.active,
                            "loss_at_keep_db": round(float(design.attenuation_db(r.keep_hz, fc, r.order)), 2),
                            "start_s": round(start / sr, 2)})
    print(f"{path.stem}: {kind}, corte {r.cutoff_hz} Hz a {6 * r.order} dB/oct, control {octave_up:g} Hz, "
          f"{'filtra' if r.active else 'no filtra'}", flush=True)
(out / "manifest.json").write_text(json.dumps({"trials_per_comparison": 1, "comparisons": comparisons},
                                              indent=1, ensure_ascii=False))
print(len(comparisons), "reactivos en", out / "manifest.json")
