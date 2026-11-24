"""El recomendador y las alternativas de methods.py sobre grabaciones reales con las notas anotadas.

- idmt: las 17 líneas de bajo eléctrico de IDMT-SMT-Bass-Single-Track (Abeßer, 2013; Zenodo,
  doi 10.5281/zenodo.7544099), 44.1 kHz, mono. Se protege la nota más grave con altura definida
  (estilos NO, BE, VI y SL; las notas muertas y los armónicos no cuentan), con el tono MIDI anotado
  en temperamento igual y La4 = 440 Hz. tuning_cents mide cuánto se aparta cada grabación de esa
  afinación: mediana, por parcial (1 a 3), de la desviación del pico espectral respecto del tono
  anotado en las notas normales de al menos 0.2 s.
- guitarset: las 360 tomas de GuitarSet (Xi y otros, 2018; Zenodo, doi 10.5281/zenodo.3371780),
  audio_mono-mic. Los valores de note_midi no son enteros (la altura de cada nota anotada en el
  conjunto); se protege el menor.

Cada grabación se evalúa tal como está (original) y con rumble y hum añadidos con la misma
distribución que las líneas tonales de la batería (contaminado: generator.contaminate, rumble
centrado entre 15 Hz y 0.6 veces la nota más grave, de -35 a -10 dB respecto del valor eficaz, una
semilla por archivo). En la original no se conoce el ruido grave: solo se miden la pérdida en la
nota y la razón con el corte máximo. Como bajo, para la regla_100, cuentan las líneas de IDMT.

    python real_annotated.py idmt ../../datos/idmt_bass            -> results/real_idmt.json
    python real_annotated.py guitarset ../../datos/guitarset       -> results/real_guitarset.json"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from _paths import RESULTS
from generator import contaminate
from methods import NOTE_LOSS_DB, base_of, corners, measure

SEED = {"idmt": 1, "guitarset": 2}
PITCHED = ("NO", "BE", "VI", "SL")


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def idmt_items(root):
    for i, p in enumerate(sorted((root / "misc" / "notes_csv").glob("*_note_parameters.csv"))):
        k = p.name[:3]
        notes = [(float(r[0]), float(r[1]), int(r[2]), r[6]) for r in csv.reader(open(p))]
        low = min(m for _, _, m, style in notes if style in PITCHED)
        longest = max(off - on for on, off, m, style in notes if m == low and style in PITCHED)
        yield dict(i=i, name=k, wav=str(root / "audio" / f"{k}.wav"), low_midi=low, low_max_dur_s=longest,
                   is_bass=True, notes=notes)


def guitarset_items(root):
    for i, p in enumerate(sorted((root / "annotation").glob("*.jams"))):
        jams = json.load(open(p))
        notes = [d for a in jams["annotations"] if a["namespace"] == "note_midi" for d in a["data"]]
        low = min(d["value"] for d in notes)
        longest = max(d["duration"] for d in notes if d["value"] < low + 0.5)
        yield dict(i=i, name=p.stem, wav=str(root / "audio_mono-mic" / f"{p.stem}_mic.wav"), low_midi=low,
                   low_max_dur_s=longest, is_bass=False, notes=None)


def spectral_peak(seg, sr, f, nfft=1 << 20):
    """Pico de |X| a menos de 60 cents de f, con interpolación parabólica en log; None en un borde."""
    x = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft))
    df = sr / nfft
    lo, hi = int(f * 2 ** (-60 / 1200) / df), int(f * 2 ** (60 / 1200) / df) + 1
    i = lo + int(np.argmax(x[lo:hi]))
    if i in (lo, hi - 1):
        return None
    a, b, c = np.log(x[i - 1:i + 2] + 1e-20)
    return (i + 0.5 * (a - c) / (a - 2 * b + c)) * df


def tuning_cents(x, sr, notes):
    out = {}
    for k in (1, 2, 3):
        dev = []
        for on, off, m, style in notes:
            if style != "NO" or off - on < 0.2:
                continue
            seg = x[int((on + 0.04) * sr):int((off - 0.01) * sr)]
            f = spectral_peak(seg, sr, k * midi_hz(m))
            if f:
                dev.append(1200 * np.log2(f / (k * midi_hz(m))))
        out[f"k{k}"] = round(float(np.median(dev)), 1) if dev else None
    return out


def evaluate(job):
    dataset, item = job
    x, sr = sf.read(item["wav"], dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    truth = midi_hz(item["low_midi"])
    row = {"name": item["name"], "low_midi": item["low_midi"], "low_max_dur_s": round(item["low_max_dur_s"], 3),
           "truth": truth, "budget": NOTE_LOSS_DB,
           "duration_s": round(len(x) / sr, 2), "sr": sr}
    if item["notes"] is not None:
        row["tuning_cents"] = tuning_cents(x, sr, item["notes"])
    # pico del espectro de largo plazo entre 70 y 140 Hz, respecto del máximo entre 70 Hz y 2 kHz: una
    # resonancia fija de la grabación (cuerpo o sala) aparece en el mismo lugar en todas las tomas
    f, p = signal.welch(x, sr, nperseg=1 << 16)
    band, top = (f > 70) & (f < 140), (f > 70) & (f < 2000)
    k = int(np.argmax(p[band]))
    row["peak_70_140"] = {"hz": round(float(f[band][k]), 2), "db_re_max": round(float(10 * np.log10(p[band][k] / p[top].max())), 1)}
    rng = np.random.default_rng([SEED[dataset], item["i"]])
    y, rumble, f_r, r_db, has_hum = contaminate(rng, x, 0.6 * truth, (-35, -10), sr)
    for condition, signal_, rum in (("original", x, None), ("contaminado", y, rumble)):
        methods, r, f_yin = corners(signal_, sr, item["is_bass"])
        row[condition] = {"source": r.source, "keep": r.keep_hz, "f0_min": r.f0_min_hz, "active": r.active,
                          "f0_yin": f_yin, **{m: measure(fc, n, truth, NOTE_LOSS_DB, rum, sr, base_of(m))
                                              for m, (fc, n) in methods.items()}}
    row["contaminado"].update(f_r=f_r, r_db=r_db, hum=has_hum)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SEED))
    ap.add_argument("root", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    items = list((idmt_items if args.dataset == "idmt" else guitarset_items)(args.root))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, row in enumerate(pool.map(evaluate, [(args.dataset, it) for it in items])):
            rows.append(row)
            if k % 20 == 0:
                print(k, row["name"], round(time.time() - t0), flush=True)
    out = RESULTS / f"real_{args.dataset}.json"
    out.write_text(json.dumps({"dataset": args.dataset, "items": rows}, indent=1))
    print("listo", len(rows), "grabaciones,", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
