"""Paridad del port C++ contra la implementación de referencia en Python.

python cpp/tests/parity.py cpp/build/hpf_recommend [carpeta_con_wavs ...]

Corre las dos versiones sobre las mismas muestras (casos sintéticos fijos, una batería
aleatoria con semilla fija y, si se dan, los WAV de las carpetas) y compara campo por campo.
Sale con código 1 si algún campo difiere más de lo tolerado. Con PARITY_DUMP=archivo.json
guarda los reportes de C++ para comparar dos corridas (determinismo).

Compara todos los campos del reporte (también dc_offset_dbfs, params y el conjunto de claves),
corre los casos fijos también a 44.1 y 96 kHz y en PCM de 24 bits, y revisa que el CLI dé JSON
válido con nombres de archivo que llevan comillas o barras invertidas."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "python"), str(root / "python" / "tests")]
from hpf_recommender import recommend                     # noqa: E402
from signals import SR, bursts, harmonic, only_harmonics, rare_low_note_over_hum, rumble   # noqa: E402

TOL = {"keep_hz": 0.15, "f0_min_hz": 0.15, "spectral_edge_hz": 0.15, "voiced_energy": 0.002,
       "loss_at_keep_db": 0.011, "phase_at_keep_deg": 0.15, "group_delay_ms": 0.011,
       "rejection_short_db": 0.15}
EXACT = ["cutoff_hz", "order", "source", "rejection_met", "f0_note", "peak_band_hz"]


def fixed_cases(sr=SR):
    t8 = np.arange(8 * sr) / sr
    hum = sum(0.004 / k * np.sin(2 * np.pi * 60 * k * t8) for k in (1, 2, 3))
    yield "bajo_E1_rumble", harmonic(41.2, 6.0, sr) + rumble(6.0, 16.0, 0.01, sr)
    yield "bajo_E1_fund_-20", harmonic(41.2, 6.0, sr, fund_db=-20) + rumble(6.0, 16.0, 0.01, sr)
    yield "acorde_E2_B2", harmonic(82.41, 6.0, sr) + harmonic(123.47, 6.0, sr)
    yield "fundamental_ausente", only_harmonics(55.0, range(2, 9), 6.0, sr)
    yield "E1_4pct", np.concatenate([harmonic(82.41, 6.0, sr), harmonic(41.2, 0.25, sr)])
    yield "percusion_rumble", bursts(8.0, sr) + rumble(8.0, 30.0, 0.02, sr)
    yield "percusion_hum", bursts(8.0, sr) + hum
    yield "estereo_contrafase", np.column_stack([bursts(6.0, sr) + harmonic(110.0, 6.0, sr), -(bursts(6.0, sr) + harmonic(110.0, 6.0, sr))])
    yield "silencio", np.zeros(2 * sr)
    yield "corta", harmonic(110.0, 0.3, sr)
    yield "pedal_E1", (harmonic(41.2, 7.2, sr, amp=0.15)
                       + np.concatenate([harmonic(f, 0.6, sr) for f in [164.8, 196.0, 220.0, 246.9] * 3])
                       + rumble(7.2, 22.0, 0.005, sr))
    yield "nota_rara_sobre_hum", rare_low_note_over_hum(sr)[1]
    yield "golpe_100_muestras", 0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(100) / sr)


def random_cases(n, seed=7):
    rng = np.random.default_rng(seed)
    for i in range(n):
        dur = float(rng.uniform(4, 9))
        if rng.random() < 0.65:
            low = 41.2034 * 2 ** (int(rng.integers(0, 25)) / 12)
            notes = [low] + [low * 2 ** (int(rng.integers(2, 20)) / 12) for _ in range(6)]
            parts = [harmonic(f, dur / 7, fund_db=float(rng.uniform(-20, 0))) for f in notes]
            x = np.concatenate(parts)
            r = rumble((len(x) + 1) / SR, float(rng.uniform(15, 0.6 * low)), float(rng.uniform(0.002, 0.03)), seed=i)
            x = x + r[:len(x)]
        else:
            x = bursts(dur, seed=i)
            x = x + rumble((len(x) + 1) / SR, float(rng.uniform(15, 80)), float(rng.uniform(0.002, 0.03)), seed=i)[:len(x)]
        if rng.random() < 0.3:
            t = np.arange(len(x)) / SR
            x = x + sum(float(rng.uniform(0.001, 0.004)) / k * np.sin(2 * np.pi * 60 * k * t) for k in (1, 2, 3))
        if rng.random() < 0.3:
            x = np.column_stack([x, x * float(rng.uniform(0.5, 1.0))])
        yield f"aleatorio_{i:03d}", x


def compare(name, py, cc, bad):
    keys_py = set(py) - {"spectrum"}
    keys_cc = set(cc) - {"file", "sr", "channels", "exported"}
    if keys_py != keys_cc:
        bad.append((name, "claves", sorted(keys_py ^ keys_cc), ""))
    if py.get("params") != cc.get("params"):
        bad.append((name, "params", py.get("params"), cc.get("params")))
    da, db = py.get("dc_offset_dbfs", []), cc.get("dc_offset_dbfs", [])
    if len(da) != len(db) or any(abs(a - b) > 0.15 for a, b in zip(da, db)):
        bad.append((name, "dc_offset_dbfs", da, db))
    for k in EXACT:
        if py.get(k) != cc.get(k):
            bad.append((name, k, py.get(k), cc.get(k)))
    for k, tol in TOL.items():
        a, b = py.get(k), cc.get(k)
        if (a is None) != (b is None) or (a is not None and abs(a - b) > tol):
            bad.append((name, k, a, b))
    ha, hb = py.get("hum"), cc.get("hum")
    if (ha is None) != (hb is None) or (ha and (ha["base_hz"] != hb["base_hz"] or ha["harmonics_hz"] != hb["harmonics_hz"]
                                                or abs(ha["level_db_re_peak"] - hb["level_db_re_peak"]) > 0.15)):
        bad.append((name, "hum", ha, hb))
    if py.get("warnings") != cc.get("warnings"):
        bad.append((name, "warnings", py.get("warnings"), cc.get("warnings")))


def run(binary, files):
    t = time.time()
    out = subprocess.run([binary, *map(str, files)], capture_output=True, text=True, check=True).stdout
    dt = time.time() - t
    data = json.loads(out)
    return (data if isinstance(data, list) else [data]), dt


def cli_checks(binary, tmp, bad):
    """El CLI tiene que dar JSON válido aunque el nombre lleve comillas o barras invertidas."""
    x = harmonic(41.2, 4.0)
    for name in ('nombre "con" comillas.wav', "nombre\\con\\barras.wav"):
        path = tmp / name
        sf.write(path, x, SR, subtype="PCM_24")
        out = subprocess.run([binary, str(path)], capture_output=True, text=True).stdout
        try:
            rep = json.loads(out)
            if rep.get("file") != name:
                bad.append(("cli", "file", name, rep.get("file")))
        except json.JSONDecodeError as e:
            bad.append(("cli", "json inválido", name, str(e)))


def main():
    binary = sys.argv[1]
    folders = [Path(p) for p in sys.argv[2:]]
    bad, n, t_py, t_cc, dump = [], 0, 0.0, 0.0, []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cli_checks(binary, tmp, bad)
        batch = []
        for name, x in list(fixed_cases()) + list(random_cases(int(os.environ.get("N_RANDOM", 120)))):
            path = tmp / f"{name}.wav"
            sf.write(path, x, SR, subtype="DOUBLE")
            batch.append(path)
        for sr in (44100, 96000):
            for name, x in fixed_cases(sr):
                path = tmp / f"{name}_{sr}.wav"
                sf.write(path, x, sr, subtype="DOUBLE")
                batch.append(path)
        for name, x in fixed_cases():
            path = tmp / f"{name}_pcm24.wav"
            sf.write(path, x, SR, subtype="PCM_24")
            batch.append(path)
        files = batch + sorted(p for f in folders for p in f.glob("*.wav"))
        for i in range(0, len(files), 16):
            chunk = files[i:i + 16]
            cc_reports, dt = run(binary, chunk)
            t_cc += dt
            dump.extend(cc_reports)
            for path, cc in zip(chunk, cc_reports):
                x, sr = sf.read(path, always_2d=True, dtype="float64")
                t = time.time()
                py = recommend(x, sr).as_dict()
                t_py += time.time() - t
                compare(path.stem, py, cc, bad)
                n += 1
    if os.environ.get("PARITY_DUMP"):
        Path(os.environ["PARITY_DUMP"]).write_text(json.dumps(dump, indent=1, ensure_ascii=False))
    for b in bad:
        print("DIFERENCIA", *b)
    print(f"{n} files, {len(bad)} differences beyond tolerance; Python {t_py:.1f} s, C++ {t_cc:.1f} s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
