"""Lo que una sola pista no dice: comodulación dentro de la pista y correlación entre pistas.

Envolvente de una banda de 1/3 de octava (Butterworth de orden 3) en tramas de 10 ms, en dB, menos
su media móvil de 1 s (para que no manden las secciones del tema). Solo cuentan las tramas a menos de
40 dB del percentil 95 de la potencia de la pista (de las dos pistas, al comparar). Un ataque es una
trama que sube 6 dB o más sobre el mínimo de los 50 ms anteriores (con 100 ms de espera entre ataques);
un ataque del cuerpo coincide con la banda si ésta tiene un ataque a ±30 ms.

    python cross_track.py "carpeta/con/wavs"      -> results/cross_track.json"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from _paths import RESULTS

HOP = 0.010
NOMINAL = [40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
           2500, 3150, 4000, 5000, 6300, 8000, 10000]
folder = Path(sys.argv[1])
cache = {}


def load(name):
    if name not in cache:
        x, sr = sf.read(str(folder / f"{name}.wav"), always_2d=True, dtype="float64")
        cache[name] = (x.mean(axis=1), sr)
    return cache[name]


def envelope(name, fc):
    x, sr = load(name)
    if fc is None:
        y = x
    else:
        y = signal.sosfilt(signal.butter(3, [fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)], btype="band", fs=sr, output="sos"), x)
    hop = int(HOP * sr)
    n = len(y) // hop
    ms = np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1)
    return 10 * np.log10(ms + 1e-12)


def detrend(db):
    k = int(round(1.0 / HOP)) | 1
    return db - np.convolve(db, np.ones(k) / k, mode="same")


def active(name):
    db = envelope(name, None)
    return db >= np.percentile(db, 95) - 40


def onsets(db, rise=6.0, back=5, wait=10):
    out, last = [], -wait
    for i in range(back, len(db)):
        if i - last >= wait and db[i] - db[i - back:i].min() >= rise:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def corr(a, b, mask):
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def body_band(name):
    return max(NOMINAL[3:], key=lambda f: np.mean(10 ** (envelope(name, f) / 10)))


out = {"within": {}, "cross": {}}
for name in ("HH_03", "BASS COUNTRYMAN_01"):
    body = body_band(name)
    eb = envelope(name, body)
    on = onsets(eb)
    act = active(name)
    rows = []
    for f in (40, 50, 63, 80, 100, 125, 160, 200, 250):
        if abs(np.log2(f / body)) <= 1 / 3:
            continue
        el = envelope(name, f)
        lo = onsets(el)
        hit = np.mean([np.any(np.abs(lo - i) <= 3) for i in on]) if len(on) else float("nan")
        rows.append({"band_hz": f, "r_body": round(corr(detrend(el), detrend(eb), act), 3), "onset_share": round(float(hit), 3)})
    out["within"][name] = {"body_hz": body, "onsets": int(len(on)), "bands": rows}
    rs = [r["r_body"] for r in rows]
    hs = [r["onset_share"] for r in rows]
    print(f"{name}: cuerpo {body} Hz, {len(on)} ataques; r con el cuerpo {min(rs):.2f} a {max(rs):.2f}; ataques acompañados {min(hs):.0%} a {max(hs):.0%}")

pairs = [("TOM 2_01", "TOM 3_01", (63, 80, 100, 125, 160, 200, 250)), ("SNR UP_01", "SNR DWN_01", (100, 125, 160, 200, 250, 315))]
for a, b, bands in pairs:
    mask = active(a) & active(b)
    rows = []
    for f in bands:
        ea, eb = envelope(a, f), envelope(b, f)
        n = min(len(ea), len(eb))
        level = float(10 * np.log10(np.mean(10 ** (eb[:n] / 10)) / np.mean(10 ** (ea[:n] / 10))))
        rows.append({"band_hz": f, "r": round(corr(detrend(ea[:n]), detrend(eb[:n]), mask[:n]), 3), "level_b_minus_a_db": round(level, 1)})
    out["cross"][f"{a} | {b}"] = rows
    print(f"{a} con {b}: r {min(r['r'] for r in rows):.2f} a {max(r['r'] for r in rows):.2f}; "
          f"{b} más fuerte por {min(r['level_b_minus_a_db'] for r in rows):.1f} a {max(r['level_b_minus_a_db'] for r in rows):.1f} dB")

others = sorted(p.stem for p in folder.glob("*.wav") if p.stem != "GTR LINE_01")
best = {}
for o in others:
    mask = active("GTR LINE_01") & active(o)
    rs = []
    for f in (40, 50, 63, 80, 100, 125, 160, 200, 250, 315):
        ea, eb = envelope("GTR LINE_01", f), envelope(o, f)
        n = min(len(ea), len(eb))
        rs.append(corr(detrend(ea[:n]), detrend(eb[:n]), mask[:n]))
    best[o] = round(max(rs), 3)
    cache.pop(o, None)
out["cross"]["GTR LINE_01 | others (max over 40-315 Hz)"] = best
top = sorted(best.items(), key=lambda kv: -kv[1])[:3]
print("GTR LINE_01 contra las demás, máximo en 40-315 Hz:", ", ".join(f"{k} {v:.2f}" for k, v in top))
(RESULTS / "cross_track.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
