"""Espectro de largo plazo en bandas de 1/3 de octava, piso estacionario y zumbido de red."""

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

SILENCE_DB = -100.0     # potencia de trama (dBFS) por debajo de esto se trata como silencio digital


def to_db(p):
    return 10.0 * np.log10(np.maximum(p, 1e-20))


def band_centers(f_lo, f_hi, per_octave=24):
    k = np.arange(np.ceil(per_octave * np.log2(f_lo / 1000.0)),
                  np.floor(per_octave * np.log2(f_hi / 1000.0)) + 1)
    return 1000.0 * 2.0 ** (k / per_octave)


@dataclass
class Spectrum:
    centers: np.ndarray       # Hz, una banda cada 1/24 oct, cada una de 1/3 oct de ancho
    mean_db: np.ndarray       # media de las tramas activas
    floor_db: np.ndarray      # parte estacionaria (percentil bajo, con corrección de sesgo)
    excess_db: np.ndarray     # media menos piso: lo que no es ruido de fondo constante
    stationary: bool          # todo el material es estacionario: no se resta el piso
    active_fraction: float
    fine_freqs: np.ndarray    # espectro fino (un bin cada ~1.5 Hz) hasta 500 Hz, para el hum
    fine_floor_db: np.ndarray
    frame_db: np.ndarray      # [trama corta, banda]: potencia de cada trama corta, para la prueba por trama
    frame_t0: float           # centro de la primera trama corta (s)
    frame_hop: float          # salto entre tramas cortas (s)

    def peak_index(self, f_lo, f_hi):
        idx = np.flatnonzero((self.centers >= f_lo) & (self.centers <= f_hi))
        return int(idx[np.argmax(self.excess_db[idx])])


def band_dof(i_lo, w_lo, i_hi, w_hi):
    """Grados de libertad de la potencia de una banda (Satterthwaite). La banda suma bins de Hann
    con pesos a_k (fraccionarios en los bordes); la potencia de dos bins vecinos correlaciona
    |rho_1|² = 4/9 y a dos bins |rho_2|² = 1/36, así que nu = 2 (sum a)² / sum_ij a_i a_j |rho_ij|².
    Dividir el ancho por el ENBW (1.5 bins) no sirve: el ENBW mide ancho de ruido, no distancia de
    decorrelación, y sobrestima nu un 30 % en bandas anchas. Una banda de menos de un bin repartida
    entre dos tiene más de 2 grados de libertad."""
    a = np.zeros(i_hi - i_lo + 1)
    if i_hi == i_lo:
        a[0] = w_hi - w_lo
    else:
        a[0], a[1:-1], a[-1] = 1.0 - w_lo, 1.0, w_hi
    var = a @ a + 2.0 * (4.0 / 9.0) * (a[:-1] @ a[1:]) + 2.0 * (1.0 / 36.0) * (a[:-2] @ a[2:])
    return 2.0 * a.sum() ** 2 / var


def _frame_powers(x, sr, centers, bw_oct, bin_hz, fine_max_hz=0.0, overlap=4, chunk=32):
    """Potencia por trama en cada banda. Los canales se suman en potencia: un contenido
    en contrafase entre L y R no desaparece, como sí pasaría con un downmix (L+R)/2."""
    n, n_ch = x.shape
    nfft = 1 << int(np.ceil(np.log2(sr / bin_hz)))
    if n < nfft:
        # señal más corta que una trama: al centro, donde la ventana pesa; al principio quedaría
        # donde Hann vale casi cero y un golpe corto se leería como silencio
        left = (nfft - n) // 2
        x = np.pad(x, ((left, nfft - n - left), (0, 0)))
        n = nfft
    hop = nfft // overlap
    win = signal.get_window("hann", nfft)
    scale = 1.0 / (nfft * np.sum(win ** 2))       # suma de bins = potencia media de la trama
    df = sr / nfft
    n_bins = nfft // 2 + 1

    # Integral exacta de la potencia entre los bordes de cada banda, interpolando dentro
    # del bin (el bin k cubre [k - 1/2, k + 1/2]·df). Así las bandas de 1/3 oct en 20 Hz,
    # más angostas que unos pocos bins, no quedan vacías ni cuantizadas.
    def edge(f):
        pos = np.clip(f / df + 0.5, 0.0, n_bins)
        i = np.minimum(np.floor(pos).astype(int), n_bins - 1)
        return i, pos - i

    f_lo = centers * 2.0 ** (-bw_oct / 2)
    f_hi = centers * 2.0 ** (bw_oct / 2)
    i_lo, w_lo = edge(f_lo)
    i_hi, w_hi = edge(f_hi)
    n_fine = int(fine_max_hz / df) + 1

    starts = np.arange(0, n - nfft + 1, hop)
    bands = np.empty((len(starts), len(centers)))
    fine = np.empty((len(starts), n_fine))
    total = np.empty(len(starts))
    views = [np.lib.stride_tricks.sliding_window_view(x[:, ch], nfft) for ch in range(n_ch)]
    for s in range(0, len(starts), chunk):
        idx = starts[s:s + chunk]
        p = np.zeros((len(idx), n_bins))
        for v in views:
            p += np.abs(np.fft.rfft(v[idx] * win, axis=1)) ** 2
        p *= scale
        p[:, 1:-1] *= 2.0
        c = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(p, axis=1)], axis=1)
        bands[s:s + len(idx)] = (c[:, i_hi] + w_hi * p[:, i_hi]) - (c[:, i_lo] + w_lo * p[:, i_lo])
        fine[s:s + len(idx)] = p[:, :n_fine]
        total[s:s + len(idx)] = c[:, -1]

    nu = np.array([band_dof(a, wa, b, wb) for a, wa, b, wb in zip(i_lo, w_lo, i_hi, w_hi)])
    return bands, fine, np.arange(n_fine) * df, total, nu, nfft


def long_term_spectrum(x, sr, active_range_db=40.0, floor_pct=10.0, fine_max_hz=500.0):
    """None si todo es silencio digital."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    centers = band_centers(12.5, min(16000.0, 0.45 * sr))
    # Dos resoluciones. Tramas largas (~1.5 Hz por bin, 0.68 s a 48 kHz) para el nivel
    # medio y el espectro fino del hum. Tramas de ~170 ms para el piso: con 0.68 s un
    # patrón de batería llena todas las tramas y parece estacionario.
    bands, fine, fine_freqs, total, _, _ = _frame_powers(x, sr, centers, 1 / 3, 1.5, fine_max_hz)
    short, _, _, short_total, nu, nfft_s = _frame_powers(x, sr, centers, 1 / 3, 6.0, overlap=2)

    total_db = to_db(total)
    valid = total_db > SILENCE_DB
    if not valid.any():
        return None
    active = valid & (total_db >= np.percentile(total_db[valid], 95) - active_range_db)
    mean = bands[active].mean(axis=0)
    short_valid = to_db(short_total) > SILENCE_DB

    # Piso estacionario por cuantil (Stahl, Fischer y Bippus, 2000, ICASSP; en la línea de la
    # estadística de mínimos de Martin, 2001): un percentil bajo de la potencia de cada banda en
    # el tiempo. Para ruido gaussiano la potencia de banda sigue ~chi² con nu grados de libertad
    # (band_dof), y el percentil queda por debajo de la media; se divide por ese cociente para no
    # subestimar el piso en las bandas graves, que tienen muy pocos bins.
    bias = stats.chi2.ppf(floor_pct / 100.0, nu) / nu
    floor = np.minimum(np.percentile(short[short_valid], floor_pct, axis=0) / bias, mean)

    audible = centers >= 20.0
    stationary = bool(np.max(mean[audible] / np.maximum(floor[audible], 1e-30)) < 2.0)
    excess = mean if stationary else np.maximum(mean - floor, 1e-3 * mean)

    return Spectrum(centers, to_db(mean), to_db(floor), to_db(excess), stationary,
                    float(active.mean()), fine_freqs,
                    to_db(np.percentile(fine[valid], floor_pct, axis=0)),
                    to_db(short), 0.5 * nfft_s / sr, (nfft_s // 2) / sr)


def spectral_edge(spec, f_floor, f_top, range_db):
    """Borde inferior de la región conectada al máximo del espectro (sin piso) que queda a
    menos de range_db de ese máximo. Una isla grave separada por un valle más profundo que
    el umbral (sangrado de otro instrumento, rumble) no cuenta como contenido propio."""
    level = spec.excess_db
    peak = spec.peak_index(f_floor, f_top)
    thr = level[peak] - range_db
    lo = int(np.searchsorted(spec.centers, f_floor))
    i = peak
    while i > lo and level[i - 1] >= thr:
        i -= 1
    if i == lo:
        return float(spec.centers[lo])
    a, b = level[i - 1], level[i]
    t = (thr - a) / (b - a)
    return float(spec.centers[i - 1] * (spec.centers[i] / spec.centers[i - 1]) ** t)


def detect_hum(spec, prominence_db=10.0):
    """Busca la serie de 50 o 60 Hz en el piso estacionario. Las notas no caen en 50/60 Hz
    exactos (G1 = 49.0, B1 = 61.7 Hz con A4 = 440), y una nota no forma parte del piso."""
    f, db = spec.fine_freqs, spec.fine_floor_db
    best = None
    for base in (50.0, 60.0):
        hits = []
        for k in range(1, 9):
            fk = k * base
            if fk * 2 ** (1 / 6) > f[-1]:
                break
            # máximo local a menos de 1 Hz de k·base (la red se desvía mucho menos que eso);
            # sin esta condición la falda de un parcial a 3 Hz pasa por pico
            i = int(round(fk / (f[1] - f[0])))
            j = i - 2 + int(np.argmax(db[i - 2:i + 3]))
            ctx = (f > fk * 2 ** (-1 / 6)) & (f < fk * 2 ** (1 / 6)) & ~((f > fk - 4) & (f < fk + 4))
            if (abs(f[j] - fk) < 1.0 and db[j] > db[j - 1] and db[j] > db[j + 1]
                    and db[j] - np.median(db[ctx]) >= prominence_db):
                hits.append(fk)
        # serie creíble: incluye la fundamental o el segundo armónico (rectificación)
        credible = len(hits) >= 2 and (base in hits or 2 * base in hits)
        if credible and (best is None or len(hits) > len(best[1])):
            best = (base, hits)
    if best is None:
        return None
    base, hits = best
    level = max(10 * np.log10(np.sum(10 ** (db[(f > h - 1.5) & (f < h + 1.5)] / 10))) for h in hits)
    return {"base_hz": base, "harmonics_hz": hits, "level_db": float(level)}
