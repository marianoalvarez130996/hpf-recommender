"""Nota más grave con YIN (de Cheveigné y Kawahara, 2002, JASA 111(4)) y verificación espectral."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from .analysis import SILENCE_DB, to_db

ANALYSIS_SR = 16000     # para f0 <= 1 kHz sobra; YIN cuesta ~3 veces menos que a 48 kHz
FRAME_SNR_DB = 10.0     # en una banda dominada por el piso, lo que una nota sobresale mientras suena


@dataclass
class Steady:
    """Bandas dominadas por el piso estacionario y, para la prueba por trama, la potencia de cada
    trama corta del espectro de largo plazo."""
    centers: np.ndarray     # Hz
    mask: np.ndarray        # piso > media - 3 dB
    frame_db: np.ndarray    # [trama corta, banda]
    floor_db: np.ndarray
    t0: float               # centro de la primera trama corta (s)
    hop: float              # salto entre tramas cortas (s)


@dataclass
class PitchResult:
    f0_min: Optional[float]
    voiced_energy: float      # parte de la energía activa que está en tramas con nota verificada
    n_voiced: int


def yin(x, sr, fmin, fmax, hop_s=0.010, threshold=0.15, batch=512):
    """f0 por trama (NaN si no hay periodo claro), potencia, centro de cada trama y la
    aperiodicidad (d' normalizada de YIN) en el periodo elegido y en el doble."""
    tau_min = max(2, int(sr / fmax))
    tau_max = int(math.ceil(sr / fmin))
    w = tau_max                       # ventana de integración = periodo más largo buscado
    flen = w + tau_max
    hop = max(1, int(round(hop_s * sr)))
    if len(x) < flen:
        empty = np.array([])
        return empty, empty, np.array([], dtype=int), empty, empty

    frames = np.lib.stride_tricks.sliding_window_view(x, flen)[::hop]
    nfft = 1 << int(math.ceil(math.log2(flen + w)))
    taus = np.arange(tau_max + 1)
    f0 = np.full(len(frames), np.nan)
    power = np.empty(len(frames))
    aper = np.full(len(frames), np.nan)
    aper2 = np.full(len(frames), np.nan)

    for s in range(0, len(frames), batch):
        fr = np.asarray(frames[s:s + batch], dtype=np.float64)
        rows = np.arange(len(fr))
        cs = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr * fr, axis=1)], axis=1)
        power[s:s + len(fr)] = cs[:, w] / w

        # d(tau) = sum_j (x_j - x_{j+tau})^2 desarrollado; la correlación cruzada por FFT
        r = np.fft.irfft(np.fft.rfft(fr, nfft) * np.conj(np.fft.rfft(fr[:, :w], nfft)), nfft)
        d = cs[:, [w]] + cs[:, taus + w] - cs[:, taus] - 2.0 * r[:, :tau_max + 1]
        d[:, 0] = 0.0
        cm = np.ones_like(d)
        cm[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-12)

        below = cm[:, tau_min:tau_max] < threshold
        has = below.any(axis=1)
        t = np.argmax(below, axis=1) + tau_min
        while True:                   # del primer cruce baja hasta el mínimo local
            nxt = np.minimum(t + 1, tau_max)
            step = has & (t + 1 < tau_max) & (cm[rows, nxt] < cm[rows, t])
            if not step.any():
                break
            t += step
        ok = has & (t < tau_max - 1)  # pegado al borde de búsqueda: no fiable

        a, b, c = cm[rows, t - 1], cm[rows, t], cm[rows, np.minimum(t + 1, tau_max)]
        den = a - 2.0 * b + c
        curved = np.abs(den) > 1e-12
        shift = np.clip(np.where(curved, 0.5 * (a - c) / np.where(curved, den, 1.0), 0.0), -1.0, 1.0)
        f0[s:s + len(fr)][ok] = sr / (t[ok] + shift[ok])
        aper[s:s + len(fr)][ok] = b[ok]
        # en el periodo doble: mínimo de t2-1, t2, t2+1 (2t no cae justo en un entero)
        t2 = np.minimum(2 * t, tau_max - 1)
        near = np.minimum(np.minimum(cm[rows, t2 - 1], cm[rows, t2]), cm[rows, t2 + 1])
        ok2 = ok & (2 * t + 1 <= tau_max)
        aper2[s:s + len(fr)][ok2] = near[ok2]

    centers = np.arange(len(frames)) * hop + flen // 2
    return f0, power, centers, aper, aper2


def verify_octave(x, sr, f0, centers, fmin, aper=None, aper2=None, n_harm=8,
                  prom_db=8.0, drop_db=25.0, nfft=4096):
    """Corrige errores de octava de YIN con el espectro de la misma trama.

    En un acorde E2+B2 YIN da el periodo común (41 Hz), y con rumble puede caer una octava
    abajo: se toma el múltiplo k·f0 más bajo que es una nota creíble. Una nota creíble
    tiene un pico propio (prom_db sobre la mediana de ±1/2 octava), a menos de drop_db
    del parcial más fuerte, y al menos uno de sus armónicos 2 o 3 también con pico: un
    pico suelto en una zona de ruido grave no es una nota.

    Con un segundo armónico mucho más fuerte que la fundamental (bajo microfoneado) el
    error va al revés, una octava arriba. f0/2 se acepta solo si además de ser creíble
    el propio YIN lo respalda: la señal es más periódica en el periodo doble que en el
    elegido (d'(2t) < d'(t)). Si nada cumple, la trama queda en NaN."""
    out = np.full(f0.shape, np.nan)
    voiced = np.flatnonzero(np.isfinite(f0))
    if voiced.size == 0:
        return out
    half = nfft // 2
    xp = np.pad(x, (half, half))
    win = signal.get_window("hann", nfft)
    res = sr / nfft                   # resolución real; el zero-padding x2 solo interpola
    bin_hz = res / 2
    f_lim = 0.45 * sr
    k = np.arange(1, n_harm + 1)

    def peak(sp, f):
        tol = max(0.03 * f, 0.5 * res)
        lo, hi = int((f - tol) / bin_hz), int(math.ceil((f + tol) / bin_hz)) + 1
        band = sp[lo:hi]
        m = int(np.argmax(band))
        floor = np.median(sp[int(f / math.sqrt(2) / bin_hz):int(math.ceil(f * math.sqrt(2) / bin_hz)) + 1])
        # máximo en el borde de la ventana = ladera de otro pico
        return band[m], (0 < m < len(band) - 1) and band[m] - floor >= prom_db

    def credible(sp, f, strongest):
        level, prominent = peak(sp, f)
        if not prominent or level < strongest - drop_db:
            return False
        return any(peak(sp, h * f)[1] for h in (2, 3) if h * f < f_lim)

    for s in range(0, voiced.size, 256):
        idx = voiced[s:s + 256]
        seg = np.stack([xp[c:c + nfft] for c in centers[idx]]) * win
        spec = 20.0 * np.log10(np.abs(np.fft.rfft(seg, 2 * nfft, axis=1)) + 1e-12)
        for j, sp in zip(idx, spec):
            fk = f0[j] * k[f0[j] * k < f_lim]
            strongest = max(peak(sp, f)[0] for f in fk)
            h = 0.5 * f0[j]
            if (h >= fmin and aper2 is not None and np.isfinite(aper2[j])
                    and aper2[j] < aper[j] and credible(sp, h, strongest)):
                out[j] = h
                continue
            for f in fk:
                if credible(sp, f, strongest):
                    out[j] = f
                    break
    return out


def _longest_run(frames, max_gap=2):
    cuts = np.flatnonzero(np.diff(frames) > max_gap)
    bounds = np.concatenate(([0], cuts + 1, [len(frames)]))
    return int(np.max(np.diff(bounds)))


def lowest_note(f0, frame_index, min_frames=12, min_run=6, width_cents=100.0):
    """f0 más grave con soporte suficiente: al menos min_frames tramas dentro de un
    semitono y un tramo de min_run tramas casi seguidas (a 10 ms: 120 ms y 60 ms).
    El tramo seguido es lo que separa una nota de errores sueltos dispersos por el tema.
    No se exige una proporción mínima de las tramas: la nota más grave de una línea de
    bajo puede sonar 1 % del tiempo o menos y sigue siendo la que hay que proteger."""
    ok = np.isfinite(f0)
    f0, frame_index = f0[ok], frame_index[ok]
    if len(f0) < min_frames:
        return None
    cents = 1200.0 * np.log2(f0 / 440.0)
    order = np.argsort(cents)
    cents, f0, frame_index = cents[order], f0[order], frame_index[order]
    ends = np.searchsorted(cents, cents + width_cents, side="right")
    for i in np.flatnonzero(ends - np.arange(len(cents)) >= min_frames):
        if _longest_run(np.sort(frame_index[i:ends[i]])) >= min_run:
            return float(np.median(f0[i:ends[i]]))
    return None


def pitch_signals(x):
    """Señales para la estimación de f0. En estéreo, la suma (L+R)/2 y la diferencia (L-R)/2: lo que
    está en contrafase desaparece de la suma, pero el espectro (que suma los canales en potencia) sí
    lo ve, y sin nota la regla de la isla puede tirarlo: un bajo en contrafase bajo unos platos más
    fuertes se perdería. Con más de dos canales, la media o, si se cancela, el canal más fuerte."""
    if x.ndim == 1 or x.shape[1] == 1:
        return [x.reshape(-1)]
    if x.shape[1] == 2:
        return [0.5 * (x[:, 0] + x[:, 1]), 0.5 * (x[:, 0] - x[:, 1])]
    mid = x.mean(axis=1)
    per_channel = np.mean(x ** 2, axis=0)
    if np.mean(mid ** 2) < 0.25 * per_channel.mean():
        return [x[:, int(np.argmax(per_channel))]]
    return [mid]


def estimate_min_f0(x, sr, fmin=27.5, fmax=1000.0, gate_db=35.0, exclude_hz=(), steady=None):
    """exclude_hz: frecuencias de hum ya detectadas. Un zumbido estacionario es periódico
    y YIN lo toma por nota; se descartan las tramas a menos de 25 cents de él. Las notas
    temperadas con A4 = 440 caen a 35 cents o más de 50·k y 60·k para k <= 4.

    steady: Steady con las bandas dominadas por el piso estacionario. Un rumble de banda angosta
    también es casi periódico, y su f0 cae en cualquier parte; una nota real entra y sale, y
    mientras suena su banda queda muy por encima del piso. La media de largo plazo no sirve para
    verlo: una nota que suena el 3 % del tema apenas la mueve. Por eso la trama se descarta solo
    si en ese momento su banda está a menos de FRAME_SNR_DB del piso: un rumble gaussiano con ~2
    grados de libertad pasa 10 dB sobre su media con probabilidad e^-10 por trama.
    Una nota sostenida todo el tema (pedal, drone) también es estacionaria, pero arrastra
    sus armónicos: si la banda de 2·f0 o la de 3·f0 también es estacionaria, la trama se
    queda. Un rumble no tiene armónicos estacionarios encima."""
    g = math.gcd(int(sr), ANALYSIS_SR)
    tracks = [(y, *yin(y, ANALYSIS_SR, fmin, fmax))
              for y in (signal.resample_poly(s, ANALYSIS_SR // g, int(sr) // g)
                        for s in pitch_signals(np.asarray(x, dtype=np.float64)))]
    if tracks[0][1].size == 0:
        return PitchResult(None, 0.0, 0)
    # Solo tramas a menos de gate_db del nivel alto del tema (el mismo umbral para suma y
    # diferencia): lo que queda por debajo es cola de reverb, ruido o sangrado lejano, y no define
    # la nota más grave.
    p_dbs = [to_db(t[2]) for t in tracks]
    if not any((p > SILENCE_DB).any() for p in p_dbs):
        return PitchResult(None, 0.0, 0)
    gate = max(np.percentile(p[p > SILENCE_DB], 95) for p in p_dbs if (p > SILENCE_DB).any()) - gate_db
    merged, e_voiced, e_active = None, 0.0, 0.0
    for (y, f0, power, centers, aper, aper2), p_db in zip(tracks, p_dbs):
        active = (p_db > SILENCE_DB) & (p_db >= gate)
        f0 = _verified_f0(y, np.where(active, f0, np.nan), centers, aper, aper2, fmin, fmax, exclude_hz, steady)
        e_voiced += float(power[np.isfinite(f0)].sum())
        e_active += float(power[active].sum())
        # la misma trama en la suma y en la diferencia cuenta una vez (12 tramas siguen siendo
        # 120 ms): queda el f0 más bajo de las dos
        merged = f0 if merged is None else np.fmin(merged, f0)
    # En energía y no en tiempo: en una batería las tramas con nota son colas de toms o
    # sangrado de la caja; pueden ser muchas y pesar poco.
    share = e_voiced / e_active if e_active > 0 else 0.0
    return PitchResult(lowest_note(merged, np.arange(len(merged))), share, int(np.isfinite(merged).sum()))


def _verified_f0(x, f0, centers, aper, aper2, fmin, fmax, exclude_hz, steady):
    """Verificación de octava y exclusiones (hum, bandas estacionarias) de una pista de f0."""
    f0 = verify_octave(x, ANALYSIS_SR, f0, centers, fmin, aper, aper2)
    f0[f0 > fmax] = np.nan
    for h in exclude_hz:
        with np.errstate(invalid="ignore"):
            f0[np.abs(1200.0 * np.log2(f0 / h)) < 25.0] = np.nan
    if steady is not None:
        def band(f):
            return np.clip(np.searchsorted(steady.centers, f * 2 ** (-1 / 48)), 0, len(steady.centers) - 1)

        ok = np.flatnonzero(np.isfinite(f0))
        b = band(f0[ok])
        drone = steady.mask[band(2 * f0[ok])] | steady.mask[band(3 * f0[ok])]
        k = np.clip(np.round((centers[ok] / ANALYSIS_SR - steady.t0) / steady.hop).astype(int),
                    0, len(steady.frame_db) - 1)
        quiet = steady.frame_db[k, b] - steady.floor_db[b] < FRAME_SNR_DB
        f0[ok[steady.mask[b] & ~drone & quiet]] = np.nan
    return f0
