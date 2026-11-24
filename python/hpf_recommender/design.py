"""HPF Butterworth elegido por especificación: pérdida máxima en la nota más grave y
rechazo mínimo del ruido grave, con el orden más bajo que cumpla ambas."""

import numpy as np
from scipy import signal

# Serie R20 de números preferentes (ISO 3): pasos de ~12 %, menos que la tolerancia de
# pérdida en la banda de paso, así que redondear hacia abajo nunca la rompe.
_R20 = np.array([1.00, 1.12, 1.25, 1.40, 1.60, 1.80, 2.00, 2.24, 2.50, 2.80,
                 3.15, 3.55, 4.00, 4.50, 5.00, 5.60, 6.30, 7.10, 8.00, 9.00])
GRID = np.round(np.concatenate([_R20 * 10, _R20 * 100, [1000.0]]), 1)


def snap_down(fc):
    i = int(np.searchsorted(GRID, fc * (1 + 1e-9), side="right")) - 1
    return float(GRID[max(0, i)])


def attenuation_db(f, fc, order):
    """Butterworth analógico. A 48 kHz y por debajo de 1 kHz el digital (bilineal con
    prewarping en fc) difiere en centésimas de dB."""
    return 10.0 * np.log10(1.0 + (fc / np.asarray(f, dtype=float)) ** (2 * order))


def chain_attenuation_db(f, fc, order, base_hz=None, base_order=0):
    """Pérdida del filtro fijo (si lo hay) más la del recomendado: en dB se suman."""
    a = attenuation_db(f, fc, order)
    return a + attenuation_db(f, base_hz, base_order) if base_hz else a


def max_cutoff(f_keep, order, max_loss_db):
    """fc más alto que deja la pérdida en f_keep en max_loss_db: de |H|² = 1/(1 + (fc/f)^2N)."""
    return f_keep * (10.0 ** (max_loss_db / 10.0) - 1.0) ** (1.0 / (2 * order))


def choose(f_keep, junk_hz, junk_need_db, max_loss_db, orders, fc_min, fc_max, snap=True,
           tol_db=1.0):
    """Devuelve (fc, orden, cumple_rechazo, faltante_dB). El faltante es cuánto queda el
    ruido grave sobre el objetivo con el orden más alto permitido; se tolera tol_db."""
    last = None
    for n in orders:
        fc = float(np.clip(max_cutoff(f_keep, n, max_loss_db), fc_min, fc_max))
        if snap:
            fc = snap_down(fc)
        short = 0.0
        if len(junk_hz):
            short = float(np.max(junk_need_db - attenuation_db(junk_hz, fc, n)))
        last = (fc, n, short <= tol_db, max(short, 0.0))
        if short <= tol_db:
            return last
    return last


def sos(fc, order, sr):
    return signal.butter(order, fc, btype="highpass", fs=sr, output="sos")


def chain_sos(fc, order, sr, base_hz=None, base_order=0):
    """El filtro fijo (base_hz, base_order) seguido del recomendado; sin base, solo el recomendado."""
    s = sos(fc, order, sr)
    return np.vstack([sos(base_hz, base_order, sr), s]) if base_hz else s


def response_at(f, fc, order, sr, base_hz=None, base_order=0):
    """Pérdida (dB), fase (grados) y retardo de grupo (ms) del filtro digital en f, con el fijo
    delante si lo hay."""
    s = chain_sos(fc, order, sr, base_hz, base_order)
    df = max(0.01, f * 1e-3)
    _, h = signal.sosfreqz(s, worN=np.array([f - df, f, f + df]), fs=sr)
    phase = np.unwrap(np.angle(h))
    gd = -(phase[2] - phase[0]) / (2 * np.pi * 2 * df)
    return float(-20 * np.log10(abs(h[1]))), float(np.degrees(phase[1])), float(gd * 1000.0)


def apply_hpf(x, sr, fc, order):
    """Causal (fase mínima), como el HPF de una consola o del plugin. x: (n,) o (n, canales)."""
    return signal.sosfilt(sos(fc, order, sr), x, axis=0)


def apply_recommendation(x, sr, rec):
    """La cadena completa de una recomendación: el filtro fijo y, si active, el recomendado."""
    y = np.asarray(x, dtype=np.float64)
    if rec.base_hz:
        y = apply_hpf(y, sr, rec.base_hz, rec.base_order)
    if rec.active and rec.cutoff_hz is not None:
        y = apply_hpf(y, sr, rec.cutoff_hz, rec.order)
    return y
