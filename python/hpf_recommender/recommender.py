from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from . import analysis, design, pitch

MAX_ORDER = 8        # pendientes de 6 a 48 dB/oct, las mismas que ofrece el plugin
EDGE_MAX_ORDER = 4   # 24 dB/oct como máximo cuando lo protegido es el borde del espectro
# Filtro fijo, siempre puesto y antes del análisis: lo que hay bajo 20 Hz no es contenido útil. A
# 24 dB/oct pierde 0.13 dB en B0 (30.9 Hz) y 0.01 dB en E1, y quita 24 dB a 10 Hz.
BASE_HZ = 20.0
BASE_ORDER = 4

_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(f):
    n = int(round(12.0 * np.log2(f / 440.0))) + 57   # A4 = 57 semitonos sobre C0
    return f"{_NOTES[n % 12]}{n // 12}"


@dataclass
class Recommendation:
    cutoff_hz: Optional[float]
    order: int = 0
    slope_db_oct: int = 0
    active: bool = False                 # si conviene filtrar: hay algo medible que quitar
    keep_hz: Optional[float] = None      # frecuencia que el filtro protege
    source: str = "ninguna"              # f0 | espectro | mixto | ninguna
    loss_at_keep_db: Optional[float] = None
    loss_budget_db: Optional[float] = None   # pérdida permitida en keep_hz: depende de lo que se protege
    base_hz: Optional[float] = None      # filtro fijo, siempre puesto, delante del recomendado
    base_order: int = 0
    phase_at_keep_deg: Optional[float] = None
    group_delay_ms: Optional[float] = None
    rejection_met: bool = True
    rejection_short_db: float = 0.0
    f0_min_hz: Optional[float] = None
    f0_note: Optional[str] = None
    voiced_energy: float = 0.0
    spectral_edge_hz: Optional[float] = None
    peak_band_hz: Optional[float] = None
    hum: Optional[dict] = None
    dc_offset_dbfs: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    spectrum: Optional[analysis.Spectrum] = field(default=None, repr=False)

    def as_dict(self):
        d = asdict(self)
        d.pop("spectrum")
        return d


def recommend(x, sr, *, max_loss_db=1.0, edge_loss_db=3.0, range_db=30.0, fmin=27.5, fmax=1000.0,
              f_floor=20.0, fc_min=16.0, fc_max=300.0, order=None, max_order=MAX_ORDER,
              tonal_energy=0.25, snap=True, base_hz=BASE_HZ):
    """Recomienda un HPF para una pista. x: (n,) o (n, canales), cualquier sample rate.

    max_loss_db: pérdida máxima permitida cuando lo protegido es una nota (fuentes con altura).
    edge_loss_db: pérdida permitida cuando lo protegido es el borde inferior del espectro (fuentes
              sin altura). El borde es donde la media ya cayó range_db bajo el máximo: con 3 dB el
              corte queda en el borde mismo, la definición clásica de frecuencia de corte. Si el
              ruido estacionario llega al borde, el borde medido es solo una cota superior y se
              usa max_loss_db.
    range_db: lo que queda a más de range_db bajo el máximo del espectro (bandas de 1/3 oct)
              no se considera contenido útil; también es el objetivo de rechazo del ruido grave.
    order: fija el orden (1 a 8, de 6 a 48 dB/oct); si es None se usa el más bajo entre 1 y
              max_order que deje el ruido grave en el objetivo.
    base_hz: filtro fijo de orden BASE_ORDER, siempre puesto. Se aplica antes de analizar, así que lo
              que hay debajo no decide nada, y el recomendado va detrás: la pérdida del fijo en la
              frecuencia protegida se descuenta del presupuesto. None o 0: sin filtro fijo.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if not np.all(np.isfinite(x)):
        raise ValueError("the signal contains NaN or Inf")
    _check_params(sr, max_loss_db, edge_loss_db, range_db, fmin, fmax, fc_min, fc_max, order, max_order)
    if base_hz is not None and (not np.isfinite(base_hz) or base_hz < 0 or (base_hz > 0 and not base_hz < 0.5 * sr)):
        raise ValueError("base_hz goes from 0 (no fixed filter) to sr/2")
    params = {"max_loss_db": max_loss_db, "edge_loss_db": edge_loss_db, "range_db": range_db, "fmin": fmin,
              "fmax": fmax, "max_order": max_order, "fc_max": fc_max, "base_hz": base_hz or None}
    base_order = BASE_ORDER if base_hz else 0

    warnings = []
    dc = [float(20 * np.log10(abs(m) + 1e-12)) for m in x.sum(axis=0) / max(len(x), 1)]
    has_dc = max(dc) > -60.0
    if has_dc:
        warnings.append(f"DC offset of {max(dc):.0f} dBFS; " +
                        (f"the fixed {base_hz:g} Hz filter removes it." if base_hz else "the HPF removes it."))
    if len(x) < 3 * sr:
        warnings.append("Signal shorter than 3 s: the estimate is unreliable.")

    if base_hz:
        x = design.apply_hpf(x, sr, base_hz, base_order)
    spec = analysis.long_term_spectrum(x, sr)
    if spec is None:
        return Recommendation(None, base_hz=base_hz or None, base_order=base_order,
                              dc_offset_dbfs=[round(v, 1) for v in dc], params=params,
                              warnings=warnings + ["Digital silence: nothing to analyse."])
    if spec.stationary:
        warnings.append("Stationary material (no dynamics): background noise cannot be separated from the signal.")

    f_top = min(16000.0, 0.45 * sr)
    edge = analysis.spectral_edge(spec, f_floor, f_top, range_db)
    ref = spec.peak_index(f_floor, f_top)
    hum = analysis.detect_hum(spec)
    # Bandas donde el piso estacionario es más de la mitad de la media: ahí no hay notas
    # que entran y salen, hay rumble, hum o un tono de equipo. Si todo el material es
    # estacionario esa distinción no existe y no se aplica.
    noisy = spec.floor_db > spec.mean_db - 3.0
    steady = None if spec.stationary else pitch.Steady(spec.centers, noisy, spec.frame_db, spec.floor_db,
                                                       spec.frame_t0, spec.frame_hop)
    p = pitch.estimate_min_f0(x, sr, fmin=fmin, fmax=fmax,
                              exclude_hz=hum["harmonics_hz"] if hum else (), steady=steady)

    # Con una fuente afinada manda la nota más grave: debajo de su fundamental el
    # instrumento no produce parciales, lo que haya es ruido o sangrado. Si las tramas
    # con nota pesan poco en la energía (batería, ambiente), el f0 puede venir de un
    # sangrado y solo puede bajar el punto protegido, nunca subirlo sobre el borde espectral.
    if p.f0_min is not None and p.voiced_energy >= tonal_energy:
        keep, source = p.f0_min, "f0"
    elif p.f0_min is not None and p.f0_min < edge:
        keep, source = p.f0_min, "mixto"
    else:
        keep, source = edge, "espectro"
    # Una nota pertenece a la fuente: se protege con max_loss_db. El borde del espectro está, por
    # definición, range_db bajo el máximo, y ahí basta edge_loss_db. Salvo que el ruido estacionario
    # llegue al borde (una banda dominada por el piso a menos de un ancho de banda de análisis, 1/3
    # de octava, debajo de él): al restar el piso se resta también la falda de la fuente, el borde
    # medido es solo una cota superior y se vuelve a max_loss_db.
    budget = max_loss_db
    if source == "espectro":
        near = (spec.centers >= edge * 2 ** (-1 / 3)) & (spec.centers < edge)
        if steady is not None and bool(np.any(noisy[near])):
            warnings.append(f"Stationary noise reaches the spectral edge ({edge:.0f} Hz): that edge is "
                            f"an upper bound and is protected with {max_loss_db:g} dB, not {edge_loss_db:g} dB.")
        else:
            budget = edge_loss_db

    # Ruido grave: bandas que superan el umbral de contenido útil y están enteras por
    # debajo de keep/sqrt(2). Tras el filtro deben quedar en ref - range_db, el mismo
    # umbral que define qué es útil. La media octava bajo keep es banda de transición:
    # ahí cae la fuga de la propia nota protegida (vibrato, bends, lóbulo de la ventana)
    # y ningún filtro razonable la separa.
    # El recomendado va detrás del fijo: entre los dos no pasan del presupuesto en keep.
    rec_budget = budget - float(design.attenuation_db(keep, base_hz, base_order)) if base_hz else budget
    rec_budget = max(rec_budget, 1e-6)
    below = spec.centers * 2 ** (1 / 6) < keep / np.sqrt(2)
    need = spec.mean_db[below] - (spec.mean_db[ref] - range_db)
    junk = need > 0
    # Una nota se mide con precisión y admite cualquier pendiente. El borde de una fuente sin altura
    # es una falda gradual que se mide con error (sube cuando el ruido estacionario la cubre), y con
    # más pendiente el mismo error se vuelve más pérdida: ahí la pendiente llega a 24 dB/oct.
    top = max_order if source != "espectro" else min(max_order, EDGE_MAX_ORDER)
    orders = [order] if order else list(range(1, top + 1))
    # Con el fijo delante, un orden cuyo corte máximo cae bajo fc_min no cabe en el presupuesto
    # (subir el corte a fc_min lo rompería) y lo subsónico ya lo quita el fijo: se descarta.
    if base_hz and not order:
        orders = [k for k in orders if design.max_cutoff(keep, k, rec_budget) >= fc_min] or orders
    fc, n, met, short = design.choose(keep, spec.centers[below][junk], need[junk],
                                      rec_budget, orders, fc_min, fc_max, snap)
    if not met:
        fix = ("on the edge of an unpitched source no steeper slope is used: treat it separately."
               if source == "espectro" and not order and top < max_order else "a steeper slope or separate treatment is needed.")
        warnings.append(f"At {6 * n} dB/oct the low-frequency noise stays {short:.0f} dB above the target; " + fix)
    if design.max_cutoff(keep, n, rec_budget) > fc_max:
        warnings.append(f"Corner limited to fc_max = {fc_max:g} Hz.")
    if design.max_cutoff(keep, n, rec_budget) < fc_min:
        warnings.append(f"Useful content reaches down to {keep:.0f} Hz: only a subsonic filter fits, "
                        f"at fc_min = {fc_min:g} Hz.")
    # Filtrar solo si hay algo medible que quitar: bandas bajo la transición sobre el umbral de
    # contenido útil, u offset DC si no hay filtro fijo que lo quite. Pestana y Reiss (2014) no
    # encontraron justificación subjetiva para el pasa-altos en pistas sin graves.
    active = bool(np.any(junk)) or (has_dc and not base_hz)
    if not active:
        quitar = f"what is left after the fixed {base_hz:g} Hz filter" if base_hz else "no DC offset"
        warnings.append(f"Below {keep / np.sqrt(2):.0f} Hz nothing exceeds the usefulness threshold ({quitar}): "
                        "the filter would remove nothing measurable and leaving it off is recommended "
                        "(the corner given is the highest that respects the allowed loss).")

    loss, phase, gd = design.response_at(keep, fc, n, sr, base_hz or None, base_order)
    if hum is not None:
        hum["level_db_re_peak"] = round(hum.pop("level_db") - float(spec.mean_db[ref]), 1)
        above = [h for h in hum["harmonics_hz"] if h > fc]
        if above and hum["level_db_re_peak"] > -60.0:
            warnings.append(f"Stationary tones at multiples of {hum['base_hz']:g} Hz above the corner "
                            f"({', '.join(f'{h:g}' for h in above)} Hz), typical of mains hum: "
                            "an HPF cannot remove them without cutting the instrument; use a notch or de-hum.")

    return Recommendation(
        cutoff_hz=fc, order=n, slope_db_oct=6 * n, active=active, keep_hz=round(float(keep), 1), source=source,
        loss_at_keep_db=round(loss, 2), loss_budget_db=budget, base_hz=base_hz or None, base_order=base_order,
        phase_at_keep_deg=round(phase, 1),
        group_delay_ms=round(gd, 2), rejection_met=met, rejection_short_db=round(short, 1),
        f0_min_hz=round(p.f0_min, 1) if p.f0_min else None,
        f0_note=note_name(p.f0_min) if p.f0_min else None,
        voiced_energy=round(p.voiced_energy, 3), spectral_edge_hz=round(edge, 1),
        peak_band_hz=round(float(spec.centers[ref]), 1), hum=hum,
        dc_offset_dbfs=[round(v, 1) for v in dc], warnings=warnings,
        params=params, spectrum=spec)


def _check_params(sr, max_loss_db, edge_loss_db, range_db, fmin, fmax, fc_min, fc_max, order, max_order):
    """Lo que el resto del código da por cierto. Fuera de esto hay divisiones por cero
    (fmin, fmax), un clip con los límites al revés (fc_max < fc_min) o ningún orden que probar."""
    finite = all(np.isfinite(v) for v in (sr, max_loss_db, edge_loss_db, range_db, fmin, fmax, fc_min, fc_max))
    if not finite or sr <= 0:
        raise ValueError("sample rate or parameters not finite")
    if max_loss_db <= 0 or edge_loss_db <= 0 or range_db <= 0:
        raise ValueError("max_loss_db, edge_loss_db and range_db must be positive")
    if not 0 < fmin < fmax:
        raise ValueError("0 < fmin < fmax is required")
    if not 0 < fc_min <= fc_max:
        raise ValueError("0 < fc_min <= fc_max is required")
    if order is not None and not 0 <= order <= MAX_ORDER:
        raise ValueError(f"order goes from 1 to {MAX_ORDER} (0 or None: the lowest that meets the target)")
    if not order and not 1 <= max_order <= MAX_ORDER:
        raise ValueError(f"max_order goes from 1 to {MAX_ORDER}")
