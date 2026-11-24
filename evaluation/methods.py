"""El recomendador y las alternativas con las que se compara, y cómo se mide cada corte. Lo usan
baselines.py (batería sintética) y real_annotated.py (grabaciones reales con notas anotadas).

Alternativas:

- fijo_40, fijo_80, fijo_120: el mismo corte para todas las pistas, a 12 dB/oct.
- regla_100: la regla de un sistema de mezcla basado en conocimiento (De Man y Reiss, 2013):
  100 Hz a 12 dB/oct en todas las pistas menos el bajo y el bombo.
- margen_yin: 20 % bajo la nota más grave de YIN sin verificación espectral (mismo YIN, misma
  compuerta de actividad y mismo soporte que el método), a 24 dB/oct; sin nota, sin filtro.
- solo_borde: el corte en el borde del espectro con la regla de fuentes sin altura (3 dB), sin
  análisis de altura.
- recomendador_borde_1dB y recomendador_borde_3dB: el recomendador con 1 dB o con 3 dB en el borde
  siempre, llegue o no el ruido estacionario hasta él; son las dos ablaciones de la regla de borde.

Cuando el recomendador decide no filtrar (no hay nada medible que quitar), cuenta como sin filtro
recomendado. Las variantes del recomendador llevan siempre delante el filtro fijo de 20 Hz: su pérdida
se suma a la del recomendado, su corte máximo es el que respeta el presupuesto que deja el fijo, y el
rumble pasa por los dos. Las alternativas se miden sin filtro fijo, como se publicaron.

Cada corte se mide contra lo que hay que proteger y su presupuesto (1 dB en una nota, 3 dB en el
borde útil de una fuente sin altura): pérdida ahí, razón entre el corte y el máximo que respeta el
presupuesto (Ec. 4 del paper) y energía del rumble conocido que quita el filtro."""
import numpy as np
from scipy import signal

from hpf_recommender import analysis, design, pitch, recommend
from hpf_recommender.analysis import SILENCE_DB, to_db
from hpf_recommender.recommender import BASE_HZ, BASE_ORDER

NOTE_LOSS_DB, EDGE_LOSS_DB, RANGE_DB = 1.0, 3.0, 30.0
FIXED_HZ = (40.0, 80.0, 120.0)
MARGIN, MARGIN_ORDER = 0.2, 4
METHODS = ["recomendador", "recomendador_borde_1dB", "recomendador_borde_3dB", "fijo_40", "fijo_80", "fijo_120",
           "regla_100", "margen_yin", "solo_borde"]


BASE = (BASE_HZ, BASE_ORDER)


def base_of(name):
    """El filtro fijo de las variantes del recomendador; las alternativas no lo tienen."""
    return BASE if name.startswith("recomendador") else None


def plain_yin_lowest(x, sr):
    """Nota más grave de YIN con la compuerta y el soporte del método, sin verificación ni exclusiones."""
    g = np.gcd(int(sr), pitch.ANALYSIS_SR)
    y = signal.resample_poly(x, pitch.ANALYSIS_SR // g, int(sr) // g)
    f0, power, _, _, _ = pitch.yin(y, pitch.ANALYSIS_SR, 27.5, 1000.0)
    if f0.size == 0:
        return None
    p_db = to_db(power)
    alive = p_db > SILENCE_DB
    if not alive.any():
        return None
    gate = np.percentile(p_db[alive], 95) - 35.0
    return pitch.lowest_note(np.where(alive & (p_db >= gate), f0, np.nan), np.arange(len(f0)))


def clean_edge(clean, sr):
    """Borde útil de una fuente limpia, con el mismo análisis y rango que el método."""
    spec = analysis.long_term_spectrum(clean[:, None], sr)
    return float(analysis.spectral_edge(spec, 20.0, min(16000.0, 0.45 * sr), RANGE_DB))


def edge_corner(f_keep, order, loss_db):
    return float(np.clip(design.snap_down(design.max_cutoff(f_keep, order, loss_db)), 16.0, 300.0))


def energy_db(r, y):
    return float(10 * np.log10(np.sum(r ** 2) / np.sum(y ** 2)))


def corners(x, sr, is_bass):
    """Corte y orden de cada método para la señal x (mono). is_bass: si la regla_100 la exceptúa."""
    r = recommend(x, sr)
    f_yin = plain_yin_lowest(x, sr)
    edge = r.spectral_edge_hz
    def chosen(rec):
        return (rec.cutoff_hz, rec.order) if rec.active else (None, 0)
    out = {"recomendador": chosen(r)}
    # la fuente, el punto protegido y la decisión de filtrar no dependen de los presupuestos: solo
    # cambia el corte en el borde
    r1 = r if r.source != "espectro" else recommend(x, sr, edge_loss_db=NOTE_LOSS_DB)
    r3 = r if r.source != "espectro" else recommend(x, sr, max_loss_db=EDGE_LOSS_DB)
    out["recomendador_borde_1dB"] = chosen(r1)
    out["recomendador_borde_3dB"] = chosen(r3)
    for f in FIXED_HZ:
        out[f"fijo_{f:.0f}"] = (f, 2)
    out["regla_100"] = (None, 0) if is_bass else (100.0, 2)
    out["margen_yin"] = ((1.0 - MARGIN) * f_yin, MARGIN_ORDER) if f_yin else (None, 0)
    out["solo_borde"] = (edge_corner(edge, 2, EDGE_LOSS_DB), 2) if edge else (None, 0)
    return out, r, f_yin


def measure(fc, order, truth, budget, rumble, sr, base=None):
    """rumble=None cuando no se conoce el ruido grave (una grabación tal como está). base: (Hz, orden)
    del filtro fijo que va delante, o None."""
    base_loss = float(design.attenuation_db(truth, *base)) if base else 0.0
    y = None if rumble is None else (design.apply_hpf(rumble, sr, *base) if base else rumble)
    if fc is None:
        return {"fc": None, "order": 0, "loss": base_loss, "ratio": 0.0,
                "removed": None if rumble is None else energy_db(rumble, y)}
    return {"fc": fc, "order": order, "loss": base_loss + float(design.attenuation_db(truth, fc, order)),
            "ratio": float(fc / design.max_cutoff(truth, order, max(budget - base_loss, 1e-6))),
            "removed": None if rumble is None else energy_db(rumble, design.apply_hpf(y, sr, fc, order))}
