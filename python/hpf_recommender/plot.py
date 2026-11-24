import numpy as np

from . import analysis, design
from .recommender import note_name


def before_after(rec, x, sr, title):
    """Espectro de largo plazo (bandas de 1/3 oct, dB sobre el máximo) antes y después del
    HPF recomendado, en completo y con zoom en graves."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    spec = rec.spectrum
    after = analysis.long_term_spectrum(design.apply_hpf(np.asarray(x, dtype=np.float64), sr,
                                                         rec.cutoff_hz, rec.order), sr)
    f = spec.centers
    ref = spec.mean_db[spec.peak_index(20.0, f[-1])]
    db_b, db_a, db_f = spec.mean_db - ref, after.mean_db - ref, spec.floor_db - ref

    C_BEFORE, C_AFTER, C_CUT, C_KEEP, C_FLOOR = "#868e96", "#e8590c", "#c92a2a", "#1971c2", "#adb5bd"
    fmt_hz = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    fc = rec.cutoff_hz
    if rec.source in ("f0", "mixto") and rec.f0_min_hz and abs(rec.keep_hz - rec.f0_min_hz) < 0.2:
        keep_txt = f"Nota más grave\n{rec.keep_hz:.0f} Hz ({note_name(rec.keep_hz)})"
    else:
        keep_txt = f"Borde del contenido útil\n{rec.keep_hz:.0f} Hz"

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1, 1.3]})
    panels = ((ax_full, (12.5, f[-1]), [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], "Espectro completo"),
              (ax_zoom, (12.5, max(4.0 * fc, 500.0)), [20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 800],
               "Zoom en graves"))
    for ax, (lo, hi), ticks, name in panels:
        ax.semilogx(f, db_f, color=C_FLOOR, lw=1.0, ls=":", label="Piso estacionario")
        ax.semilogx(f, db_b, color=C_BEFORE, lw=1.6, label="Antes")
        ax.semilogx(f, db_a, color=C_AFTER, lw=2.0, label="Después")
        ax.fill_between(f, db_a, db_b, where=db_b > db_a + 1.0, interpolate=True,
                        color=C_CUT, alpha=0.22, lw=0, label="Lo que se elimina")
        ax.axhline(-rec.params["range_db"], color=C_FLOOR, lw=0.8)
        ax.axvline(fc, color=C_CUT, lw=2.0, ls="--")
        ax.annotate(f"HPF {fc:g} Hz\n{rec.slope_db_oct} dB/oct", (fc, 6), xytext=(-8, -8),
                    textcoords="offset points", ha="right", va="top", color=C_CUT,
                    fontsize=11, fontweight="bold")
        ax.axvline(rec.keep_hz, color=C_KEEP, lw=1.5, ls=":")
        ax.annotate(keep_txt, (rec.keep_hz, 6), xytext=(8, -8), textcoords="offset points",
                    ha="left", va="top", color=C_KEEP, fontsize=10)
        if rec.hum:
            for h in rec.hum["harmonics_hz"]:
                ax.axvline(h, color=C_CUT, lw=0.6, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-80, 8)
        ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
        ax.xaxis.set_major_formatter(fmt_hz)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", alpha=0.45)
        ax.grid(True, which="minor", alpha=0.12)
        ax.set_ylabel("dB sobre el máximo (1/3 oct)")
        ax.set_title(name, loc="left", fontsize=10, color="#495057")
    ax_zoom.set_xlabel("Frecuencia (Hz)")
    ax_full.legend(loc="lower center", fontsize=9, framealpha=0.9, ncol=4)
    fig.suptitle(f"{title}  ·  HPF recomendado: {fc:g} Hz ({rec.slope_db_oct} dB/oct)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
