"""Calibración de la corrección chi² del piso con ruido gaussiano estacionario (blanco y rosa).
Para cada banda: percentil 10 / media medido en tramas cortas contra el modelo chi2.ppf(0.1, nu)/nu.
Escribe results/floor_calibration.json."""
import json

import numpy as np
from scipy import signal, stats

from _paths import RESULTS
from hpf_recommender import analysis

SR, DUR = 48000, 300.0
rng = np.random.default_rng(11)
n = int(DUR * SR)
pink = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                      [1, -2.494956002, 2.017265875, -0.522189400], rng.standard_normal(n))
signals = {"white": 0.05 * rng.standard_normal(n), "pink": 0.1 * pink / np.std(pink)}
centers = analysis.band_centers(12.5, 16000.0)
out = {"centers": centers.round(2).tolist()}
for name, x in signals.items():
    bands, _, _, _, nu, nfft = analysis._frame_powers(x[:, None], SR, centers, 1 / 3, 6.0, overlap=2)
    df = SR / nfft
    measured = 10 * np.log10(np.percentile(bands, 10, axis=0) / bands.mean(axis=0))
    model = 10 * np.log10(stats.chi2.ppf(0.1, nu) / nu)
    # la fórmula del ancho entre el ENBW, con nu >= 2, para comparar
    nu_enbw = 2 * np.maximum((centers * (2 ** (1 / 6) - 2 ** (-1 / 6))) / df / 1.5, 1.0)
    enbw = 10 * np.log10(stats.chi2.ppf(0.1, nu_enbw) / nu_enbw)
    sel = centers >= 20
    out[name] = {"measured_db": measured.round(3).tolist(), "model_db": model.round(3).tolist(), "enbw_db": enbw.round(3).tolist(),
                 "frames": int(bands.shape[0])}
    err, err_e = (measured - model)[sel], (measured - enbw)[sel]
    print(f"{name}: piso estimado / media real, bandas >= 20 Hz. Satterthwaite: mediana {np.median(err):+.2f}, "
          f"rango {err.min():+.2f} a {err.max():+.2f} dB. Ancho/ENBW: mediana {np.median(err_e):+.2f}, rango {err_e.min():+.2f} a {err_e.max():+.2f} dB")
    for f in (20, 25, 31.5, 40, 50, 63, 100, 250, 1000, 4000):
        b = int(np.argmin(np.abs(np.log(centers / f))))
        print(f"   {centers[b]:7.1f} Hz: medido {measured[b]:6.2f}, Satterthwaite {model[b]:6.2f}, ENBW {enbw[b]:6.2f} dB")
(RESULTS / "floor_calibration.json").write_text(json.dumps(out))
