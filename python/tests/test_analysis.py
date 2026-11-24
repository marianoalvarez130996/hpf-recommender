import numpy as np
import pytest

from hpf_recommender import analysis
from signals import SR, bursts


def test_band_dof_known_cases():
    # un bin entero: la potencia es exponencial, 2 grados de libertad
    assert analysis.band_dof(5, 0.0, 5, 1.0) == pytest.approx(2.0)
    # dos medios bins vecinos (correlación de potencia 4/9): más de 2, menos de 4
    assert analysis.band_dof(5, 0.5, 6, 0.5) == pytest.approx(2.0 / (0.5 + 2 * (4 / 9) * 0.25))
    # banda ancha de m bins: nu -> 2 m / (35/18), no 2 m / 1.5, que es lo que da dividir por el ENBW
    m = 400
    assert analysis.band_dof(10, 0.0, 10 + m, 0.0) / m == pytest.approx(36 / 35, rel=0.01)


def test_floor_subtraction_recovers_intermittent_content():
    # Ruido estacionario, golpes entre 200 Hz y 8 kHz (para que la pista no sea estacionaria) y un
    # tono de 31.5 Hz que suena el 10 % del tiempo y aporta a la media de su banda 1.5 veces lo que
    # aporta el ruido. En esa banda el exceso (media - piso) tiene que ser todo lo que no es el
    # ruido. Con nu = 2 en una banda que tiene casi 3 grados de libertad, el piso del ruido sale
    # 3 dB alto y el exceso se encoge unos 5 dB.
    rng = np.random.default_rng(4)
    n = 120 * SR
    noise = 0.02 * rng.standard_normal(n)
    t = np.arange(n) / SR
    ramp = np.clip(np.minimum(t % 10.0, 1.0 - t % 10.0) / 0.05, 0.0, 1.0) * (t % 10.0 < 1.0)
    tone = 1.9e-3 * np.sin(2 * np.pi * 31.5 * t) * ramp
    music = bursts(120.0, seed=4)
    mix = analysis.long_term_spectrum((noise + tone + music)[:, None], SR)
    only = analysis.long_term_spectrum(noise[:, None], SR)
    b = int(np.argmin(np.abs(np.log(mix.centers / 31.5))))
    assert not mix.stationary
    expected = 10 * np.log10(10 ** (mix.mean_db[b] / 10) - 10 ** (only.mean_db[b] / 10))
    assert expected - only.mean_db[b] == pytest.approx(1.76, abs=0.6)      # el caso está bien armado
    assert mix.excess_db[b] == pytest.approx(expected, abs=1.5)
