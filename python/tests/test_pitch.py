import numpy as np
import pytest

from hpf_recommender.pitch import estimate_min_f0
from signals import SR, harmonic, only_harmonics

E1, E2, B2 = 41.20, 82.41, 123.47


def cents(a, b):
    return abs(1200 * np.log2(a / b))


def test_bass_low_e():
    r = estimate_min_f0(harmonic(E1, 2.0), SR)
    assert cents(r.f0_min, E1) < 20
    assert r.voiced_energy > 0.9


def test_weak_fundamental_is_not_an_octave_up():
    # bajo con micrófono: fundamental 20 dB bajo el segundo armónico
    r = estimate_min_f0(harmonic(E1, 2.0, fund_db=-20.0), SR)
    assert cents(r.f0_min, E1) < 20


def test_power_chord_gives_lowest_note_not_common_period():
    x = harmonic(E2, 2.0) + harmonic(B2, 2.0)
    r = estimate_min_f0(x, SR)
    assert cents(r.f0_min, E2) < 20          # YIN solo daría 41 Hz


def test_missing_fundamental_protects_lowest_partial():
    # sin energía en 55 Hz no hay nada que proteger ahí: lo más grave es 110 Hz
    r = estimate_min_f0(only_harmonics(55.0, range(2, 9), 2.0), SR)
    assert cents(r.f0_min, 110.0) < 20


def test_rare_low_note_is_kept():
    x = np.concatenate([harmonic(E2, 3.0), harmonic(E1, 0.5)])
    r = estimate_min_f0(x, SR)
    assert cents(r.f0_min, E1) < 20


def test_noise_has_no_note():
    x = 0.1 * np.random.default_rng(3).standard_normal(3 * SR)
    r = estimate_min_f0(x, SR)
    assert r.f0_min is None
    assert r.voiced_energy < 0.05
