import numpy as np
import pytest

from hpf_recommender import design


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("loss", [0.5, 1.0, 3.0])
def test_max_cutoff_gives_exact_loss(order, loss):
    fc = design.max_cutoff(80.0, order, loss)
    assert design.attenuation_db(80.0, fc, order) == pytest.approx(loss, abs=1e-9)


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
@pytest.mark.parametrize("order", [2, 4])
def test_digital_filter_matches_analog_below_1k(sr, order):
    for keep in (30.0, 82.4, 400.0):
        fc = design.max_cutoff(keep, order, 1.0)
        loss, phase, gd = design.response_at(keep, fc, order, sr)
        assert loss == pytest.approx(1.0, abs=0.02)
        assert phase > 0 and gd > 0


def test_snap_down():
    assert design.snap_down(100.0) == 100.0
    assert design.snap_down(99.9) == 90.0
    for fc in np.geomspace(20, 900, 50):
        s = design.snap_down(fc)
        assert s <= fc and s in design.GRID
        assert s > fc / 1.13            # paso R20 ~12 %


def test_choose_takes_lowest_order_that_rejects():
    keep = 100.0
    junk_hz, need = np.array([25.0]), np.array([20.0])
    fc, n, met, short = design.choose(keep, junk_hz, need, 1.0, [2, 3, 4], 16, 300, tol_db=0.0)
    assert met and short == 0.0
    assert design.attenuation_db(25.0, fc, n) >= 20.0
    fc_prev = design.snap_down(design.max_cutoff(keep, n - 1, 1.0))
    assert design.attenuation_db(25.0, fc_prev, n - 1) < 20.0


def test_choose_reports_shortfall():
    fc, n, met, short = design.choose(100.0, np.array([70.0]), np.array([40.0]), 1.0, [2, 3, 4], 16, 300)
    assert not met and n == 4 and short > 1.0
