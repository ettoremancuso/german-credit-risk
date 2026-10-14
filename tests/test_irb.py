import numpy as np
import pytest
from scipy.stats import norm

from irb import PD_FLOOR_RETAIL, recalibrate, risk_weight


def test_risk_weight_matches_hand_calculation():
    # PD 1%, LGD 45%, worked out step by step from CRR Art. 154
    pd_, lgd = 0.01, 0.45
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    rho = 0.03 * w + 0.16 * (1 - w)
    k = lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho)) - pd_ * lgd
    assert risk_weight(pd_, lgd) == pytest.approx(12.5 * k)
    assert 0.40 < risk_weight(pd_, lgd) < 0.50   # sanity: ~45% for a 1% PD consumer loan


def test_risk_weight_increases_with_pd_in_normal_range():
    grid = np.linspace(0.001, 0.20, 200)
    assert np.all(np.diff(risk_weight(grid, 0.45)) > 0)


def test_pd_floor_is_applied():
    assert risk_weight(0.0, 0.45) == pytest.approx(risk_weight(PD_FLOOR_RETAIL, 0.45))


def test_recalibration_hits_target_and_keeps_ranking():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.01, 0.9, 500)
    new = recalibrate(raw, 0.03)
    assert new.mean() == pytest.approx(0.03, abs=1e-6)
    assert np.array_equal(np.argsort(raw), np.argsort(new))
