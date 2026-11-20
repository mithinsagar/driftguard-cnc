import math

import numpy as np

from driftguard.calibration.conformal import (
    ACI,
    conformal_pvalue,
    min_alpha,
    mondrian_thresholds,
    split_conformal_threshold,
    weighted_conformal_threshold,
)


def test_split_conformal_hand_example():
    cal = np.arange(1, 20)  # 19 scores 1..19
    # alpha = 0.10: k = ceil(20 * 0.9) = 18 -> 18th smallest = 18
    assert split_conformal_threshold(cal, 0.10) == 18
    # alpha = 0.05: k = ceil(20 * 0.95) = 19 -> 19
    assert split_conformal_threshold(cal, 0.05) == 19
    # alpha = 0.01: k = ceil(19.8) = 20 > 19 -> no finite threshold
    assert math.isinf(split_conformal_threshold(cal, 0.01))
    assert math.isclose(min_alpha(19), 0.05)


def test_pvalue_definition():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    assert conformal_pvalue(cal, 3.5) == (1 + 1) / 5
    assert conformal_pvalue(cal, 0.0) == 1.0


def test_false_alarm_rate_is_controlled_under_exchangeability():
    rng = np.random.default_rng(0)
    alpha, fars = 0.1, []
    for _ in range(2000):
        cal = rng.normal(size=99)
        test = rng.normal()
        fars.append(test > split_conformal_threshold(cal, alpha))
    assert np.mean(fars) <= alpha + 0.02


def test_threshold_monotone_in_alpha():
    cal = np.random.default_rng(1).normal(size=200)
    ts = [split_conformal_threshold(cal, a) for a in (0.01, 0.02, 0.05, 0.1, 0.2)]
    assert all(a >= b for a, b in zip(ts, ts[1:], strict=False))


def test_weighted_equals_split_with_uniform_weights():
    cal = np.random.default_rng(2).normal(size=99)
    for a in (0.05, 0.1, 0.2):
        assert weighted_conformal_threshold(cal, np.ones(99), a) == split_conformal_threshold(cal, a)


def test_mondrian_per_group():
    cal = np.concatenate([np.arange(1, 20), np.arange(101, 120)])
    g = np.array(["a"] * 19 + ["b"] * 19)
    t = mondrian_thresholds(cal, g, 0.1)
    assert t["a"] == 18 and t["b"] == 118


def test_aci_moves_toward_target():
    aci = ACI(cal_scores=np.arange(100.0), alpha=0.05, gamma=0.05)
    for _ in range(20):
        aci.feedback(flagged_normal=True)  # too many false alarms -> alpha_t falls -> threshold rises
    assert aci.alpha_t < 0.05
