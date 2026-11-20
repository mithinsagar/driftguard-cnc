"""The service's online calibrator must make exactly the decisions of policy P5 in the streaming study."""
import numpy as np

from driftguard.calibration.adaptive import AdaptiveCalibrator
from driftguard.calibration.stream import simulate


def test_matches_streaming_policy_p5():
    rng = np.random.default_rng(3)
    cal = rng.normal(size=300)
    scores = np.r_[rng.normal(0.4, 1.0, 400), rng.normal(3, 1, 15), rng.normal(0.4, 1.0, 300)]  # shifted healthy + burst
    bad = np.r_[np.zeros(400, bool), np.ones(15, bool), np.zeros(300, bool)]
    delay, alpha, gamma = 5, 0.05, 0.005
    ref = simulate(scores, bad, cal, alpha, "P5_delayed_aci", delay=delay, gamma=gamma, buffer_size=300)
    cal_obj = AdaptiveCalibrator(cal, alpha=alpha, gamma=gamma, buffer_size=300)
    flags = []
    for t, s in enumerate(scores):
        flags.append(cal_obj.record(str(t), s))
        j = t - delay  # the label of run j arrives now
        if j >= 0:
            cal_obj.feedback(str(j), healthy=not bad[j])
    assert np.array_equal(np.array(flags), ref["flags"])


def test_faulty_confirmations_never_enter_the_buffer():
    cal_obj = AdaptiveCalibrator(np.zeros(50) + 1.0, alpha=0.1)
    cal_obj.record("a", 99.0)
    cal_obj.feedback("a", healthy=False)
    assert max(cal_obj.buf.buf) == 1.0 and cal_obj.n_confirmed_faulty == 1


def test_unknown_or_repeated_feedback_is_rejected():
    cal_obj = AdaptiveCalibrator(np.arange(100.0))
    cal_obj.record("r1", 5.0)
    assert cal_obj.feedback("r1", True) and not cal_obj.feedback("r1", True) and not cal_obj.feedback("nope", True)
