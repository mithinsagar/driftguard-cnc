import os

import numpy as np
import pytest

from driftguard.calibration.conformal import similarity_weights, weighted_conformal_threshold


def test_weighted_threshold_is_infinite_when_test_point_is_far_from_all_calibration():
    rng = np.random.default_rng(0)
    cal_scores, cal_emb = rng.normal(size=200), rng.normal(size=(200, 4))
    far = np.full(4, 50.0)  # an unseen process far from every calibration recording
    w = similarity_weights(cal_emb, far, bandwidth=1.0)
    assert np.isinf(weighted_conformal_threshold(cal_scores, w, 0.05))
    ess = w.sum() ** 2 / max((w**2).sum(), 1e-300)
    assert ess < 1 / 0.05  # the fallback rule in eval/engine.py routes this case to pooled split conformal


def test_w1acas_runs_and_controls_false_alarms_on_exchangeable_data():
    pytest.importorskip("tsfm_public")
    os.environ["DRIFTGUARD_W1ACAS_INPROC"] = "1"
    from driftguard.calibration.ibm_w1acas import w1acas_flags

    rng = np.random.default_rng(1)
    flags = w1acas_flags(rng.normal(size=300), rng.normal(size=600), 0.05)
    assert flags.mean() < 0.12
