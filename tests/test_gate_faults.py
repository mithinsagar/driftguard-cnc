import numpy as np
import pytest

from driftguard.serve.gate import check_window


def _healthy(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(2000) / 2000
    base = np.stack([np.sin(2 * np.pi * f * t) for f in (37, 91, 143)], axis=1)
    return (base + 0.3 * rng.normal(size=(2000, 3))).astype(np.float32)


def test_healthy_windows_pass():
    fails = [check_window(_healthy(s)) for s in range(300)]
    assert sum(f is not None for f in fails) == 0


@pytest.mark.parametrize("fault", ["invalid", "flatline", "clipping", "stuck", "shape"])
def test_injected_faults_detected(fault):
    w = _healthy()
    if fault == "invalid":
        w[100, 1] = np.nan
    elif fault == "flatline":
        w[400:1000, 2] = 0.0
    elif fault == "clipping":
        hi = np.percentile(w[:, 0], 97)
        w[:, 0] = np.minimum(w[:, 0], hi)
    elif fault == "stuck":
        w[1200:1400, 0] = 0.123
    elif fault == "shape":
        w = w[:1500]
    assert check_window(w) == fault
