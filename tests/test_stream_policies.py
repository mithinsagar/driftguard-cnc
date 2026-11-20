"""Mechanism behind claim C2 on synthetic, perfectly exchangeable healthy scores."""
import numpy as np

from driftguard.calibration.stream import order_stream, simulate


def _stream(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), np.zeros(n, dtype=bool), rng.normal(size=300)


def test_naive_admission_drags_threshold_down_even_without_faults():
    s, bad, cal = _stream()
    naive = simulate(s, bad, cal, 0.05, "P1_naive")
    frozen = simulate(s, bad, cal, 0.05, "P0_frozen")
    assert naive["threshold_end"] < naive["threshold_start"]
    assert naive["far_after_first_third"] > 2 * 0.05
    assert abs(frozen["far"] - 0.05) < 0.02


def test_delayed_label_admission_holds_the_target():
    s, bad, cal = _stream(seed=1)
    r = simulate(s, bad, cal, 0.05, "P2_delayed", delay=20)
    assert abs(r["far"] - 0.05) < 0.02


def test_aci_long_run_error_near_target():
    s, bad, cal = _stream(seed=2)
    r = simulate(s + 0.3, bad, cal, 0.05, "P3_aci", delay=20, gamma=0.02)  # shifted healthy stream
    frozen = simulate(s + 0.3, bad, cal, 0.05, "P0_frozen")
    assert abs(r["far"] - 0.05) < abs(frozen["far"] - 0.05)


def test_burst_order_keeps_all_files():
    t = np.repeat(np.arange(10), 10)
    idx = np.tile(np.arange(10), 10)
    is_bad = np.zeros(100, bool)
    is_bad[::17] = True
    o = order_stream(t, idx, is_bad, "burst")
    assert sorted(o.tolist()) == list(range(100))
    cut = int((~is_bad).sum()) // 3
    assert is_bad[o][cut:cut + is_bad.sum()].all()
