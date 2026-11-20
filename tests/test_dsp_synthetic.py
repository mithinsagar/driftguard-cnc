import numpy as np

from driftguard.features.dsp import BAND_EDGES, FEATURE_NAMES, FS, WIN, features, windows


def _sine(freq, amp=1.0, n=WIN, axes=3, offset=0.0):
    t = np.arange(n) / FS
    s = amp * np.sin(2 * np.pi * freq * t) + offset
    return np.repeat(s[:, None], axes, axis=1).astype(np.float32)


def _col(name):
    return FEATURE_NAMES.index(name)


def test_feature_count():
    assert len(FEATURE_NAMES) == 81


def test_windowing_shape_and_mean_removal():
    sig = _sine(50, offset=9.81, n=5000)
    w = windows(sig)
    assert w.shape == (4, WIN, 3)  # starts 0, 1000, 2000, 3000
    assert np.allclose(w.mean(axis=1), 0, atol=1e-4)


def test_short_signal_gives_no_windows():
    assert windows(np.zeros((1999, 3), np.float32)).shape[0] == 0


def test_rms_crest_of_sine():
    f = features(windows(_sine(50, amp=2.0)))
    assert np.isclose(f[0, _col("x_rms")], 2.0 / np.sqrt(2), rtol=1e-3)
    assert np.isclose(f[0, _col("x_crest")], np.sqrt(2), rtol=1e-2)
    assert np.isclose(f[0, _col("x_kurtosis")], 1.5, rtol=1e-2)  # kurtosis of a sine is 1.5


def test_spectral_energy_lands_in_right_band():
    freq = 120.0
    f = features(windows(_sine(freq)))
    band = int(np.searchsorted(BAND_EDGES, freq) - 1)
    bands = [f[0, _col(f"x_band{i:02d}")] for i in range(16)]
    assert int(np.argmax(bands)) == band
    assert abs(f[0, _col("x_centroid")] - freq) < 15


def test_cross_axis_correlation_identical_axes():
    f = features(windows(_sine(80)))
    assert np.isclose(f[0, _col("corr_xy")], 1.0, atol=1e-4)
