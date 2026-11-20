"""Windowing and vibration features.

Windows are 1.0 s (2,000 samples x 3 axes) with a 0.5 s hop; each window has its per-axis mean removed so that
gravity and mounting offsets do not look like anomalies.
Features per axis (26): RMS, std, peak-to-peak, crest factor, kurtosis, skewness, zero-crossing rate,
spectral centroid, spectral flatness, 85% roll-off and 16 log-spaced Welch band energies (5 Hz to 1 kHz).
Cross-axis (3): Pearson correlation X-Y, X-Z, Y-Z. Total 81.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

FS = 2000
WIN = 2000
HOP = 1000
N_BANDS = 16
BAND_EDGES = np.geomspace(5.0, 1000.0, N_BANDS + 1)
AXES = ("x", "y", "z")
PER_AXIS = ["rms", "std", "p2p", "crest", "kurtosis", "skew", "zcr", "centroid", "flatness", "rolloff85"] + [
    f"band{i:02d}" for i in range(N_BANDS)
]
FEATURE_NAMES = [f"{a}_{n}" for a in AXES for n in PER_AXIS] + ["corr_xy", "corr_xz", "corr_yz"]
_EPS = 1e-12


def windows(signal: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    """(n, 3) signal -> (n_windows, win, 3) array, per-window per-axis mean removed. Partial tail dropped."""
    n = signal.shape[0]
    if n < win:
        return np.empty((0, win, 3), dtype=np.float32)
    starts = np.arange(0, n - win + 1, hop)
    idx = starts[:, None] + np.arange(win)[None, :]
    w = signal[idx].astype(np.float32)
    return w - w.mean(axis=1, keepdims=True)


def features(w: np.ndarray) -> np.ndarray:
    """(n_windows, win, 3) -> (n_windows, 81) float32 feature matrix."""
    if len(w) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    x = w.astype(np.float64)
    rms = np.sqrt((x**2).mean(axis=1))
    std = x.std(axis=1)
    p2p = x.max(axis=1) - x.min(axis=1)
    peak = np.abs(x).max(axis=1)
    crest = peak / (rms + _EPS)
    c = x - x.mean(axis=1, keepdims=True)
    m2 = (c**2).mean(axis=1) + _EPS
    kurt = (c**4).mean(axis=1) / m2**2
    skew = (c**3).mean(axis=1) / m2**1.5
    zcr = (np.diff(np.signbit(x), axis=1) != 0).mean(axis=1)

    f, pxx = welch(x, fs=FS, nperseg=512, noverlap=256, window="hann", axis=1)  # pxx: (n, n_freq, 3)
    p = pxx + _EPS
    total = p.sum(axis=1)
    centroid = (f[None, :, None] * p).sum(axis=1) / total
    flatness = np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)
    cum = np.cumsum(p, axis=1) / total[:, None, :]
    rolloff = f[np.argmax(cum >= 0.85, axis=1)]
    bands = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True):
        sel = (f >= lo) & (f < hi) if hi < BAND_EDGES[-1] else (f >= lo) & (f <= hi)
        bands.append(np.log10(p[:, sel, :].sum(axis=1) + _EPS))
    bands = np.stack(bands, axis=1)  # (n, 16, 3)

    per_axis = np.stack([rms, std, p2p, crest, kurt, skew, zcr, centroid, flatness, rolloff], axis=1)  # (n, 10, 3)
    per_axis = np.concatenate([per_axis, bands], axis=1)  # (n, 26, 3)
    per_axis = per_axis.transpose(0, 2, 1).reshape(len(x), -1)  # axis-major: x_*, y_*, z_*

    sd = c.std(axis=1) + _EPS
    z = c / sd[:, None, :]
    corr = np.stack([(z[:, :, 0] * z[:, :, 1]).mean(1), (z[:, :, 0] * z[:, :, 2]).mean(1), (z[:, :, 1] * z[:, :, 2]).mean(1)], 1)
    return np.concatenate([per_axis, corr], axis=1).astype(np.float32)
