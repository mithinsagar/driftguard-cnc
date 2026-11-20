"""Deterministic sensor health gate. Runs before any model; sensor faults are a different alarm from process anomalies."""
from __future__ import annotations

import numpy as np

WIN = 2000


def check_window(w: np.ndarray, flat_tol: float = 1e-6, clip_share: float = 0.005, stuck_run: int = 100) -> str | None:
    """Return a fault code for one raw (2000, 3) window, or None if healthy."""
    if w.ndim != 2 or w.shape != (WIN, 3):
        return "shape"
    if not np.isfinite(w).all():
        return "invalid"
    for a in range(3):
        x = w[:, a]
        rng = float(x.max() - x.min())
        # flatline: any 200-sample stretch (0.1 s) with essentially zero variation over at least 0.2 s total
        seg = x[: (len(x) // 200) * 200].reshape(-1, 200)
        if (seg.std(axis=1) <= flat_tol * max(1.0, abs(float(x.mean())))).sum() >= 2:
            return "flatline"
        if rng > 0:
            at_edge = np.isclose(x, x.max()) | np.isclose(x, x.min())
            if at_edge.mean() >= clip_share and at_edge.sum() >= 10:
                return "clipping"
        change = np.flatnonzero(np.diff(x) != 0)
        runs = np.diff(np.r_[-1, change, len(x) - 1])
        if runs.max() > stuck_run:
            return "stuck"
    return None
