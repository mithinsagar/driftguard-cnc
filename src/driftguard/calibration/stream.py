"""Streaming deployment study (claim C2): calibration-buffer contamination and label delay.

Held-out recordings are replayed in chronological order. Operator labels arrive `delay` files late. Policies:
  P0_frozen    threshold from the initial calibration set, never updated
  P1_naive     every file whose score is at or below the current threshold is admitted to the buffer immediately
               (a widely used self-training rule; it truncates the buffer and drags the threshold down)
  P2_delayed   a file is admitted only after its label arrives and confirms it is healthy
  P3_aci       frozen buffer; the working alpha is updated by ACI from delayed false-alarm feedback
  P5_delayed_aci  delayed-label admission (P2) AND an ACI correction of the working alpha from the same delayed labels:
               the buffer adapts the threshold to the new regime, ACI keeps the achieved false-alarm rate on target
Scenarios: "natural" order, and "burst", where the fold's real anomalous files are moved into one contiguous block
starting one third into the stream (no data is synthesised, only the order changes).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from driftguard.calibration.conformal import ACI, RollingBuffer, split_conformal_threshold


def order_stream(t_order: np.ndarray, index: np.ndarray, is_bad: np.ndarray, scenario: str, seed: int = 0,
                 burst_at: float = 1 / 3) -> np.ndarray:
    base = np.lexsort((index, t_order))
    if scenario == "natural":
        return base
    if scenario != "burst":
        raise ValueError(scenario)
    good = base[~is_bad[base]]
    bad = base[is_bad[base]]
    cut = int(len(good) * burst_at)
    return np.concatenate([good[:cut], bad, good[cut:]])


def simulate(scores: np.ndarray, is_bad: np.ndarray, cal_scores: np.ndarray, alpha: float, policy: str,
             delay: int = 20, gamma: float = 0.01, buffer_size: int = 300, label_rate: float = 1.0, seed: int = 0,
             w1_kw: dict | None = None) -> dict:
    """Replay `scores` (already in stream order). Returns per-step thresholds and flags plus summary metrics."""
    n = len(scores)
    if policy == "P4_w1acas":
        from driftguard.calibration.ibm_w1acas import w1acas_pvalues

        pv = w1acas_pvalues(cal_scores, scores, alpha, **(w1_kw or {}))
        flags = pv < alpha
        good, late = ~is_bad, np.arange(n) >= n // 3
        return {"thresholds": np.full(n, np.nan), "flags": flags, "pvalues": pv,
                "far": float(flags[good].mean()) if good.any() else float("nan"),
                "far_after_first_third": float(flags[good & late].mean()) if (good & late).any() else float("nan"),
                "recall": float(flags[is_bad].mean()) if is_bad.any() else float("nan"),
                "threshold_start": float("nan"), "threshold_end": float("nan")}
    thr = np.empty(n)
    flags = np.zeros(n, dtype=bool)
    buf = RollingBuffer(cal_scores, size=buffer_size)
    aci = ACI(np.asarray(cal_scores, float), alpha, gamma=gamma)
    pending: deque = deque()
    labelled = np.random.default_rng(seed).random(n) < label_rate  # which files ever receive an operator label
    for t in range(n):
        if policy == "P3_aci":
            q = aci.threshold()
        elif policy == "P5_delayed_aci":
            q = buf.threshold(float(np.clip(aci.alpha_t, 1e-4, 0.5)))
        elif policy == "P0_frozen":
            q = split_conformal_threshold(cal_scores, alpha)
        else:
            q = buf.threshold(alpha)
        thr[t] = q
        flags[t] = scores[t] > q
        if policy == "P1_naive" and not flags[t]:
            buf.admit(scores[t])
        pending.append(t)
        while pending and pending[0] <= t - delay:
            j = pending.popleft()
            if not labelled[j]:
                continue
            if policy in ("P2_delayed", "P5_delayed_aci") and not is_bad[j]:
                buf.admit(scores[j])
            if policy in ("P3_aci", "P5_delayed_aci") and not is_bad[j]:
                aci.feedback(flagged_normal=bool(flags[j]))
    good = ~is_bad
    third = n // 3
    late = np.arange(n) >= third
    return {
        "thresholds": thr, "flags": flags,
        "far": float(flags[good].mean()) if good.any() else float("nan"),
        "far_after_first_third": float(flags[good & late].mean()) if (good & late).any() else float("nan"),
        "recall": float(flags[is_bad].mean()) if is_bad.any() else float("nan"),
        "threshold_start": float(thr[0]), "threshold_end": float(thr[-1]),
    }


def rolling_far(flags: np.ndarray, is_bad: np.ndarray, window: int = 100) -> np.ndarray:
    """False-alarm rate over the last `window` healthy files, aligned to stream positions (nan until warm)."""
    out = np.full(len(flags), np.nan)
    hist: deque = deque(maxlen=window)
    for t in range(len(flags)):
        if not is_bad[t]:
            hist.append(flags[t])
        if len(hist) == window:
            out[t] = np.mean(hist)
    return out
