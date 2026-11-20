"""Calibration rules that turn an anomaly score into a decision at a user-chosen false-alarm rate alpha.

All rules flag a unit when its score is strictly greater than the threshold.
- static_percentile: the common practice (a fixed percentile of validation scores); no guarantee.
- split_conformal_threshold: finite-sample false-alarm control under exchangeability (Bates et al.).
- mondrian_thresholds: split conformal within known groups (machining process).
- weighted_conformal_threshold: calibration points reweighted by similarity to the test unit
  (Tibshirani et al. 2019 style, with the test point's own mass placed at +infinity).
- ACI: adaptive conformal inference (Gibbs and Candes 2021), driven by possibly delayed error feedback.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def static_percentile(scores: np.ndarray, q: float = 95.0) -> float:
    return float(np.percentile(np.asarray(scores, dtype=float), q))


def split_conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """k-th smallest calibration score, k = ceil((n+1)(1-alpha)); +inf when alpha is below 1/(n+1)."""
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(s[k - 1])


def conformal_pvalue(cal_scores: np.ndarray, score: float) -> float:
    s = np.asarray(cal_scores, dtype=float)
    return float((1 + np.sum(s >= score)) / (len(s) + 1))


def min_alpha(n_cal: int) -> float:
    """Smallest false-alarm level a split-conformal threshold can promise with n_cal calibration points."""
    return 1.0 / (n_cal + 1)


def mondrian_thresholds(cal_scores: np.ndarray, groups: np.ndarray, alpha: float) -> dict:
    cal_scores, groups = np.asarray(cal_scores, float), np.asarray(groups)
    return {g: split_conformal_threshold(cal_scores[groups == g], alpha) for g in np.unique(groups)}


def weighted_conformal_threshold(cal_scores: np.ndarray, weights: np.ndarray, alpha: float, test_weight: float = 1.0) -> float:
    """Smallest t with sum_{s_i <= t} p_i >= 1 - alpha, where p_i = w_i / (sum w + w_test) and the test point's
    mass w_test sits at +inf. Returns +inf when the calibration mass cannot reach 1 - alpha."""
    s = np.asarray(cal_scores, float)
    w = np.clip(np.asarray(weights, float), 0, None)
    order = np.argsort(s)
    s, w = s[order], w[order]
    total = w.sum() + test_weight
    if total <= 0:
        return math.inf
    cdf = np.cumsum(w) / total
    idx = np.searchsorted(cdf, 1 - alpha - 1e-12, side="left")
    if idx >= len(s):
        return math.inf
    return float(s[idx])


def similarity_weights(cal_emb: np.ndarray, test_emb: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian kernel weights from Euclidean distance between file-level embeddings."""
    d2 = ((np.asarray(cal_emb) - np.asarray(test_emb)[None, :]) ** 2).sum(axis=1)
    return np.exp(-d2 / (2 * bandwidth**2))


@dataclass
class ACI:
    """Adaptive conformal inference over a fixed calibration set, with error feedback that may arrive late.

    alpha_t+1 = alpha_t + gamma * (alpha - err_t), clipped to [1e-4, 0.5]; err_t = 1 if a confirmed-normal unit was flagged.
    """

    cal_scores: np.ndarray
    alpha: float
    gamma: float = 0.01
    alpha_t: float = field(init=False)

    def __post_init__(self):
        self.alpha_t = self.alpha

    def threshold(self) -> float:
        return split_conformal_threshold(self.cal_scores, float(np.clip(self.alpha_t, 1e-4, 0.5)))

    def feedback(self, flagged_normal: bool) -> None:
        err = 1.0 if flagged_normal else 0.0
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - err), 1e-4, 0.5))


@dataclass
class RollingBuffer:
    """Calibration buffer of the last `size` admitted normal scores."""

    initial: np.ndarray
    size: int = 300
    buf: deque = field(init=False)

    def __post_init__(self):
        self.buf = deque(np.asarray(self.initial, float)[-self.size:], maxlen=self.size)

    def threshold(self, alpha: float) -> float:
        return split_conformal_threshold(np.fromiter(self.buf, float), alpha)

    def admit(self, score: float) -> None:
        self.buf.append(float(score))
