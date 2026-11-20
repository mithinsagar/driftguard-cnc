"""Baseline C5: IBM Research W1-ACAS (Martinez Gil et al., ICLR 2026, arXiv 2604.20122), run with IBM's own code.

We call `tsfm_public.toolkit.conformal.AdaptiveWeightedConformalScoreWrapper` from the `granite-tsfm` package
(Apache-2.0) on our stream of file-level nonconformity scores, with the defaults of IBM's reference script
(n_epochs=1, n_batch_update=10, lr=1e-3, prior_past_weights_value="proximity", uniform initial weighting).
Difference from IBM's pipeline: their scores are forecast errors of a time-series foundation model and calibration
starts from the first 1/alpha points of the stream; here the scores are our scorers' file scores and calibration starts
from the healthy calibration recordings. An alarm is raised when the returned p-value is below alpha.
"""
from __future__ import annotations

import math

import numpy as np


def w1acas_pvalues(cal_scores: np.ndarray, stream_scores: np.ndarray, alpha: float, n_epochs: int = 1, n_batch_update: int = 10,
                   lr: float = 1e-3, prior_past_weights_value="proximity") -> np.ndarray:
    try:
        from tsfm_public.toolkit.conformal import AdaptiveWeightedConformalScoreWrapper
    except ImportError as e:  # pragma: no cover
        raise ImportError("Baseline C5 needs IBM's granite-tsfm: pip install granite-tsfm") from e
    cal = np.asarray(cal_scores, dtype=float)
    window = int(min(max(len(cal), math.ceil(1 / alpha) + 1), 5000))
    awcs = AdaptiveWeightedConformalScoreWrapper(
        false_alarm=alpha, window_size=window, weighting="uniform",
        weighting_params={"n_batch_update": n_batch_update, "conformal_weights_update": False, "stride": 1, "lr": lr,
                          "n_epochs": n_epochs, "prior_past_weights_value": prior_past_weights_value},
    )
    awcs.fit(cal)
    return np.asarray(awcs.predict(np.asarray(stream_scores, dtype=float)), dtype=float)


def w1acas_flags(cal_scores, stream_scores, alpha, **kw) -> np.ndarray:
    return w1acas_pvalues(cal_scores, stream_scores, alpha, **kw) < alpha
