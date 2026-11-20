"""Window-level anomaly scorers. Higher score = more anomalous.

B0  RMS of the Z axis (the heuristic most plants already have)
B1  Isolation Forest on the 81 standardised DSP features
B2  XGBoost classifier on features (supervised reference; trained only on non-test files)
AE  compact 1D-CNN autoencoder on raw 1 s windows (see autoencoder.py)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from driftguard.features.dsp import FEATURE_NAMES

Z_RMS = FEATURE_NAMES.index("z_rms")


class RMSScorer:
    name = "B0_rms"

    def fit(self, X):
        return self

    def score(self, X):
        return np.asarray(X[:, Z_RMS], dtype=float)


class IForestScorer:
    name = "B1_iforest"

    def __init__(self, seed: int = 0, max_train: int = 60000):
        self.seed, self.max_train = seed, max_train

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            X = X[rng.choice(len(X), self.max_train, replace=False)]
        self.scaler = StandardScaler().fit(X)
        self.model = IsolationForest(n_estimators=300, random_state=self.seed, n_jobs=-1).fit(self.scaler.transform(X))
        return self

    def score(self, X):
        return -self.model.score_samples(self.scaler.transform(X))


class XGBScorer:
    name = "B2_xgb_supervised"

    def __init__(self, seed: int = 0, max_train: int = 120000):
        self.seed, self.max_train = seed, max_train

    def fit(self, X, y):
        from xgboost import XGBClassifier

        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            keep = rng.choice(len(X), self.max_train, replace=False)
            X, y = X[keep], y[keep]
        pos = max(1, int(y.sum()))
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(y) - pos) / pos, random_state=self.seed, n_jobs=1, tree_method="hist", eval_metric="aucpr",
        ).fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


def aggregate(window_scores: np.ndarray, file_idx: np.ndarray, n_files: int, rule: str) -> np.ndarray:
    """Window scores -> one score per file. rule: max | mean | top10 (mean of the top 10% windows, at least 1)."""
    out = np.full(n_files, np.nan)
    order = np.argsort(file_idx, kind="stable")
    fi, ws = file_idx[order], window_scores[order]
    bounds = np.flatnonzero(np.diff(fi)) + 1
    for chunk_f, chunk_s in zip(np.split(fi, bounds), np.split(ws, bounds), strict=True):
        if len(chunk_s) == 0:
            continue
        if rule == "max":
            v = chunk_s.max()
        elif rule == "mean":
            v = chunk_s.mean()
        elif rule == "top10":
            k = max(1, int(np.ceil(0.1 * len(chunk_s))))
            v = np.sort(chunk_s)[-k:].mean()
        else:
            raise ValueError(rule)
        out[chunk_f[0]] = v
    return out
