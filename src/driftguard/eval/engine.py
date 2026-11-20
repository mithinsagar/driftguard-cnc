"""Fold runner: fit scorers on healthy files, aggregate to file scores, calibrate, evaluate on the held-out group.

Calibration strategies (all at a user-chosen false-alarm level alpha):
  C0_static     100*(1 - alpha) percentile of validation healthy file scores (common practice, no guarantee)
  C1_split      split conformal on calibration healthy files (pooled)
  C2_mondrian   split conformal per machining process; pooled fallback when the test process is unseen
  C4_weighted   similarity-weighted conformal: calibration files weighted by a Gaussian kernel on file embeddings;
                kernel width chosen by a nested pseudo-shift on validation data only (never on test)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from driftguard.calibration.conformal import (
    mondrian_thresholds,
    similarity_weights,
    split_conformal_threshold,
    static_percentile,
    weighted_conformal_threshold,
)
from driftguard.models.scorers import aggregate

ALPHAS = (0.01, 0.02, 0.05, 0.10)
AGG_RULES = ("max", "mean", "top10")
BANDWIDTH_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
SHIFT_GROUP = {"lopo": "process", "lomo": "machine", "time": "t_order"}


@dataclass
class FoldData:
    name: str
    proto: str
    idx: dict  # role -> np.ndarray of manifest row indices


def fold_indices(df: pd.DataFrame, fold: dict, proto: str) -> FoldData:
    pos = pd.Series(np.arange(len(df)), index=df["file_id"])
    roles = ("train_good", "val_good", "cal_good", "val_bad", "test")
    return FoldData(fold["name"], proto, {r: pos.loc[fold[r]].to_numpy() for r in roles})


def file_embeddings(X: np.ndarray, file_idx: np.ndarray, n_files: int, train_rows: np.ndarray, dims: int = 16):
    """Mean standardised window features per file, projected with PCA fitted on training files."""
    tr_mask = np.isin(file_idx, train_rows)
    sc = StandardScaler().fit(X[tr_mask])
    Z = sc.transform(X)
    sums = np.zeros((n_files, X.shape[1]))
    np.add.at(sums, file_idx, Z)
    cnt = np.bincount(file_idx, minlength=n_files)[:, None]
    emb = sums / np.maximum(cnt, 1)
    pca = PCA(n_components=dims, random_state=0).fit(emb[train_rows])
    return pca.transform(emb)


def choose_aggregation(window_scores, file_idx, n_files, fd: FoldData) -> tuple[str, dict]:
    val = np.concatenate([fd.idx["val_good"], fd.idx["val_bad"]])
    y = np.r_[np.zeros(len(fd.idx["val_good"])), np.ones(len(fd.idx["val_bad"]))]
    aps = {}
    for rule in AGG_RULES:
        s = aggregate(window_scores, file_idx, n_files, rule)[val]
        ok = ~np.isnan(s)
        aps[rule] = float(average_precision_score(y[ok], s[ok])) if y[ok].sum() > 0 else float("nan")
    best = max(aps, key=lambda r: -1 if math.isnan(aps[r]) else aps[r])
    return best, aps


def choose_bandwidth(file_scores, emb, df, fd: FoldData, alpha: float = 0.05) -> tuple[float, dict]:
    """Nested pseudo-shift on validation data: hold out each shift group among non-test files in turn."""
    col = SHIFT_GROUP[fd.proto]
    cal, val = fd.idx["cal_good"], fd.idx["val_good"]
    groups = df[col].to_numpy()
    cal_emb = emb[cal]
    d = np.sqrt(((cal_emb[:, None, :] - cal_emb[None, :, :]) ** 2).sum(-1))
    med = float(np.median(d[np.triu_indices(len(cal), 1)]))
    errs = {}
    for c in BANDWIDTH_FACTORS:
        h = c * med
        per_group = []
        for g in np.unique(groups[val]):
            pv = val[groups[val] == g]
            pc = cal[groups[cal] != g]
            if len(pv) < 5 or len(pc) < 20:
                continue
            flags = [file_scores[i] > weighted_conformal_threshold(file_scores[pc], similarity_weights(emb[pc], emb[i], h), alpha)
                     for i in pv]
            per_group.append(abs(np.mean(flags) - alpha))
        errs[c] = float(np.mean(per_group)) if per_group else float("inf")
    best = min(errs, key=errs.get)
    return best * med, {"factor": best, "median_dist": med, "pseudo_shift_errors": errs}


def thresholds_for_test(file_scores, emb, df, fd: FoldData, alpha: float, bandwidth: float) -> dict:
    """Return a per-test-file threshold array for each calibration strategy."""
    cal, val, test = fd.idx["cal_good"], fd.idx["val_good"], fd.idx["test"]
    n = len(test)
    out = {
        "C0_static": np.full(n, static_percentile(file_scores[val], 100 * (1 - alpha))),
        "C1_split": np.full(n, split_conformal_threshold(file_scores[cal], alpha)),
    }
    pooled = out["C1_split"][0]
    mt = mondrian_thresholds(file_scores[cal], df["process"].to_numpy()[cal], alpha)
    out["C2_mondrian"] = np.array([mt.get(p, pooled) for p in df["process"].to_numpy()[test]])
    out["C4_weighted"] = np.array([
        weighted_conformal_threshold(file_scores[cal], similarity_weights(emb[cal], emb[i], bandwidth), alpha) for i in test
    ])
    return out


def evaluate(file_scores, df, fd: FoldData, thr: dict, alpha: float) -> dict:
    test = fd.idx["test"]
    y = (df["label"].to_numpy()[test] == "bad").astype(int)
    s = file_scores[test]
    res = {}
    for name, t in thr.items():
        flag = s > t
        n_good, n_bad = int((y == 0).sum()), int(y.sum())
        res[name] = {
            "alpha": alpha, "n_good": n_good, "n_bad": n_bad,
            "false_alarms": int(flag[y == 0].sum()), "far": float(flag[y == 0].mean()) if n_good else float("nan"),
            "detected": int(flag[y == 1].sum()), "recall": float(flag[y == 1].mean()) if n_bad else float("nan"),
            "inf_thresholds": int(np.isinf(t).sum()),
        }
    return res


def threshold_free(file_scores, df, fd: FoldData) -> dict:
    test = fd.idx["test"]
    y = (df["label"].to_numpy()[test] == "bad").astype(int)
    s = file_scores[test]
    if 0 < y.sum() < len(y):
        return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
                "prevalence": float(y.mean())}
    return {"pr_auc": float("nan"), "roc_auc": float("nan"), "prevalence": float(y.mean())}
