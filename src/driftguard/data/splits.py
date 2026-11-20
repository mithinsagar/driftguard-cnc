"""Leak-free, file-level split protocols.

Every fold assigns whole recordings (file_id) to exactly one role:
  train_good  healthy files used to fit scorers and feature scalers
  val_good    healthy files used for model selection (aggregation rule, kernel bandwidth)
  cal_good    healthy files used only to calibrate thresholds
  val_bad     anomalous files from the non-test part, used only for model selection and the supervised reference
  test        every file of the held-out group (healthy and anomalous)

Protocols
  lopo   leave-one-process-out: test = one machining process (all machines)
  lomo   leave-one-machine-out: test = one machine (M03 has only 9 anomalous files: indicative)
  time   chronological: train on earlier timeframes, test on the latest ones
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("train_good", "val_good", "cal_good", "val_bad", "test")


def _split_good(good: pd.DataFrame, seed: int, frac=(0.6, 0.2, 0.2)) -> dict:
    """Stratified (machine x process) split of healthy files into train / val / cal."""
    rng = np.random.default_rng(seed)
    out = {"train_good": [], "val_good": [], "cal_good": []}
    for _, g in good.groupby(["machine", "process"], sort=True):
        ids = g["file_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = max(1, int(round(frac[0] * n)))
        n_va = int(round(frac[1] * n)) if n >= 3 else 0
        out["train_good"] += ids[:n_tr].tolist()
        out["val_good"] += ids[n_tr:n_tr + n_va].tolist()
        out["cal_good"] += ids[n_tr + n_va:].tolist()
    return out


def make_fold(df: pd.DataFrame, test_mask: np.ndarray, seed: int, name: str) -> dict:
    rest = df[~test_mask]
    fold = _split_good(rest[rest.label == "good"], seed)
    fold["val_bad"] = rest.loc[rest.label == "bad", "file_id"].tolist()
    fold["test"] = df.loc[test_mask, "file_id"].tolist()
    fold["name"] = name
    return fold


def time_cutoff(df: pd.DataFrame, test_share: float = 0.3) -> int:
    """Smallest t_order such that files at or after it make up about `test_share` of the data and include anomalies."""
    orders = sorted(df["t_order"].unique())
    for t in orders:
        tail = df[df.t_order >= t]
        if len(tail) / len(df) <= test_share and (tail.label == "bad").any():
            return int(t)
    return int(orders[-1])


def all_folds(df: pd.DataFrame, seed: int = 0) -> dict:
    folds = {"lopo": [], "lomo": [], "time": []}
    for op in sorted(df.process.unique()):
        folds["lopo"].append(make_fold(df, (df.process == op).to_numpy(), seed, f"lopo_{op}"))
    for m in sorted(df.machine.unique()):
        folds["lomo"].append(make_fold(df, (df.machine == m).to_numpy(), seed, f"lomo_{m}"))
    cut = time_cutoff(df)
    folds["time"].append(make_fold(df, (df.t_order >= cut).to_numpy(), seed, f"time_from_{cut}"))
    return folds


def check_no_leakage(fold: dict) -> None:
    seen = {}
    for role in ROLES:
        for fid in fold[role]:
            if fid in seen:
                raise AssertionError(f"{fold['name']}: {fid} in both {seen[fid]} and {role}")
            seen[fid] = role


def save(folds: dict, out_dir: Path = Path("splits")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for proto, fl in folds.items():
        (out_dir / f"{proto}.json").write_text(json.dumps(fl, indent=1))


def load(proto: str, out_dir: Path = Path("splits")) -> list[dict]:
    return json.loads((out_dir / f"{proto}.json").read_text())
