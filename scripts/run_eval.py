"""make eval: run scorers x calibration strategies over the split protocols and write results/ with provenance.

Usage: python scripts/run_eval.py --proto lopo lomo time --scorers B0 B1 B2 AE --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driftguard.data import splits  # noqa: E402
from driftguard.data.dataset import load_signal  # noqa: E402
from driftguard.eval import engine  # noqa: E402
from driftguard.features.dsp import windows  # noqa: E402
from driftguard.models.scorers import IForestScorer, RMSScorer, XGBScorer, aggregate  # noqa: E402

RES = Path("results")
AE_EPOCHS = 8
# IBM granite-tsfm provides baseline C5; it runs in a spawned worker (see calibration/ibm_w1acas.py).
USE_W1ACAS = importlib.util.find_spec("tsfm_public") is not None
_SIG: dict[int, np.ndarray] = {}


def provenance(seed: int, extra: dict | None = None) -> dict:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "uncommitted"
    card = json.loads(Path("data/README.md").read_text())
    return {"git_commit": commit, "data_checksum": card["manifest_sha256_of_hashes"], "seed": seed,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), **(extra or {})}


def signal(df: pd.DataFrame, row: int) -> np.ndarray:
    if row not in _SIG:
        _SIG[row] = load_signal(df.path.iloc[row])
    return _SIG[row]


def ae_window_scores(df, fd, file_idx, seed, log) -> tuple[np.ndarray, dict]:
    from driftguard.models.autoencoder import AEScorer

    rng = np.random.default_rng(seed)

    def sample(rows, per_file):
        out = []
        for r in rows:
            w = windows(signal(df, r))
            if len(w):
                out.append(w[rng.choice(len(w), min(per_file, len(w)), replace=False)])
        return np.concatenate(out)

    tr = sample(fd.idx["train_good"], 24)
    va = sample(fd.idx["val_good"], 6)
    t0 = time.time()
    model = AEScorer(seed=seed, epochs=AE_EPOCHS).fit(tr, va, log=log)
    fit_s = time.time() - t0
    if fd.proto == "time" and seed == 0:  # the deployment model: export_edge.py quantizes exactly this network
        import torch

        Path("artifacts").mkdir(exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.model.state_dict().items()}, "mu": model.mu, "sd": model.sd,
                    "seed": seed, "fold": fd.name, "epochs": AE_EPOCHS}, "artifacts/ae_time_fold.pt")
    scores = np.empty(len(file_idx), dtype=float)
    for r in range(len(df)):
        sel = np.flatnonzero(file_idx == r)
        if len(sel):
            scores[sel] = model.score(windows(signal(df, r)))
    return scores, {"train_windows": len(tr), "fit_seconds": round(fit_s, 1), "params": model.n_params()}


def run(protos, scorer_names, seed, quick=False):
    df = pd.read_csv("data/manifest.csv")
    cache = np.load("data/cache/features.npz")
    X, file_idx = cache["X"], cache["file_idx"]
    n_files = len(df)
    y_win = (df.label.to_numpy()[file_idx] == "bad").astype(int)
    for proto in protos:
        folds = splits.load(proto)
        if quick:
            folds = folds[:3]
        out = {name: {"protocol": proto, "scorer": name, "folds": [], "provenance": provenance(seed)} for name in scorer_names}
        file_scores_store = {name: {} for name in scorer_names}
        for fold in folds:
            fd = engine.fold_indices(df, fold, proto)
            emb = engine.file_embeddings(X, file_idx, n_files, fd.idx["train_good"])
            train_mask = np.isin(file_idx, fd.idx["train_good"])
            for name in scorer_names:
                t0 = time.time()
                info = {}
                if name == "B0":
                    ws = RMSScorer().score(X)
                elif name == "B1":
                    ws = IForestScorer(seed=seed).fit(X[train_mask]).score(X)
                elif name == "B2":
                    sup_rows = np.concatenate([fd.idx["train_good"], fd.idx["val_bad"]])
                    m = np.isin(file_idx, sup_rows)
                    ws = XGBScorer(seed=seed).fit(X[m], y_win[m]).score(X)
                elif name == "AE":
                    ws, info = ae_window_scores(df, fd, file_idx, seed, log=lambda s, f=fd.name: print(f"  [{f}] {s}", flush=True))
                else:
                    raise ValueError(name)
                rule, aps = engine.choose_aggregation(ws, file_idx, n_files, fd)
                fs = aggregate(ws, file_idx, n_files, rule)
                fs = np.where(np.isnan(fs), np.nanmedian(fs), fs)
                bw, bw_info = engine.choose_bandwidth(fs, emb, df, fd)
                rec = {"fold": fd.name, "aggregation": rule, "val_pr_auc_by_rule": aps, "bandwidth": bw_info,
                       "threshold_free": engine.threshold_free(fs, df, fd), "by_alpha": {}, "scorer_info": info,
                       "n_cal": int(len(fd.idx["cal_good"]))}
                for a in engine.ALPHAS:
                    thr = engine.thresholds_for_test(fs, emb, df, fd, a, bw)
                    extra = {"C5_ibm_w1acas": engine.w1acas_test_flags(fs, df, fd, a)} if USE_W1ACAS else None
                    rec["by_alpha"][str(a)] = engine.evaluate(fs, df, fd, thr, a, extra)
                rec["seconds"] = round(time.time() - t0, 1)
                out[name]["folds"].append(rec)
                file_scores_store[name][fd.name] = fs.astype(np.float32)
                tf = rec["threshold_free"]
                c = rec["by_alpha"]["0.05"]
                print(f"{proto} {fd.name} {name}: agg={rule} PR-AUC={tf['pr_auc']:.3f} | FAR@5%: "
                      + " ".join(f"{k}={v['far']:.3f}" for k, v in c.items()) + f" | {rec['seconds']}s", flush=True)
        (RES / "raw" / proto).mkdir(parents=True, exist_ok=True)
        (RES / "scores").mkdir(parents=True, exist_ok=True)
        for name in scorer_names:
            tag = f"{name}_seed{seed}" + ("_quick" if quick else "")
            (RES / "raw" / proto / f"{tag}.json").write_text(json.dumps(out[name], indent=1))
            np.savez_compressed(RES / "scores" / f"{proto}_{tag}.npz", **file_scores_store[name])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", nargs="+", default=["lopo", "lomo", "time"])
    ap.add_argument("--scorers", nargs="+", default=["B0", "B1", "B2", "AE"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ae-epochs", type=int, default=8)
    a = ap.parse_args()
    if "B2" in a.scorers and "AE" in a.scorers:
        raise SystemExit("Run B2 (XGBoost) and AE (PyTorch) in separate invocations: their OpenMP runtimes crash together on macOS.")
    if "B2" in a.scorers:
        import xgboost  # noqa: F401
    AE_EPOCHS = a.ae_epochs
    run(a.proto, a.scorers, a.seed, a.quick)
    _ = hashlib  # provenance helpers keep hashlib available for future checksum fields
