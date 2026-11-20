"""Few-shot recalibration for a new machining process (claim C1b).

For each leave-one-process-out fold, the first k healthy recordings of the unseen process (in chronological order) are
confirmed by an operator and added to calibration; false alarms are measured on the remaining healthy recordings of that
process and recall on its anomalous recordings. Strategies:
  pooled+k     split conformal on old calibration files plus the k new ones
  weighted+k   similarity-weighted conformal on the same set (new files get large weights automatically)
  new-only     split conformal on the k new files alone (no finite threshold until k >= 1/alpha - 1)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driftguard.calibration.conformal import similarity_weights, split_conformal_threshold, weighted_conformal_threshold  # noqa: E402
from driftguard.data import splits  # noqa: E402
from driftguard.eval import engine  # noqa: E402

KS = (0, 1, 2, 3, 5, 10, 20)


def main(scorer: str, seed: int, alpha: float):
    df = pd.read_csv("data/manifest.csv")
    cache = np.load("data/cache/features.npz")
    X, file_idx = cache["X"], cache["file_idx"]
    store = np.load(f"results/scores/lopo_{scorer}_seed{seed}.npz")
    raw = {f["fold"]: f for f in json.loads(Path(f"results/raw/lopo/{scorer}_seed{seed}.json").read_text())["folds"]}
    rows = []
    for fold in splits.load("lopo"):
        fd = engine.fold_indices(df, fold, "lopo")
        fs = store[fd.name]
        emb = engine.file_embeddings(X, file_idx, len(df), fd.idx["train_good"])
        bw_info = raw[fd.name]["bandwidth"]
        h = bw_info["factor"] * bw_info["median_dist"]
        test = fd.idx["test"]
        good = test[df.label.to_numpy()[test] == "good"]
        bad = test[df.label.to_numpy()[test] == "bad"]
        good = good[np.lexsort((df["index"].to_numpy()[good], df.t_order.to_numpy()[good]))]
        cal = fd.idx["cal_good"]
        for k in KS:
            if k >= len(good) - 10:
                continue
            new, ev_good = good[:k], good[k:]
            ev = np.concatenate([ev_good, bad])
            y = np.r_[np.zeros(len(ev_good)), np.ones(len(bad))].astype(bool)
            calk = np.concatenate([cal, new])
            t_pool = split_conformal_threshold(fs[calk], alpha)
            t_new = split_conformal_threshold(fs[new], alpha) if k else np.inf
            t_w = np.array([weighted_conformal_threshold(fs[calk], similarity_weights(emb[calk], emb[i], h), alpha) for i in ev])
            for name, t in (("pooled+k", t_pool), ("weighted+k", t_w), ("new-only", t_new)):
                flag = fs[ev] > t
                rows.append({"fold": fd.name, "k": k, "strategy": name, "far": float(flag[~y].mean()),
                             "false_alarms": int(flag[~y].sum()), "n_good": int((~y).sum()),
                             "detected": int(flag[y].sum()), "n_bad": int(y.sum())})
    res = pd.DataFrame(rows)
    res["abs_err"] = (res.far - alpha).abs()
    summ = res.groupby(["strategy", "k"]).agg(mean_far=("far", "mean"), mean_abs_err=("abs_err", "mean"), folds=("fold", "nunique"),
                                              false_alarms=("false_alarms", "sum"), n_good=("n_good", "sum"),
                                              detected=("detected", "sum"), n_bad=("n_bad", "sum")).reset_index()
    summ["pooled_far"] = summ.false_alarms / summ.n_good
    summ["pooled_recall"] = summ.detected / summ.n_bad.where(summ.n_bad > 0)
    tag = f"fewshot_lopo_{scorer}_seed{seed}_a{alpha}"
    Path(f"results/{tag}.json").write_text(json.dumps({"alpha": alpha, "scorer": scorer, "rows": rows,
                                                       "summary": summ.to_dict(orient="records")}, indent=1, default=float))
    print(summ.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {"pooled+k": "#8a99a8", "weighted+k": "#2f7d4f", "new-only": "#6d5bb0"}
    for st, g in summ.groupby("strategy"):
        g = g.sort_values("k")
        ax.plot(g.k, g.mean_abs_err, "o-", color=colors[st], label=st, lw=1.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(list(KS), [str(k) for k in KS])
    ax.set_xlabel("confirmed healthy recordings of the new process added to calibration (k)")
    ax.set_ylabel(f"mean |achieved FAR - {alpha}| across processes")
    ax.set_title("How fast does the false-alarm promise recover for a new machining process?", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"reports/figures/{tag}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="B1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    main(a.scorer, a.seed, a.alpha)
