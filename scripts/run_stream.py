"""Streaming contamination study (claim C2) from saved file scores. Writes results/stream_*.json and figures."""
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
from driftguard.calibration.stream import order_stream, rolling_far, simulate  # noqa: E402
from driftguard.data import splits  # noqa: E402
from driftguard.eval import engine  # noqa: E402

POLICIES = ("P0_frozen", "P1_naive", "P2_delayed", "P3_aci")
COLORS = {"P0_frozen": "#8a99a8", "P1_naive": "#b04a4a", "P2_delayed": "#4a7fb5", "P3_aci": "#5b8a52"}


def main(proto: str, scorer: str, seed: int, alpha: float, delays=(5, 20, 50), gammas=(0.005, 0.01, 0.05)):
    df = pd.read_csv("data/manifest.csv")
    store = np.load(f"results/scores/{proto}_{scorer}_seed{seed}.npz")
    rows, curves = [], {}
    for fold in splits.load(proto):
        fd = engine.fold_indices(df, fold, proto)
        fs = store[fd.name]
        test = fd.idx["test"]
        is_bad_all = (df.label.to_numpy()[test] == "bad")
        cal = fs[fd.idx["cal_good"]]
        for scen in ("natural", "burst"):
            order = order_stream(df.t_order.to_numpy()[test], df["index"].to_numpy()[test], is_bad_all, scen, seed)
            s, b = fs[test][order], is_bad_all[order]
            configs = [("P0_frozen", {}), ("P1_naive", {})] + [("P2_delayed", {"delay": d}) for d in delays] + \
                      [("P3_aci", {"delay": 20, "gamma": g}) for g in gammas]
            for pol, kw in configs:
                r = simulate(s, b, cal, alpha, pol, **kw)
                rows.append({"fold": fd.name, "scenario": scen, "policy": pol, **kw, "n": len(s), "n_bad": int(b.sum()),
                             **{k: r[k] for k in ("far", "far_after_first_third", "recall", "threshold_start", "threshold_end")}})
                if (pol, kw.get("delay", 20), kw.get("gamma", 0.01)) in {("P0_frozen", 20, 0.01), ("P1_naive", 20, 0.01),
                                                                          ("P2_delayed", 20, 0.01), ("P3_aci", 20, 0.01)}:
                    curves[(fd.name, scen, pol)] = (r["thresholds"], rolling_far(r["flags"], b), b)
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    tag = f"stream_{proto}_{scorer}_seed{seed}_a{alpha}"
    summary = res.groupby(["scenario", "policy", "delay", "gamma"], dropna=False)[["far", "far_after_first_third", "recall"]].mean()
    Path(f"results/{tag}.json").write_text(json.dumps({
        "protocol": proto, "scorer": scorer, "alpha": alpha, "rows": res.to_dict(orient="records"),
        "summary": summary.reset_index().to_dict(orient="records")}, indent=1, default=float))
    print(summary.round(3).to_string())

    # Figure: the fold with the most test files, burst scenario.
    big = res.groupby("fold").n.max().idxmax()
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for pol in POLICIES:
        thr, rf, b = curves[(big, "burst", pol)]
        axes[0].plot(thr, color=COLORS[pol], label=pol, lw=1.4)
        axes[1].plot(rf, color=COLORS[pol], lw=1.4)
    xs = np.flatnonzero(b)
    for ax in axes:
        ax.axvspan(xs.min(), xs.max(), color="#f3d6d6", alpha=0.6, lw=0)
    axes[1].axhline(alpha, color="black", ls="--", lw=1, label=f"target alpha={alpha}")
    axes[0].set_ylabel("threshold")
    axes[1].set_ylabel("rolling FAR (last 100 healthy)")
    axes[1].set_xlabel(f"files in stream order ({big}; shaded = burst of real anomalous files)")
    axes[0].legend(fontsize=7, ncol=4)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"reports/figures/{tag}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="lomo")
    ap.add_argument("--scorer", default="B1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    main(a.proto, a.scorer, a.seed, a.alpha)
