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

POLICIES = ("P0_frozen", "P1_naive", "P2_delayed", "P3_aci", "P4_w1acas")
COLORS = {"P0_frozen": "#8a99a8", "P1_naive": "#b04a4a", "P2_delayed": "#4a7fb5", "P3_aci": "#5b8a52", "P4_w1acas": "#c0392b"}


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
                      [("P3_aci", {"delay": 20, "gamma": g}) for g in gammas] + [("P4_w1acas", {})]
            for pol, kw in configs:
                r = simulate(s, b, cal, alpha, pol, **kw)
                rows.append({"fold": fd.name, "scenario": scen, "policy": pol, **kw, "n": len(s), "n_bad": int(b.sum()),
                             **{k: r[k] for k in ("far", "far_after_first_third", "recall", "threshold_start", "threshold_end")}})
                if (pol, kw.get("delay", 20), kw.get("gamma", 0.01)) in {("P0_frozen", 20, 0.01), ("P1_naive", 20, 0.01),
                                                                          ("P2_delayed", 20, 0.01), ("P3_aci", 20, 0.01),
                                                                          ("P4_w1acas", 20, 0.01)}:
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
        if not np.all(np.isnan(thr)):
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


def sweep(proto: str, scorer: str, seed: int, alphas=(0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15), scenario: str = "burst",
          label_rates=(1.0,)):
    """Achieved FAR vs recall for each policy across alpha (pooled over folds): fair comparison at matched FAR."""
    df = pd.read_csv("data/manifest.csv")
    store = np.load(f"results/scores/{proto}_{scorer}_seed{seed}.npz")
    configs = [("P0_frozen", {}), ("P1_naive", {}), ("P2_delayed", {"delay": 5}), ("P2_delayed", {"delay": 20}),
               ("P3_aci", {"delay": 20, "gamma": 0.005}), ("P4_w1acas", {}),
               ("P5_delayed_aci", {"delay": 5, "gamma": 0.005}), ("P5_delayed_aci", {"delay": 20, "gamma": 0.005})]
    configs += [("P5_delayed_aci", {"delay": 5, "gamma": 0.005, "label_rate": r}) for r in label_rates if r < 1.0]
    rows = []
    for fold in splits.load(proto):
        fd = engine.fold_indices(df, fold, proto)
        fs, test = store[fd.name], fd.idx["test"]
        is_bad = df.label.to_numpy()[test] == "bad"
        order = order_stream(df.t_order.to_numpy()[test], df["index"].to_numpy()[test], is_bad, scenario, seed)
        s, b, cal = fs[test][order], is_bad[order], fs[fd.idx["cal_good"]]
        for a in alphas:
            for pol, kw in configs:
                r = simulate(s, b, cal, a, pol, **kw)
                fl = r["flags"]
                rows.append({"fold": fd.name, "alpha": a, "policy": pol + (f"_d{kw['delay']}" if pol in ("P2_delayed", "P5_delayed_aci") else "")
                             + (f"_labels{int(100 * kw['label_rate'])}" if kw.get("label_rate", 1.0) < 1.0 else ""),
                             "false_alarms": int(fl[~b].sum()), "n_good": int((~b).sum()), "detected": int(fl[b].sum()), "n_bad": int(b.sum())})
    res = pd.DataFrame(rows)
    g = res.groupby(["policy", "alpha"])[["false_alarms", "n_good", "detected", "n_bad"]].sum().reset_index()
    g["far"], g["recall"] = g.false_alarms / g.n_good, g.detected / g.n_bad
    tag = f"stream_sweep_{proto}_{scorer}_seed{seed}_{scenario}" + ("_labelrates" if len(label_rates) > 1 else "")
    Path(f"results/{tag}.json").write_text(json.dumps(g.to_dict(orient="records"), indent=1, default=float))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cols = {"P0_frozen": "#8a99a8", "P1_naive": "#b04a4a", "P2_delayed_d5": "#1f5fa8", "P2_delayed_d20": "#6f9fd8",
            "P3_aci": "#5b8a52", "P4_w1acas": "#c0392b", "P5_delayed_aci_d5": "#111111", "P5_delayed_aci_d20": "#666666"}
    for pol, gg in g.groupby("policy"):
        gg = gg.sort_values("alpha")
        ax.plot(gg.far, gg.recall, "o-", color=cols.get(pol, "k"), label=pol, lw=1.6, ms=4)
    ax.set_xlabel("achieved false-alarm rate (pooled over folds)")
    ax.set_ylabel("recall of anomalous recordings")
    ax.set_title(f"Streaming, {proto}, {scenario} order, scorer {scorer}: each point is one alpha", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"reports/figures/{tag}.png", dpi=160)
    plt.close(fig)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="lomo")
    ap.add_argument("--scorer", default="B1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--scenario", default="burst")
    ap.add_argument("--label-rates", nargs="+", type=float, default=[1.0])
    a = ap.parse_args()
    if a.sweep:
        sweep(a.proto, a.scorer, a.seed, scenario=a.scenario, label_rates=tuple(a.label_rates))
    else:
        main(a.proto, a.scorer, a.seed, a.alpha)
