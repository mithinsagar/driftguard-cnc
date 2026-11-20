"""Held-out streaming evaluation: hyperparameters are chosen on the first half of every stream, results are reported on the
second half only. Both the proposed policy (P5) and IBM W1-ACAS get a tuning grid; frozen, naive and default variants are
reported alongside. This answers "were the settings picked on the same data they are scored on?" with a clean protocol.

Policies run online over the whole stream (state carries over), but the selection metric uses only first-half positions
and every reported metric uses only second-half positions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driftguard.calibration.stream import order_stream, simulate  # noqa: E402
from driftguard.data import splits  # noqa: E402
from driftguard.eval import engine  # noqa: E402

ALPHAS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15)
TUNE_ALPHA = 0.05
P5_GRID = [{"gamma": g, "buffer_size": b} for g, b in itertools.product((0.002, 0.005, 0.01, 0.02), (100, 300, 600))]
W1_GRID = [{"lr": lr, "n_epochs": e, "n_batch_update": nb} for lr, e, nb in itertools.product((1e-3, 1e-2), (1, 3), (10, 50))]


def streams(proto, scorer, seed, scenario):
    df = pd.read_csv("data/manifest.csv")
    store = np.load(f"results/scores/{proto}_{scorer}_seed{seed}.npz")
    out = []
    for fold in splits.load(proto):
        fd = engine.fold_indices(df, fold, proto)
        fs, test = store[fd.name], fd.idx["test"]
        is_bad = df.label.to_numpy()[test] == "bad"
        # the burst of real faulty runs lands in the evaluation half (75% into the stream); tuning sees a quiet period
        order = order_stream(df.t_order.to_numpy()[test], df["index"].to_numpy()[test], is_bad, scenario, seed, burst_at=0.75)
        s, b = fs[test][order], is_bad[order]
        half = np.arange(len(s)) >= len(s) // 2
        out.append({"fold": fd.name, "scores": s, "bad": b, "cal": fs[fd.idx["cal_good"]], "eval": half})
    return out


def counts(flags, bad, mask):
    return int(flags[mask & ~bad].sum()), int((mask & ~bad).sum()), int(flags[mask & bad].sum()), int((mask & bad).sum())


def pooled(sts, alpha, policy, part, **kw):
    fa = ng = de = nb = 0
    for st in sts:
        r = simulate(st["scores"], st["bad"], st["cal"], alpha, policy, **kw)
        mask = st["eval"] if part == "eval" else ~st["eval"]
        a, b, c, d = counts(r["flags"], st["bad"], mask)
        fa, ng, de, nb = fa + a, ng + b, de + c, nb + d
    return {"far": fa / ng if ng else np.nan, "recall": de / nb if nb else np.nan, "false_alarms": fa, "n_good": ng, "detected": de, "n_bad": nb}


def select(sts, policy, grid, fixed, wrap=None):
    """Pick the config with the smallest |FAR - alpha| on the tuning halves at TUNE_ALPHA; ties broken by recall."""
    best, best_key, scores = None, None, []
    for cfg in grid:
        kw = {**fixed, **({"w1_kw": cfg} if wrap == "w1" else cfg)}
        m = pooled(sts, TUNE_ALPHA, policy, "tune", **kw)
        key = (round(abs(m["far"] - TUNE_ALPHA), 4), -(m["recall"] if m["recall"] == m["recall"] else 0))
        scores.append({"config": cfg, **m})
        if best_key is None or key < best_key:
            best, best_key = cfg, key
    return best, scores


def run(proto, scorer, seed, scenario):
    sts = streams(proto, scorer, seed, scenario)
    p5_k5, p5_scores = select(sts, "P5_delayed_aci", P5_GRID, {"delay": 5})
    p5_k20, _ = select(sts, "P5_delayed_aci", P5_GRID, {"delay": 20})
    w1_best, w1_scores = select(sts, "P4_w1acas", W1_GRID, {}, wrap="w1")
    variants = {
        "P0_frozen": ("P0_frozen", {}),
        "P1_naive": ("P1_naive", {}),
        "P2_delayed_d5": ("P2_delayed", {"delay": 5}),
        "P4_w1acas_default": ("P4_w1acas", {}),
        "P4_w1acas_tuned": ("P4_w1acas", {"w1_kw": w1_best}),
        "P5_default_d5": ("P5_delayed_aci", {"delay": 5, "gamma": 0.005, "buffer_size": 300}),
        "P5_tuned_d5": ("P5_delayed_aci", {"delay": 5, **p5_k5}),
        "P5_tuned_d20": ("P5_delayed_aci", {"delay": 20, **p5_k20}),
    }
    rows = []
    for name, (pol, kw) in variants.items():
        for a in ALPHAS:
            rows.append({"policy": name, "alpha": a, **pooled(sts, a, pol, "eval", **kw)})
    res = {"protocol": proto, "scorer": scorer, "seed": seed, "scenario": scenario, "tune_alpha": TUNE_ALPHA,
           "selected": {"P5_d5": p5_k5, "P5_d20": p5_k20, "W1ACAS": w1_best}, "tuning_P5": p5_scores, "tuning_W1": w1_scores, "eval": rows}
    Path("results/holdout").mkdir(parents=True, exist_ok=True)
    Path(f"results/holdout/{proto}_{scorer}_seed{seed}_{scenario}.json").write_text(json.dumps(res, indent=1, default=float))
    g = pd.DataFrame(rows)
    at5 = g[g.alpha == 0.05].set_index("policy")
    print(f"{proto} {scorer} seed{seed} {scenario} selected P5={p5_k5} W1={w1_best}")
    for pol in ("P0_frozen", "P4_w1acas_default", "P4_w1acas_tuned", "P5_default_d5", "P5_tuned_d5"):
        print(f"   {pol:<18} FAR {100 * at5.loc[pol, 'far']:.1f}%  recall {100 * at5.loc[pol, 'recall']:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", nargs="+", default=["lomo", "time"])
    ap.add_argument("--scorers", nargs="+", default=["B0", "B1", "B2", "AE"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--scenarios", nargs="+", default=["burst", "natural"])
    a = ap.parse_args()
    for proto in a.proto:
        for scorer in a.scorers:
            for seed in a.seeds:
                if not Path(f"results/scores/{proto}_{scorer}_seed{seed}.npz").exists():
                    continue
                for scen in a.scenarios:
                    run(proto, scorer, seed, scen)
