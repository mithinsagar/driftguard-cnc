"""Aggregate results/raw/* into results/summary.json, README tables (docs/results_tables.md) and figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driftguard.eval.engine import ALPHAS  # noqa: E402

STRATS = ["C0_static", "C1_split", "C2_mondrian", "C4_weighted", "C4f_weighted_fallback", "C5_ibm_w1acas"]
LABEL = {"C0_static": "static percentile", "C1_split": "split conformal", "C2_mondrian": "Mondrian (per process)",
         "C4_weighted": "similarity-weighted (raw)", "C4f_weighted_fallback": "similarity-weighted + ESS fallback", "C5_ibm_w1acas": "IBM W1-ACAS (ICLR 2026)"}
COLOR = {"C0_static": "#b7793a", "C1_split": "#8a99a8", "C2_mondrian": "#6d5bb0", "C4_weighted": "#7fb98f",
         "C4f_weighted_fallback": "#1b6b3a", "C5_ibm_w1acas": "#c0392b"}
SCORER_NAME = {"B0": "RMS heuristic", "B1": "Isolation Forest", "B2": "XGBoost (supervised)", "AE": "1D-CNN autoencoder"}


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return (float(lo), float(hi))


def boot_mean(x: np.ndarray, n: int = 10000, seed: int = 0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def load_runs(seed_glob: str = "*_seed*.json"):
    runs = []
    for p in sorted(Path("results/raw").glob(f"*/{seed_glob}")):
        if "_quick" in p.name:
            continue
        runs.append(json.loads(p.read_text()))
    return runs


def summarise(runs):
    rows = []
    for run in runs:
        proto, scorer, seed = run["protocol"], run["scorer"], run["provenance"]["seed"]
        for a in ALPHAS:
            for st in STRATS:
                if st not in run["folds"][0]["by_alpha"][str(a)]:
                    continue
                per = [f["by_alpha"][str(a)][st] for f in run["folds"]]
                fars = np.array([r["far"] for r in per], float)
                err = np.abs(fars - a)
                fa, ng = sum(r["false_alarms"] for r in per), sum(r["n_good"] for r in per)
                det, nb = sum(r["detected"] for r in per), sum(r["n_bad"] for r in per)
                rows.append({
                    "protocol": proto, "scorer": scorer, "seed": seed, "alpha": a, "strategy": st, "n_folds": len(per),
                    "pooled_far": fa / ng if ng else np.nan, "pooled_far_ci": clopper_pearson(fa, ng),
                    "mean_abs_cal_error": float(np.nanmean(err)), "mean_abs_cal_error_ci": boot_mean(err),
                    "violation_share": float(np.nanmean(fars > a + 0.02)),
                    "pooled_recall": det / nb if nb else np.nan, "pooled_recall_ci": clopper_pearson(det, nb),
                    "detected": det, "n_bad": nb, "false_alarms": fa, "n_good": ng,
                    "folds_far": fars.tolist(),
                })
        prs = [f["threshold_free"]["pr_auc"] for f in run["folds"]]
        rows.append({"protocol": proto, "scorer": scorer, "seed": seed, "alpha": None, "strategy": "threshold_free",
                     "mean_pr_auc": float(np.nanmean(prs)), "mean_pr_auc_ci": boot_mean(np.array(prs, float)),
                     "mean_roc_auc": float(np.nanmean([f["threshold_free"]["roc_auc"] for f in run["folds"]]))})
    return pd.DataFrame(rows)


def paired_improvement(runs, a=0.05, base="C1_split", new="C4_weighted"):
    out = []
    for run in runs:
        d = [abs(f["by_alpha"][str(a)][base]["far"] - a) - abs(f["by_alpha"][str(a)][new]["far"] - a) for f in run["folds"]]
        d = np.array(d, float)
        out.append({"protocol": run["protocol"], "scorer": run["scorer"], "alpha": a, "mean_reduction_in_cal_error": float(np.nanmean(d)),
                    "ci95": boot_mean(d), "folds_improved": int(np.nansum(d > 0)), "folds_worse": int(np.nansum(d < 0)), "n_folds": int(len(d))})
    return out


def fmt_ci(v, ci, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v)} [{f(ci[0])}, {f(ci[1])}]"


def seed_span(vals, pct=True):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) == 0:
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    return f(v.mean()) if len(v) == 1 else f"{f(v.mean())} [{f(v.min())}, {f(v.max())}]"


def tables(df: pd.DataFrame, paired) -> str:
    """Seed-aggregated tables: mean over training seeds, [lowest, highest seed] in brackets, seed-0 bootstrap CIs in JSON."""
    out = ["<!-- generated by scripts/make_tables.py from results/raw; do not edit -->", "",
           "Values are means over training seeds; brackets give the lowest and highest seed. The RMS heuristic is deterministic "
           "and has one seed. Bootstrap confidence intervals per seed are in `results/summary.json`.", ""]
    thr = df[df.strategy == "threshold_free"]
    out += ["### Detection quality (threshold-free, file level)", "", "| Protocol | Scorer | Seeds | Mean PR-AUC | Mean ROC-AUC |", "|---|---|---|---|---|"]
    for (proto, sc), g in thr.groupby(["protocol", "scorer"]):
        pr, roc = seed_span(g.mean_pr_auc, pct=False), seed_span(g.mean_roc_auc, pct=False)
        out.append(f"| {proto} | {SCORER_NAME.get(sc, sc)} | {g.seed.nunique()} | {pr} | {roc} |")
    for proto in ["lopo", "lomo", "time"]:
        sub = df[(df.protocol == proto) & (df.alpha == 0.05)]
        if sub.empty:
            continue
        out += ["", f"### False-alarm control at α = 5%, protocol `{proto}`", "",
                "| Scorer | Strategy | Seeds | Achieved FAR | Mean abs. calibration error | Folds violating (FAR > 7%) | Recall |",
                "|---|---|---|---|---|---|---|"]
        for (sc, st), g in sub.groupby(["scorer", "strategy"]):
            out.append(f"| {SCORER_NAME.get(sc, sc)} | {LABEL[st]} | {g.seed.nunique()} | {seed_span(g.pooled_far)} | {seed_span(g.mean_abs_cal_error)} | "
                       f"{seed_span(g.violation_share)} of {int(g.n_folds.iloc[0])} | {seed_span(g.pooled_recall)} |")
    out += ["", "### Paired comparison: similarity-weighted vs split conformal (α = 5%, seed 0)", "",
            "| Protocol | Scorer | Mean reduction in abs. calibration error [95% CI] | Folds better / worse |", "|---|---|---|---|"]
    for p in paired:
        out.append(f"| {p['protocol']} | {SCORER_NAME.get(p['scorer'], p['scorer'])} | {fmt_ci(p['mean_reduction_in_cal_error'], p['ci95'])} | "
                   f"{p['folds_improved']} / {p['folds_worse']} of {p['n_folds']} |")
    return "\n".join(out) + "\n"


def figure_target_vs_achieved(df: pd.DataFrame, scorer: str):
    protos = [p for p in ["lopo", "lomo", "time"] if not df[(df.protocol == p) & (df.scorer == scorer)].empty]
    fig, axes = plt.subplots(1, len(protos), figsize=(4.2 * len(protos), 3.8), squeeze=False)
    for ax, proto in zip(axes[0], protos, strict=True):
        sub = df[(df.protocol == proto) & (df.scorer == scorer) & df.alpha.notna()]
        for st in STRATS:
            s = sub[sub.strategy == st].sort_values("alpha")
            if s.empty:
                continue
            lo = [c[0] for c in s.pooled_far_ci]
            hi = [c[1] for c in s.pooled_far_ci]
            ax.plot(s.alpha, s.pooled_far, "o-", color=COLOR[st], label=LABEL[st], lw=1.6, ms=4)
            ax.fill_between(s.alpha, lo, hi, color=COLOR[st], alpha=0.12, lw=0)
        ax.plot([0, 0.105], [0, 0.105], "k--", lw=1, label="target = achieved")
        ax.set_xlim(0, 0.105)
        ax.set_title(f"{proto}: {SCORER_NAME.get(scorer, scorer)}", fontsize=10)
        ax.set_xlabel("target false-alarm rate α")
        ax.set_ylabel("achieved false-alarm rate (pooled)")
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"reports/figures/target_vs_achieved_{scorer}.png", dpi=160)
    plt.close(fig)


def figure_per_fold(df: pd.DataFrame, scorer: str, proto: str = "lopo", a: float = 0.05):
    sub = df[(df.protocol == proto) & (df.scorer == scorer) & (df.alpha == a)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    strats = [st for st in STRATS if not sub[sub.strategy == st].empty]
    for i, st in enumerate(strats):
        fars = np.array(sub[sub.strategy == st].folds_far.iloc[0])
        ax.scatter(np.full(len(fars), i) + np.linspace(-0.15, 0.15, len(fars)), fars, color=COLOR[st], s=18)
    ax.axhline(a, color="black", ls="--", lw=1)
    ax.set_xticks(range(len(strats)), [LABEL[s] for s in strats], fontsize=7)
    ax.set_ylabel("achieved FAR per held-out process")
    ax.set_title(f"Leave-one-process-out, α = {a}: every dot is one unseen machining process ({SCORER_NAME.get(scorer, scorer)})", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"reports/figures/per_fold_far_{proto}_{scorer}.png", dpi=160)
    plt.close(fig)


def main():
    runs = load_runs()
    df = summarise(runs)
    seed0 = [r for r in runs if r["provenance"]["seed"] == 0]
    paired = paired_improvement(seed0, 0.05)
    Path("results/summary.json").write_text(json.dumps({"rows": df.to_dict(orient="records"), "paired": paired}, indent=1, default=float))
    Path("docs/results_tables.md").write_text(tables(df, paired))
    d0 = df[df.seed == 0]
    for sc in d0.scorer.unique():
        figure_target_vs_achieved(d0, sc)
        figure_per_fold(d0, sc)
    print(Path("docs/results_tables.md").read_text())


if __name__ == "__main__":
    main()
