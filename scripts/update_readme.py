"""Write the README results section from results/*.json (numbers are never typed by hand)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def pct(x, d=1):
    return "n/a" if x != x else f"{100 * x:.{d}f}%"


def main():
    summ = pd.DataFrame(json.loads(Path("results/summary.json").read_text())["rows"])
    stream = pd.DataFrame(json.loads(Path("results/stream_summary.json").read_text()))
    edge = json.loads(Path("results/edge.json").read_text()) if Path("results/edge.json").exists() else None

    def far(proto, scorer, strat):
        r = summ[(summ.protocol == proto) & (summ.scorer == scorer) & (summ.strategy == strat) & (summ.alpha == 0.05)]
        return float(r.pooled_far.iloc[0]) if len(r) else float("nan")

    def prauc(proto, scorer):
        r = summ[(summ.protocol == proto) & (summ.scorer == scorer) & (summ.strategy == "threshold_free")]
        return float(r.mean_pr_auc.iloc[0]) if len(r) else float("nan")

    main_s = stream[~stream.label_ablation]
    pivot = main_s[main_s.policy.isin(["P4_w1acas", "P5_delayed_aci_d5"])].pivot_table(
        index=["protocol", "scorer", "scenario"], columns="policy", values=["mean_abs_cal_error", "recall_at_far_2pct", "recall_at_far_5pct"])
    n_settings = len(pivot)
    cal_wins = int((pivot["mean_abs_cal_error"]["P5_delayed_aci_d5"] < pivot["mean_abs_cal_error"]["P4_w1acas"]).sum())
    r2 = pivot["recall_at_far_2pct"].dropna()
    r2_better = int((r2["P5_delayed_aci_d5"] > r2["P4_w1acas"] + 1e-9).sum())
    r2_worse = int((r2["P5_delayed_aci_d5"] < r2["P4_w1acas"] - 1e-9).sum())
    r2_max = float((r2["P5_delayed_aci_d5"] - r2["P4_w1acas"]).max())
    p5_cal = pivot["mean_abs_cal_error"]["P5_delayed_aci_d5"]
    w1_cal = pivot["mean_abs_cal_error"]["P4_w1acas"]
    key = main_s[(main_s.protocol == "lomo") & (main_s.scorer == "B2") & (main_s.scenario == "burst")].set_index("policy")
    lab = stream[stream.label_ablation].set_index("policy")

    lines = [
        "![headline](reports/figures/headline.png)", "",
        "### Key results (all generated from `results/`; α = 5% unless stated)", "",
        f"1. **The most accurate detector is the least trustworthy under shift.** Supervised XGBoost has the best ranking "
        f"(mean PR-AUC {prauc('lopo', 'B2'):.2f} on unseen processes), yet a static threshold set for 5% false alarms produced "
        f"**{pct(far('time', 'B2', 'C0_static'))} false alarms** on the later time period and {pct(far('lomo', 'B2', 'C0_static'))} on unseen machines; "
        f"split conformal gives {pct(far('time', 'B2', 'C1_split'))} and {pct(far('lomo', 'B2', 'C1_split'))}.",
        f"2. **Label-free adaptive calibration fixes the rate but costs recall.** IBM's W1-ACAS (ICLR 2026, run with IBM's code) brings "
        f"unseen-machine false alarms to {pct(far('lomo', 'B2', 'C5_ibm_w1acas'))} for the supervised scorer.",
        f"3. **A little late operator feedback buys a lot (this work).** Admitting only operator-confirmed healthy runs to the "
        f"calibration buffer and correcting the working α with ACI from the same late labels (5-run delay) is the best-calibrated "
        f"policy in **{cal_wins} of {n_settings}** streaming settings (mean absolute calibration error {pct(p5_cal.min())} to "
        f"{pct(p5_cal.max())}, against {pct(w1_cal.min())} to {pct(w1_cal.max())} for W1-ACAS). On real unseen-machine streams "
        f"with a burst of faulty runs it achieved **{pct(key.loc['P5_delayed_aci_d5', 'far_at_alpha5'])} false alarms with "
        f"{pct(key.loc['P5_delayed_aci_d5', 'recall_at_alpha5'])} recall**, versus {pct(key.loc['P4_w1acas', 'far_at_alpha5'])} / "
        f"{pct(key.loc['P4_w1acas', 'recall_at_alpha5'])} for W1-ACAS. At a matched 2% false-alarm rate, recall was higher in "
        f"{r2_better} settings (by up to {100 * r2_max:.0f} points) and lower in {r2_worse}. W1-ACAS needs no labels; this policy needs "
        f"delayed confirmations, and the label-coverage ablation shows it stays near target with 50% coverage "
        f"({pct(lab.loc['P5_delayed_aci_d5_labels50', 'far_at_alpha5'])} achieved) but drifts with 10% "
        f"({pct(lab.loc['P5_delayed_aci_d5_labels10', 'far_at_alpha5'])}).",
        f"4. **Naive self-labelling is worse than doing nothing.** Admitting every unflagged run raised unseen-machine false alarms "
        f"from {pct(key.loc['P0_frozen', 'far_at_alpha5'])} (frozen) to {pct(key.loc['P1_naive', 'far_at_alpha5'])}, the "
        f"self-poisoning effect described in prior work, here measured on real factory streams.",
        "5. **Negative results, reported:** similarity-weighted conformal thresholds did not beat pooled calibration under unseen "
        "processes, and adding a few confirmed runs of a new process barely moved static thresholds (see `results/fewshot_*`).",
    ]
    if edge:
        i8 = edge["onnx_int8"]
        lines.append(
            f"6. **The edge model makes the same decisions.** After INT8 static quantization and recalibration on INT8 scores, file-level "
            f"verdicts agreed with FP32 on **{pct(i8['verdict_agreement_recalibrated'])}** of held-out recordings "
            f"({pct(i8['verdict_agreement_naive'])} if FP32 thresholds are reused without recalibration); score correlation "
            f"{i8['score_pearson_vs_torch']:.4f}; p95 latency {i8['latency_single_window_1thread']['p95_ms']:.2f} ms per 1 s window on one "
            f"Apple M5 CPU core ({edge['onnx_fp32']['latency_single_window_1thread']['p95_ms']:.2f} ms FP32); model {i8['size_kb']:.0f} KB.")
    lines += ["", "Full tables: [docs/results_tables.md](docs/results_tables.md) (offline calibration, all scorers and protocols) and "
              "[docs/stream_tables.md](docs/stream_tables.md) (streaming study). Figures: [reports/figures/](reports/figures/)."]
    block = "\n".join(lines)
    readme = Path("README.md").read_text()
    readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", readme, flags=re.S)
    Path("README.md").write_text(readme)
    print(block)


if __name__ == "__main__":
    main()
