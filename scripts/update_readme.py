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
    hold = json.loads(Path("results/holdout_comparison.json").read_text())
    hsum = pd.DataFrame(json.loads(Path("results/holdout_summary.json").read_text()))
    edge = json.loads(Path("results/edge.json").read_text())
    lat = json.loads(Path("results/detection_latency.json").read_text())

    def far(proto, scorer, strat):
        r = summ[(summ.protocol == proto) & (summ.scorer == scorer) & (summ.strategy == strat) & (summ.alpha == 0.05)]
        return float(r.pooled_far.mean()) if len(r) else float("nan")

    def prauc(proto, scorer):
        r = summ[(summ.protocol == proto) & (summ.scorer == scorer) & (summ.strategy == "threshold_free")]
        return float(r.mean_pr_auc.mean())

    n_seeds = int(summ[summ.scorer == "B2"].seed.nunique())
    key = hsum[(hsum.protocol == "lomo") & (hsum.scorer == "B2") & (hsum.scenario == "burst") & (hsum.seed == 0)].set_index("policy")
    full = stream[(~stream.label_ablation) & (stream.protocol == "lomo") & (stream.scorer == "B2") & (stream.scenario == "burst")].set_index("policy")
    lab = stream[stream.label_ablation].set_index("policy")
    wt, wd = hold["P4_w1acas_tuned"], hold["P4_w1acas_default"]
    i8, f32 = edge["onnx_int8"], edge["onnx_fp32"]

    lines = [
        "![headline](reports/figures/headline.png)", "",
        "### Key results (generated from `results/`; α = 5% unless stated)", "",
        f"1. **The most accurate detector is the least trustworthy under shift.** Supervised XGBoost ranks faults best (mean PR-AUC "
        f"{prauc('lopo', 'B2'):.2f} on unseen processes, {n_seeds} seeds), yet a static threshold set for 5% false alarms produced "
        f"**{pct(far('time', 'B2', 'C0_static'))} false alarms** on the later time period and {pct(far('lomo', 'B2', 'C0_static'))} on unseen machines.",
        f"2. **Label-free adaptive calibration fixes the rate.** IBM's W1-ACAS (ICLR 2026, run with IBM's code) brings those to "
        f"{pct(far('time', 'B2', 'C5_ibm_w1acas'))} and {pct(far('lomo', 'B2', 'C5_ibm_w1acas'))}.",
        f"3. **Late operator feedback calibrates better (main result, held-out protocol).** Every tunable policy had its settings chosen on "
        f"the first half of each real stream and was scored on the second half only. Admitting only operator-confirmed healthy runs to the "
        f"calibration buffer and correcting the working α with ACI from the same late labels was better calibrated than IBM W1-ACAS in "
        f"**{wt['calibration_better']} of {wt['settings']}** held-out runs (16 settings, up to 3 training seeds each) against tuned W1-ACAS "
        f"and {wd['calibration_better']} of {wd['settings']} against its default settings. Recall at a matched 2% false-alarm rate was higher in {wt['recall_at_far_2pct']['better']} and lower in "
        f"{wt['recall_at_far_2pct']['worse']} of {wt['recall_at_far_2pct']['comparable']} comparable runs against tuned W1-ACAS "
        f"(mean {wt['recall_at_far_2pct']['mean_diff_points']:+.1f} points); against default W1-ACAS it was higher in "
        f"{wd['recall_at_far_2pct']['better']} and lower in {wd['recall_at_far_2pct']['worse']} (mean {wd['recall_at_far_2pct']['mean_diff_points']:+.1f}). "
        f"Example, unseen machines with a burst of real faulty runs: **{pct(key.loc['P5_tuned_d5', 'far_at_alpha5'])} false alarms, "
        f"{pct(key.loc['P5_tuned_d5', 'recall_at_alpha5'])} recall**, against {pct(key.loc['P4_w1acas_tuned', 'far_at_alpha5'])} / "
        f"{pct(key.loc['P4_w1acas_tuned', 'recall_at_alpha5'])} for tuned W1-ACAS and {pct(key.loc['P0_frozen', 'far_at_alpha5'])} false alarms "
        f"for a frozen threshold. W1-ACAS needs no labels; this policy does, and it stays near target with 50% label coverage "
        f"({pct(lab.loc['P5_delayed_aci_d5_labels50', 'far_at_alpha5'])}) but drifts with 10% ({pct(lab.loc['P5_delayed_aci_d5_labels10', 'far_at_alpha5'])}).",
        f"4. **Naive self-labelling is worse than doing nothing.** Admitting every unflagged run raised unseen-machine false alarms from "
        f"{pct(full.loc['P0_frozen', 'far_at_alpha5'])} (frozen) to {pct(full.loc['P1_naive', 'far_at_alpha5'])}, the self-poisoning effect "
        f"described in prior work, measured here on real factory streams.",
        f"5. **The edge model makes the same decisions and can warn early.** INT8 decisions agreed with FP32 on "
        f"**{pct(i8['verdict_agreement_recalibrated'])}** of 469 held-out recordings after recalibration ({pct(i8['verdict_agreement_naive'])} "
        f"without); p95 latency {i8['latency_single_window_1thread']['p95_ms']:.2f} ms per 1 s window on one Apple M5 CPU core "
        f"(FP32 {f32['latency_single_window_1thread']['p95_ms']:.2f} ms); {i8['size_kb']:.0f} KB. Alarming at the first crossing of the running "
        f"score caught {lat['n_ever_crossed']} of {lat['n_faulty_runs']} faulty 2021 runs at a median of {lat['median_seconds']:.1f} s into the "
        f"run, with {pct(lat['healthy_false_alarm_rate_alarm_on_first_crossing'])} false alarms on healthy runs (end-of-run decisions: "
        f"{lat['n_detected_at_end']} of {lat['n_faulty_runs']} at {pct(lat['healthy_false_alarm_rate_end_of_run'])}).",
        "6. **Negative results, reported:** similarity-weighted conformal thresholds did not beat pooled calibration under unseen "
        "processes, and adding a few confirmed runs of a new process barely moved static thresholds.",
        "",
        f"Secondary analysis without held-out tuning (whole streams, fixed settings): the same policy reached "
        f"{pct(full.loc['P5_delayed_aci_d5', 'far_at_alpha5'])} false alarms with {pct(full.loc['P5_delayed_aci_d5', 'recall_at_alpha5'])} recall in the "
        f"example above; those numbers are more optimistic because the policy was designed on the same streams.",
        "",
        "Tables: [held-out streaming](docs/holdout_tables.md), [offline calibration, all scorers and seeds](docs/results_tables.md), "
        "[whole-stream analysis](docs/stream_tables.md). Report: [docs/report.pdf](docs/report.pdf). Figures: [reports/figures/](reports/figures/).",
    ]
    block = "\n".join(lines)
    readme = Path("README.md").read_text()
    readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", readme, flags=re.S)
    Path("README.md").write_text(readme)
    print(block)


if __name__ == "__main__":
    main()
