"""Build docs/report.pdf: a short technical report whose numbers are all read from results/ (Markdown -> HTML -> Chrome PDF)."""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import markdown
import pandas as pd

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pct(x, d=1):
    return "n/a" if x != x else f"{100 * x:.{d}f}%"


def img(path, width="100%"):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return f'<img src="data:{mime};base64,{data}" style="width:{width}">'


def main():
    summ = pd.DataFrame(json.loads(Path("results/summary.json").read_text())["rows"])
    hold = json.loads(Path("results/holdout_comparison.json").read_text())
    hsum = pd.DataFrame(json.loads(Path("results/holdout_summary.json").read_text()))
    edge = json.loads(Path("results/edge.json").read_text())
    lat = json.loads(Path("results/detection_latency.json").read_text())
    stream = pd.DataFrame(json.loads(Path("results/stream_summary.json").read_text()))

    def far(proto, scorer, strat):
        r = summ[(summ.protocol == proto) & (summ.scorer == scorer) & (summ.strategy == strat) & (summ.alpha == 0.05)]
        return float(r.pooled_far.mean())

    def prauc(proto, scorer):
        return float(summ[(summ.protocol == proto) & (summ.scorer == scorer) & (summ.strategy == "threshold_free")].mean_pr_auc.mean())

    key = hsum[(hsum.protocol == "lomo") & (hsum.scorer == "B2") & (hsum.scenario == "burst") & (hsum.seed == 0)].set_index("policy")
    wt, wd = hold["P4_w1acas_tuned"], hold["P4_w1acas_default"]
    lab = stream[stream.label_ablation].set_index("policy")
    i8 = edge["onnx_int8"]

    md = f"""
# DriftGuard-CNC: false-alarm control for CNC vibration monitoring under process, machine and time shift

**Mithin Sagar S**, B.Tech CSE (AI and ML), VIT Chennai. Code and data pipeline: github.com/mithinsagar/driftguard-cnc

## Abstract

Vibration-based anomaly detectors on CNC machines are usually judged by ranking metrics, while a factory needs a decision rule
whose false-alarm rate holds after the product mix, the machine or the season changes. On the public Bosch CNC Machining
dataset (1,702 recordings, 3 machines, 15 processes, 2018 to 2021) we build a leak-free, recording-level benchmark for
calibrated decisions under three shift types and evaluate four scorers and six calibration rules, including IBM's W1-ACAS
(ICLR 2026) run with the authors' code. The best-ranking scorer, a supervised XGBoost model (mean PR-AUC {prauc('lopo', 'B2'):.2f} on
unseen processes), produces {pct(far('time', 'B2', 'C0_static'))} false alarms on a later period when a static threshold is set for 5%.
We then study streaming deployment with operator labels that arrive late. A policy that admits only operator-confirmed
healthy runs to the calibration buffer and corrects the working level with adaptive conformal inference from the same late
labels was better calibrated than tuned W1-ACAS in {wt['calibration_better']} of {wt['settings']} held-out settings, where all
hyperparameters were chosen on the first half of each stream and results measured on the second half. Recall at a matched
false-alarm rate was higher on average ({wt['recall_at_far_2pct']['mean_diff_points']:+.1f} points at 2%) but not in every setting. An INT8
edge model reproduces {pct(i8['verdict_agreement_recalibrated'])} of full-precision decisions after recalibration.

## 1. Problem

A monitoring system that alarms too often is muted by operators within days. The engineer's real requirement is a budget:
"at most α of healthy runs may raise an alarm". Conformal calibration gives that guarantee only if new healthy runs resemble
the calibration set; brownfield plants violate this constantly as programs, machines and conditions change.

## 2. Data and protocol

The Bosch CNC Machining dataset (Tnani, Feil, Diepold, Procedia CIRP 2022; CC-BY-4.0) records tri-axial acceleration at
2 kHz on three milling machines. Labels are per recording (1,632 healthy, 70 anomalous; machine M03 has only 9 anomalous).
Every split assigns whole recordings to one role. Protocols: leave-one-process-out (15 folds), leave-one-machine-out (3 folds)
and chronological (train 2019 to 2020, test 2021). Within the non-test part, healthy recordings are split 60/20/20 into
training, validation and calibration. Features: 81 DSP features per 1 s window; scorers: RMS heuristic, Isolation Forest,
supervised XGBoost (reference), and a 150k-parameter 1D-CNN autoencoder. Window scores are aggregated to a recording score
with a rule chosen on validation data. Isolation Forest and XGBoost were trained with three seeds; the autoencoder with three seeds on the machine and time folds and one on the process folds.

## 3. Calibration rules and streaming policies

Offline rules: static percentile, split conformal, Mondrian (per process), similarity-weighted conformal (raw and with an
effective-sample-size fallback) and W1-ACAS. Streaming: held-out recordings are replayed in time order, naturally and with a
contiguous burst of the real faulty runs; labels arrive k runs late. Policies: frozen threshold, naive self-labelling,
delayed-label admission, ACI from delayed errors, W1-ACAS (label-free), and delayed-label admission plus ACI (proposed, P5).

## 4. Results

{img("reports/figures/headline.png")}

**Static rules break under shift.** For the supervised scorer at α = 5%, static and split-conformal thresholds gave
{pct(far('lomo', 'B2', 'C0_static'))} and {pct(far('lomo', 'B2', 'C1_split'))} false alarms on unseen machines and {pct(far('time', 'B2', 'C0_static'))} and
{pct(far('time', 'B2', 'C1_split'))} on the later period; W1-ACAS brought them to {pct(far('lomo', 'B2', 'C5_ibm_w1acas'))} and {pct(far('time', 'B2', 'C5_ibm_w1acas'))}.

**Held-out streaming comparison.** On the second halves of the streams, P5 was better calibrated than W1-ACAS in
{wt['calibration_better']} of {wt['settings']} settings (tuned W1-ACAS) and {wd['calibration_better']} of {wd['settings']} (default W1-ACAS).
Recall at a matched 2% false-alarm rate was higher in {wt['recall_at_far_2pct']['better']} and lower in {wt['recall_at_far_2pct']['worse']} of
{wt['recall_at_far_2pct']['comparable']} comparable settings against tuned W1-ACAS (mean {wt['recall_at_far_2pct']['mean_diff_points']:+.1f} points), and
higher in {wd['recall_at_far_2pct']['better']}, lower in {wd['recall_at_far_2pct']['worse']} against default W1-ACAS (mean
{wd['recall_at_far_2pct']['mean_diff_points']:+.1f} points). On unseen machines with a fault burst (supervised scorer), P5 reached
{pct(key.loc['P5_tuned_d5', 'far_at_alpha5'])} false alarms with {pct(key.loc['P5_tuned_d5', 'recall_at_alpha5'])} recall, against
{pct(key.loc['P4_w1acas_tuned', 'far_at_alpha5'])} / {pct(key.loc['P4_w1acas_tuned', 'recall_at_alpha5'])} (tuned W1-ACAS) and
{pct(key.loc['P0_frozen', 'far_at_alpha5'])} false alarms for a frozen threshold. Naive self-labelling was worse than a frozen threshold.

**Label coverage.** P5 depends on operator confirmations: with 50% of runs labelled it stayed near target
({pct(lab.loc['P5_delayed_aci_d5_labels50', 'far_at_alpha5'])} at α = 5%); with 10% it drifted to {pct(lab.loc['P5_delayed_aci_d5_labels10', 'far_at_alpha5'])}.

**Edge model.** INT8 static quantization of the autoencoder agreed with FP32 on {pct(i8['verdict_agreement_recalibrated'])} of 469 held-out
recordings after recalibration ({pct(i8['verdict_agreement_naive'])} when FP32 thresholds were reused); p95 latency
{i8['latency_single_window_1thread']['p95_ms']:.2f} ms per 1 s window on one Apple M5 CPU core; {i8['size_kb']:.0f} KB. Alarming at the first
crossing of the running score caught {lat['n_ever_crossed']} of {lat['n_faulty_runs']} faulty 2021 runs at a median of
{lat['median_seconds']:.1f} s into the run, at a healthy false-alarm rate of {pct(lat['healthy_false_alarm_rate_alarm_on_first_crossing'])}.

**Negative results.** Similarity-weighted thresholds did not improve on pooled calibration under unseen processes, and a few
confirmed runs of a new process did not move static thresholds.

## 5. Limitations

70 anomalous recordings (17 in the 2021 test period) limit statistical power; streaming results pool three machine folds and
one time fold. Operator labels are simulated as correct and delayed by a fixed number of runs. W1-ACAS was designed for
foundation-model forecast residuals; here it calibrates our scorers' recording scores. Latency was measured on a laptop CPU.
The deployed autoencoder ranks anomalies worse than a simple RMS rule on this data.

## 6. Related work and contribution

Conformal anomaly detection (Bates et al.), weighted conformal under shift (Tibshirani et al.), adaptive conformal inference
(Gibbs and Candès), the self-poisoning of self-labelled recalibration (arXiv 2607.21673) and W1-ACAS (Martinez Gil et al.,
ICLR 2026) are prior work. The contribution here is empirical: a leak-free calibrated-decision benchmark on real factory data,
a held-out comparison of label-aware and label-free online calibration against the strongest available baseline, and an
edge deployment whose decisions match full precision. All numbers are reproduced by `make verify` and the scripts in the
repository.
"""
    css = """@page { size: A4; margin: 15mm 16mm; } body { font: 9.6pt/1.42 Charter, Georgia, serif; color: #1d2733; }
    h1 { font: 700 15pt/1.2 'Helvetica Neue', Arial; color: #12304f; margin: 0 0 3mm; } h2 { font: 700 11pt 'Helvetica Neue', Arial; color: #12304f; margin: 4mm 0 1.5mm; break-after: avoid; } h2 + p img, img { break-inside: avoid; }
    p { margin: 0 0 2mm; text-align: justify; } img { margin: 1mm 0 2mm; } code { font-size: 8.4pt; }"""
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{markdown.markdown(md)}</body></html>"
    Path("docs/report.html").write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={Path('docs/report.pdf').resolve()}",
                    Path("docs/report.html").resolve().as_uri()], check=True, capture_output=True)
    Path("docs/report.html").unlink()
    print("docs/report.pdf written")


if __name__ == "__main__":
    main()
