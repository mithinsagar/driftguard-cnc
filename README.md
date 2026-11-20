# DriftGuard-CNC

**False-alarm-controlled vibration anomaly detection on Bosch's CNC machining data, tested under the shifts that break real deployments: a new machining process, a new machine, a later time period.**

An engineer chooses a false-alarm budget α ("at most 5 in 100 healthy runs may raise an alarm"). DriftGuard-CNC sets thresholds to meet it, measures honestly whether the promise survives shift, keeps sensor faults separate from process anomalies, and ships an INT8 model that makes the same decisions as the full-precision one.

<!-- RESULTS:START -->
![headline](reports/figures/headline.png)

### Key results (all generated from `results/`; α = 5% unless stated)

1. **The most accurate detector is the least trustworthy under shift.** Supervised XGBoost has the best ranking (mean PR-AUC 0.90 on unseen processes), yet a static threshold set for 5% false alarms produced **44.2% false alarms** on the later time period and 18.0% on unseen machines; split conformal gives 47.1% and 19.9%.
2. **Label-free adaptive calibration fixes the rate but costs recall.** IBM's W1-ACAS (ICLR 2026, run with IBM's code) brings unseen-machine false alarms to 5.5% for the supervised scorer.
3. **A little late operator feedback buys a lot (this work).** Admitting only operator-confirmed healthy runs to the calibration buffer and correcting the working α with ACI from the same late labels (5-run delay) is the best-calibrated policy in **16 of 16** streaming settings (mean absolute calibration error 0.1% to 0.4%, against 0.4% to 2.7% for W1-ACAS). On real unseen-machine streams with a burst of faulty runs it achieved **5.3% false alarms with 87.1% recall**, versus 4.3% / 70.0% for W1-ACAS. At a matched 2% false-alarm rate, recall was higher in 10 settings (by up to 30 points) and lower in 3. W1-ACAS needs no labels; this policy needs delayed confirmations, and the label-coverage ablation shows it stays near target with 50% coverage (6.1% achieved) but drifts with 10% (12.0%).
4. **Naive self-labelling is worse than doing nothing.** Admitting every unflagged run raised unseen-machine false alarms from 19.9% (frozen) to 27.6%, the self-poisoning effect described in prior work, here measured on real factory streams.
5. **Negative results, reported:** similarity-weighted conformal thresholds did not beat pooled calibration under unseen processes, and adding a few confirmed runs of a new process barely moved static thresholds (see `results/fewshot_*`).
6. **The edge model makes the same decisions.** After INT8 static quantization and recalibration on INT8 scores, file-level verdicts agreed with FP32 on **99.8%** of held-out recordings (99.6% if FP32 thresholds are reused without recalibration); score correlation 0.9984; p95 latency 1.22 ms per 1 s window on one Apple M5 CPU core (1.20 ms FP32); model 188 KB.

Full tables: [docs/results_tables.md](docs/results_tables.md) (offline calibration, all scorers and protocols) and [docs/stream_tables.md](docs/stream_tables.md) (streaming study). Figures: [reports/figures/](reports/figures/).
<!-- RESULTS:END -->

## Demo

![dashboard](reports/figures/dashboard.jpg)

`make demo` starts the FastAPI service and the replay dashboard on http://localhost:8000. Pick any held-out recording from the chronological test fold, move the false-alarm budget α, and inject a sensor fault: a disconnected axis returns `SENSOR_FAULT`, never a process verdict. The screenshot shows a real anomalous run (M01, OP07, August 2021) scored by the INT8 model at 2.71 against a conformal threshold of 0.84, p-value 0.0043 (the smallest the 234-recording calibration set can produce). The deployed autoencoder also misses faulty runs, and the dashboard shows ground truth next to every decision.

API: `POST /v1/score` (n x 3 samples at 2 kHz) returns decision, file score, threshold, conformal p-value, window scores and latency; `PUT /v1/config` changes α and rejects values below what the calibration set can support; the service refuses to start if the calibration file was computed for a different model hash.

## Validate it yourself (about 15 minutes on a laptop)

```bash
git clone <this repository> && cd driftguard-cnc
make setup && make data     # downloads 0.9 GB from Bosch's repository, verifies 1,702 files by git blob SHA-1
make verify                 # re-runs Isolation Forest on 3 process folds and compares with committed results
make test                   # leakage, DSP, conformal maths, streaming policies, sensor gate, ONNX API tests
```

## Why this is hard

- **Thresholds do not transfer.** A detector tuned on one set of machining processes alarms constantly on another, so operators mute it.
- **Random window splits lie.** Windows from one recording share its process and tool state; splitting them randomly leaks the test set into training. Every split here is by recording.
- **Few anomalies.** 70 anomalous recordings among 1,702 (M03 has only 9), so recall estimates carry wide intervals, and they are reported with them.

## Data

[Bosch CNC Machining dataset](https://github.com/boschresearch/CNC_Machining) (Tnani, Feil, Diepold, Procedia CIRP 2022; data CC-BY-4.0): tri-axial Bosch CISS accelerometer at 2 kHz on three brownfield CNC milling machines (M01 to M03), 15 processes (OP00 to OP14), October 2018 to August 2021. `make data` downloads every file from the official repository, verifies each against its git blob SHA-1, checks the counts (1,702 recordings: 1,632 good, 70 bad), writes the audit in [docs/data_audit.md](docs/data_audit.md) and the split manifests in [splits/](splits/).

## Method

| Stage | What it does |
|---|---|
| Sensor health gate | Deterministic checks (NaN, flatline, clipping, stuck values, shape) before any model; returns `SENSOR_FAULT`, never a silent pass |
| Features | 1 s windows, 0.5 s hop, per-window mean removal; 81 DSP features (RMS, kurtosis, crest, spectral centroid, flatness, roll-off, 16 Welch bands per axis, cross-axis correlation) |
| Scorers | B0 RMS heuristic, B1 Isolation Forest, B2 XGBoost (supervised reference), AE compact 1D-CNN autoencoder (150k parameters, MSE + log-spectral loss), all trained on healthy recordings only (except B2) |
| Window to file | Aggregation rule (max, mean, top-10% mean) chosen on validation recordings |
| Calibration | C0 static percentile, C1 split conformal, C2 Mondrian per process, C4 similarity-weighted conformal (kernel width chosen by a nested pseudo-shift on validation data), C5 IBM W1-ACAS (their implementation) |
| Streaming calibration | Held-out recordings replayed in time order (natural, and with a contiguous burst of the real faulty runs); operator labels arrive k runs late. Policies: frozen, naive self-labelling, delayed-label admission, ACI from delayed errors, IBM W1-ACAS (label-free), and delayed-label admission + ACI (this work); swept over α to compare recall at matched false-alarm rate; label-coverage ablation |
| Edge | ONNX FP32 and static INT8 (QDQ, per-channel), recalibrated on INT8 scores; single-core latency |
| Serving | FastAPI service with SHA-256 binding between model and calibration, plus a replay dashboard |

## Evaluation protocols

| Protocol | Test set | Shift |
|---|---|---|
| `lopo` | one machining process at a time (15 folds) | unseen process |
| `lomo` | one machine at a time (3 folds) | unseen machine |
| `time` | latest timeframes | later period |

Within the non-test part, healthy recordings are split 60/20/20 into train, validation and calibration, stratified by machine and process. Anomalous recordings outside the test group are used only for model selection and the supervised reference.

## Related work and what is new

See [docs/related_work.md](docs/related_work.md). In short: conformal anomaly detection, weighted conformal, ACI, the self-poisoning of self-labelled recalibration, and adaptive conformal monitoring (IBM W1-ACAS, ICLR 2026) are prior work. New here: a leak-free calibrated-decision benchmark on this real factory dataset; process-aware weighted calibration evaluated against those baselines under three shift types; a label-delay study on real chronological streams; and decision-level parity of an INT8 edge model.

## Reproduce

```bash
make setup      # Python 3.12 venv with pinned dependencies
make data       # download + verify (about 0.9 GB), audit, features, splits
make eval       # all scorers x calibration strategies x protocols
make stream     # streaming label-delay study
make export     # ONNX FP32 + INT8, parity and latency
make tables     # results/summary.json, docs/results_tables.md, figures
make verify     # fast re-run compared with committed results
make test
make demo       # http://localhost:8000
```

## Limitations

- Latency is measured on an Apple M5 CPU core, not on an industrial PC.
- Labels are per recording; per-window anomaly labels do not exist and are never invented.
- Similarity-weighted calibration gives approximate, not guaranteed, false-alarm control under shift.

## Licence

Code: MIT. Data: CC-BY-4.0 (Bosch), not redistributed here. IBM W1-ACAS is used as an installed dependency (Apache-2.0).
