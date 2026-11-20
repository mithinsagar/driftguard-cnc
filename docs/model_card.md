# Model card: DriftGuard-CNC deployed model

## Intended use

Flag anomalous CNC milling runs from tri-axial vibration at 2 kHz, at a false-alarm rate chosen by a reliability engineer, as decision support for operators. Not a safety system: an alarm is a prompt for a human check, and a sensor fault is reported separately from a process anomaly.

## Out of scope

Machines, sensors or sampling rates other than those in the Bosch CNC dataset without recalibration; per-window fault localisation (labels are per recording); any claim of guaranteed false-alarm control when new recordings do not resemble the calibration set.

## Deployed artefacts

| File | What it is |
|---|---|
| `artifacts/ae_int8.onnx` | 1D-CNN autoencoder (150,179 parameters), static INT8 (QDQ, per-channel), 188 KB |
| `artifacts/calibration.json` | per-axis normalisation, the window-to-file aggregation rule chosen on validation data, 234 healthy calibration file scores, validation scores for the static baseline, model SHA-256 |

The service refuses to start if the SHA-256 of the loaded model differs from the one recorded in the calibration file.

## Training and calibration data

Chronological fold of the Bosch CNC Machining dataset: trained on healthy recordings up to August 2020, calibrated on held-out healthy recordings from the same period, tested on February and August 2021 (452 healthy, 17 anomalous recordings). All splits by recording, never by window (`splits/time.json`).

## Evaluation

<!-- EDGE:START -->
- Window-to-file aggregation: `mean` (chosen on validation data by `run_eval.py`).
- Full precision at α = 5% on the 2021 test period: achieved false-alarm rate 3.1%, recall 47%.
- INT8 after recalibration on INT8 scores: verdict agreement with FP32 99.79% of 469 held-out recordings (1 flip); 99.57% if FP32 thresholds are reused; achieved false-alarm rate 3.1%, recall 41%; score correlation 0.9984.
- Size 188 KB (FP32 596 KB). Latency p95 1.22 ms per 1 s window on one Apple M5 CPU core (FP32 1.20 ms); not measured on industrial hardware.
- This scorer ranks anomalies less well than the RMS heuristic or the supervised reference on this data (see `docs/results_tables.md`); it is deployed because it is unsupervised, small and fast, and the calibration layer is model-agnostic.
<!-- EDGE:END -->

## What the false-alarm promise means

The conformal threshold keeps the false-alarm rate at or below α when new healthy recordings are exchangeable with the calibration recordings. Under a new process, machine or time period this can fail badly (see `reports/figures/target_vs_achieved_*.png`). The service makes its decisions with the online delayed-label calibrator: confirmed-healthy runs update the threshold, an ACI correction keeps the working α on budget, and faulty confirmations never enter the calibration set. In the streaming study this policy keeps achieved rates close to target under machine and time shift when operator confirmations arrive with a delay, but its control is long-run and empirical, not a finite-sample guarantee, and it needs label coverage of roughly 25 to 50% of runs to stay near target.

## Known failure modes

- Misses faulty runs whose vibration stays within the healthy reconstruction range (see the recall figures above).
- Process-conditional (Mondrian) thresholds can become too conservative on a new machine and lose most recall.
- Similarity-weighted thresholds return no finite value for recordings unlike all calibration data; the fallback reverts to pooled calibration.
- The sensor gate detects NaN, flatlines, clipping, stuck values and wrong shapes; it does not detect a slowly drifting sensor gain.

## Licences

Code MIT. Data CC-BY-4.0 (Bosch), not redistributed. IBM granite-tsfm (Apache-2.0) used as an installed dependency for the W1-ACAS baseline.
