# Related work and what is new

Novelty search run on 20 November 2026 (arXiv, Google Scholar via web search, GitHub). Re-run before publishing.

## Closest prior work

| Work | What it does | Relation to DriftGuard-CNC |
|---|---|---|
| Tnani, Feil, Diepold, *Smart Data Collection System for Brownfield CNC Milling Machines*, Procedia CIRP 107 (2022), [doi](https://doi.org/10.1016/j.procir.2022.04.022), [data](https://github.com/boschresearch/CNC_Machining) | Publishes the dataset (3 machines, 15 processes, 2 kHz CISS accelerometer, Oct 2018 to Aug 2021) and motivates process variation | We build a calibrated-decision benchmark on it. The search found no published work that applies conformal false-alarm control or reports false-alarm rates under process, machine or time shift on this dataset. |
| Student and project repositories on this dataset (for example One-Class SVM and Isolation Forest notebooks) | Unsupervised detectors with accuracy metrics | No calibrated thresholds, no leak-free cross-process protocol. |
| Bates, Candès, Lei, Romano, Sesia, *Testing for outliers with conformal p-values*, Annals of Statistics 2023, [arXiv](https://arxiv.org/abs/2104.08279) | Finite-sample false-alarm control under exchangeability | Our split-conformal baseline (C1). |
| Tibshirani, Barber, Candès, Ramdas, *Conformal prediction under covariate shift*, NeurIPS 2019, [arXiv](https://arxiv.org/abs/1904.06019) | Weighted conformal with likelihood-ratio weights | Basis of our similarity-weighted rule (C4), which uses kernel weights on file embeddings instead of a known likelihood ratio, so its guarantee is approximate. |
| Gibbs and Candès, *Adaptive conformal inference under distribution shift*, NeurIPS 2021, [arXiv](https://arxiv.org/abs/2106.00170) | Long-run error control with immediate feedback | Our ACI policy, here driven by **delayed** labels. |
| Martinez Gil, O'Donncha, Gifford, Zhou, Patel, Vaculin (IBM Research), *Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring* (W1-ACAS), ICLR 2026, [arXiv](https://arxiv.org/abs/2604.20122), [code](https://github.com/ibm-granite/granite-tsfm/tree/main/notebooks/hfdemo/adaptive_conformal_tsad) | Online adaptive weighted conformal p-values over a stream of nonconformity scores, weights learned with a Wasserstein-1 objective | Run as a baseline with IBM's own implementation (`tsfm_public.toolkit.conformal.AdaptiveWeightedConformalScoreWrapper`, Apache-2.0) on our score streams. Not our contribution. |
| *Self-Poisoning in Adaptive Out-of-Distribution Detection: A Sharp-Threshold Theory and Certified Label-Free Calibration* (2026), [arXiv](https://arxiv.org/abs/2607.21673) | Theory of how self-labelled recalibration drives thresholds down; certified label-free correction | **Establishes the self-poisoning mechanism.** Our streaming study does not claim to discover it; we measure it on real factory streams and compare label-delay-aware policies. |
| CGC-AD (GitHub, 2026) | Reports a ratchet effect of adaptive thresholds under self-labelling on industrial and medical detectors | Same mechanism, different data. |
| *Breakpoint based online anomaly detection*, [arXiv 2402.03565](https://arxiv.org/abs/2402.03565) | Uses the detector's own outputs as calibration labels | An example of the self-labelling practice our naive policy models. |
| *Robust conformal outlier detection under contaminated reference data*, [arXiv 2502.04807](https://arxiv.org/abs/2502.04807) | Conservative validity with contaminated calibration data, using labelled outliers | Offline; complementary. |
| Conformal anomaly detection for industrial cyber-physical systems (Reliability Engineering and System Safety, 2026), [PolyU](https://research.polyu.edu.hk/en/publications/conformal-machine-learning-for-reliable-anomaly-detection-in-indu/) | Excludes significant anomalies from calibration to keep false-alarm guarantees | Offline calibration set; not label delay. |

## What is new here, and what the experiments said

| Claim (pre-registered) | Outcome |
|---|---|
| **Benchmark.** An open, leak-free, file-level benchmark for calibrated anomaly decisions on the Bosch CNC dataset with three shift protocols, committed split manifests, four scorers, six calibration strategies including IBM's W1-ACAS, and confidence intervals | Delivered (`splits/`, `results/`, `docs/results_tables.md`) |
| **C1.** Similarity-weighted conformal thresholds track the target better than pooled calibration on unseen processes | **Refuted.** Raw weighting collapses to infinite thresholds when a process is unlike all calibration data (no alarms, no recall); with an effective-sample-size fallback it behaves like pooled split conformal. Reported as a negative result. |
| **C1b.** A few confirmed runs of a new process restore calibration quickly | **Refuted for static rules.** About 300 older calibration recordings dominate; calibrating on the new process alone needs at least 19 runs at α = 5%. |
| **C2 (narrowed after the novelty search).** Under label delay on real chronological factory streams, label-aware online calibration beats label-free and self-labelled calibration | **Supported.** Delayed-label admission with an ACI correction from the same late labels is the best-calibrated policy in every streaming setting tested and has higher recall at matched low false-alarm rates in most settings, against IBM W1-ACAS. Its cost is label dependence, quantified by a label-coverage ablation. The self-poisoning of naive self-labelling is prior work; we measure it on real streams. |
| **C3.** INT8 decisions match FP32 after recalibration | See `results/edge.json` |

What is **not** claimed: inventing conformal anomaly detection, weighted conformal, ACI, delayed-label admission as an idea on its own, the self-poisoning mechanism, or adaptive conformal monitoring with foundation models. The contribution is the measured combination on real factory data, against the strongest available baseline run with its authors' code.
