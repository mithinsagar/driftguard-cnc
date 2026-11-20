# Pre-registered plan and outcomes

The hypotheses below were written on 20 November 2026 in the project blueprint (`Mithin_Sagar_AI_Portfolio_Blueprint.pdf`, Part A, section A.9.2) before any experiment was run. Outcomes link to generated results. Targets were hypotheses, never results.

| Id | Hypothesis as pre-registered | Outcome | Evidence |
|---|---|---|---|
| H1 | Under leave-one-process-out, the static rule exceeds α + 0.03 on most folds at α = 5% | **Partly.** The static rule violates on 20% to 53% of process folds depending on the scorer; the large failures are under machine shift (static 18% pooled for the supervised scorer; static 47% and split conformal 58% for the autoencoder on M03) and time shift (static 44% for the supervised scorer) | `docs/results_tables.md` |
| H2 | Pooled split conformal reduces calibration error relative to static but still violates on some folds | **Mixed.** Split conformal is not better than the static rule under machine or time shift; both assume exchangeability that the shift breaks | `docs/results_tables.md` |
| H3 (C1) | Similarity-weighted conformal reaches mean calibration error ≤ 2% across process folds at matched recall | **Refuted.** Raw weighting degenerates to infinite thresholds on unseen processes; with an effective-sample-size fallback it matches pooled calibration | `docs/results_tables.md`, `docs/related_work.md` |
| H4 (C2) | In the fault-burst stream the naive rule exceeds 2α and delayed-label admission stays within α ± 0.02 | **Half supported.** The naive part held: self-labelling reached 27.6% against 19.9% frozen (supervised scorer, unseen machines). The delayed-label part failed as stated: admission alone overshot to 8.5%. A post-hoc policy adding an ACI correction from the same late labels (P5) held 5.3% with 87% recall and was the best-calibrated policy in every streaming setting | `docs/stream_tables.md`, `reports/figures/headline.png` |
| H5 (C3) | INT8 verdict agreement ≥ 99% after recalibration; p95 ≤ 10 ms per window on one CPU core | **Supported.** See `results/edge.json` and the model card | `docs/model_card.md` |

Added after review, to remove the selection concern: a held-out streaming protocol in which every tunable policy (P5 and W1-ACAS) has its settings chosen on the first half of each stream and is scored on the second half only (`scripts/run_holdout.py`, `docs/holdout_tables.md`). The README and report lead with these held-out numbers.

Added after the novelty search (before running the streaming study): IBM W1-ACAS as baseline C5, run with IBM's code. Added after observing H3: the effective-sample-size fallback (C4f), reported alongside raw C4. Added after observing H4's overshoot: policy P5 (delayed-label admission plus ACI). Both additions are reported as such; neither replaced a pre-registered comparison.
