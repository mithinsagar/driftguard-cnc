"""Fill the evaluation block of docs/model_card.md from results/edge.json."""
import json
import re
from pathlib import Path

e = json.loads(Path("results/edge.json").read_text())
i8, f32, t = e["onnx_int8"], e["onnx_fp32"], e["torch_fp32"]
n_test = 469
flips = round((1 - i8["verdict_agreement_recalibrated"]) * n_test)
block = "\n".join([
    f"- Window-to-file aggregation: `{e['aggregation']}` (chosen on validation data by `run_eval.py`).",
    f"- Full precision at α = 5% on the 2021 test period: achieved false-alarm rate {100 * t['far']:.1f}%, recall {100 * t['recall']:.0f}%.",
    f"- INT8 after recalibration on INT8 scores: verdict agreement with FP32 {100 * i8['verdict_agreement_recalibrated']:.2f}% of {n_test} held-out "
    f"recordings ({flips} flip{'s' if flips != 1 else ''}); {100 * i8['verdict_agreement_naive']:.2f}% if FP32 thresholds are reused; achieved false-alarm "
    f"rate {100 * i8['far_recalibrated']:.1f}%, recall {100 * i8['recall_recalibrated']:.0f}%; score correlation {i8['score_pearson_vs_torch']:.4f}.",
    f"- Size {i8['size_kb']:.0f} KB (FP32 {f32['size_kb']:.0f} KB). Latency p95 {i8['latency_single_window_1thread']['p95_ms']:.2f} ms per 1 s window on one "
    f"Apple M5 CPU core (FP32 {f32['latency_single_window_1thread']['p95_ms']:.2f} ms); not measured on industrial hardware.",
    "- This scorer ranks anomalies less well than the RMS heuristic or the supervised reference on this data (see `docs/results_tables.md`); "
    "it is deployed because it is unsupervised, small and fast, and the calibration layer is model-agnostic.",
])
p = Path("docs/model_card.md")
p.write_text(re.sub(r"<!-- EDGE:START -->.*<!-- EDGE:END -->", f"<!-- EDGE:START -->\n{block}\n<!-- EDGE:END -->", p.read_text(), flags=re.S))
print(block)
