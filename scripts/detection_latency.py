"""Detection latency of the deployed INT8 model: seconds into a faulty run until its running file score crosses the threshold.

The running file score after w windows uses the deployed aggregation rule over the windows seen so far (window w ends at
1.0 + 0.5 * (w - 1) seconds). Reported for the faulty runs that are detected by the end of the run, on the 2021 test fold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driftguard.calibration.conformal import split_conformal_threshold  # noqa: E402
from driftguard.data.dataset import load_signal  # noqa: E402
from driftguard.features.dsp import windows  # noqa: E402
from driftguard.serve.scoring import OnnxScorer  # noqa: E402


def running(ws: np.ndarray, rule: str) -> np.ndarray:
    if rule == "mean":
        return np.cumsum(ws) / np.arange(1, len(ws) + 1)
    if rule == "max":
        return np.maximum.accumulate(ws)
    return np.array([np.sort(ws[: i + 1])[-max(1, int(np.ceil(0.1 * (i + 1)))):].mean() for i in range(len(ws))])


def main():
    cal = json.loads(Path("artifacts/calibration.json").read_text())
    df = pd.read_csv("data/manifest.csv").set_index("file_id")
    scorer = OnnxScorer("artifacts/" + cal["model"], cal["mu"], cal["sd"])
    thr = split_conformal_threshold(np.asarray(cal["cal_scores"]), cal["default_alpha"])
    rows, healthy_ever, healthy_end = [], [], []
    for fid in cal["replay_files"]:
        ws = scorer.score(windows(load_signal(df.loc[fid, "path"])))
        run = running(ws, cal["aggregation"])
        if df.loc[fid, "label"] != "bad":
            healthy_ever.append(bool((run > thr).any()))
            healthy_end.append(bool(run[-1] > thr))
            continue
        crossed = np.flatnonzero(run > thr)
        duration = 1.0 + 0.5 * (len(ws) - 1)
        if len(crossed):
            t = 1.0 + 0.5 * crossed[0]
            rows.append({"file_id": fid, "detected": True, "seconds": t, "run_seconds": duration, "fraction_of_run": t / duration,
                         "detected_at_end": bool(run[-1] > thr)})
        else:
            rows.append({"file_id": fid, "detected": False, "run_seconds": duration})
    det = [r for r in rows if r["detected"]]
    secs = np.array([r["seconds"] for r in det])
    frac = np.array([r["fraction_of_run"] for r in det])
    out = {"alpha": cal["default_alpha"], "aggregation": cal["aggregation"], "threshold": thr, "n_faulty_runs": len(rows),
           "n_ever_crossed": len(det), "n_detected_at_end": int(sum(r.get("detected_at_end", False) for r in det)),
           "median_seconds": float(np.median(secs)) if len(det) else None, "p90_seconds": float(np.percentile(secs, 90)) if len(det) else None,
           "median_fraction_of_run": float(np.median(frac)) if len(det) else None,
           "healthy_false_alarm_rate_end_of_run": float(np.mean(healthy_end)),
           "healthy_false_alarm_rate_alarm_on_first_crossing": float(np.mean(healthy_ever)), "runs": rows}
    Path("results/detection_latency.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "runs"})


if __name__ == "__main__":
    main()
