"""DriftGuard-CNC service: sensor gate -> INT8 ONNX autoencoder -> file score -> calibrated decision.

Decisions use the online delayed-label calibrator (policy P5 of the streaming study): the threshold adapts as operators
confirm runs as healthy, possibly many runs later, and ACI keeps the achieved false-alarm rate on the chosen budget.
The fixed split-conformal and static-percentile thresholds are reported alongside for comparison.

Run: uvicorn driftguard.serve.app:app --port 8000   (artifacts/ produced by `make export`)
The service refuses to start if the calibration file was computed for a different model (SHA-256 binding).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from driftguard.calibration.adaptive import AdaptiveCalibrator
from driftguard.calibration.conformal import conformal_pvalue, min_alpha, split_conformal_threshold, static_percentile
from driftguard.data.dataset import load_signal
from driftguard.features.dsp import windows
from driftguard.serve.gate import check_window
from driftguard.serve.scoring import OnnxScorer

ART = Path(os.environ.get("DRIFTGUARD_ARTIFACTS", "artifacts"))
STATIC = Path(__file__).parent / "static"


class Config(BaseModel):
    alpha: float = Field(0.05, gt=0, lt=0.5)


class ScoreRequest(BaseModel):
    samples: list[list[float]] = Field(..., description="n x 3 accelerations (X, Y, Z) at 2 kHz")


class Feedback(BaseModel):
    run_id: str
    healthy: bool = Field(..., description="operator confirmation: true = the run was healthy, false = it was faulty")


def _load():
    cal = json.loads((ART / "calibration.json").read_text())
    model = ART / cal["model"]
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != cal["model_sha256"]:
        raise RuntimeError(f"calibration was computed for {cal['model_sha256'][:12]}, loaded model is {digest[:12]}")
    return cal, OnnxScorer(str(model), cal["mu"], cal["sd"])


CAL, SCORER = _load()
CAL_SCORES = np.asarray(CAL["cal_scores"], dtype=float)
ADAPTIVE = AdaptiveCalibrator(CAL_SCORES, alpha=CAL["default_alpha"])
LOCK = threading.Lock()
MANIFEST = pd.read_csv("data/manifest.csv").set_index("file_id") if Path("data/manifest.csv").exists() else None
DEMO = json.loads((ART / "demo_streams.json").read_text()) if (ART / "demo_streams.json").exists() else {"scenes": []}
app = FastAPI(title="DriftGuard-CNC", version="1.1")


def fixed_thresholds(alpha: float) -> dict:
    return {"conformal": split_conformal_threshold(CAL_SCORES, alpha),
            "static_percentile": static_percentile(np.asarray(CAL["val_scores"]), 100 * (1 - alpha))}


def file_score(win_scores: np.ndarray) -> float:
    """Window scores -> file score with the aggregation rule recorded in the calibration file (chosen on validation data)."""
    rule = CAL.get("aggregation", "top10")
    if rule == "max":
        return float(win_scores.max())
    if rule == "mean":
        return float(win_scores.mean())
    k = max(1, int(np.ceil(0.1 * len(win_scores))))
    return float(np.sort(win_scores)[-k:].mean())


def decide(signal: np.ndarray, record: bool = True) -> dict:
    t0 = time.perf_counter()
    w = windows(signal)
    if len(w) == 0:
        raise HTTPException(422, "need at least 2,000 samples (1 s at 2 kHz)")
    faults = [check_window(x) for x in w]
    if any(faults):
        bad = next(f for f in faults if f)
        return {"decision": "SENSOR_FAULT", "fault": bad, "faulty_windows": int(sum(bool(f) for f in faults)),
                "latency_ms": round((time.perf_counter() - t0) * 1e3, 2), "model_sha256": CAL["model_sha256"]}
    ws = SCORER.score(w)
    s = file_score(ws)
    with LOCK:
        thr = ADAPTIVE.threshold()
        run_id = uuid.uuid4().hex[:12]
        flagged = ADAPTIVE.record(run_id, s) if record else bool(s > thr)
        state = ADAPTIVE.state()
    fixed = fixed_thresholds(ADAPTIVE.alpha)
    return {"run_id": run_id if record else None, "decision": "ANOMALY" if flagged else "NORMAL", "file_score": s,
            "threshold": thr, "alpha": ADAPTIVE.alpha, "alpha_t": state["alpha_t"],
            "fixed_conformal_threshold": fixed["conformal"], "static_threshold": fixed["static_percentile"],
            "p_value_vs_initial_calibration": conformal_pvalue(CAL_SCORES, s), "window_scores": ws.tolist(), "n_windows": len(w),
            "latency_ms": round((time.perf_counter() - t0) * 1e3, 2), "model_sha256": CAL["model_sha256"], "calibrator": state}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/v1/health")
def health():
    return {"status": "ok", "model_sha256": CAL["model_sha256"][:16], "n_cal": len(CAL_SCORES)}


@app.get("/v1/config")
def get_config():
    with LOCK:
        state = ADAPTIVE.state()
    return {"alpha": ADAPTIVE.alpha, "min_alpha": min_alpha(len(CAL_SCORES)), **fixed_thresholds(ADAPTIVE.alpha), "adaptive": state,
            "n_cal": len(CAL_SCORES), "fold": CAL["fold"], "aggregation": CAL.get("aggregation", "top10")}


@app.put("/v1/config")
def put_config(cfg: Config):
    if cfg.alpha < min_alpha(len(CAL_SCORES)):
        raise HTTPException(422, f"alpha below {min_alpha(len(CAL_SCORES)):.4f} cannot be promised with this calibration set")
    with LOCK:
        ADAPTIVE.reset(alpha=cfg.alpha)
    return get_config()


@app.post("/v1/score")
def score(req: ScoreRequest):
    return decide(np.asarray(req.samples, dtype=np.float32))


@app.post("/v1/feedback")
def feedback(fb: Feedback):
    with LOCK:
        ok = ADAPTIVE.feedback(fb.run_id, fb.healthy)
        state = ADAPTIVE.state()
    if not ok:
        raise HTTPException(404, "unknown or already confirmed run id")
    return state


@app.post("/v1/reset")
def reset():
    with LOCK:
        ADAPTIVE.reset()
        return ADAPTIVE.state()


@app.get("/v1/replay/files")
def replay_files():
    if MANIFEST is None:
        return []
    rows = MANIFEST.loc[[f for f in CAL["replay_files"] if f in MANIFEST.index]]
    return [{"file_id": i, "machine": r.machine, "process": r.process, "label": r.label, "timeframe": r.timeframe}
            for i, r in rows.iterrows()]


@app.get("/v1/replay/{file_id}")
def replay(file_id: str, fault: str | None = None):
    if MANIFEST is None or file_id not in MANIFEST.index:
        raise HTTPException(404, "unknown file")
    sig = load_signal(MANIFEST.loc[file_id, "path"]).copy()
    if fault == "flatline":
        sig[len(sig) // 3: len(sig) // 3 + 1200, 2] = 0.0
    elif fault == "invalid":
        sig[len(sig) // 2, 0] = np.nan
    out = decide(sig)
    z = sig[:, 2]
    step = max(1, len(z) // 1500)
    out.update({"file_id": file_id, "label": MANIFEST.loc[file_id, "label"], "process": MANIFEST.loc[file_id, "process"],
                "machine": MANIFEST.loc[file_id, "machine"], "waveform_z": np.nan_to_num(z[::step]).tolist()})
    return out


@app.get("/v1/demo/streams")
def demo_streams():
    """Precomputed replays of real held-out streams through four calibration policies (scripts/make_demo_streams.py)."""
    return DEMO
