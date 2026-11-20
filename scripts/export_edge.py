"""make export: train the deployment autoencoder (chronological fold), export ONNX FP32 and static INT8,
check parity and decision agreement, measure single-core latency, and write artifacts/ + results/edge.json (claim C3).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driftguard.calibration.conformal import split_conformal_threshold  # noqa: E402
from driftguard.data import splits  # noqa: E402
from driftguard.data.dataset import load_signal  # noqa: E402
from driftguard.eval import engine  # noqa: E402
from driftguard.features.dsp import windows  # noqa: E402
from driftguard.models.autoencoder import AEScorer  # noqa: E402
from driftguard.models.scorers import aggregate  # noqa: E402
from driftguard.serve.scoring import OnnxScorer  # noqa: E402

ART = Path("artifacts")
ALPHA = 0.05


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def file_scores(score_fn, df, rows, rule):
    wins, fidx = [], []
    for k, r in enumerate(rows):
        w = windows(load_signal(df.path.iloc[r]))
        wins.append(score_fn(w) if len(w) else np.empty(0))
        fidx.append(np.full(len(w), k))
    return aggregate(np.concatenate(wins), np.concatenate(fidx), len(rows), rule)


def latency(fn, x, warmup=100, runs=2000):
    for _ in range(warmup):
        fn(x)
    ts = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter_ns()
        fn(x)
        ts[i] = (time.perf_counter_ns() - t0) / 1e6
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "p99_ms": float(np.percentile(ts, 99))}


def main(seed: int = 0):
    ART.mkdir(exist_ok=True)
    df = pd.read_csv("data/manifest.csv")
    fold = splits.load("time")[0]
    fd = engine.fold_indices(df, fold, "time")
    rng = np.random.default_rng(seed)

    def sample(rows, k):
        out = []
        for r in rows:
            w = windows(load_signal(df.path.iloc[r]))
            if len(w):
                out.append(w[rng.choice(len(w), min(k, len(w)), replace=False)])
        return np.concatenate(out)

    tr = sample(fd.idx["train_good"], 24)
    ae = AEScorer(seed=seed)
    ckpt = ART / "ae_time_fold.pt"
    if ckpt.exists():  # the network evaluated by run_eval.py on the chronological fold
        from driftguard.models.autoencoder import ConvAE

        state = torch.load(ckpt, weights_only=False)
        ae.model = ConvAE()
        ae.model.load_state_dict(state["state_dict"])
        ae.mu, ae.sd = state["mu"], state["sd"]
    else:
        ae.fit(tr, sample(fd.idx["val_good"], 6))
    model = ae.model.to("cpu").eval()

    fp32 = ART / "ae_fp32.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 2000), fp32, input_names=["x"], output_names=["x_hat"],
                      dynamic_axes={"x": {0: "batch"}, "x_hat": {0: "batch"}}, opset_version=17, dynamo=False)

    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    calib = sample(fd.idx["train_good"][rng.permutation(len(fd.idx["train_good"]))[:250]], 2)[:500]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(((calib[i:i + 1] - ae.mu) / ae.sd).astype(np.float32).transpose(0, 2, 1) for i in range(len(calib)))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"x": x}

    pre = ART / "ae_fp32_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    int8 = ART / "ae_int8.onnx"
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

    s_fp32 = OnnxScorer(str(fp32), ae.mu, ae.sd)
    s_int8 = OnnxScorer(str(int8), ae.mu, ae.sd)
    # use exactly the aggregation rule chosen on validation data for this fold by run_eval.py
    rule = json.loads(Path("results/raw/time/AE_seed0.json").read_text())["folds"][0]["aggregation"]
    cal, test = fd.idx["cal_good"], fd.idx["test"]
    y = (df.label.to_numpy()[test] == "bad")
    res = {"alpha": ALPHA, "aggregation": rule, "params": ae.n_params()}
    scores = {}
    for name, fn in (("torch_fp32", ae.score), ("onnx_fp32", s_fp32.score), ("onnx_int8", s_int8.score)):
        scores[name] = {"cal": file_scores(fn, df, cal, rule), "test": file_scores(fn, df, test, rule)}
        if name == "onnx_int8":
            scores[name]["val"] = file_scores(fn, df, fd.idx["val_good"], rule)
    ref = scores["torch_fp32"]
    thr_ref = split_conformal_threshold(ref["cal"], ALPHA)
    for name in ("onnx_fp32", "onnx_int8"):
        s = scores[name]
        thr_naive = thr_ref
        thr_recal = split_conformal_threshold(s["cal"], ALPHA)
        v_ref = ref["test"] > thr_ref
        res[name] = {
            "score_pearson_vs_torch": float(np.corrcoef(ref["test"], s["test"])[0, 1]),
            "verdict_agreement_naive": float(np.mean((s["test"] > thr_naive) == v_ref)),
            "verdict_agreement_recalibrated": float(np.mean((s["test"] > thr_recal) == v_ref)),
            "far_recalibrated": float(np.mean((s["test"] > thr_recal)[~y])), "recall_recalibrated": float(np.mean((s["test"] > thr_recal)[y])),
            "size_kb": round(Path(ART / ("ae_fp32.onnx" if name == "onnx_fp32" else "ae_int8.onnx")).stat().st_size / 1024, 1),
        }
    res["torch_fp32"] = {"far": float(np.mean(v_ref[~y])), "recall": float(np.mean(v_ref[y]))}
    one = ((tr[:1] - ae.mu) / ae.sd).astype(np.float32)
    proc = psutil.Process()
    for name, sc in (("onnx_fp32", s_fp32), ("onnx_int8", s_int8)):
        rss0 = proc.memory_info().rss
        res[name]["latency_single_window_1thread"] = latency(sc.score, tr[:1])
        res[name]["rss_delta_mb"] = round((proc.memory_info().rss - rss0) / 2**20, 1)
    _ = one
    res["hardware"] = "Apple M5 CPU, ONNX Runtime CPUExecutionProvider, 1 intra-op thread; latency = forward pass + NumPy loss"
    Path("results").mkdir(exist_ok=True)
    Path("results/edge.json").write_text(json.dumps(res, indent=1))
    calib_doc = {"model": "ae_int8.onnx", "model_sha256": sha256(int8), "mu": ae.mu.tolist(), "sd": ae.sd.tolist(),
                 "aggregation": rule, "cal_scores": scores["onnx_int8"]["cal"].tolist(), "default_alpha": ALPHA,
                 "val_scores": scores["onnx_int8"]["val"].tolist(), "fold": fd.name,
                 "replay_files": df.file_id.to_numpy()[test].tolist()}
    (ART / "calibration.json").write_text(json.dumps(calib_doc))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
