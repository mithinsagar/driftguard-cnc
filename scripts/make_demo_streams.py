"""Precompute dashboard replay streams from evaluation file scores (artifacts/demo_streams.json).

Each scene replays real held-out recordings in time order through four policies; the scores come from the scorer that was
trained without the held-out machine or period, exactly as in the streaming study.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driftguard.calibration.stream import order_stream, rolling_far, simulate  # noqa: E402
from driftguard.data import splits  # noqa: E402
from driftguard.eval import engine  # noqa: E402

SCENES = [
    ("lomo", "lomo_M03", "AE", "Unseen machine M03, autoencoder"),
    ("lomo", "lomo_M02", "B2", "Unseen machine M02, supervised XGBoost"),
    ("time", "time_from_202102", "B2", "Year 2021 after training on 2019 and 2020, supervised XGBoost"),
]
POLICIES = {"frozen": ("P0_frozen", {}), "naive": ("P1_naive", {}), "w1acas": ("P4_w1acas", {}),
            "adaptive": ("P5_delayed_aci", {"delay": 5, "gamma": 0.005, "buffer_size": 300})}
ALPHA = 0.05


def main():
    df = pd.read_csv("data/manifest.csv")
    scenes = []
    for proto, fold_name, scorer, title in SCENES:
        fold = next(f for f in splits.load(proto) if f["name"] == fold_name)
        fd = engine.fold_indices(df, fold, proto)
        fs = np.load(f"results/scores/{proto}_{scorer}_seed0.npz")[fold_name]
        test = fd.idx["test"]
        bad = df.label.to_numpy()[test] == "bad"
        order = order_stream(df.t_order.to_numpy()[test], df["index"].to_numpy()[test], bad, "natural")
        s, b = fs[test][order], bad[order]
        scene = {"title": title, "alpha": ALPHA, "delay": 5, "scores": np.round(s, 5).tolist(), "bad": b.astype(int).tolist(),
                 "file_ids": df.file_id.to_numpy()[test][order].tolist(), "policies": {}}
        for key, (pol, kw) in POLICIES.items():
            r = simulate(s, b, fs[fd.idx["cal_good"]], ALPHA, pol, **kw)
            scene["policies"][key] = {"flags": r["flags"].astype(int).tolist(),
                                      "threshold": [round(float(x), 5) if np.isfinite(x) else None for x in r["thresholds"]],
                                      "rolling_far": [None if np.isnan(x) else round(float(x), 4) for x in rolling_far(r["flags"], b, 50)],
                                      "far": r["far"], "recall": r["recall"]}
        scenes.append(scene)
        print(title, {k: (round(v["far"], 3), round(v["recall"], 3)) for k, v in scene["policies"].items()})
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/demo_streams.json").write_text(json.dumps({"scenes": scenes}))


if __name__ == "__main__":
    main()
