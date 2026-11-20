from pathlib import Path

import numpy as np
import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "calibration.json").exists(), reason="run `make export` first")


@pytest.fixture(scope="module")
def client():
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    from fastapi.testclient import TestClient

    from driftguard.serve.app import app

    return TestClient(app)


def _signal(seed=0, n=6000):
    return np.random.default_rng(seed).normal(size=(n, 3)).astype(float)


def test_health(client):
    assert client.get("/v1/health").json()["status"] == "ok"


def test_score_returns_run_id_and_sensor_faults_are_separate(client):
    r = client.post("/v1/score", json={"samples": _signal().tolist()}).json()
    assert r["decision"] in {"NORMAL", "ANOMALY"} and r["n_windows"] == 5 and r["run_id"]
    sig = _signal(1)
    sig[1000:2500, 1] = 0.0
    assert client.post("/v1/score", json={"samples": sig.tolist()}).json()["decision"] == "SENSOR_FAULT"


def test_operator_feedback_updates_the_calibrator(client):
    client.post("/v1/reset")
    run = client.post("/v1/score", json={"samples": _signal(2).tolist()}).json()["run_id"]
    state = client.post("/v1/feedback", json={"run_id": run, "healthy": True}).json()
    assert state["confirmed_healthy"] == 1
    assert client.post("/v1/feedback", json={"run_id": run, "healthy": True}).status_code == 404


def test_alpha_below_minimum_rejected(client):
    assert client.put("/v1/config", json={"alpha": 0.0001}).status_code == 422


def test_demo_streams_available(client):
    scenes = client.get("/v1/demo/streams").json()["scenes"]
    assert all({"frozen", "adaptive", "w1acas"} <= set(s["policies"]) for s in scenes)
