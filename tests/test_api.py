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


def test_health(client):
    assert client.get("/v1/health").json()["status"] == "ok"


def test_score_and_fault(client):
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(6000, 3)).astype(float)
    r = client.post("/v1/score", json={"samples": sig.tolist()}).json()
    assert r["decision"] in {"NORMAL", "ANOMALY"} and r["n_windows"] == 5
    sig[1000:2500, 1] = 0.0
    r = client.post("/v1/score", json={"samples": sig.tolist()}).json()
    assert r["decision"] == "SENSOR_FAULT"


def test_alpha_below_minimum_rejected(client):
    assert client.put("/v1/config", json={"alpha": 0.0001}).status_code == 422
