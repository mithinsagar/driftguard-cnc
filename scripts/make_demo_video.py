"""Record a captioned walkthrough of the dashboard (docs/demo/driftguard_demo.mp4, about 70 s).

Needs the service running with the dataset (`make demo`), Playwright with a local Chrome (`pip install playwright`;
uses the installed Google Chrome) and ffmpeg. Captions match docs/demo/narration.md, so a voice-over can be recorded on
top: `ffmpeg -i docs/demo/driftguard_demo.mp4 -i voice.m4a -c:v copy -c:a aac -shortest demo_with_voice.mp4`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:8000"
OUT = ROOT / "docs" / "demo" / "driftguard_demo.mp4"
SIZE = {"width": 1280, "height": 800}

CAPTION_JS = """(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div'); el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);max-width:1040px;padding:14px 22px;'
      + 'background:rgba(8,12,18,.88);color:#f2f5f8;font:500 19px/1.45 -apple-system,Segoe UI,sans-serif;border-radius:10px;'
      + 'border:1px solid rgba(255,255,255,.12);z-index:9999;text-align:center;box-shadow:0 6px 30px rgba(0,0,0,.4)';
    document.body.appendChild(el);
  }
  el.textContent = text; el.style.display = text ? 'block' : 'none';
}"""


def api(path: str, method: str = "GET") -> dict:
    req = urllib.request.Request(BASE + path, method=method)
    return json.load(urllib.request.urlopen(req, timeout=120))


def pick_files() -> tuple[str, str]:
    """A held-out healthy recording the model calls NORMAL and a faulty one it calls ANOMALY (checked, not assumed)."""
    files = api("/v1/replay/files")
    good = next(f["file_id"] for f in files if f["label"] == "good" and f["machine"] == "M02"
                and api(f"/v1/replay/{f['file_id']}")["decision"] == "NORMAL")
    bad = next(f["file_id"] for f in files if f["label"] == "bad" and api(f"/v1/replay/{f['file_id']}")["decision"] == "ANOMALY")
    api("/v1/reset", "POST")
    return good, bad


def main() -> None:
    good, bad = pick_files()
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport=SIZE, record_video_dir=str(tmp), record_video_size=SIZE)
        page = ctx.new_page()
        page.goto(BASE + "/")
        page.wait_for_function("document.querySelectorAll('#file option').length > 0")

        def say(text: str, seconds: float) -> None:
            page.evaluate(CAPTION_JS, text)
            page.wait_for_timeout(int(seconds * 1000))

        def score(file_id: str, fault: str = "") -> None:
            page.select_option("#file", file_id)
            page.select_option("#fault", fault)
            page.click("#go")
            page.wait_for_function("!['SCORING', 'IDLE', ''].includes(document.getElementById('badge').textContent)", timeout=60000)

        page.click("#tabLive")
        say("DriftGuard-CNC flags faulty CNC milling runs from vibration, while keeping false alarms within a budget the engineer sets.", 6)
        say("Live decisions: score a held-out recording. An INT8 ONNX autoencoder scores each 1-second window against a calibrated threshold.", 5)
        score(good)
        say("A healthy run from machine M02: NORMAL.", 4)
        page.click("#confirmOk")
        say("The operator confirms it later. Only confirmed healthy runs enter the calibration set, and the threshold adapts (delayed labels + ACI).", 6)
        score(bad)
        say("A faulty run: ANOMALY.", 4)
        score(good, "flatline")
        say("A disconnected sensor axis returns SENSOR FAULT, never a process verdict.", 5)
        page.click("#tabStream")
        page.select_option("#scene", "1")
        say("Streaming replay: real runs from a machine the model never saw, in time order, at a 5% false-alarm budget.", 5)
        page.click("#play")
        page.wait_for_timeout(5000)
        say("The frozen threshold climbs past 50% false alarms over the last 50 healthy runs. IBM W1-ACAS recovers after a spike.", 6)
        say("The delayed-label policy holds the budget while catching more of the faulty runs.", 5)
        page.select_option("#scene", "2")
        page.click("#play")
        say("Same check on a later year (2021) after training on 2019 and 2020.", 6)
        say("Code, data checks and results: github.com/mithinsagar/driftguard-cnc", 4)
        video = page.video.path()
        ctx.close()
        browser.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264", "-crf", "24", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.1f} MB (healthy {good}, faulty {bad})")


if __name__ == "__main__":
    main()
