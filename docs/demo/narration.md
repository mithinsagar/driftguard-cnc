# Demo narration (66 s)

Read along with `driftguard_demo.mp4`; each line matches the caption on screen at that time. Record it as one audio
file (for example `voice.m4a` from QuickTime or Voice Memos) and add it to the video:

```bash
ffmpeg -i docs/demo/driftguard_demo.mp4 -i voice.m4a -c:v copy -c:a aac -shortest demo_with_voice.mp4
```

| Time | Say |
|---|---|
| 0:00 | DriftGuard-CNC flags faulty CNC milling runs from vibration, and it keeps false alarms within a budget the engineer sets. |
| 0:06 | Here I score a held-out recording. An INT8 ONNX autoencoder scores every one-second window against a calibrated threshold. |
| 0:13 | This healthy run from machine M02 comes back normal. |
| 0:17 | When the operator confirms it later, only confirmed healthy runs go into the calibration set, so the threshold adapts without learning from faults. |
| 0:25 | A faulty run is flagged as an anomaly. |
| 0:30 | If a sensor axis disconnects, the service says sensor fault. It never turns a broken sensor into a process verdict. |
| 0:36 | Now a real stream from a machine the model never saw, in time order, with a five percent false-alarm budget. |
| 0:47 | A frozen threshold climbs past fifty percent false alarms. IBM's adaptive method recovers after a spike. |
| 0:53 | The delayed-label policy holds the budget and catches more of the faulty runs. |
| 0:58 | And the same check on a later year, after training on 2019 and 2020. |
| 1:03 | The code, the data checks and every result are on GitHub. |
