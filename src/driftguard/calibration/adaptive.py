"""Online delayed-label calibrator used by the service (policy P5 of the streaming study).

Each scored run is remembered with its score and whether it raised an alarm. When an operator later confirms a run:
  - confirmed healthy: its score enters the calibration buffer, and ACI receives "was this healthy run flagged?"
  - confirmed faulty: nothing enters the buffer (faulty runs must never become the definition of normal)
The threshold is the split-conformal quantile of the buffer at the ACI working level alpha_t.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from driftguard.calibration.conformal import ACI, RollingBuffer


@dataclass
class AdaptiveCalibrator:
    cal_scores: np.ndarray
    alpha: float = 0.05
    gamma: float = 0.005
    buffer_size: int = 300
    max_pending: int = 10000
    buf: RollingBuffer = field(init=False)
    aci: ACI = field(init=False)
    pending: OrderedDict = field(init=False)
    n_confirmed_healthy: int = field(init=False, default=0)
    n_confirmed_faulty: int = field(init=False, default=0)

    def __post_init__(self):
        self.reset()

    def reset(self, alpha: float | None = None) -> None:
        if alpha is not None:
            self.alpha = alpha
        self.buf = RollingBuffer(np.asarray(self.cal_scores, float), size=self.buffer_size)
        self.aci = ACI(np.asarray(self.cal_scores, float), self.alpha, gamma=self.gamma)
        self.pending = OrderedDict()
        self.n_confirmed_healthy = self.n_confirmed_faulty = 0

    @property
    def alpha_t(self) -> float:
        return float(np.clip(self.aci.alpha_t, 1e-4, 0.5))

    def threshold(self) -> float:
        return self.buf.threshold(self.alpha_t)

    def record(self, run_id: str, score: float) -> bool:
        flagged = bool(score > self.threshold())
        self.pending[run_id] = (float(score), flagged)
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)
        return flagged

    def feedback(self, run_id: str, healthy: bool) -> bool:
        """Apply a (possibly late) operator confirmation. Returns False if the run id is unknown or already confirmed."""
        item = self.pending.pop(run_id, None)
        if item is None:
            return False
        score, flagged = item
        if healthy:
            self.buf.admit(score)
            self.aci.feedback(flagged_normal=flagged)
            self.n_confirmed_healthy += 1
        else:
            self.n_confirmed_faulty += 1
        return True

    def state(self) -> dict:
        return {"alpha": self.alpha, "alpha_t": self.alpha_t, "threshold": self.threshold(), "buffer_size": len(self.buf.buf),
                "confirmed_healthy": self.n_confirmed_healthy, "confirmed_faulty": self.n_confirmed_faulty,
                "awaiting_confirmation": len(self.pending), "gamma": self.gamma}
