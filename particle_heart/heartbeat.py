from __future__ import annotations

import math


class HeartbeatSystem:
    def __init__(self, bpm: float = 72.0) -> None:
        self._period = 60.0 / bpm
        self.beat = 0.0
        self.shockwave = 0.0
        self.bloom_intensity = 0.0

    def update(self, elapsed: float) -> None:
        phase = (elapsed % self._period) / self._period
        p1 = math.exp(-((phase - 0.12) ** 2) / 0.0035)
        p2 = math.exp(-((phase - 0.30) ** 2) / 0.0055) * 0.5
        p3 = math.exp(-((phase - 0.37) ** 2) / 0.008) * 0.25
        self.beat = 0.28 + 0.72 * (p1 + p2 + p3 * 0.3)
        self.shockwave = p1 * 0.85 + p3
        self.bloom_intensity = 0.30 + 0.18 * (p1 + p2 * 0.6 + p3 * 0.4)
