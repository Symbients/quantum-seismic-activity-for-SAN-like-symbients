"""CUSUM (Cumulative Sum) change-point detector.

Detects sudden shifts in the mean level of acceleration/vibration.
Maintains running upper and lower cumulative sums — when either exceeds
a threshold, a change-point (event onset) is declared.

This gives sub-sample precision on exactly *when* an event started,
complementing STA/LTA which tells you *that* something happened.
"""

import numpy as np

from quantum_seismic.detectors.base import DetectorOutput


class CUSUMDetector:
    """Bilateral CUSUM detector for abrupt mean shifts."""

    name = "cusum"

    def __init__(
        self,
        threshold: float = 0.05,
        drift: float = 0.002,
        decay: float = 0.999,
    ):
        self.threshold = threshold
        self.drift = drift  # allowance for normal variation
        self.decay = decay  # slow decay to prevent unbounded growth
        self._s_high: float = 0.0  # upper cumulative sum
        self._s_low: float = 0.0  # lower cumulative sum
        self._mean: float = 0.0  # running mean estimate
        self._n: int = 0

    def process(self, samples: np.ndarray, sample_rate: int) -> DetectorOutput:
        signal = np.zeros_like(samples)
        triggered = False
        max_cusum = 0.0

        for i, x in enumerate(samples):
            # Update running mean with slow EMA
            self._n += 1
            alpha = min(0.01, 2.0 / (self._n + 1))
            self._mean = self._mean * (1 - alpha) + x * alpha

            deviation = x - self._mean

            # Update bilateral CUSUM
            self._s_high = max(0, self._s_high + deviation - self.drift)
            self._s_low = max(0, self._s_low - deviation - self.drift)

            # Apply decay to prevent unbounded accumulation during quiet periods
            self._s_high *= self.decay
            self._s_low *= self.decay

            cusum_val = max(self._s_high, self._s_low)
            signal[i] = cusum_val

            if cusum_val > max_cusum:
                max_cusum = cusum_val

            if cusum_val > self.threshold:
                triggered = True

        return DetectorOutput(
            name=self.name,
            value=float(max_cusum),
            triggered=triggered,
            signal=signal,
            metadata={
                "s_high": float(self._s_high),
                "s_low": float(self._s_low),
                "running_mean": float(self._mean),
                "threshold": self.threshold,
            },
        )

    def reset(self) -> None:
        self._s_high = 0.0
        self._s_low = 0.0
        self._mean = 0.0
        self._n = 0
