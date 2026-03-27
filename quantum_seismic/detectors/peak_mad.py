"""Peak/MAD (Median Absolute Deviation) outlier detector.

MAD is a robust measure of variability — unlike standard deviation, it's
not sensitive to the outliers it's trying to detect. The Peak/MAD ratio
measures how many MADs the peak sample is from the median.

Score interpretation:
- < 3:  Normal variation
- 3-5:  Mild anomaly (light touch, ambient change)
- 5-10: Significant event (knock, footstep)
- > 10: Major event (loud impact, earthquake)

This is the "how abnormal was this?" detector — it gives a severity score
independent of what type of event occurred.
"""

import numpy as np

from quantum_seismic.detectors.base import DetectorOutput


class PeakMADDetector:
    """Peak-to-MAD ratio outlier detection."""

    name = "peak_mad"

    def __init__(self, window_s: float = 0.5, threshold: float = 5.0):
        self.window_s = window_s
        self.threshold = threshold
        self._buffer: np.ndarray | None = None

    def process(self, samples: np.ndarray, sample_rate: int) -> DetectorOutput:
        window_size = max(10, int(self.window_s * sample_rate))

        # Maintain rolling buffer
        if self._buffer is not None:
            combined = np.concatenate([self._buffer, samples])
        else:
            combined = samples

        # Keep only what we need
        if len(combined) > window_size:
            combined = combined[-window_size:]
        self._buffer = combined.copy()

        # Compute MAD over the full window
        median = np.median(combined)
        mad = np.median(np.abs(combined - median))

        # Avoid division by zero — use a small floor
        mad = max(mad, 1e-10)

        # Compute per-sample Peak/MAD score for the current chunk
        signal = np.abs(samples - median) / mad

        peak_score = float(np.max(signal))
        mean_score = float(np.mean(signal))
        triggered = peak_score > self.threshold

        return DetectorOutput(
            name=self.name,
            value=peak_score,
            triggered=triggered,
            signal=signal,
            metadata={
                "median": float(median),
                "mad": float(mad),
                "peak_score": peak_score,
                "mean_score": mean_score,
                "severity": _severity_label(peak_score),
            },
        )

    def reset(self) -> None:
        self._buffer = None


def _severity_label(score: float) -> str:
    if score < 3:
        return "normal"
    elif score < 5:
        return "mild"
    elif score < 10:
        return "significant"
    else:
        return "major"
