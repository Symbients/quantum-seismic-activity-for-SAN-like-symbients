"""Kurtosis detector — 4th statistical moment for spike detection.

Kurtosis measures the "tailedness" of a distribution:
- Gaussian noise: kurtosis ~ 3 (excess kurtosis ~ 0)
- Sharp spikes (knocks): kurtosis >> 3
- Flat/uniform signal: kurtosis < 3

High kurtosis = impulsive events (knocks, cracks, impacts)
Low kurtosis = sustained/broadband events (earthquake, vibration)

This distinction is what allows classification: STA/LTA says "event happened",
kurtosis says "was it sharp or sustained?"
"""

import numpy as np

from quantum_seismic.detectors.base import DetectorOutput


class KurtosisDetector:
    """Rolling kurtosis computed over a sliding window."""

    name = "kurtosis"

    def __init__(self, window_s: float = 0.05, threshold: float = 6.0):
        self.window_s = window_s
        self.threshold = threshold  # excess kurtosis threshold for "impulsive"
        self._buffer: np.ndarray | None = None

    def process(self, samples: np.ndarray, sample_rate: int) -> DetectorOutput:
        window_size = max(4, int(self.window_s * sample_rate))

        # Prepend previous tail for continuity
        if self._buffer is not None:
            extended = np.concatenate([self._buffer, samples])
        else:
            extended = samples

        # Store tail for next chunk
        self._buffer = samples[-window_size:] if len(samples) >= window_size else samples.copy()

        # Compute rolling kurtosis
        signal = np.zeros(len(samples))
        max_kurt = 0.0

        for i in range(len(samples)):
            # Window into extended array
            ext_i = i + (len(extended) - len(samples))
            start = max(0, ext_i - window_size + 1)
            window = extended[start : ext_i + 1]

            if len(window) < 4:
                continue

            # Compute excess kurtosis: E[(X-mu)^4] / E[(X-mu)^2]^2 - 3
            mu = np.mean(window)
            centered = window - mu
            var = np.mean(centered**2)

            if var < 1e-20:
                signal[i] = 0.0
                continue

            kurt = np.mean(centered**4) / (var**2) - 3.0
            signal[i] = kurt

            if abs(kurt) > max_kurt:
                max_kurt = abs(kurt)

        peak_kurtosis = float(np.max(np.abs(signal)))
        triggered = peak_kurtosis > self.threshold

        return DetectorOutput(
            name=self.name,
            value=peak_kurtosis,
            triggered=triggered,
            signal=signal,
            metadata={
                "window_s": self.window_s,
                "threshold": self.threshold,
                "peak_kurtosis": peak_kurtosis,
            },
        )

    def reset(self) -> None:
        self._buffer = None
