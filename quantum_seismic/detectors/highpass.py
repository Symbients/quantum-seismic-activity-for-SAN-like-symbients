"""High-Pass Filter — strips low-frequency content (gravity analog / DC offset) to isolate impacts.

For accelerometers, this removes gravity. For microphones, this removes
low-frequency ambient rumble (HVAC, traffic) so we see only transient events.

Uses a 4th-order Butterworth filter for flat passband response.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from quantum_seismic.detectors.base import DetectorOutput


class HighPassFilter:
    """Butterworth high-pass filter as a streaming detector stage."""

    name = "highpass"

    def __init__(self, cutoff_hz: float = 20.0, order: int = 4):
        self.cutoff_hz = cutoff_hz
        self.order = order
        self._sos: np.ndarray | None = None
        self._zi: np.ndarray | None = None
        self._prev_sample_rate: int | None = None

    def _init_filter(self, sample_rate: int) -> None:
        """(Re)initialize filter coefficients when sample rate changes."""
        nyquist = sample_rate / 2.0
        normalized_cutoff = self.cutoff_hz / nyquist
        self._sos = butter(self.order, normalized_cutoff, btype="high", output="sos")
        # Initialize filter state for seamless streaming
        from scipy.signal import sosfilt_zi

        self._zi = sosfilt_zi(self._sos) * 0.0  # start from zero
        self._prev_sample_rate = sample_rate

    def process(self, samples: np.ndarray, sample_rate: int) -> DetectorOutput:
        if self._prev_sample_rate != sample_rate or self._sos is None:
            self._init_filter(sample_rate)

        filtered, self._zi = sosfilt(self._sos, samples, zi=self._zi)

        # Trigger if filtered energy is significantly above noise floor
        rms = float(np.sqrt(np.mean(filtered**2)))

        return DetectorOutput(
            name=self.name,
            value=rms,
            triggered=rms > 0.01,  # basic threshold, tunable
            signal=filtered,
            metadata={"cutoff_hz": self.cutoff_hz, "rms": rms},
        )

    def reset(self) -> None:
        self._zi = None
        self._prev_sample_rate = None
