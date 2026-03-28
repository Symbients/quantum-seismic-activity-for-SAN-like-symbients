"""Temporal aggregation — rolling statistics over multiple time windows."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowStats:
    """Statistics for a single time window."""

    rms: float = 0.0
    peak: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    count: int = 0


class RollingAggregator:
    """Maintains rolling statistics over 1min, 1hr, and 24hr windows.

    Receives raw amplitude values and computes RMS, peak, mean, std
    for each window. Thread-safe.
    """

    def __init__(self, sample_rate: int = 100):
        """sample_rate: expected samples per second for sizing buffers."""
        self._lock = threading.Lock()

        # Store downsampled values (1 value per ~100ms block)
        # 1min = 600 values, 1hr = 36000, 24hr = 864000
        self._buf_1min: deque[float] = deque(maxlen=600)
        self._buf_1hr: deque[float] = deque(maxlen=36_000)
        self._buf_24hr: deque[float] = deque(maxlen=864_000)

        # Accumulator for downsampling incoming high-rate data
        self._block: list[float] = []
        self._block_interval_s: float = 0.1  # downsample to 10Hz
        self._last_block_time: float = time.monotonic()

    def push(self, values: np.ndarray) -> None:
        """Push a chunk of raw amplitude values."""
        # Compute block-level RMS for downsampling
        rms = float(np.sqrt(np.mean(values**2)))

        now = time.monotonic()
        with self._lock:
            self._block.append(rms)

            if now - self._last_block_time >= self._block_interval_s:
                if self._block:
                    block_rms = float(np.sqrt(np.mean(np.array(self._block) ** 2)))
                    self._buf_1min.append(block_rms)
                    self._buf_1hr.append(block_rms)
                    self._buf_24hr.append(block_rms)
                    self._block.clear()
                self._last_block_time = now

    def stats_1min(self) -> WindowStats:
        return self._compute(self._buf_1min)

    def stats_1hr(self) -> WindowStats:
        return self._compute(self._buf_1hr)

    def stats_24hr(self) -> WindowStats:
        return self._compute(self._buf_24hr)

    def _compute(self, buf: deque[float]) -> WindowStats:
        with self._lock:
            if not buf:
                return WindowStats()
            arr = np.array(buf)
        return WindowStats(
            rms=float(np.sqrt(np.mean(arr**2))),
            peak=float(np.max(np.abs(arr))),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            count=len(arr),
        )


def classify_seismic_regime(rms_1min: float) -> str:
    """Classify vibration regime from RMS acceleration (g)."""
    if rms_1min < 0.002:
        return "quiet"
    elif rms_1min < 0.01:
        return "mild"
    elif rms_1min < 0.05:
        return "active"
    else:
        return "intense"


def classify_acoustic_regime(db_1min: float) -> str:
    """Classify acoustic regime from approximate dB SPL."""
    if db_1min < 30:
        return "silent"
    elif db_1min < 45:
        return "quiet"
    elif db_1min < 60:
        return "conversation"
    elif db_1min < 75:
        return "busy"
    else:
        return "loud"


def rms_to_db(rms: float, reference: float = 1.0) -> float:
    """Convert RMS amplitude to approximate dB.

    For microphone input normalized to [-1, 1], this gives a relative
    dB scale. Not calibrated to true SPL without hardware reference.
    """
    if rms < 1e-10:
        return 0.0
    return 20 * np.log10(rms / reference) + 94  # offset to approximate SPL
