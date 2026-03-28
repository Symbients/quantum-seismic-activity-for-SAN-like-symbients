"""Simple Voice Activity Detection (VAD) using energy + zero-crossing rate.

No ML or external dependencies — uses two signal-level features that
reliably distinguish speech from ambient noise:

1. Short-term energy (STE): speech has higher energy than background
2. Zero-crossing rate (ZCR): speech (especially voiced) has lower ZCR
   than white/pink noise, but higher than silence

Decision logic:
- High energy + moderate ZCR → speech
- High energy + very high ZCR → noise (not speech)
- Low energy → no speech regardless of ZCR
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class VoiceActivityDetector:
    """Streaming VAD that tracks speech presence over time."""

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_ms: float = 20.0,
        energy_threshold: float = 0.005,
        zcr_low: float = 0.02,
        zcr_high: float = 0.25,
        hangover_frames: int = 15,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.energy_threshold = energy_threshold
        self.zcr_low = zcr_low  # below this = silence
        self.zcr_high = zcr_high  # above this = noise, not speech
        self.hangover_frames = hangover_frames  # frames to hold speech state after drop

        self._lock = threading.Lock()
        self._speech_now = False
        self._hangover_counter = 0

        # Track speech ratio over last ~10 seconds
        self._decisions: deque[bool] = deque(maxlen=int(10_000 / frame_ms))
        self._buffer: list[float] = []

    def process(self, samples: np.ndarray) -> None:
        """Process a chunk of audio samples."""
        self._buffer.extend(samples.tolist())

        while len(self._buffer) >= self.frame_size:
            frame = np.array(self._buffer[: self.frame_size])
            self._buffer = self._buffer[self.frame_size :]
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> None:
        # Short-term energy (RMS)
        energy = float(np.sqrt(np.mean(frame**2)))

        # Zero-crossing rate (normalized to 0-1)
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        zcr = crossings / len(frame)

        # Decision
        is_speech = (
            energy > self.energy_threshold
            and zcr > self.zcr_low
            and zcr < self.zcr_high
        )

        with self._lock:
            if is_speech:
                self._speech_now = True
                self._hangover_counter = self.hangover_frames
            else:
                if self._hangover_counter > 0:
                    self._hangover_counter -= 1
                    # Keep speech state during hangover (bridges short pauses)
                else:
                    self._speech_now = False

            self._decisions.append(self._speech_now)

    @property
    def speech_detected(self) -> bool:
        """Is speech currently detected?"""
        with self._lock:
            return self._speech_now

    @property
    def speech_ratio(self) -> float:
        """Fraction of recent frames classified as speech (0.0-1.0)."""
        with self._lock:
            if not self._decisions:
                return 0.0
            return sum(self._decisions) / len(self._decisions)

    def reset(self) -> None:
        with self._lock:
            self._speech_now = False
            self._hangover_counter = 0
            self._decisions.clear()
            self._buffer.clear()
