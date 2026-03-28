"""Microphone as a vibration sensor — captures surface vibrations through the MacBook chassis."""

import time

import numpy as np
import sounddevice as sd

from quantum_seismic.sensors.base import SensorChunk, SensorConfig, SensorSource


class MicrophoneSource(SensorSource):
    """Reads the built-in microphone as a vibration/seismic sensor.

    The MacBook's microphone picks up mechanical vibrations transmitted through
    the chassis and desk surface. At 44.1kHz, it provides 440x the temporal
    resolution of a typical 100Hz accelerometer.
    """

    def __init__(self, config: SensorConfig | None = None):
        super().__init__(config)
        self._stream: sd.InputStream | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size,
            channels=self.config.channels,
            device=self.config.device,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            pass  # drop status messages silently in production

        # Flatten to 1D, convert to float64 for processing precision
        samples = indata[:, 0].astype(np.float64)

        chunk = SensorChunk(
            data=samples,
            timestamp=time.monotonic() - self._start_time,
            sample_rate=self.config.sample_rate,
            source_name="microphone",
            metadata={"device": str(self.config.device or "default")},
        )
        self._emit(chunk)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
