"""Real hardware accelerometer via IOKit HID (Apple Silicon Macs).

Uses the macimu library to read the Bosch BMI286 IMU through Apple's
undocumented SPU (Sensor Processing Unit) HID interface.

Requirements:
- Apple Silicon Mac with M2/M3/M4 Pro or higher (not base M1)
- Must run as root (sudo) for IOKit HID access to vendor-page devices
- macimu package: `uv pip install macimu`

The accelerometer reports at ~800Hz native, providing X/Y/Z acceleration
in g-force (1g = 9.81 m/s²). We accumulate samples into chunks matching
the configured chunk_size before emitting to the pipeline.
"""

import os
import threading
import time

import numpy as np

from quantum_seismic.sources.base import SensorChunk, SensorConfig, SensorSource


class AccelerometerSource(SensorSource):
    """Reads the hardware accelerometer via macimu's IOKit HID bindings.

    Combines X/Y/Z into a single magnitude signal for the detection pipeline,
    matching the same interface as MicrophoneSource and SimulatorSource.
    """

    # Native IMU sample rate (approximate — actual rate depends on hardware)
    NATIVE_RATE_HZ = 800

    def __init__(self, config: SensorConfig | None = None, use_mock: bool = False):
        super().__init__(config or SensorConfig(sample_rate=800, chunk_size=256))
        self._use_mock = use_mock
        self._imu = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        try:
            from macimu import IMU
        except ImportError:
            raise RuntimeError(
                "macimu is required for accelerometer access. "
                "Install it with: uv pip install macimu"
            )

        if self._use_mock:
            self._imu = IMU.mock()
        else:
            if os.geteuid() != 0:
                raise PermissionError(
                    "Accelerometer access requires root. Run with: sudo quantum-seismic --source accel"
                )
            self._imu = IMU()

        self._imu.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._running = True

    def _read_loop(self) -> None:
        """Accumulate IMU samples into chunks and emit them."""
        start_time = time.monotonic()
        buffer = []

        for sample in self._imu.stream_accel():
            if self._stop_event.is_set():
                break

            # sample is (x, y, z) in g-force
            x, y, z = sample.x, sample.y, sample.z

            # Compute magnitude: sqrt(x² + y² + z²)
            # This collapses 3-axis into a single signal for the pipeline.
            # At rest, magnitude ≈ 1.0g (gravity). The high-pass filter
            # in the pipeline strips this, leaving only dynamic events.
            magnitude = (x**2 + y**2 + z**2) ** 0.5
            buffer.append(magnitude)

            if len(buffer) >= self.config.chunk_size:
                samples = np.array(buffer, dtype=np.float64)
                buffer.clear()

                chunk = SensorChunk(
                    data=samples,
                    timestamp=time.monotonic() - start_time,
                    sample_rate=self.config.sample_rate,
                    source_name="accelerometer",
                    metadata={"last_xyz": (x, y, z)},
                )
                self._emit(chunk)

    def stop(self) -> None:
        self._stop_event.set()
        if self._imu is not None:
            self._imu.stop()
            self._imu = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._running = False

    @staticmethod
    def is_available() -> bool:
        """Check if accelerometer hardware is present (doesn't require root)."""
        try:
            from macimu import IMU

            return IMU.available()
        except (ImportError, Exception):
            return False
