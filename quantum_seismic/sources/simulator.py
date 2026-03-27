"""Synthetic seismic event simulator for testing and demos."""

import threading
import time

import numpy as np

from quantum_seismic.sources.base import SensorChunk, SensorConfig, SensorSource


class SimulatorSource(SensorSource):
    """Generates synthetic seismic data with realistic event signatures.

    Event types and their signal characteristics:
    - knock:     Sharp transient, high kurtosis, very short duration
    - footstep:  Rhythmic low-frequency pulses, moderate amplitude
    - typing:    Continuous low-amplitude high-frequency noise
    - door_slam: Medium transient with reverb tail
    - earthquake: Sustained broadband with P-wave onset + S-wave buildup
    - ambient:   Background noise floor (always present)
    """

    # Event generation parameters
    EVENT_PARAMS = {
        "knock": {
            "amplitude": 0.7,
            "duration_s": 0.01,
            "freq_hz": 800,
            "decay": 200,
            "probability": 0.02,
        },
        "footstep": {
            "amplitude": 0.15,
            "duration_s": 0.05,
            "freq_hz": 40,
            "decay": 30,
            "probability": 0.01,
        },
        "typing": {
            "amplitude": 0.03,
            "duration_s": 0.005,
            "freq_hz": 1200,
            "decay": 400,
            "probability": 0.15,
        },
        "door_slam": {
            "amplitude": 0.4,
            "duration_s": 0.15,
            "freq_hz": 120,
            "decay": 15,
            "probability": 0.005,
        },
        "earthquake": {
            "amplitude": 0.3,
            "duration_s": 3.0,
            "freq_hz": 5,
            "decay": 0.5,
            "probability": 0.001,
        },
    }

    def __init__(
        self,
        config: SensorConfig | None = None,
        noise_floor: float = 0.001,
        seed: int | None = None,
    ):
        super().__init__(config)
        self._noise_floor = noise_floor
        self._rng = np.random.default_rng(seed)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._active_events: list[dict] = []

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._running = False

    def _generate_loop(self) -> None:
        chunk_duration = self.config.chunk_size / self.config.sample_rate
        start_time = time.monotonic()
        sample_offset = 0

        while not self._stop_event.is_set():
            t0 = sample_offset / self.config.sample_rate
            samples = self._generate_chunk(t0)

            chunk = SensorChunk(
                data=samples,
                timestamp=time.monotonic() - start_time,
                sample_rate=self.config.sample_rate,
                source_name="simulator",
                metadata={"events_active": len(self._active_events)},
            )
            self._emit(chunk)

            sample_offset += self.config.chunk_size

            # Sleep to match real-time rate
            elapsed = time.monotonic() - start_time
            expected = sample_offset / self.config.sample_rate
            if expected > elapsed:
                time.sleep(expected - elapsed)

    def _generate_chunk(self, t0: float) -> np.ndarray:
        n = self.config.chunk_size
        sr = self.config.sample_rate
        t = t0 + np.arange(n) / sr

        # Base: ambient noise
        signal = self._rng.normal(0, self._noise_floor, n)

        # Spawn new events probabilistically
        for event_type, params in self.EVENT_PARAMS.items():
            if self._rng.random() < params["probability"]:
                self._active_events.append(
                    {
                        "type": event_type,
                        "start_time": t0 + self._rng.random() * (n / sr),
                        **params,
                    }
                )

        # Render active events
        still_active = []
        for event in self._active_events:
            dt = t - event["start_time"]
            mask = dt >= 0
            if not np.any(mask):
                still_active.append(event)
                continue

            age = np.max(dt[mask])
            if age > event["duration_s"] * 3:  # allow 3x duration for decay tail
                continue  # event expired

            still_active.append(event)

            # Damped sinusoid model
            env = np.where(mask, np.exp(-event["decay"] * np.maximum(dt, 0)), 0)
            carrier = np.sin(2 * np.pi * event["freq_hz"] * dt)
            signal += event["amplitude"] * env * carrier

        self._active_events = still_active
        return np.clip(signal, -1.0, 1.0)

    def inject_event(self, event_type: str, delay_s: float = 0.0) -> None:
        """Manually inject an event (for testing)."""
        if event_type not in self.EVENT_PARAMS:
            raise ValueError(f"Unknown event type: {event_type}. Use: {list(self.EVENT_PARAMS)}")
        params = self.EVENT_PARAMS[event_type]
        self._active_events.append(
            {
                "type": event_type,
                "start_time": time.monotonic() + delay_s,
                **params,
            }
        )
