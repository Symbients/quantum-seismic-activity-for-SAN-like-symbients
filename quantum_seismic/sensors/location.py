"""Location tracking via CoreLocation (macOS).

Uses pyobjc CoreLocation bindings to get GPS/WiFi positioning.
Falls back gracefully if CoreLocation is unavailable or permission denied.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class LocationReading:
    latitude: float
    longitude: float
    altitude_m: float
    accuracy_m: float
    timestamp: float  # monotonic


class LocationSensor:
    """Tracks device location via CoreLocation."""

    def __init__(self, update_interval_s: float = 60.0):
        self.update_interval_s = update_interval_s
        self._current: LocationReading | None = None
        self._history: list[LocationReading] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._available = False

    def start(self) -> None:
        try:
            import CoreLocation  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        import CoreLocation
        import objc  # noqa: F401

        manager = CoreLocation.CLLocationManager.alloc().init()

        while not self._stop_event.is_set():
            location = manager.location()
            if location is not None:
                coord = location.coordinate()
                reading = LocationReading(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    altitude_m=location.altitude(),
                    accuracy_m=location.horizontalAccuracy(),
                    timestamp=time.monotonic(),
                )
                with self._lock:
                    self._current = reading
                    self._history.append(reading)
                    # Keep last 1440 readings (~24h at 1/min)
                    if len(self._history) > 1440:
                        self._history = self._history[-1440:]

            self._stop_event.wait(self.update_interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def current(self) -> LocationReading | None:
        with self._lock:
            return self._current

    @property
    def history(self) -> list[LocationReading]:
        with self._lock:
            return list(self._history)

    @property
    def available(self) -> bool:
        return self._available
