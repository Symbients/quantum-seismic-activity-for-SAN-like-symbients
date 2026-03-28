"""Location tracking via CoreLocation (macOS).

Uses pyobjc CoreLocation bindings to get GPS/WiFi positioning.
Detects movement between locations and maintains a visit log.
Falls back gracefully if CoreLocation is unavailable or permission denied.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LocationReading:
    latitude: float
    longitude: float
    altitude_m: float
    accuracy_m: float
    timestamp: float  # monotonic


@dataclass
class LocationVisit:
    """A stay at a location."""

    latitude: float
    longitude: float
    label: str  # reverse-geocoded or user-defined
    arrived: str  # ISO timestamp
    departed: str = ""


# Minimum movement (meters) to count as a location change
MOVEMENT_THRESHOLD_M = 100.0


class LocationSensor:
    """Tracks device location via CoreLocation with movement detection."""

    def __init__(self, update_interval_s: float = 60.0):
        self.update_interval_s = update_interval_s
        self._current: LocationReading | None = None
        self._history: list[LocationReading] = []
        self._visits_today: list[LocationVisit] = []
        self._current_visit: LocationVisit | None = None
        self._stationary_since: float = 0.0
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

        self._stationary_since = time.monotonic()
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
                    self._process_reading(reading)

            self._stop_event.wait(self.update_interval_s)

    def _process_reading(self, reading: LocationReading) -> None:
        """Process a new reading — detect movement, update visits."""
        prev = self._current
        self._current = reading
        self._history.append(reading)
        if len(self._history) > 1440:
            self._history = self._history[-1440:]

        if prev is None:
            # First reading — start a visit
            self._start_visit(reading)
            self._stationary_since = reading.timestamp
            return

        distance = _haversine_m(
            prev.latitude, prev.longitude,
            reading.latitude, reading.longitude,
        )

        if distance > MOVEMENT_THRESHOLD_M:
            # Moved to a new location
            self._end_visit()
            self._start_visit(reading)
            self._stationary_since = reading.timestamp

    def _start_visit(self, reading: LocationReading) -> None:
        now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        self._current_visit = LocationVisit(
            latitude=round(reading.latitude, 5),
            longitude=round(reading.longitude, 5),
            label=f"{reading.latitude:.3f},{reading.longitude:.3f}",
            arrived=now_iso,
        )

    def _end_visit(self) -> None:
        if self._current_visit:
            now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
            self._current_visit.departed = now_iso
            self._visits_today.append(self._current_visit)
            # Keep last 50 visits
            if len(self._visits_today) > 50:
                self._visits_today = self._visits_today[-50:]
            self._current_visit = None

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
    def is_stationary(self) -> bool:
        with self._lock:
            if self._current is None:
                return True
            return (time.monotonic() - self._stationary_since) > 120  # 2min = stationary

    @property
    def stationary_since_iso(self) -> str:
        with self._lock:
            if self._current_visit:
                return self._current_visit.arrived
            return ""

    @property
    def visits_today(self) -> list[LocationVisit]:
        with self._lock:
            visits = list(self._visits_today)
            if self._current_visit:
                visits.append(self._current_visit)
            return visits

    @property
    def locations_today_labels(self) -> list[str]:
        """Unique location labels visited today."""
        seen = []
        for v in self.visits_today:
            if v.label not in seen:
                seen.append(v.label)
        return seen

    @property
    def available(self) -> bool:
        return self._available


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    R = 6_371_000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))
