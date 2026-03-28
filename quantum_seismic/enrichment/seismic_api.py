"""External seismic data from USGS and JMA APIs."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from math import acos, cos, radians, sin


@dataclass
class SeismicEvent:
    magnitude: float
    place: str
    distance_km: float
    time_ago_s: float
    url: str = ""


class SeismicAPI:
    """Fetches recent earthquake data from USGS Earthquake API.

    Periodically polls the USGS GeoJSON feed for nearby events.
    Falls back gracefully on network errors.
    """

    USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"

    def __init__(
        self,
        poll_interval_s: float = 300.0,  # every 5 minutes
        max_distance_km: float = 500.0,
    ):
        self.poll_interval_s = poll_interval_s
        self.max_distance_km = max_distance_km
        self._events: list[SeismicEvent] = []
        self._warnings: list[str] = []
        self._last_fetch: str = ""
        self._lat: float | None = None
        self._lon: float | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def set_location(self, lat: float, lon: float) -> None:
        with self._lock:
            self._lat = lat
            self._lon = lon

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._fetch()
            except Exception:
                pass
            self._stop_event.wait(self.poll_interval_s)

    def _fetch(self) -> None:
        req = urllib.request.Request(
            self.USGS_URL,
            headers={"User-Agent": "quantum-seismic/0.2"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        now_ms = time.time() * 1000
        events = []

        with self._lock:
            lat, lon = self._lat, self._lon

        for feature in data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            eq_lon, eq_lat = coords[0], coords[1]
            mag = props.get("mag", 0)
            place = props.get("place", "unknown")
            event_time_ms = props.get("time", now_ms)
            url = props.get("url", "")

            dist = _haversine(lat, lon, eq_lat, eq_lon) if lat is not None else 0
            time_ago = (now_ms - event_time_ms) / 1000

            if lat is None or dist <= self.max_distance_km:
                events.append(
                    SeismicEvent(
                        magnitude=mag,
                        place=place,
                        distance_km=round(dist, 1),
                        time_ago_s=time_ago,
                        url=url,
                    )
                )

        # Sort by distance
        events.sort(key=lambda e: e.distance_km)

        warnings = []
        for e in events:
            if e.magnitude >= 4.0 and e.distance_km < 200:
                warnings.append(f"M{e.magnitude} {e.place} ({e.distance_km}km away)")

        with self._lock:
            self._events = events[:10]  # keep nearest 10
            self._warnings = warnings
            from quantum_seismic.snapshot import EnvironmentSnapshot

            self._last_fetch = EnvironmentSnapshot.now_iso()

    @property
    def nearest_event(self) -> str:
        with self._lock:
            if not self._events:
                return "no recent events"
            e = self._events[0]
            hours = e.time_ago_s / 3600
            return f"M{e.magnitude} {e.place} {e.distance_km}km {hours:.0f}h ago"

    @property
    def warnings(self) -> list[str]:
        with self._lock:
            return list(self._warnings)

    @property
    def last_fetch(self) -> str:
        with self._lock:
            return self._last_fetch

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)


def _haversine(lat1: float | None, lon1: float | None, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    if lat1 is None or lon1 is None:
        return 0.0
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * acos(min(1, (1 - 2 * a) ** 0.5 if a < 0.5 else 0))
