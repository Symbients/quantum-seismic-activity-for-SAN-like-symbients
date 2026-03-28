"""SQLite persistence for temporal data across daemon restarts.

Stores rolling aggregates so the daemon doesn't lose 24hr context
when it restarts. Also persists location visit history.

Database: ~/.quantum-seismic/state.db
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


DEFAULT_DB_PATH = Path.home() / ".quantum-seismic" / "state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor TEXT NOT NULL,        -- 'accel' or 'mic'
    rms REAL NOT NULL,
    peak REAL NOT NULL,
    timestamp REAL NOT NULL      -- unix epoch
);

CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    label TEXT NOT NULL,
    arrived TEXT NOT NULL,       -- ISO timestamp
    departed TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_samples_sensor_ts ON samples(sensor, timestamp);
CREATE INDEX IF NOT EXISTS idx_visits_arrived ON visits(arrived);
"""


class StateStore:
    """SQLite-backed persistence for daemon state."""

    # How often to flush to disk (seconds)
    FLUSH_INTERVAL = 10.0
    # Retention: keep 48 hours of samples
    RETENTION_S = 48 * 3600

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._pending_samples: list[tuple[str, float, float, float]] = []
        self._last_flush: float = 0.0

    def open(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._flush()
            self._conn.close()
            self._conn = None

    def record_sample(self, sensor: str, rms: float, peak: float) -> None:
        """Buffer a sample for batch insert."""
        now = time.time()
        with self._lock:
            self._pending_samples.append((sensor, rms, peak, now))
            if now - self._last_flush > self.FLUSH_INTERVAL:
                self._flush_locked()

    def _flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._conn or not self._pending_samples:
            return
        self._conn.executemany(
            "INSERT INTO samples (sensor, rms, peak, timestamp) VALUES (?, ?, ?, ?)",
            self._pending_samples,
        )
        self._conn.commit()
        self._pending_samples.clear()
        self._last_flush = time.time()

    def load_window(self, sensor: str, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Load RMS and peak values from the last N seconds.

        Returns (rms_array, peak_array).
        """
        if not self._conn:
            return np.array([]), np.array([])

        cutoff = time.time() - seconds
        with self._lock:
            self._flush_locked()
            cursor = self._conn.execute(
                "SELECT rms, peak FROM samples WHERE sensor = ? AND timestamp > ? ORDER BY timestamp",
                (sensor, cutoff),
            )
            rows = cursor.fetchall()

        if not rows:
            return np.array([]), np.array([])

        data = np.array(rows)
        return data[:, 0], data[:, 1]

    def prune(self) -> int:
        """Delete samples older than retention period. Returns count deleted."""
        if not self._conn:
            return 0
        cutoff = time.time() - self.RETENTION_S
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM samples WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cursor.rowcount

    def record_visit(self, latitude: float, longitude: float, label: str, arrived: str) -> int:
        """Record a location visit. Returns the row ID."""
        if not self._conn:
            return -1
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO visits (latitude, longitude, label, arrived) VALUES (?, ?, ?, ?)",
                (latitude, longitude, label, arrived),
            )
            self._conn.commit()
            return cursor.lastrowid

    def end_visit(self, visit_id: int, departed: str) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE visits SET departed = ? WHERE id = ?", (departed, visit_id)
            )
            self._conn.commit()

    def load_visits_today(self) -> list[dict]:
        """Load today's visits."""
        if not self._conn:
            return []
        # Approximate: last 24 hours
        cutoff = time.time() - 86400
        from datetime import datetime, timezone

        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "SELECT latitude, longitude, label, arrived, departed "
                "FROM visits WHERE arrived > ? ORDER BY arrived",
                (cutoff_iso,),
            )
            return [
                {
                    "latitude": r[0],
                    "longitude": r[1],
                    "label": r[2],
                    "arrived": r[3],
                    "departed": r[4],
                }
                for r in cursor.fetchall()
            ]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
