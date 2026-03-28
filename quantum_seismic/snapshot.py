"""Structured environment snapshot — the output format for agent context."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class SeismicState:
    """Accelerometer-derived physical vibration state."""

    rms_1min: float = 0.0  # root-mean-square acceleration (g)
    rms_1hr: float = 0.0
    rms_24hr: float = 0.0
    peak_1hr: float = 0.0  # max acceleration in window
    regime: str = "unknown"  # quiet / mild / active / intense
    events_1hr: int = 0  # significant vibration events


@dataclass
class AcousticState:
    """Microphone-derived ambient sound state."""

    db_1min: float = 0.0  # approximate dB SPL
    db_1hr: float = 0.0
    db_24hr: float = 0.0
    regime: str = "unknown"  # silent / quiet / conversation / busy / loud
    speech_detected: bool = False
    noise_trend: str = "stable"  # rising / falling / stable


@dataclass
class LocationState:
    """GPS/WiFi location state."""

    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    accuracy_m: float | None = None
    label: str = "unknown"  # user-assigned or reverse-geocoded
    since: str = ""  # ISO timestamp of arrival at current location
    stationary: bool = True
    locations_today: list[str] = field(default_factory=list)


@dataclass
class VisualState:
    """Webcam-derived visual context."""

    summary: str = ""  # natural language description from vision model
    occupants: int | None = None
    last_capture: str = ""  # ISO timestamp


@dataclass
class ExternalSeismicState:
    """Real seismic data from USGS/JMA APIs."""

    nearest_event: str = ""  # e.g. "M2.1 Chiba 45km 3h ago"
    warnings: list[str] = field(default_factory=list)
    last_fetch: str = ""


@dataclass
class EnvironmentSnapshot:
    """Complete environment snapshot for agent context injection."""

    timestamp: str = ""
    uptime_s: float = 0.0
    seismic: SeismicState = field(default_factory=SeismicState)
    acoustic: AcousticState = field(default_factory=AcousticState)
    location: LocationState = field(default_factory=LocationState)
    visual: VisualState = field(default_factory=VisualState)
    external_seismic: ExternalSeismicState = field(default_factory=ExternalSeismicState)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_context_block(self) -> str:
        """Format as an XML-tagged block for system prompt injection."""
        return f"<environment>\n{self.to_json(indent=2)}\n</environment>"

    @staticmethod
    def now_iso() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")
