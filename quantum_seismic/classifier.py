"""Rule-based event classifier using detector fingerprints.

Each event type has a distinct signature across the 5 detectors.
No ML needed — the physics determines the fingerprint.

Fingerprint matrix (what each event looks like to each detector):

| Event      | STA/LTA  | Kurtosis   | Peak/MAD | Duration | CUSUM    |
|------------|----------|------------|----------|----------|----------|
| Knock      | High >5  | Very high  | High >8  | <50ms    | Sharp    |
| Typing     | Low ~1.5 | Medium 3-8 | Low <4   | Ongoing  | Flat     |
| Footstep   | Medium   | Low-med    | Medium   | ~100ms   | Moderate |
| Door slam  | Med-high | High       | Medium   | 100-300ms| Sharp    |
| Earthquake | Sustained| Low <3     | Ramps up | Seconds+ | Ramp     |
"""

from dataclasses import dataclass, field
from enum import Enum

from quantum_seismic.detectors.base import DetectorOutput


class EventType(Enum):
    KNOCK = "knock"
    TYPING = "typing"
    FOOTSTEP = "footstep"
    DOOR_SLAM = "door_slam"
    EARTHQUAKE = "earthquake"
    UNKNOWN = "unknown"
    AMBIENT = "ambient"


@dataclass
class ClassifiedEvent:
    """A classified seismic event."""

    event_type: EventType
    confidence: float  # 0.0 - 1.0
    timestamp: float  # seconds since start
    duration_s: float
    severity: str  # from Peak/MAD: normal, mild, significant, major
    fingerprint: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.timestamp:8.2f}s] {self.event_type.value:12s} "
            f"confidence={self.confidence:.0%} severity={self.severity}"
        )


class EventClassifier:
    """Classifies events based on detector output fingerprints."""

    # Minimum time between events of the same type (debounce)
    MIN_EVENT_GAP_S = 0.1

    def __init__(self):
        self._last_event_time: dict[EventType, float] = {}
        self._event_history: list[ClassifiedEvent] = []
        # Track sustained triggers for earthquake detection
        self._sustained_trigger_count: int = 0
        self._sustained_trigger_threshold: int = 10  # chunks

    def classify(
        self,
        outputs: dict[str, DetectorOutput],
        timestamp: float,
        chunk_duration: float,
    ) -> ClassifiedEvent | None:
        """Classify the current chunk based on all detector outputs.

        Returns a ClassifiedEvent if an event is detected, None otherwise.
        """
        sta_lta = outputs.get("sta_lta")
        kurtosis = outputs.get("kurtosis")
        peak_mad = outputs.get("peak_mad")
        cusum = outputs.get("cusum")
        highpass = outputs.get("highpass")

        # Check if any detector triggered
        any_triggered = any(o.triggered for o in outputs.values())

        if not any_triggered:
            self._sustained_trigger_count = 0
            return None

        # Track sustained triggers
        self._sustained_trigger_count += 1

        # Extract fingerprint values
        fp = {
            "sta_lta_ratio": sta_lta.value if sta_lta else 0,
            "kurtosis": kurtosis.value if kurtosis else 0,
            "peak_mad_score": peak_mad.value if peak_mad else 0,
            "cusum": cusum.value if cusum else 0,
            "highpass_rms": highpass.value if highpass else 0,
            "sustained_chunks": self._sustained_trigger_count,
        }

        severity = "normal"
        if peak_mad:
            severity = peak_mad.metadata.get("severity", "normal")

        # Classification rules (ordered by specificity)
        event_type, confidence = self._match_fingerprint(fp)

        # Suppress low-confidence unknowns (noise reduction)
        if event_type == EventType.UNKNOWN and confidence < 0.5:
            return None

        # Debounce
        last = self._last_event_time.get(event_type, -999)
        if timestamp - last < self.MIN_EVENT_GAP_S:
            return None

        event = ClassifiedEvent(
            event_type=event_type,
            confidence=confidence,
            timestamp=timestamp,
            duration_s=chunk_duration,
            severity=severity,
            fingerprint=fp,
        )

        self._last_event_time[event_type] = timestamp
        self._event_history.append(event)

        return event

    def _match_fingerprint(self, fp: dict) -> tuple[EventType, float]:
        """Match fingerprint to event type using rule-based logic."""

        sta = fp["sta_lta_ratio"]
        kurt = fp["kurtosis"]
        pmad = fp["peak_mad_score"]
        sustained = fp["sustained_chunks"]

        # Earthquake: sustained STA/LTA elevation + low kurtosis (broadband)
        if sustained > self._sustained_trigger_threshold and kurt < 5:
            confidence = min(0.9, 0.5 + sustained * 0.02)
            return EventType.EARTHQUAKE, confidence

        # Knock: very high kurtosis + high STA/LTA + high peak/MAD
        if kurt > 10 and sta > 4 and pmad > 6:
            return EventType.KNOCK, 0.85

        # Door slam: high kurtosis but not as extreme, medium STA/LTA
        if kurt > 6 and sta > 2.5 and pmad > 4:
            return EventType.DOOR_SLAM, 0.7

        # Footstep: moderate across the board, STA/LTA in medium range
        if 1.5 < sta < 4 and kurt < 8 and 2 < pmad < 7:
            return EventType.FOOTSTEP, 0.6

        # Typing: low STA/LTA, modest kurtosis, low severity
        if sta < 2.5 and kurt < 8 and pmad < 5:
            return EventType.TYPING, 0.5

        # Knock (relaxed): very impulsive regardless of other metrics
        if kurt > 15:
            return EventType.KNOCK, 0.65

        return EventType.UNKNOWN, 0.3

    @property
    def history(self) -> list[ClassifiedEvent]:
        return list(self._event_history)

    def reset(self) -> None:
        self._last_event_time.clear()
        self._event_history.clear()
        self._sustained_trigger_count = 0
