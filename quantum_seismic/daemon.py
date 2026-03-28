"""EnvironmentDaemon — continuous ambient sensing with temporal aggregation.

Runs all sensors in background threads, maintains rolling statistics,
and produces EnvironmentSnapshot on demand for agent context injection.
"""

from __future__ import annotations

import time

from quantum_seismic.enrichment.seismic_api import SeismicAPI
from quantum_seismic.sensors.base import SensorChunk, SensorConfig
from quantum_seismic.sensors.location import LocationSensor
from quantum_seismic.sensors.webcam import WebcamSensor
from quantum_seismic.snapshot import (
    AcousticState,
    EnvironmentSnapshot,
    ExternalSeismicState,
    LocationState,
    SeismicState,
    VisualState,
)
from quantum_seismic.temporal import (
    RollingAggregator,
    classify_acoustic_regime,
    classify_seismic_regime,
    rms_to_db,
)


class EnvironmentDaemon:
    """Orchestrates all sensors and produces environment snapshots.

    Usage:
        daemon = EnvironmentDaemon()
        daemon.start()

        # At any point:
        snapshot = daemon.snapshot()
        print(snapshot.to_json())

        daemon.stop()
    """

    def __init__(
        self,
        *,
        enable_accel: bool = True,
        enable_mic: bool = True,
        enable_location: bool = True,
        enable_webcam: bool = True,
        enable_seismic_api: bool = True,
        webcam_interval_s: float = 300.0,
        webcam_model: str = "claude-haiku-4-5-20251001",
        seismic_poll_s: float = 300.0,
        accel_sample_rate: int = 800,
        mic_sample_rate: int = 44100,
    ):
        self._enable_accel = enable_accel
        self._enable_mic = enable_mic
        self._enable_location = enable_location
        self._enable_webcam = enable_webcam
        self._enable_seismic_api = enable_seismic_api

        # Temporal aggregators
        self._accel_agg = RollingAggregator()
        self._mic_agg = RollingAggregator()

        # Sensor instances (created on start)
        self._accel_source = None
        self._mic_source = None
        self._location = LocationSensor()
        self._webcam = WebcamSensor(
            capture_interval_s=webcam_interval_s,
            model=webcam_model,
        )
        self._seismic_api = SeismicAPI(poll_interval_s=seismic_poll_s)

        self._accel_sample_rate = accel_sample_rate
        self._mic_sample_rate = mic_sample_rate
        self._start_time: float = 0.0
        self._running = False

    def start(self) -> None:
        """Start all enabled sensors."""
        self._start_time = time.monotonic()

        # Accelerometer
        if self._enable_accel:
            try:
                from quantum_seismic.sensors.accelerometer import AccelerometerSource

                if AccelerometerSource.is_available():
                    config = SensorConfig(
                        sample_rate=self._accel_sample_rate, chunk_size=256
                    )
                    self._accel_source = AccelerometerSource(config)
                    self._accel_source.on_chunk(self._on_accel_chunk)
                    self._accel_source.start()
            except (ImportError, PermissionError):
                pass  # accelerometer not available or no root

        # Microphone
        if self._enable_mic:
            try:
                from quantum_seismic.sensors.microphone import MicrophoneSource

                config = SensorConfig(
                    sample_rate=self._mic_sample_rate, chunk_size=2048
                )
                self._mic_source = MicrophoneSource(config)
                self._mic_source.on_chunk(self._on_mic_chunk)
                self._mic_source.start()
            except Exception:
                pass

        # Location
        if self._enable_location:
            self._location.start()

        # Webcam
        if self._enable_webcam:
            self._webcam.start()

        # External seismic API
        if self._enable_seismic_api:
            # Feed location to seismic API when available
            if self._location.current:
                loc = self._location.current
                self._seismic_api.set_location(loc.latitude, loc.longitude)
            self._seismic_api.start()

        self._running = True

    def _on_accel_chunk(self, chunk: SensorChunk) -> None:
        self._accel_agg.push(chunk.data)

    def _on_mic_chunk(self, chunk: SensorChunk) -> None:
        self._mic_agg.push(chunk.data)

    def snapshot(self) -> EnvironmentSnapshot:
        """Build a point-in-time environment snapshot."""
        now_iso = EnvironmentSnapshot.now_iso()
        uptime = time.monotonic() - self._start_time if self._running else 0.0

        # Seismic (accelerometer)
        accel_1min = self._accel_agg.stats_1min()
        accel_1hr = self._accel_agg.stats_1hr()
        accel_24hr = self._accel_agg.stats_24hr()
        seismic = SeismicState(
            rms_1min=round(accel_1min.rms, 6),
            rms_1hr=round(accel_1hr.rms, 6),
            rms_24hr=round(accel_24hr.rms, 6),
            peak_1hr=round(accel_1hr.peak, 6),
            regime=classify_seismic_regime(accel_1min.rms),
        )

        # Acoustic (microphone)
        mic_1min = self._mic_agg.stats_1min()
        mic_1hr = self._mic_agg.stats_1hr()
        mic_24hr = self._mic_agg.stats_24hr()
        db_1min = rms_to_db(mic_1min.rms)
        db_1hr = rms_to_db(mic_1hr.rms)
        db_24hr = rms_to_db(mic_24hr.rms)

        # Trend: compare 1min to 1hr
        if mic_1hr.rms > 0 and mic_1min.rms > 0:
            ratio = mic_1min.rms / mic_1hr.rms
            if ratio > 1.5:
                trend = "rising"
            elif ratio < 0.7:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        acoustic = AcousticState(
            db_1min=round(db_1min, 1),
            db_1hr=round(db_1hr, 1),
            db_24hr=round(db_24hr, 1),
            regime=classify_acoustic_regime(db_1min),
            noise_trend=trend,
        )

        # Location
        loc_reading = self._location.current
        location = LocationState()
        if loc_reading:
            location = LocationState(
                latitude=round(loc_reading.latitude, 5),
                longitude=round(loc_reading.longitude, 5),
                altitude_m=round(loc_reading.altitude_m, 1),
                accuracy_m=round(loc_reading.accuracy_m, 1),
                stationary=True,  # TODO: detect movement from history
            )
            # Update seismic API with location
            self._seismic_api.set_location(loc_reading.latitude, loc_reading.longitude)

        # Visual
        visual = VisualState(
            summary=self._webcam.summary,
            occupants=self._webcam.occupants,
            last_capture=self._webcam.last_capture,
        )

        # External seismic
        external = ExternalSeismicState(
            nearest_event=self._seismic_api.nearest_event,
            warnings=self._seismic_api.warnings,
            last_fetch=self._seismic_api.last_fetch,
        )

        return EnvironmentSnapshot(
            timestamp=now_iso,
            uptime_s=round(uptime, 1),
            seismic=seismic,
            acoustic=acoustic,
            location=location,
            visual=visual,
            external_seismic=external,
        )

    def stop(self) -> None:
        """Stop all sensors."""
        if self._accel_source:
            self._accel_source.stop()
        if self._mic_source:
            self._mic_source.stop()
        self._location.stop()
        self._webcam.stop()
        self._seismic_api.stop()
        self._running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
