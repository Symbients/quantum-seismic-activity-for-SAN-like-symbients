"""Processing pipeline — connects sensor source → detectors → classifier."""

import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from quantum_seismic.classifier import ClassifiedEvent, EventClassifier
from quantum_seismic.detectors.base import DetectorOutput
from quantum_seismic.detectors.cusum import CUSUMDetector
from quantum_seismic.detectors.highpass import HighPassFilter
from quantum_seismic.detectors.kurtosis import KurtosisDetector
from quantum_seismic.detectors.peak_mad import PeakMADDetector
from quantum_seismic.detectors.sta_lta import STALTADetector
from quantum_seismic.sources.base import SensorChunk, SensorSource


@dataclass
class PipelineState:
    """Snapshot of the full pipeline state for one chunk."""

    timestamp: float
    raw_chunk: SensorChunk
    filtered_signal: np.ndarray
    detector_outputs: dict[str, DetectorOutput]
    event: ClassifiedEvent | None
    chunks_processed: int

    @property
    def any_triggered(self) -> bool:
        return any(o.triggered for o in self.detector_outputs.values())


# Callback for UI updates
StateCallback = Callable[[PipelineState], None]


class Pipeline:
    """Full processing pipeline from sensor source to classified events."""

    def __init__(self, source: SensorSource):
        self.source = source

        # Initialize detectors
        self.highpass = HighPassFilter(cutoff_hz=20.0)
        self.sta_lta = STALTADetector()
        self.cusum = CUSUMDetector()
        self.kurtosis = KurtosisDetector()
        self.peak_mad = PeakMADDetector()

        self.classifier = EventClassifier()

        self._callbacks: list[StateCallback] = []
        self._chunks_processed: int = 0
        self._lock = threading.Lock()

    def on_state(self, callback: StateCallback) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        self.source.on_chunk(self._process_chunk)
        self.source.start()

    def stop(self) -> None:
        self.source.stop()

    def _process_chunk(self, chunk: SensorChunk) -> None:
        with self._lock:
            self._chunks_processed += 1
            sr = chunk.sample_rate

            # Stage 1: High-pass filter (preprocessing)
            hp_out = self.highpass.process(chunk.data, sr)
            filtered = hp_out.signal

            # Stage 2: Run all detectors on the filtered signal
            sta_out = self.sta_lta.process(filtered, sr)
            cusum_out = self.cusum.process(filtered, sr)
            kurt_out = self.kurtosis.process(filtered, sr)
            pmad_out = self.peak_mad.process(filtered, sr)

            outputs = {
                "highpass": hp_out,
                "sta_lta": sta_out,
                "cusum": cusum_out,
                "kurtosis": kurt_out,
                "peak_mad": pmad_out,
            }

            # Stage 3: Classify
            chunk_duration = len(chunk.data) / sr
            event = self.classifier.classify(outputs, chunk.timestamp, chunk_duration)

            # Build state snapshot
            state = PipelineState(
                timestamp=chunk.timestamp,
                raw_chunk=chunk,
                filtered_signal=filtered,
                detector_outputs=outputs,
                event=event,
                chunks_processed=self._chunks_processed,
            )

            # Emit to UI
            for cb in self._callbacks:
                cb(state)

    def reset(self) -> None:
        self.highpass.reset()
        self.sta_lta.reset()
        self.cusum.reset()
        self.kurtosis.reset()
        self.peak_mad.reset()
        self.classifier.reset()
        self._chunks_processed = 0

    @property
    def events(self) -> list[ClassifiedEvent]:
        return self.classifier.history

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
