"""Abstract base for sensor data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SensorConfig:
    """Configuration for a sensor source."""

    sample_rate: int = 44100  # Hz
    chunk_size: int = 1024  # samples per callback
    channels: int = 1
    device: int | str | None = None  # None = system default


@dataclass
class SensorChunk:
    """A chunk of sensor data with timing info."""

    data: np.ndarray  # shape: (chunk_size,) — normalized float64 [-1, 1]
    timestamp: float  # seconds since source start
    sample_rate: int
    source_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.data**2)))


# Callback type: receives a SensorChunk
ChunkCallback = Callable[[SensorChunk], None]


class SensorSource(ABC):
    """Abstract sensor source that streams chunks to a callback."""

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()
        self._running = False
        self._callbacks: list[ChunkCallback] = []

    def on_chunk(self, callback: ChunkCallback) -> None:
        self._callbacks.append(callback)

    def _emit(self, chunk: SensorChunk) -> None:
        for cb in self._callbacks:
            cb(chunk)

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @property
    def running(self) -> bool:
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
