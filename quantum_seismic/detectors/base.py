"""Base detector interface."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class DetectorOutput:
    """Output from a single detector for one chunk."""

    name: str  # detector name
    value: float  # primary scalar output (e.g., STA/LTA ratio, kurtosis value)
    triggered: bool  # whether this detector considers an event active
    signal: np.ndarray  # transformed signal (same length as input chunk)
    metadata: dict = field(default_factory=dict)


class Detector(Protocol):
    """Protocol for all detector algorithms."""

    name: str

    def process(self, samples: np.ndarray, sample_rate: int) -> DetectorOutput:
        """Process a chunk of samples and return detector output."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...
