"""STA/LTA (Short-Term Average / Long-Term Average) Ratio detector.

Classic seismological trigger algorithm. Compares energy in a short recent
window against a longer background window. When the ratio spikes, something
just happened.

Runs at 3 timescales simultaneously:
- Fast:   STA=0.01s / LTA=0.5s   — catches sharp transients (knocks)
- Medium: STA=0.05s / LTA=2.0s   — catches moderate events (footsteps)
- Slow:   STA=0.2s  / LTA=10.0s  — catches sustained events (earthquake onset)
"""

import numpy as np

from quantum_seismic.detectors.base import DetectorOutput


class STALTADetector:
    """Multi-timescale STA/LTA ratio detector."""

    name = "sta_lta"

    # (sta_seconds, lta_seconds, trigger_threshold)
    TIMESCALES = [
        (0.01, 0.5, 3.0),  # fast
        (0.05, 2.0, 2.5),  # medium
        (0.2, 10.0, 2.0),  # slow
    ]

    def __init__(self):
        # Running averages for each timescale
        self._sta_buffers: list[np.ndarray | None] = [None] * len(self.TIMESCALES)
        self._lta_buffers: list[np.ndarray | None] = [None] * len(self.TIMESCALES)
        self._sta_values: list[float] = [0.0] * len(self.TIMESCALES)
        self._lta_values: list[float] = [0.0] * len(self.TIMESCALES)
        self._initialized = False

    def process(self, samples: np.ndarray, sample_rate: int) -> DetectorOutput:
        # Work with squared amplitude (energy)
        energy = samples**2

        ratios = []
        triggered_any = False
        timescale_results = []

        for i, (sta_s, lta_s, threshold) in enumerate(self.TIMESCALES):
            # Exponential moving average decay factors
            sta_samples = max(1, int(sta_s * sample_rate))
            lta_samples = max(1, int(lta_s * sample_rate))
            alpha_sta = 2.0 / (sta_samples + 1)
            alpha_lta = 2.0 / (lta_samples + 1)

            # Update running averages sample-by-sample for accuracy
            sta = self._sta_values[i]
            lta = self._lta_values[i]

            # Process in blocks for speed: use recursive EMA formula
            for e in energy:
                sta = sta * (1 - alpha_sta) + e * alpha_sta
                lta = lta * (1 - alpha_lta) + e * alpha_lta

            self._sta_values[i] = sta
            self._lta_values[i] = lta

            ratio = sta / max(lta, 1e-10)
            ratios.append(ratio)

            triggered = ratio > threshold
            if triggered:
                triggered_any = True

            timescale_results.append(
                {
                    "label": ["fast", "medium", "slow"][i],
                    "sta_s": sta_s,
                    "lta_s": lta_s,
                    "ratio": float(ratio),
                    "threshold": threshold,
                    "triggered": triggered,
                }
            )

        # Primary value: max ratio across timescales
        max_ratio = max(ratios)

        # Build a per-sample ratio signal for visualization (using fast timescale)
        sta_s, lta_s, _ = self.TIMESCALES[0]
        alpha_sta = 2.0 / (max(1, int(sta_s * sample_rate)) + 1)
        alpha_lta = 2.0 / (max(1, int(lta_s * sample_rate)) + 1)
        signal = np.zeros_like(samples)
        sta_viz = self._sta_values[0]
        lta_viz = self._lta_values[0]
        for j, e in enumerate(energy):
            sta_viz = sta_viz * (1 - alpha_sta) + e * alpha_sta
            lta_viz = lta_viz * (1 - alpha_lta) + e * alpha_lta
            signal[j] = sta_viz / max(lta_viz, 1e-10)

        return DetectorOutput(
            name=self.name,
            value=float(max_ratio),
            triggered=triggered_any,
            signal=signal,
            metadata={"timescales": timescale_results},
        )

    def reset(self) -> None:
        self._sta_values = [0.0] * len(self.TIMESCALES)
        self._lta_values = [0.0] * len(self.TIMESCALES)
        self._initialized = False
