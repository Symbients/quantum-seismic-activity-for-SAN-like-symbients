"""Terminal UI — real-time waveform, detector gauges, and event log."""

from collections import deque

import numpy as np
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, RichLog, Static

from quantum_seismic.pipeline import PipelineState

# ──────────────────────────────────────────────────────────────────────
# Sparkline waveform using Unicode block characters
# ──────────────────────────────────────────────────────────────────────

SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: np.ndarray, width: int = 80) -> str:
    """Render a numpy array as a unicode sparkline."""
    if len(values) == 0:
        return " " * width

    # Downsample to fit width
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width, dtype=int)
        values = values[indices]

    # Normalize to 0-1
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-10:
        normalized = np.full_like(values, 0.5)
    else:
        normalized = (values - vmin) / (vmax - vmin)

    chars = [SPARK_CHARS[int(v * (len(SPARK_CHARS) - 1))] for v in normalized]
    return "".join(chars)


# ──────────────────────────────────────────────────────────────────────
# Widgets
# ──────────────────────────────────────────────────────────────────────


class WaveformDisplay(Static):
    """Shows real-time waveform sparkline."""

    DEFAULT_CSS = """
    WaveformDisplay {
        height: 5;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def __init__(self):
        super().__init__("")
        self._buffer: deque[float] = deque(maxlen=4000)

    def update_data(self, samples: np.ndarray) -> None:
        # Take every Nth sample to keep buffer manageable
        stride = max(1, len(samples) // 100)
        for s in samples[::stride]:
            self._buffer.append(float(s))
        self._render()

    def _render(self) -> None:
        if not self._buffer:
            return
        arr = np.array(self._buffer)
        width = max(20, self.size.width - 4) if self.size.width > 0 else 80

        # Raw waveform (centered around zero)
        line1 = sparkline(np.abs(arr), width)
        # Envelope
        window = min(50, len(arr) // 4) if len(arr) > 4 else 1
        if window > 1:
            kernel = np.ones(window) / window
            envelope = np.convolve(np.abs(arr), kernel, mode="same")
            line2 = sparkline(envelope, width)
        else:
            line2 = line1

        self.update(f"[bold]Waveform[/]\n{line1}\n[dim]Envelope[/]\n{line2}")


class DetectorGauge(Static):
    """Shows a single detector's value as a horizontal bar."""

    DEFAULT_CSS = """
    DetectorGauge {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, detector_name: str, max_val: float = 10.0):
        super().__init__("")
        self.detector_name = detector_name
        self.max_val = max_val

    def update_value(self, value: float, triggered: bool) -> None:
        bar_width = 30
        fill = min(bar_width, int((value / self.max_val) * bar_width))
        empty = bar_width - fill

        if triggered:
            bar = f"[bold red]{'█' * fill}[/][dim]{'░' * empty}[/]"
            indicator = "[bold red]TRIG[/]"
        else:
            bar = f"[green]{'█' * fill}[/][dim]{'░' * empty}[/]"
            indicator = "[dim]    [/]"

        name = f"{self.detector_name:10s}"
        val = f"{value:8.2f}"
        self.update(f"{name} {bar} {val} {indicator}")


class DetectorPanel(Container):
    """Panel showing all detector gauges."""

    DEFAULT_CSS = """
    DetectorPanel {
        height: 9;
        border: solid $secondary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("[bold]Detectors[/]")
        yield DetectorGauge("highpass", max_val=0.1)
        yield DetectorGauge("sta_lta", max_val=10.0)
        yield DetectorGauge("cusum", max_val=0.2)
        yield DetectorGauge("kurtosis", max_val=20.0)
        yield DetectorGauge("peak_mad", max_val=15.0)

    def update_outputs(self, outputs: dict) -> None:
        gauges = self.query(DetectorGauge)
        for gauge in gauges:
            if gauge.detector_name in outputs:
                out = outputs[gauge.detector_name]
                gauge.update_value(out.value, out.triggered)


class EventLog(RichLog):
    """Scrolling log of classified events."""

    DEFAULT_CSS = """
    EventLog {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    """


class StatusBar(Static):
    """Shows pipeline stats."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $primary-background;
        padding: 0 1;
    }
    """

    chunks_processed = reactive(0)
    events_detected = reactive(0)
    source_name = reactive("--")

    def render(self) -> str:
        return (
            f"Chunks: {self.chunks_processed:,}  |  "
            f"Events: {self.events_detected:,}  |  "
            f"Source: {self.source_name}"
        )


# ──────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────


class SeismicApp(App):
    """Quantum Seismic — desk seismograph."""

    TITLE = "Quantum Seismic"
    SUB_TITLE = "desk seismograph"

    CSS = """
    Screen {
        layout: vertical;
    }

    #top-row {
        height: auto;
        max-height: 16;
    }

    #waveform-col {
        width: 2fr;
    }

    #detector-col {
        width: 1fr;
    }

    #event-log {
        height: 1fr;
        min-height: 6;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reset", "Reset"),
        ("space", "inject_knock", "Inject knock"),
    ]

    def __init__(self, pipeline=None, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
        self._event_count = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            with Vertical(id="waveform-col"):
                yield WaveformDisplay()
            with Vertical(id="detector-col"):
                yield DetectorPanel()
        yield Label("[bold]Event Log[/]", id="event-log-label")
        yield EventLog(id="event-log", max_lines=200, markup=True)
        yield StatusBar()
        yield Footer()

    def on_mount(self) -> None:
        if self._pipeline:
            self._pipeline.on_state(self._on_pipeline_state)

    def _on_pipeline_state(self, state: PipelineState) -> None:
        """Called from pipeline thread — schedule UI update on main thread."""
        self.call_from_thread(self._update_ui, state)

    def _update_ui(self, state: PipelineState) -> None:
        # Update waveform
        waveform = self.query_one(WaveformDisplay)
        waveform.update_data(state.filtered_signal)

        # Update detector gauges
        panel = self.query_one(DetectorPanel)
        panel.update_outputs(state.detector_outputs)

        # Update status
        status = self.query_one(StatusBar)
        status.chunks_processed = state.chunks_processed
        status.source_name = state.raw_chunk.source_name

        # Log events
        if state.event is not None:
            self._event_count += 1
            status.events_detected = self._event_count
            log = self.query_one(EventLog)

            evt = state.event
            color = {
                "knock": "red",
                "door_slam": "yellow",
                "footstep": "cyan",
                "typing": "dim",
                "earthquake": "bold red",
                "unknown": "white",
                "ambient": "dim",
            }.get(evt.event_type.value, "white")

            log.write(
                Text.from_markup(
                    f"[{color}]{evt.timestamp:8.2f}s  "
                    f"{evt.event_type.value:12s}  "
                    f"conf={evt.confidence:.0%}  "
                    f"severity={evt.severity}  "
                    f"sta/lta={evt.fingerprint.get('sta_lta_ratio', 0):.1f}  "
                    f"kurt={evt.fingerprint.get('kurtosis', 0):.1f}  "
                    f"p/mad={evt.fingerprint.get('peak_mad_score', 0):.1f}[/]"
                )
            )

    def action_reset(self) -> None:
        if self._pipeline:
            self._pipeline.reset()
            self._event_count = 0
            log = self.query_one(EventLog)
            log.clear()

    def action_inject_knock(self) -> None:
        """Inject a test knock event (only works with simulator source)."""
        if self._pipeline:
            source = self._pipeline.source
            if hasattr(source, "inject_event"):
                source.inject_event("knock")
