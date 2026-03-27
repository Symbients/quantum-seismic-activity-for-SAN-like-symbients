"""CLI entry point for quantum-seismic."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="quantum-seismic",
        description="Desk seismograph — seismic impact detection using MacBook sensors",
    )
    parser.add_argument(
        "--source",
        choices=["mic", "sim"],
        default="sim",
        help="Sensor source: mic (microphone) or sim (simulator, default)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Samples per processing chunk (default: 2048)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Audio device name or index (default: system default)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without UI — print events to stdout as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for simulator (for reproducible demos)",
    )

    args = parser.parse_args()

    # Build source
    from quantum_seismic.sources.base import SensorConfig

    config = SensorConfig(
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        device=int(args.device) if args.device and args.device.isdigit() else args.device,
    )

    if args.source == "mic":
        from quantum_seismic.sources.microphone import MicrophoneSource

        source = MicrophoneSource(config)
    else:
        from quantum_seismic.sources.simulator import SimulatorSource

        source = SimulatorSource(config, seed=args.seed)

    # Build pipeline
    from quantum_seismic.pipeline import Pipeline

    pipeline = Pipeline(source)

    if args.headless:
        _run_headless(pipeline)
    else:
        _run_ui(pipeline)


def _run_headless(pipeline):
    """Print events as JSON lines to stdout."""
    import json
    import signal
    import time

    def on_state(state):
        if state.event is not None:
            evt = state.event
            record = {
                "timestamp": evt.timestamp,
                "event": evt.event_type.value,
                "confidence": evt.confidence,
                "severity": evt.severity,
                "fingerprint": evt.fingerprint,
            }
            print(json.dumps(record), flush=True)

    pipeline.on_state(on_state)

    def shutdown(*_):
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    pipeline.start()
    print('{"status": "running", "source": "' + type(pipeline.source).__name__ + '"}', flush=True)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pipeline.stop()


def _run_ui(pipeline):
    """Run the Textual terminal UI."""
    from quantum_seismic.ui import SeismicApp

    app = SeismicApp(pipeline=pipeline)

    # Start pipeline when app mounts (handled in on_mount)
    # Stop pipeline when app exits
    pipeline.start()
    try:
        app.run()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
