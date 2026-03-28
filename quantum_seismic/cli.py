"""CLI entry point — run the daemon and print snapshots or start an agent."""

import argparse
import asyncio
import json
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="quantum-seismic",
        description="Ambient environmental sensing for AI agents",
    )
    parser.add_argument(
        "--mode",
        choices=["daemon", "snapshot", "agent"],
        default="daemon",
        help="daemon: run continuously and print snapshots; snapshot: single snapshot; agent: start interactive agent",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Snapshot interval in seconds for daemon mode (default: 10)",
    )
    parser.add_argument(
        "--no-accel",
        action="store_true",
        help="Disable accelerometer (skip sudo requirement)",
    )
    parser.add_argument(
        "--no-mic",
        action="store_true",
        help="Disable microphone",
    )
    parser.add_argument(
        "--no-location",
        action="store_true",
        help="Disable GPS/location tracking",
    )
    parser.add_argument(
        "--no-webcam",
        action="store_true",
        help="Disable webcam capture",
    )
    parser.add_argument(
        "--no-seismic-api",
        action="store_true",
        help="Disable external seismic API",
    )
    parser.add_argument(
        "--webcam-interval",
        type=float,
        default=300.0,
        help="Webcam capture interval in seconds (default: 300)",
    )

    args = parser.parse_args()

    from quantum_seismic.daemon import EnvironmentDaemon

    daemon = EnvironmentDaemon(
        enable_accel=not args.no_accel,
        enable_mic=not args.no_mic,
        enable_location=not args.no_location,
        enable_webcam=not args.no_webcam,
        enable_seismic_api=not args.no_seismic_api,
        webcam_interval_s=args.webcam_interval,
    )

    if args.mode == "snapshot":
        daemon.start()
        time.sleep(2)  # let sensors warm up
        print(daemon.snapshot().to_json(indent=2))
        daemon.stop()

    elif args.mode == "agent":
        _run_agent(daemon)

    else:  # daemon
        _run_daemon(daemon, args.interval)


def _run_daemon(daemon, interval: float):
    """Run daemon and print periodic snapshots as JSONL."""

    def shutdown(*_):
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    daemon.start()
    print(json.dumps({"status": "running", "interval_s": interval}), flush=True)

    while True:
        time.sleep(interval)
        print(daemon.snapshot().to_json(), flush=True)


def _run_agent(daemon):
    """Start an interactive agent with environment context."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
    except ImportError:
        print(
            "claude-agent-sdk is required for agent mode. "
            "Install with: uv pip install 'quantum-seismic[agent]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from quantum_seismic.agent import agent_hook, system_prompt

    daemon.start()
    time.sleep(2)  # sensor warm-up

    async def run():
        print("Environment daemon running. Type your message (Ctrl+C to quit):\n")
        while True:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue

            async for message in query(
                prompt=user_input,
                options=ClaudeAgentOptions(
                    system_prompt=system_prompt(),
                    allowed_tools=["Read", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
                    hooks={
                        "UserPromptSubmit": [
                            HookMatcher(matcher=".*", hooks=[agent_hook(daemon)])
                        ]
                    },
                ),
            ):
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            print(block.text)
                elif hasattr(message, "result"):
                    print(f"\n[{message.subtype}]")

            print()

    try:
        asyncio.run(run())
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
