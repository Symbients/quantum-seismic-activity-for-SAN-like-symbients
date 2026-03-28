"""Webcam capture + vision model description.

Takes periodic snapshots via ImageIO/AVFoundation and sends them to
a cheap vision model (Haiku) for natural language descriptions.
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class WebcamSensor:
    """Periodic webcam capture with vision model descriptions."""

    def __init__(
        self,
        capture_interval_s: float = 300.0,  # every 5 minutes
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.capture_interval_s = capture_interval_s
        self.model = model
        self._current_summary: str = ""
        self._current_occupants: int | None = None
        self._last_capture: str = ""
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._capture_and_describe()
            except Exception:
                pass  # non-critical — don't crash the daemon
            self._stop_event.wait(self.capture_interval_s)

    def _capture_and_describe(self) -> None:
        """Capture a frame and describe it with a vision model."""
        image_path = self._capture_frame()
        if image_path is None:
            return

        description = self._describe_image(image_path)
        if description:
            with self._lock:
                self._current_summary = description.get("summary", "")
                self._current_occupants = description.get("occupants")
                from quantum_seismic.snapshot import EnvironmentSnapshot

                self._last_capture = EnvironmentSnapshot.now_iso()

        # Clean up temp file
        try:
            Path(image_path).unlink()
        except OSError:
            pass

    def _capture_frame(self) -> str | None:
        """Capture a single frame from the webcam using imagesnap (macOS)."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()

        try:
            # imagesnap is a lightweight macOS webcam capture tool
            # Install: brew install imagesnap
            result = subprocess.run(
                ["imagesnap", "-q", "-w", "1.0", tmp.name],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and Path(tmp.name).stat().st_size > 0:
                return tmp.name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: ffmpeg
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-f", "avfoundation", "-framerate", "1",
                    "-i", "0", "-frames:v", "1", "-y", "-loglevel", "quiet",
                    tmp.name,
                ],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and Path(tmp.name).stat().st_size > 0:
                return tmp.name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    def _describe_image(self, image_path: str) -> dict | None:
        """Send image to a vision model for description."""
        try:
            import anthropic
        except ImportError:
            return None

        image_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this webcam image in one concise sentence for "
                                "environmental context. Focus on: setting (office/home/outdoor), "
                                "lighting, number of people visible, activity level. "
                                "Reply as JSON: {\"summary\": \"...\", \"occupants\": N}"
                            ),
                        },
                    ],
                }
            ],
        )

        import json

        try:
            text = response.content[0].text
            # Handle both raw JSON and markdown-wrapped JSON
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return {"summary": response.content[0].text, "occupants": None}

    @property
    def summary(self) -> str:
        with self._lock:
            return self._current_summary

    @property
    def occupants(self) -> int | None:
        with self._lock:
            return self._current_occupants

    @property
    def last_capture(self) -> str:
        with self._lock:
            return self._last_capture

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
