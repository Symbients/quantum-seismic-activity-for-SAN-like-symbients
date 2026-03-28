# quantum-seismic

Ambient environmental sensing for AI agents. Continuously samples MacBook sensors and produces structured context that enriches every agent turn with physical environmental awareness.

## What it does

Runs a background daemon that samples sensors, builds temporal aggregates, and injects an `<environment>` block into every agent message via the Claude Agent SDK.

```
Continuous Sensors          Temporal Aggregation         Agent Context
──────────────────         ──────────────────────       ──────────────
Accelerometer (800Hz) ──┐   Rolling stats:
Microphone (44.1kHz)  ──┤   • 1min / 1hr / 24hr       ┌──────────────┐
CoreLocation (GPS)    ──┤──►  averages & trends    ──► │ <environment>│──► Claude Agent SDK
Webcam (periodic)     ──┤   • Regime detection          │ JSON block   │    query() hook
USGS Seismic API      ──┘   • Visual summaries          │ per turn     │
                                                        └──────────────┘
```

### What the agent sees each turn

```json
{
  "timestamp": "2026-03-27T09:15:00+09:00",
  "uptime_s": 3600.0,
  "seismic": {
    "rms_1min": 0.003, "rms_1hr": 0.002, "rms_24hr": 0.002,
    "peak_1hr": 0.015, "regime": "mild", "events_1hr": 2
  },
  "acoustic": {
    "db_1min": 42.3, "db_1hr": 38.1, "db_24hr": 40.5,
    "regime": "quiet", "speech_detected": false, "noise_trend": "stable"
  },
  "location": {
    "latitude": 35.6812, "longitude": 139.7671,
    "label": "unknown", "stationary": true
  },
  "visual": {
    "summary": "person at desk, laptop open, daylight from window, coffee cup",
    "occupants": 1
  },
  "external_seismic": {
    "nearest_event": "M2.1 Chiba 45km 3h ago",
    "warnings": []
  }
}
```

## Install

```bash
# Core (mic only, no sudo needed)
uv pip install -e .

# With hardware accelerometer (M2/M3/M4 Pro+, requires sudo)
uv pip install -e ".[accel]"

# With all sensors + agent SDK
uv pip install -e ".[all]"
```

## Usage

### As a library (Claude Agent SDK)

```python
from quantum_seismic import EnvironmentDaemon, enrich_prompt, system_prompt
from claude_agent_sdk import query, ClaudeAgentOptions

daemon = EnvironmentDaemon(enable_accel=False)  # no sudo needed
daemon.start()

# enrich_prompt() appends live sensor data as an <environment> block
async for message in query(
    prompt=enrich_prompt(daemon, user_input),
    options=ClaudeAgentOptions(
        system_prompt=system_prompt(),
    ),
):
    ...

daemon.stop()
```

### As a daemon (JSONL to stdout)

```bash
# Print environment snapshots every 10s
quantum-seismic --mode daemon --interval 10

# Single snapshot
quantum-seismic --mode snapshot

# Mic only (no sudo, no extras)
quantum-seismic --no-accel --no-webcam --no-location

# With accelerometer (requires sudo)
sudo quantum-seismic --mode daemon
```

### Sensor sources

| Sensor | What | Rate | Requires |
|--------|------|------|----------|
| Accelerometer | Bosch BMI286 via IOKit HID | ~800Hz | `sudo`, `.[accel]`, M2+ Pro |
| Microphone | Ambient sound via PortAudio | 44.1kHz | Nothing |
| Location | GPS/WiFi via CoreLocation | 1/min | `.[location]` |
| Webcam | Snapshots → Haiku vision description | 1/5min | `.[webcam]`, `imagesnap` or `ffmpeg` |
| USGS API | Regional earthquake data | 1/5min | Internet |

## For SAN-like Symbients

This is the sensory substrate for Symbient Awareness Networks. The daemon gives agents embodied environmental awareness — they don't just process text, they sense the physical world around the machine they run on.

## License

MIT
