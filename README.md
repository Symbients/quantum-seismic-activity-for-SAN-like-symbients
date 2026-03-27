# quantum-seismic

Seismic impact detection using MacBook sensors. Turns your laptop into a desk seismograph.

## What it does

Captures vibrations through your MacBook's microphone (or a synthetic simulator), runs 5 parallel detection algorithms, and classifies events in real-time.

### Detection Pipeline

| Algorithm | Role | What it catches |
|-----------|------|-----------------|
| **High-Pass Filter** | Preprocessing | Strips DC offset / low-frequency rumble to isolate dynamic events |
| **STA/LTA Ratio** | Primary trigger | "Something just happened" at 3 timescales (fast/medium/slow) |
| **CUSUM** | Onset precision | Pinpoints exact start of a shift (sub-sample accuracy) |
| **Kurtosis** | Shape classifier | Sharp spike (knock) vs sustained (vibration) vs rolling (footsteps) |
| **Peak/MAD** | Magnitude scorer | "How abnormal was this?" on a severity scale |

### Event Classification

The 5 detector outputs form a fingerprint. Different events have distinct signatures:

| Event | STA/LTA | Kurtosis | Peak/MAD | Duration |
|-------|---------|----------|----------|----------|
| Knock | High spike | Very high | High | <50ms |
| Typing | Low (~1.5) | Medium | Low | Continuous |
| Footstep | Rhythmic | Low-medium | Medium | ~100ms repeating |
| Door slam | Medium-high | High | Medium | 100-300ms |
| Earthquake | Sustained | Low (broadband) | Ramps up | Seconds+ |

## Install

```bash
# Requires Python 3.11+ and uv
uv pip install -e .

# For hardware accelerometer support (M2/M3/M4 Pro+)
uv pip install -e ".[accel]"
```

## Usage

```bash
# Run with simulator (default) — generates synthetic seismic events
quantum-seismic

# Run with hardware accelerometer (requires sudo, M2/M3/M4 Pro+)
sudo quantum-seismic --source accel

# Run with microphone — vibration detection via chassis acoustics
quantum-seismic --source mic

# Headless mode — JSON event stream to stdout
quantum-seismic --headless

# Custom sample rate and chunk size
quantum-seismic --sample-rate 22050 --chunk-size 1024
```

### Keyboard shortcuts (in UI mode)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset detectors and event log |
| `Space` | Inject a test knock (simulator only) |

## Architecture

```
Sensor Source          Detection Pipeline          Classification
─────────────         ──────────────────          ──────────────
                      ┌─── High-Pass ───┐
Accelerometer ┐       │    (preprocess)  │
Microphone  ──┤─────► ├─── STA/LTA ─────┤
Simulator   ──┘       ├─── CUSUM ───────┤ ──────► Rule-based    ──► Terminal UI
                      ├─── Kurtosis ────┤         Classifier         + Event Log
                      └─── Peak/MAD ────┘
```

### Sensor sources

| Source | API | Rate | Requires | Best for |
|--------|-----|------|----------|----------|
| **Accelerometer** | IOKit HID → Bosch BMI286 via Apple SPU | ~800Hz | `sudo`, M2/M3/M4 Pro+ | True seismic detection, gravity-referenced |
| **Microphone** | PortAudio → built-in mic | 44.1kHz | Nothing | Acoustic vibration, highest temporal resolution |
| **Simulator** | Synthetic event generator | Configurable | Nothing | Testing, demos |

The accelerometer reads the undocumented Bosch BMI286 IMU via Apple's Sensor Processing Unit using [macimu](https://github.com/olvvier/apple-silicon-accelerometer). The microphone captures mechanical vibrations through the chassis — a legitimate technique used in seismology (acoustic seismometry).

## For SAN-like Symbients

This project provides the sensory substrate for Symbient Awareness Networks. The classified event stream can feed into higher-level awareness systems that build temporal models of their physical environment.

## License

MIT
