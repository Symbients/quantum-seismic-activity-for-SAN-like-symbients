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
```

## Usage

```bash
# Run with simulator (default) — generates synthetic seismic events
quantum-seismic

# Run with microphone — real vibration detection
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
Microphone ──┐        │    (preprocess)  │
             ├──────► ├─── STA/LTA ─────┤
Simulator  ──┘        ├─── CUSUM ───────┤ ──────► Rule-based    ──► Terminal UI
                      ├─── Kurtosis ────┤         Classifier         + Event Log
                      └─── Peak/MAD ────┘
```

### Why microphone?

Apple Silicon Macs don't expose the hardware accelerometer to userspace (CMMotionManager is iOS-only). The built-in microphone picks up mechanical vibrations through the chassis at 44.1kHz — 440x the resolution of a typical 100Hz accelerometer. This is a legitimate technique used in seismology (acoustic seismometry).

## For SAN-like Symbients

This project provides the sensory substrate for Symbient Awareness Networks. The classified event stream can feed into higher-level awareness systems that build temporal models of their physical environment.

## License

MIT
