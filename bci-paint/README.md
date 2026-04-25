# BCI Paint

A brain-computer-interface driven painting tool built for the **g.tec Unicorn Hybrid Black** EEG headset. The user paints a simple coloring-book scene entirely hands-free using two neural signals:

| Mechanism | What the user does | What the system does |
|---|---|---|
| **SSVEP** (region selection) | Stares at a flickering canvas region for ~4 s | Detects the matching frequency in occipital EEG (PO7/PO8) and fills that region |
| **Mental imagery** (colour selection) | Thinks about a trained colour | Classifies frontal alpha patterns (Fz, C3, Cz, C4) → selects the recalled colour |

---

## Architecture

```
bci-paint/
├── main.py                  # CLI entry point
├── config.py                # All tunable constants
├── requirements.txt
├── .env                     # Optional Unicorn device serial
│
├── bci/
│   ├── headset.py           # Unicorn Hybrid Black wrapper + synthetic mock
│   ├── ssvep.py             # Rolling FFT / SNR SSVEP detector
│   └── color_classifier.py  # LDA mental-imagery colour classifier
│
├── ui/
│   ├── app.py               # Main window + CONNECTING→TRAINING→PAINTING state machine
│   ├── canvas.py            # Coloring-book paint canvas (6 bordered regions)
│   ├── training_screen.py   # Colour stimulus + EEG epoch recorder
│   └── ssvep_overlay.py     # Per-region flickering overlays (QTimer)
│
└── utils/
    └── signal_processing.py # Band-power, Butterworth filter, feature extractor
```

---

## Requirements

- Python ≥ 3.10
- **g.tec Unicorn Hybrid Black** headset + **Unicorn Suite** installed (or use `--mock` for a demo without hardware)
- macOS / Linux / Windows (tested on macOS 14+)

### UnicornPy SDK

`UnicornPy` is **not on PyPI** — it ships inside the g.tec Unicorn Suite installer.
After installing Unicorn Suite, copy the native binding into your environment:

| Platform | File to copy | Destination |
|---|---|---|
| macOS / Linux | `UnicornPy.so` | `.venv/lib/pythonX.Y/site-packages/` |
| Windows | `UnicornPy.pyd` | `.venv/Lib/site-packages/` |

---

## Installation

```bash
cd "bci-paint"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# then copy UnicornPy.so / UnicornPy.pyd as described above
```

---

## Running

### With the real Unicorn headset

Pair the headset via the Unicorn Suite or OS Bluetooth settings, then:

```bash
python main.py
```

If you have multiple paired Unicorn devices, set the serial in `.env`:

```
UNICORN_DEVICE_SERIAL=UN-2019.05.51
```

### Without hardware (mock / demo)

Generates synthetic EEG internally so you can explore the full UI flow:

```bash
python main.py --mock
```

Add `--debug-click` to also enable mouse clicks for filling regions manually
(useful during UI development):

```bash
python main.py --mock --debug-click
```

---

## Session flow

```
CONNECTING  ──►  TRAINING  ──►  READY  ──►  PAINTING
                                              │
                              ◄──────── Retrain button
```

### TRAINING phase

- The screen cycles through **2 colours × 10 trials** of 2-second stimulus patches.
- EEG is recorded for each patch and stored as labelled epochs.
- After all trials a linear-discriminant-analysis (LDA) classifier is trained on
  band-power features extracted from the frontal channels (Fz, C3, Cz, C4).
- Leave-one-out cross-validation accuracy is printed to the status bar.

### PAINTING phase

Two BCI loops run in parallel at 10 Hz:

**Colour selection (mental imagery)**
- Every tick: classify the last 3 s of frontal EEG → update the *Active Colour* swatch.
- The participant thinks about the desired colour; the model identifies it within a few seconds.

**Region selection (SSVEP)**
- Each of the 4 canvas regions flickers at its own frequency (8, 10, 12, 15 Hz).
- PO7 / PO8 occipital EEG is buffered for 4 s; an FFT is computed and the SNR at each
  target frequency is measured.
- When SNR exceeds the threshold for 4 consecutive seconds the region is confirmed and
  filled with the currently selected colour.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start painting (from READY screen) |
| `T` | Go to training screen |
| `R` | Reset canvas (during PAINTING) |

---

## Configuration

All parameters are in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 250 Hz | g.tec Unicorn Hybrid Black native sample rate |
| `CHANNEL_NAMES` | Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 | Fixed Unicorn electrode layout |
| `N_TRAINING_COLORS` | 2 | Colours used in training (2–4) |
| `TRAINING_TRIALS_PER_COLOR` | 10 | EEG epochs recorded per colour |
| `SSVEP_SNR_THRESHOLD` | 3.5 | Minimum occipital SNR to count as a detected region |
| `STARE_CONFIRM_SEC` | 4.0 | How long user must sustain stare to confirm a region |
| `FLICKER_OVERLAY_ALPHA` | 120 | Brightness of the flickering white overlay (0–255) |
| `SSVEP_FREQUENCIES` | 8, 10, 12, 15 Hz | One per canvas region |

---

## How the BCI signals work

### SSVEP (region targeting)

The visual cortex responds at exactly the frequency of a flickering stimulus
(and its harmonics).  Placing a semi-transparent white overlay on each canvas
region and toggling it at region-specific frequencies causes the occipital
channels (PO7/PO8) to show a spectral peak at the frequency the participant is gazing at.
A Welch PSD is computed over a 4-second window and the signal-to-noise ratio
at each target frequency is compared against the background noise floor.

### Mental imagery (colour selection)

Different visual experiences (and their recall) produce distinguishable patterns
in the EEG alpha band (~8–13 Hz) and in overall frontal activity.  During
training, a coloured patch is shown and the resulting EEG epoch is labelled.
After training, when the participant mentally recalls the same colour the model
detects the familiar activity pattern and selects that colour.

> **Note:** Mental imagery classification is the harder of the two tasks.
> With dry electrodes and 10 trials per colour, expect 60–80 % accuracy.
> Increasing `TRAINING_TRIALS_PER_COLOR` or adding a gamma feature improves results.
