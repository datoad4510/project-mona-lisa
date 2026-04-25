"""
BCI Paint — Central configuration.

All tunable constants live here so the rest of the codebase imports from a
single location.
"""

# ── EEG hardware ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 250          # Hz (g.tec Unicorn Hybrid Black)
N_CHANNELS  = 8
# Unicorn electrode layout (fixed order from UnicornPy SDK)
CHANNEL_NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

# Best channels for SSVEP (parieto-occipital)
SSVEP_CHANNELS         = ["PO7", "PO8"]
SSVEP_CHANNEL_INDICES  = [CHANNEL_NAMES.index(c) for c in SSVEP_CHANNELS]

# Best channels for colour mental imagery (frontal/central)
COLOR_CHANNELS         = ["Fz", "C3", "Cz", "C4"]
COLOR_CHANNEL_INDICES  = [CHANNEL_NAMES.index(c) for c in COLOR_CHANNELS]

# ── SSVEP detection ───────────────────────────────────────────────────────────
SSVEP_WINDOW_SEC         = 4.0                          # analysis window (s)
SSVEP_WINDOW_SAMPLES     = int(SSVEP_WINDOW_SEC * SAMPLE_RATE)
SSVEP_SNR_THRESHOLD      = 3.5                          # min SNR to confirm
# One flickering frequency per canvas region (Hz) — one per circle
SSVEP_FREQUENCIES        = [8.0, 10.0, 12.0, 15.0]

# ── Colour training & classification ─────────────────────────────────────────
TRAINING_COLORS = [
    {"name": "Red",    "rgb": (231,  76,  60)},
    {"name": "Yellow", "rgb": (241, 196,  15)},
    {"name": "Blue",   "rgb": (52,  152, 219)},
    {"name": "Green",  "rgb": (39,  174,  96)},
]
N_TRAINING_COLORS           = 4    # all four colours
TRAINING_TRIALS_PER_COLOR   = 10
TRAINING_STIMULUS_MS        = 2000  # display colour patch (ms)
TRAINING_ISI_MS             = 1000  # inter-stimulus interval  (ms)
COLOR_EPOCH_SEC             = 2.0   # EEG epoch used for feature extraction (s)
COLOR_EPOCH_SAMPLES         = int(COLOR_EPOCH_SEC * SAMPLE_RATE)
RECALL_DURATION_SEC         = 3.0   # how long user mentally recalls colour
RECALL_SAMPLES              = int(RECALL_DURATION_SEC * SAMPLE_RATE)

# ── Frequency bands (Hz) ──────────────────────────────────────────────────────
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ── Canvas regions ────────────────────────────────────────────────────────────
# Four circles arranged in a 2 × 2 grid, well separated so they never touch.
# Each circle maps to one SSVEP frequency and one training colour.
# Geometry: normalised (x0, y0, x1, y1) bounding-box of the ellipse.
#
#  Layout (normalised coordinates, canvas 700 × 550):
#
#     [Circle 0 — 8 Hz]       [Circle 1 — 10 Hz]
#
#     [Circle 2 — 12 Hz]      [Circle 3 — 15 Hz]
#
# Radius ≈ 0.15 → pixel radius ≈ 82 px.
# Horizontal gap between circles: 0.20 norm ≈ 140 px.
# Vertical gap between circles:   0.14 norm ≈  77 px.

CANVAS_REGIONS = [
    {
        "id": 0, "label": "Circle 1",
        "ssvep_freq": 8.0,
        "shape": "ellipse",
        "coords": (0.10, 0.13, 0.40, 0.47),   # top-left circle
        "default_color": (220, 220, 220),
    },
    {
        "id": 1, "label": "Circle 2",
        "ssvep_freq": 10.0,
        "shape": "ellipse",
        "coords": (0.60, 0.13, 0.90, 0.47),   # top-right circle
        "default_color": (220, 220, 220),
    },
    {
        "id": 2, "label": "Circle 3",
        "ssvep_freq": 12.0,
        "shape": "ellipse",
        "coords": (0.10, 0.57, 0.40, 0.91),   # bottom-left circle
        "default_color": (220, 220, 220),
    },
    {
        "id": 3, "label": "Circle 4",
        "ssvep_freq": 15.0,
        "shape": "ellipse",
        "coords": (0.60, 0.57, 0.90, 0.91),   # bottom-right circle
        "default_color": (220, 220, 220),
    },
]

# ── UI ────────────────────────────────────────────────────────────────────────
CANVAS_W = 700
CANVAS_H = 550
FLICKER_OVERLAY_ALPHA = 120   # 0-255  transparency of the flicker overlay
STARE_CONFIRM_SEC     = 4.0   # seconds of sustained SSVEP to confirm region
