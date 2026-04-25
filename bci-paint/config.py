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

# Best channels for colour mental imagery (all 8 channels; even-numbered
# channels 2,4,6,8 — C3, C4, PO7, PO8 — are weighted more heavily)
COLOR_CHANNELS         = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
COLOR_CHANNEL_INDICES  = [CHANNEL_NAMES.index(c) for c in COLOR_CHANNELS]
# Per-channel significance weights (channels 2,4,6,8 in 1-indexed = C3,C4,PO7,PO8)
COLOR_CHANNEL_WEIGHTS  = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

# ── SSVEP detection ───────────────────────────────────────────────────────────
SSVEP_WINDOW_SEC         = 4.0                          # analysis window (s)
SSVEP_WINDOW_SAMPLES     = int(SSVEP_WINDOW_SEC * SAMPLE_RATE)
SSVEP_SNR_THRESHOLD      = 3.5                          # min SNR to confirm
# One flickering frequency per canvas region (Hz) — one per circle
SSVEP_FREQUENCIES        = [8.0, 10.0, 12.0]

# ── Colour training & classification ─────────────────────────────────────────
TRAINING_COLORS = [
    {"name": "Red",    "rgb": (231,  76,  60)},
    {"name": "Yellow", "rgb": (241, 196,  15)},
    {"name": "Blue",   "rgb": (52,  152, 219)},
]
N_TRAINING_COLORS           = 3    # Red, Yellow, Blue
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
# Three circles: two on top, one bottom-centre.
# Each circle maps to one SSVEP frequency and one training colour.
# Geometry: normalised (x0, y0, x1, y1) bounding-box of the ellipse.
#
#  Layout (normalised coordinates, canvas 700 × 550):
#
#     [Circle 0 — 8 Hz]       [Circle 1 — 10 Hz]
#
#              [Circle 2 — 12 Hz]
#

CANVAS_REGIONS = [
    {
        "id": 0, "label": "Circle 1",
        "ssvep_freq": 8.0,
        "shape": "ellipse",
        "coords": (0.10, 0.08, 0.42, 0.50),   # top-left
        "default_color": (220, 220, 220),
    },
    {
        "id": 1, "label": "Circle 2",
        "ssvep_freq": 10.0,
        "shape": "ellipse",
        "coords": (0.58, 0.08, 0.90, 0.50),   # top-right
        "default_color": (220, 220, 220),
    },
    {
        "id": 2, "label": "Circle 3",
        "ssvep_freq": 12.0,
        "shape": "ellipse",
        "coords": (0.34, 0.55, 0.66, 0.97),   # bottom-centre
        "default_color": (220, 220, 220),
    },
]

# ── UI ────────────────────────────────────────────────────────────────────────
CANVAS_W = 700
CANVAS_H = 550
FLICKER_OVERLAY_ALPHA = 120   # 0-255  transparency of the flicker overlay
STARE_CONFIRM_SEC     = 4.0   # seconds of sustained SSVEP to confirm region
