"""
config.py — Central path configuration for the Voice Emotion project.
All other scripts import from here so you only change paths in ONE place.
"""

from pathlib import Path

# ── Root of the project (the folder that contains this file) ──────────
ROOT = Path(__file__).parent

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR      = ROOT / "data"          # raw RAVDESS .wav files land here
RAVDESS_DIR   = DATA_DIR / "ravdess"   # extracted Actor_xx folders

# ── Saved model artifacts ─────────────────────────────────────────────
MODEL_DIR     = ROOT / "model"         # model.pkl, scaler.pkl, etc.

# ── Plots and reports ─────────────────────────────────────────────────
OUTPUTS_DIR   = ROOT / "outputs"

# ── Audio settings (must match the training notebook) ─────────────────
SAMPLE_RATE   = 44100
DURATION_SEC  = 3.0
N_MFCC        = 40
N_MELS        = 64

# ── Emotion labels (RAVDESS coding) ──────────────────────────────────
EMOTION_DICT = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fear
    "07": 6,  # disgust
    "08": 7,  # surprise
}

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry",   "fear", "disgust", "surprise"
]

EMOTION_EMOJI = {
    "neutral":  "😐",
    "calm":     "😌",
    "happy":    "😄",
    "sad":      "😢",
    "angry":    "😠",
    "fear":     "😨",
    "disgust":  "🤢",
    "surprise": "😲",
}

EMOTION_COLOR = {
    "neutral":  "#94a3b8",
    "calm":     "#67e8f9",
    "happy":    "#fbbf24",
    "sad":      "#60a5fa",
    "angry":    "#f87171",
    "fear":     "#c084fc",
    "disgust":  "#4ade80",
    "surprise": "#fb923c",
}

# ── Create dirs if they don't exist ───────────────────────────────────
for d in (DATA_DIR, RAVDESS_DIR, MODEL_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
