"""
train.py — Multi-dataset Speech Emotion Recognition Training
Trains on RAVDESS + CREMA-D + TESS combined for better generalisation.

Datasets:
  RAVDESS  → https://zenodo.org/record/1188976
  CREMA-D  → https://github.com/CheyneyComputerScience/CREMA-D  (AudioWAV folder)
  TESS     → https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Set paths below then run:
    python train.py

Output: model/model.pkl, scaler.pkl, selector.pkl, emotion_labels.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import glob
import zipfile
import urllib.request
import joblib
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from collections import Counter

import librosa
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

from config import (MODEL_DIR, DATA_DIR, OUTPUTS_DIR,
                    EMOTION_LABELS, SAMPLE_RATE, DURATION_SEC)
from src.feature_extractor import EnhancedAudioFeatureExtractor

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# DATASET PATHS — edit these to match where you saved the datasets
# ─────────────────────────────────────────────────────────────────────

RAVDESS_DIR = DATA_DIR / "ravdess"          # existing
CREMAD_DIR  = Path(r"C:\Users\fares\Downloads\CREMA-D\AudioWAV")
TESS_DIR    = Path(r"C:\Users\fares\Downloads\TESS Toronto emotional speech set data")

# Set to False to skip a dataset if you haven't downloaded it yet
USE_RAVDESS = True
USE_CREMAD  = True
USE_TESS    = True


# ─────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────

# RAVDESS filename: 03-01-{code}-... emotion codes
RAVDESS_MAP = {
    "01": "neutral", "02": "calm",    "03": "happy", "04": "sad",
    "05": "angry",   "06": "fear",    "07": "disgust", "08": "surprise",
}

# CREMA-D filename: 1001_DFA_{CODE}_XX.wav
CREMAD_MAP = {
    "ANG": "angry",   "DIS": "disgust", "FEA": "fear",
    "HAP": "happy",   "NEU": "neutral", "SAD": "sad",
    # calm and surprise not in CREMA-D
}

# TESS folder names (e.g. OAF_angry, YAF_ps)
TESS_MAP = {
    "angry":   "angry",   "disgust": "disgust", "fear":    "fear",
    "happy":   "happy",   "neutral": "neutral", "sad":     "sad",
    "ps":      "surprise",  # "pleasant surprise"
    # calm not in TESS
}

# Emotion → integer index
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}


# ─────────────────────────────────────────────────────────────────────
# Augmenter
# ─────────────────────────────────────────────────────────────────────

class AudioAugmenter:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def _fix(self, audio):
        target = int(SAMPLE_RATE * DURATION_SEC)
        if len(audio) > target: return audio[:target]
        return np.pad(audio, (0, target - len(audio)))

    def augment(self, audio, sr, copies=3):
        base = self._fix(audio)
        out  = [base]
        fns  = ["noise", "pitch", "stretch", "gain"]
        for _ in range(copies):
            x  = base.copy()
            fn = self.rng.choice(fns)
            if fn == "noise":
                x = x + 0.003 * self.rng.standard_normal(len(x))
            elif fn == "pitch":
                steps = int(self.rng.choice([-2, -1, 1, 2]))
                x = librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)
            elif fn == "stretch":
                rate = float(self.rng.choice([0.9, 1.1]))
                x = librosa.effects.time_stretch(x, rate=rate)
            elif fn == "gain":
                x = x * float(self.rng.uniform(0.8, 1.2))
            out.append(self._fix(x))
        return out


# ─────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────

extractor = EnhancedAudioFeatureExtractor(sr=SAMPLE_RATE, duration=DURATION_SEC)
augmenter = AudioAugmenter()


def _extract(path, label_idx, augment_copies=3):
    """Load one file, augment, extract features. Returns list of (features, label)."""
    try:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION_SEC)
        versions  = augmenter.augment(audio, sr, copies=augment_copies)
        return [(extractor.extract_features(v, sr), label_idx) for v in versions]
    except Exception as e:
        return []


def load_ravdess(augment_copies=3) -> tuple[list, list]:
    print("\n📂 Loading RAVDESS…")
    files = glob.glob(str(RAVDESS_DIR / "**" / "*.wav"), recursive=True)
    if not files:
        print("   ⚠️  No files found — skipping RAVDESS")
        return [], []

    X, y = [], []
    for path in tqdm(files, desc="  RAVDESS"):
        parts = Path(path).stem.split("-")
        if len(parts) < 3: continue
        emotion = RAVDESS_MAP.get(parts[2])
        if emotion is None: continue
        for feat, lbl in _extract(path, EMOTION_TO_IDX[emotion], augment_copies):
            X.append(feat); y.append(lbl)

    print(f"   ✅ {len(X)} samples (after augmentation)")
    return X, y


def load_cremad(augment_copies=2) -> tuple[list, list]:
    print("\n📂 Loading CREMA-D…")
    if not CREMAD_DIR.exists():
        print(f"   ⚠️  Not found at {CREMAD_DIR} — skipping")
        return [], []

    files = glob.glob(str(CREMAD_DIR / "*.wav"))
    if not files:
        files = glob.glob(str(CREMAD_DIR / "**" / "*.wav"), recursive=True)
    if not files:
        print("   ⚠️  No .wav files found — skipping")
        return [], []

    X, y = [], []
    for path in tqdm(files, desc="  CREMA-D"):
        parts = Path(path).stem.split("_")
        if len(parts) < 3: continue
        emotion = CREMAD_MAP.get(parts[2].upper())
        if emotion is None: continue
        for feat, lbl in _extract(path, EMOTION_TO_IDX[emotion], augment_copies):
            X.append(feat); y.append(lbl)

    print(f"   ✅ {len(X)} samples (after augmentation)")
    return X, y


def load_tess(augment_copies=2) -> tuple[list, list]:
    print("\n📂 Loading TESS…")
    if not TESS_DIR.exists():
        print(f"   ⚠️  Not found at {TESS_DIR} — skipping")
        return [], []

    # TESS structure: TESS_DIR / OAF_angry / OAF_back_angry.wav
    # or:             TESS_DIR / angry / *.wav  (Kaggle version)
    files = glob.glob(str(TESS_DIR / "**" / "*.wav"), recursive=True)
    if not files:
        print("   ⚠️  No .wav files found — skipping")
        return [], []

    X, y = [], []
    for path in tqdm(files, desc="  TESS   "):
        # Emotion is the last part of the parent folder name
        # e.g. OAF_angry → angry, YAF_ps → ps
        folder = Path(path).parent.name.lower()
        emotion_key = folder.split("_")[-1]   # "angry" from "OAF_angry"
        emotion = TESS_MAP.get(emotion_key)
        if emotion is None:
            # Try matching folder directly
            emotion = TESS_MAP.get(folder)
        if emotion is None: continue
        for feat, lbl in _extract(path, EMOTION_TO_IDX[emotion], augment_copies):
            X.append(feat); y.append(lbl)

    print(f"   ✅ {len(X)} samples (after augmentation)")
    return X, y


# ─────────────────────────────────────────────────────────────────────
# Download RAVDESS if needed
# ─────────────────────────────────────────────────────────────────────

def maybe_download_ravdess():
    if any(RAVDESS_DIR.glob("**/*.wav")):
        print(f"✅ RAVDESS already present at {RAVDESS_DIR}")
        return
    url      = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = DATA_DIR / "ravdess.zip"
    print("📥 Downloading RAVDESS (~600 MB)…")
    urllib.request.urlretrieve(url, zip_path)
    print("📦 Extracting…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAVDESS_DIR)
    zip_path.unlink()
    print(f"✅ RAVDESS ready at {RAVDESS_DIR}")


# ─────────────────────────────────────────────────────────────────────
# Feature selection
# ─────────────────────────────────────────────────────────────────────

def select_features(X, y, k=100):
    print(f"\n🔍 SelectKBest (k={k})…")
    sel  = SelectKBest(f_classif, k=k)
    Xs   = sel.fit_transform(X, y)
    print(f"   {X.shape[1]} → {Xs.shape[1]} features")
    return Xs, sel


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaler
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # Class weights — fix imbalance (calm will be minority)
    classes      = np.unique(y_tr)
    weights      = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))
    print(f"\n⚖️  Class weights:")
    for idx, w in class_weight.items():
        print(f"   {EMOTION_LABELS[idx]:10s}: {w:.3f}")

    # SVM — fixed best params instead of GridSearch
    # C=10, rbf, scale consistently works best for SER tasks
    print("\n⏳ Training SVM (C=10, rbf, scale)…")
    best_svm = SVC(
        C=10, kernel="rbf", gamma="scale",
        probability=True, class_weight=class_weight, random_state=42
    )
    best_svm.fit(X_tr_sc, y_tr)
    print(f"   SVM accuracy: {accuracy_score(y_te, best_svm.predict(X_te_sc)):.4f}")

    # Ensemble
    print("\n⏳ Training Ensemble (SVM + RF)…")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight=class_weight, random_state=42, n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[("svm", best_svm), ("rf", rf)],
        voting="soft"
    )
    ensemble.fit(X_tr_sc, y_tr)

    # Evaluate both
    acc_svm = accuracy_score(y_te, best_svm.predict(X_te_sc))
    acc_ens = accuracy_score(y_te, ensemble.predict(X_te_sc))
    print(f"\n📊 SVM accuracy:      {acc_svm:.4f}")
    print(f"📊 Ensemble accuracy: {acc_ens:.4f}")

    final_model = ensemble if acc_ens >= acc_svm else best_svm
    y_pred      = final_model.predict(X_te_sc)

    print(f"\n✨ Final: {'Ensemble' if final_model is ensemble else 'SVM'}")
    print("\n" + classification_report(y_te, y_pred, target_names=EMOTION_LABELS))

    return final_model, scaler, y_te, y_pred


# ─────────────────────────────────────────────────────────────────────
# Plots + save
# ─────────────────────────────────────────────────────────────────────

def save_plots(y_te, y_pred, dataset_counts):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

    # Dataset contribution bar
    datasets = list(dataset_counts.keys())
    counts   = list(dataset_counts.values())
    colors   = ["#6366f1", "#22c55e", "#f59e0b"]
    axes[1].bar(datasets, counts, color=colors[:len(datasets)], alpha=0.85)
    axes[1].set_title("Training Samples per Dataset", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Sample count (after augmentation)")
    for i, (d, c) in enumerate(zip(datasets, counts)):
        axes[1].text(i, c + 50, str(c), ha="center", fontweight="bold")

    plt.tight_layout()
    path = OUTPUTS_DIR / "training_results.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved → {path}")


def save_artifacts(model, scaler, selector):
    joblib.dump(model,    MODEL_DIR / "model.pkl")
    joblib.dump(scaler,   MODEL_DIR / "scaler.pkl")
    if selector:
        joblib.dump(selector, MODEL_DIR / "selector.pkl")
    with open(MODEL_DIR / "emotion_labels.json", "w") as f:
        json.dump(EMOTION_LABELS, f)
    print(f"\n✅ Artifacts saved → {MODEL_DIR}")
    for p in MODEL_DIR.iterdir():
        print(f"   {p.name}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Dataset SER Training")
    print("  Datasets: RAVDESS + CREMA-D + TESS")
    print("=" * 60)

    # Download RAVDESS if needed
    if USE_RAVDESS:
        maybe_download_ravdess()

    # Load all datasets
    X_all, y_all = [], []
    dataset_counts = {}

    if USE_RAVDESS:
        Xr, yr = load_ravdess(augment_copies=3)
        X_all += Xr; y_all += yr
        dataset_counts["RAVDESS"] = len(Xr)

    if USE_CREMAD:
        Xc, yc = load_cremad(augment_copies=2)
        X_all += Xc; y_all += yc
        dataset_counts["CREMA-D"] = len(Xc)

    if USE_TESS:
        Xt, yt = load_tess(augment_copies=2)
        X_all += Xt; y_all += yt
        dataset_counts["TESS"] = len(Xt)

    if not X_all:
        print("\n❌ No data loaded. Check your dataset paths.")
        exit(1)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=int)

    print(f"\n📊 Combined dataset: {X.shape[0]} samples · {X.shape[1]} features")
    print("\nClass distribution:")
    counts = Counter(y)
    for idx, emo in enumerate(EMOTION_LABELS):
        bar = "█" * (counts.get(idx, 0) // 50)
        print(f"  {emo:10s}: {counts.get(idx,0):5d}  {bar}")

    # Feature selection
    X_sel, selector = select_features(X, y, k=100)

    # Train
    final_model, scaler, y_te, y_pred = train(X_sel, y)

    # Save plots
    save_plots(y_te, y_pred, dataset_counts)

    # Save artifacts
    save_artifacts(final_model, scaler, selector)

    print("\n🎉 Training complete!")
    print("   Drop the model/ folder into your app and restart.")