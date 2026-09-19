"""
validate.py — Model Validation on Synthetic Emotion Pair Clips

Runs all .mp3/.wav files in emotion_pairs_synthetic through the trained model
and computes:
  - Exact pair match accuracy (model predicts the exact same pair)
  - Partial match accuracy  (at least 1 emotion in the pair is correct)
  - Per-pair breakdown table
  - Top-3 confusion pairs
  - Confidence statistics
  - Exports: CSV raw results + PDF report + Excel summary

Usage:
    python validate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
import json
import glob
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from itertools import combinations
from collections import defaultdict, Counter
from datetime import datetime

from config import MODEL_DIR, OUTPUTS_DIR, EMOTION_LABELS, EMOTION_COLOR, EMOTION_EMOJI, SAMPLE_RATE, DURATION_SEC
from src.feature_extractor import EnhancedAudioFeatureExtractor
from src.preprocessor      import AudioPreprocessor

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

PAIRS_DIR = Path(r"C:\Users\fares\Desktop\GRADUATION PROJECT ENCS5300\emotion_pairs_synthetic")

ALL_PAIRS   = list(combinations(
    ['neutral','calm','happy','sad','angry','fear','disgust','surprise'], 2))
PAIR_LOOKUP = {frozenset(p): p for p in ALL_PAIRS}

DARK_BG  = "#0f172a"
PANEL_BG = "#1e293b"
TEXT_COL = "#e2e8f0"
GRID_COL = "#334155"


# ─────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────

def load_artifacts():
    model    = joblib.load(MODEL_DIR / "model.pkl")
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    sel_path = MODEL_DIR / "selector.pkl"
    selector = joblib.load(sel_path) if sel_path.exists() else None
    with open(MODEL_DIR / "emotion_labels.json") as f:
        labels = json.load(f)
    print(f"✅  Model    → {type(model).__name__}")
    print(f"    Emotions → {labels}")
    return model, scaler, selector, labels


# ─────────────────────────────────────────────────────────────────────
# Inference pipeline
# ─────────────────────────────────────────────────────────────────────

extractor    = EnhancedAudioFeatureExtractor(sr=SAMPLE_RATE, duration=DURATION_SEC)
preprocessor = AudioPreprocessor(sr=SAMPLE_RATE)

CALIBRATION_WEIGHTS = {
    'neutral': 1.20, 'calm': 1.10, 'happy': 1.00, 'sad': 1.00,
    'angry':   0.90, 'fear': 0.90, 'disgust': 0.40, 'surprise': 0.95,
}

def calibrate(raw: dict) -> dict:
    weighted = {k: v * CALIBRATION_WEIGHTS.get(k, 1.0) for k, v in raw.items()}
    total    = sum(weighted.values()) or 1
    normed   = {k: v/total for k, v in weighted.items()}
    thresh   = {k: v if v >= 0.10 else 0.0 for k, v in normed.items()}
    if sum(thresh.values()) == 0:
        thresh = normed
    total2 = sum(thresh.values()) or 1
    return {k: v/total2 for k, v in thresh.items()}

def predict(audio_path, model, scaler, selector, labels):
    """Returns (pred_pair, pred_score, all_pair_probs, raw_probs) or None."""
    try:
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=DURATION_SEC)
        if len(audio) < int(SAMPLE_RATE * 0.5):
            return None
        # Apply pre-emphasis only (no VAD for validation files)
        audio    = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)
        features = extractor.extract_features(audio, SAMPLE_RATE).reshape(1, -1)
        if selector is not None:
            features = selector.transform(features)
        fs    = scaler.transform(features)
        probs = model.predict_proba(fs)[0]
        raw   = {labels[i]: float(probs[i]) for i in range(len(labels))}
        cal   = calibrate(raw)

        ranked  = sorted(cal.items(), key=lambda x: x[1], reverse=True)
        e1, e2  = ranked[0][0], ranked[1][0]
        pair    = PAIR_LOOKUP.get(frozenset([e1, e2]), (e1, e2))

        pair_probs = {(pa, pb): cal.get(pa,0)+cal.get(pb,0) for pa,pb in ALL_PAIRS}
        total      = sum(pair_probs.values()) or 1
        pair_probs = {k: v/total for k, v in pair_probs.items()}

        return pair, pair_probs[pair], pair_probs, cal
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────
# Run validation
# ─────────────────────────────────────────────────────────────────────

def run_validation(model, scaler, selector, labels):
    print(f"\n🔍 Scanning: {PAIRS_DIR}")
    audio_files = (
        glob.glob(str(PAIRS_DIR / "**" / "*.mp3"), recursive=True) +
        glob.glob(str(PAIRS_DIR / "**" / "*.wav"), recursive=True)
    )
    if not audio_files:
        print("❌ No audio files found. Check PAIRS_DIR in config.")
        return pd.DataFrame()

    print(f"   Found {len(audio_files)} clips across all pairs\n")

    records = []
    errors  = 0

    for path in audio_files:
        path       = Path(path)
        folder     = path.parent.name          # e.g. "happy_sad"
        parts      = folder.split("_", 1)
        if len(parts) != 2:
            continue
        true_e1, true_e2 = parts[0], parts[1]
        true_pair = PAIR_LOOKUP.get(frozenset([true_e1, true_e2]))
        if true_pair is None:
            continue

        result = predict(path, model, scaler, selector, labels)
        if result is None:
            errors += 1
            continue

        pred_pair, pred_score, pair_probs, raw_probs = result

        exact_match   = (pred_pair == true_pair)
        partial_match = (pred_pair[0] in true_pair or pred_pair[1] in true_pair)

        # Top-3 pairs
        top3 = sorted(pair_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_pairs = [p for p, _ in top3]
        in_top3 = true_pair in top3_pairs

        records.append({
            "file":           path.name,
            "true_pair":      f"{true_e1.capitalize()} – {true_e2.capitalize()}",
            "true_e1":        true_e1,
            "true_e2":        true_e2,
            "pred_pair":      f"{pred_pair[0].capitalize()} – {pred_pair[1].capitalize()}",
            "pred_e1":        pred_pair[0],
            "pred_e2":        pred_pair[1],
            "confidence":     round(pred_score, 4),
            "exact_match":    exact_match,
            "partial_match":  partial_match,
            "in_top3":        in_top3,
            **{f"p_{e}": round(raw_probs.get(e, 0), 4) for e in labels}
        })

        status = "✅" if exact_match else ("🟡" if partial_match else "❌")
        print(f"  {status}  {true_e1:8s}–{true_e2:8s}  →  "
              f"{pred_pair[0]:8s}–{pred_pair[1]:8s}  "
              f"({pred_score*100:.0f}%)")

    df = pd.DataFrame(records)
    print(f"\n   {len(df)} clips processed  |  {errors} errors")
    return df


# ─────────────────────────────────────────────────────────────────────
# Compute metrics
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "total_clips":         n,
        "exact_match_acc":     f"{df['exact_match'].mean()*100:.1f}%",
        "partial_match_acc":   f"{df['partial_match'].mean()*100:.1f}%",
        "top3_acc":            f"{df['in_top3'].mean()*100:.1f}%",
        "mean_confidence":     f"{df['confidence'].mean()*100:.1f}%",
        "std_confidence":      f"{df['confidence'].std()*100:.1f}%",
        "best_pair":           df.groupby("true_pair")["exact_match"].mean().idxmax(),
        "worst_pair":          df.groupby("true_pair")["exact_match"].mean().idxmin(),
        "validation_date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_type":          "VotingClassifier (SVM + RF)",
        "training_datasets":   "RAVDESS + CREMA-D + TESS",
    }


# ─────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────

def _style(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor(GRID_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    if title: ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=8)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.5)


def plot_per_pair_accuracy(df, ax):
    acc = df.groupby("true_pair")["exact_match"].mean().sort_values() * 100
    colors = ["#22c55e" if v >= 50 else "#f87171" for v in acc.values]
    ax.barh(acc.index, acc.values, color=colors, alpha=0.85)
    ax.axvline(acc.mean(), color="#fbbf24", linestyle="--", linewidth=1.5,
               label=f"Mean {acc.mean():.1f}%")
    ax.set_xlabel("Exact Match Accuracy (%)", color=TEXT_COL)
    ax.set_xlim(0, 105)
    for i, v in enumerate(acc.values):
        ax.text(v+1, i, f"{v:.0f}%", va="center", color=TEXT_COL, fontsize=7)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)
    _style(ax, "Per-Pair Exact Match Accuracy")


def plot_match_breakdown(df, ax):
    exact   = df["exact_match"].sum()
    partial = (df["partial_match"] & ~df["exact_match"]).sum()
    wrong   = (~df["partial_match"]).sum()
    n       = len(df)
    labels  = ["Exact match", "Partial match\n(1 emotion correct)", "Wrong"]
    values  = [exact, partial, wrong]
    colors  = ["#22c55e", "#fbbf24", "#f87171"]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        textprops={"color": TEXT_COL, "fontsize": 8}
    )
    for at in autotexts: at.set_color(TEXT_COL); at.set_fontsize(9)
    ax.set_facecolor(PANEL_BG)
    _style(ax, "Prediction Breakdown")


def plot_confidence_by_correctness(df, ax):
    correct   = df[df["exact_match"]]["confidence"] * 100
    incorrect = df[~df["exact_match"]]["confidence"] * 100
    bins = np.linspace(0, 100, 25)
    ax.hist(correct,   bins=bins, color="#22c55e", alpha=0.7, label=f"Correct ({len(correct)})")
    ax.hist(incorrect, bins=bins, color="#f87171", alpha=0.7, label=f"Wrong ({len(incorrect)})")
    ax.set_xlabel("Confidence (%)", color=TEXT_COL)
    ax.set_ylabel("Count", color=TEXT_COL)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)
    _style(ax, "Confidence Distribution: Correct vs Wrong")


def plot_confusion_emotions(df, ax):
    """How often each true emotion appears in wrong predictions."""
    wrong = df[~df["exact_match"]]
    if wrong.empty:
        ax.text(0.5, 0.5, "No wrong predictions!", ha="center",
                va="center", transform=ax.transAxes, color=TEXT_COL)
        _style(ax, "Most Confused Emotions")
        return
    # Count how often each true emotion was confused
    e1_counts = Counter(wrong["true_e1"].tolist() + wrong["true_e2"].tolist())
    emos   = list(e1_counts.keys())
    counts = list(e1_counts.values())
    colors = [EMOTION_COLOR.get(e, "#6366f1") for e in emos]
    ax.bar(emos, counts, color=colors, alpha=0.85)
    ax.set_ylabel("Times confused", color=TEXT_COL)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _style(ax, "Which Emotions Are Most Often Confused?")


def plot_top_misclassifications(df, ax):
    wrong = df[~df["exact_match"]][["true_pair","pred_pair"]].copy()
    if wrong.empty:
        ax.text(0.5, 0.5, "No misclassifications!", ha="center",
                va="center", transform=ax.transAxes, color=TEXT_COL)
        _style(ax, "Top Misclassifications")
        return
    pairs = wrong.groupby(["true_pair","pred_pair"]).size().reset_index(name="count")
    pairs = pairs.sort_values("count", ascending=False).head(10)
    labels_y = [f"{r.true_pair}\n→ {r.pred_pair}" for _, r in pairs.iterrows()]
    ax.barh(labels_y[::-1], pairs["count"].values[::-1],
            color="#f87171", alpha=0.8)
    ax.set_xlabel("Count", color=TEXT_COL)
    _style(ax, "Top 10 Misclassification Patterns")


def generate_figure(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(20, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    ax1 = fig.add_subplot(gs[:, 0])   # full left column — per pair accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    plot_per_pair_accuracy(df, ax1)
    plot_match_breakdown(df, ax2)
    plot_confidence_by_correctness(df, ax3)
    plot_confusion_emotions(df, ax4)
    plot_top_misclassifications(df, ax5)

    fig.suptitle("Emotion Pair Model Validation Report",
                 color=TEXT_COL, fontsize=15, fontweight="bold", y=0.99)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────

def export_csv(df, path):
    df.to_csv(path, index=False)
    print(f"✅ CSV  → {path}")


def export_excel(df, metrics, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Results", index=False)

        pd.DataFrame(list(metrics.items()),
                     columns=["Metric","Value"]).to_excel(
            writer, sheet_name="Summary", index=False)

        per_pair = df.groupby("true_pair").agg(
            clips       = ("exact_match", "count"),
            exact_acc   = ("exact_match",  lambda x: f"{x.mean()*100:.1f}%"),
            partial_acc = ("partial_match", lambda x: f"{x.mean()*100:.1f}%"),
            top3_acc    = ("in_top3",       lambda x: f"{x.mean()*100:.1f}%"),
            mean_conf   = ("confidence",    lambda x: f"{x.mean()*100:.1f}%"),
        ).reset_index()
        per_pair.to_excel(writer, sheet_name="Per-Pair Accuracy", index=False)

    print(f"✅ Excel → {path}")


def export_pdf(df, metrics, path):
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage,
                                    HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    title_s = ParagraphStyle("t", fontSize=20, fontName="Helvetica-Bold",
                              alignment=TA_CENTER,
                              textColor=colors.HexColor("#6366f1"), spaceAfter=6)
    sub_s   = ParagraphStyle("s", fontSize=11, alignment=TA_CENTER,
                              textColor=colors.HexColor("#64748b"), spaceAfter=20)
    h2_s    = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1e293b"),
                              spaceAfter=8, spaceBefore=14)
    foot_s  = ParagraphStyle("f", fontSize=8, alignment=TA_CENTER,
                              textColor=colors.HexColor("#94a3b8"))

    def tbl(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#334155")),
            ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),
                                             colors.HexColor("#f1f5f9")]),
            ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWHEIGHT",     (0,0),(-1,-1), 16),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        return t

    story = [
        Spacer(1, 1*cm),
        Paragraph("🎭 Emotion Pair Model Validation", title_s),
        Paragraph("RAVDESS + CREMA-D + TESS Trained Model · Synthetic TTS Validation Set", sub_s),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 0.4*cm),
    ]

    # Summary table
    story.append(Paragraph("Validation Summary", h2_s))
    sdata = [["Metric","Value"]] + [[k,v] for k,v in metrics.items()]
    story.append(tbl(sdata, [9*cm, 7*cm]))
    story.append(Spacer(1, 0.5*cm))

    # Per-pair table
    story.append(Paragraph("Per-Pair Accuracy", h2_s))
    per_pair = df.groupby("true_pair").agg(
        clips       = ("exact_match", "count"),
        exact_acc   = ("exact_match",  lambda x: f"{x.mean()*100:.1f}%"),
        partial_acc = ("partial_match", lambda x: f"{x.mean()*100:.1f}%"),
        top3_acc    = ("in_top3",       lambda x: f"{x.mean()*100:.1f}%"),
        mean_conf   = ("confidence",    lambda x: f"{x.mean()*100:.1f}%"),
    ).reset_index().sort_values("exact_acc", ascending=False)

    pdata = [["Pair","Clips","Exact","Partial","Top-3","Conf"]]
    for _, r in per_pair.iterrows():
        pdata.append([r.true_pair, str(r.clips), r.exact_acc,
                      r.partial_acc, r.top3_acc, r.mean_conf])
    story.append(tbl(pdata, [5.5*cm,1.5*cm,2*cm,2*cm,2*cm,2*cm]))
    story.append(PageBreak())

    # Charts
    story.append(Paragraph("Visualisations", h2_s))
    fig = generate_figure(df)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    buf.seek(0); plt.close(fig)
    story.append(RLImage(buf, width=16*cm, height=13*cm))
    story += [
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 0.2*cm),
        Paragraph(f"Generated · Birzeit University · {datetime.now().strftime('%Y-%m-%d %H:%M')}", foot_s),
    ]
    doc.build(story)
    print(f"✅ PDF  → {path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Emotion Pair Model Validation")
    print("=" * 60)

    model, scaler, selector, labels = load_artifacts()
    df = run_validation(model, scaler, selector, labels)

    if df.empty:
        print("❌ No results to report.")
        exit(1)

    metrics = compute_metrics(df)

    print("\n" + "=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    # Export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    export_csv(df,                   OUTPUTS_DIR / f"validation_{ts}.csv")
    export_excel(df, metrics,        OUTPUTS_DIR / f"validation_{ts}.xlsx")
    export_pdf(df, metrics,          OUTPUTS_DIR / f"validation_report_{ts}.pdf")

    print(f"\n📁 All exports saved to: {OUTPUTS_DIR}")
    print("\n🎉 Validation complete!")
