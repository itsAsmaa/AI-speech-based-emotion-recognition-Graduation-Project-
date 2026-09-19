"""
src/reporter.py
Session recorder + multi-format report generator with modality analysis.

Each record stores:
  - voice-only pair prediction
  - face top emotion (if detected)
  - fused pair prediction
  - whether face changed the top pair
  - confidence delta (fused - voice_only)

Reports answer the research question:
  "How much does face expression detection improve emotion pair recognition?"
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path
import threading
import io

from config import OUTPUTS_DIR, EMOTION_EMOJI, EMOTION_COLOR

# ─────────────────────────────────────────────────────────────────────
DARK_BG  = "#0f172a"
PANEL_BG = "#1e293b"
TEXT_COL = "#e2e8f0"
GRID_COL = "#334155"

ALL_EMOTIONS = ['neutral','calm','happy','sad','angry','fear','disgust','surprise']


# ─────────────────────────────────────────────────────────────────────
# Session Recorder
# ─────────────────────────────────────────────────────────────────────

class SessionRecorder:
    def __init__(self):
        self._lock    = threading.Lock()
        self._records = []
        self._start   = None
        self._active  = False

    def start(self):
        with self._lock:
            self._records = []
            self._start   = datetime.now()
            self._active  = True

    def stop(self):
        with self._lock:
            self._active = False

    def record(self,
               voice_pair:   tuple,
               voice_score:  float,
               voice_probs:  dict,
               fused_pair:   tuple,
               fused_score:  float,
               fused_probs:  dict,
               face_detected: bool,
               face_top:     str | None,
               face_score:   float | None,
               source:       str):
        if not self._active:
            return
        e1v, e2v = voice_pair
        e1f, e2f = fused_pair
        pair_changed = (voice_pair != fused_pair)
        conf_delta   = round(fused_score - voice_score, 4)

        row = {
            "timestamp_ms":      int((datetime.now() - self._start).total_seconds() * 1000),
            "source":            source,
            # Voice-only
            "voice_e1":          e1v,
            "voice_e2":          e2v,
            "voice_pair":        f"{e1v.capitalize()} – {e2v.capitalize()}",
            "voice_score":       round(voice_score, 4),
            # Face
            "face_detected":     face_detected,
            "face_top_emotion":  face_top or "—",
            "face_confidence":   round(face_score, 4) if face_score else 0.0,
            # Fused
            "fused_e1":          e1f,
            "fused_e2":          e2f,
            "fused_pair":        f"{e1f.capitalize()} – {e2f.capitalize()}",
            "fused_score":       round(fused_score, 4),
            # Research metrics
            "pair_changed_by_face": pair_changed,
            "confidence_delta":     conf_delta,
        }
        for emo in ALL_EMOTIONS:
            row[f"v_{emo}"] = round(voice_probs.get(emo, 0.0), 4)
            row[f"f_{emo}"] = round(fused_probs.get(emo, 0.0), 4)

        with self._lock:
            self._records.append(row)

    def get_dataframe(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(self._records) if self._records else pd.DataFrame()

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._records) == 0

    def get_summary(self) -> dict:
        df = self.get_dataframe()
        if df.empty:
            return {}

        n           = len(df)
        dur         = df["timestamp_ms"].max() / 1000
        face_rate   = df["face_detected"].mean() * 100
        changed     = df["pair_changed_by_face"].sum()
        changed_pct = changed / n * 100
        mean_delta  = df["confidence_delta"].mean() * 100
        std_delta   = df["confidence_delta"].std() * 100
        top_fused   = df["fused_pair"].value_counts()
        top_voice   = df["voice_pair"].value_counts()
        agree_rate  = (df["voice_pair"] == df["fused_pair"]).mean() * 100

        return {
            "session_start":          self._start.strftime("%Y-%m-%d %H:%M:%S") if self._start else "—",
            "duration_sec":           round(dur, 1),
            "total_predictions":      n,
            "face_detection_rate":    f"{face_rate:.1f}%",
            "voice_face_agreement":   f"{agree_rate:.1f}%",
            "face_changed_pair":      f"{changed} / {n}  ({changed_pct:.1f}%)",
            "mean_confidence_delta":  f"{mean_delta:+.2f}%",
            "std_confidence_delta":   f"{std_delta:.2f}%",
            "top_fused_pair":         top_fused.index[0] if len(top_fused) else "—",
            "top_voice_pair":         top_voice.index[0] if len(top_voice) else "—",
            "unique_fused_pairs":     df["fused_pair"].nunique(),
            "mean_voice_confidence":  f"{df['voice_score'].mean()*100:.1f}%",
            "mean_fused_confidence":  f"{df['fused_score'].mean()*100:.1f}%",
        }


# ─────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────

def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=8)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.5)


# ─────────────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────────────

def plot_pair_frequency(df, ax):
    counts = df["fused_pair"].value_counts().head(10)
    colors = [EMOTION_COLOR.get(row.split(" – ")[0].lower(), "#6366f1")
              for row in counts.index]
    ax.barh(counts.index[::-1], counts.values[::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Count", color=TEXT_COL)
    _style_ax(ax, "Fused Pair Frequency (top 10)")


def plot_confidence_comparison(df, ax):
    """Voice-only vs fused confidence over time."""
    t  = df["timestamp_ms"] / 1000
    v  = df["voice_score"] * 100
    fu = df["fused_score"] * 100
    ax.plot(t, v,  color="#818cf8", linewidth=1.2, alpha=0.7, label="Voice only")
    ax.plot(t, fu, color="#22c55e", linewidth=1.5, alpha=0.9, label="Fused (V+F)")
    ax.fill_between(t, v, fu, where=(fu >= v), alpha=0.15,
                    color="#22c55e", label="Face helped")
    ax.fill_between(t, v, fu, where=(fu < v),  alpha=0.15,
                    color="#f87171", label="Face hurt")
    ax.set_xlabel("Time (s)", color=TEXT_COL)
    ax.set_ylabel("Confidence (%)", color=TEXT_COL)
    ax.set_ylim(0, 100)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, fontsize=8)
    _style_ax(ax, "Voice-Only vs Fused Confidence Over Time")


def plot_face_impact(df, ax):
    """Bar: % predictions where face changed vs kept pair."""
    detected    = df["face_detected"].sum()
    not_detect  = len(df) - detected
    changed     = df["pair_changed_by_face"].sum()
    unchanged   = detected - changed

    labels = ["No face\ndetected", "Face detected\n(pair unchanged)", "Face detected\n(pair changed)"]
    values = [not_detect, unchanged, changed]
    colors = ["#475569", "#60a5fa", "#f472b6"]
    bars   = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}\n({pct:.1f}%)", ha="center", color=TEXT_COL, fontsize=8)
    ax.set_ylabel("Predictions", color=TEXT_COL)
    _style_ax(ax, "Face Detection Impact on Pair Prediction")


def plot_confidence_delta_hist(df, ax):
    """Distribution of confidence delta (fused - voice_only)."""
    delta = df["confidence_delta"] * 100
    pos   = delta[delta >= 0]
    neg   = delta[delta < 0]
    bins  = np.linspace(delta.min() - 1, delta.max() + 1, 30)
    ax.hist(pos, bins=bins, color="#22c55e", alpha=0.7, label="Face helped")
    ax.hist(neg, bins=bins, color="#f87171", alpha=0.7, label="Face hurt")
    ax.axvline(0, color=TEXT_COL, linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(delta.mean(), color="#fbbf24", linewidth=1.5,
               linestyle="--", label=f"Mean {delta.mean():+.2f}%")
    ax.set_xlabel("Confidence Delta (%)", color=TEXT_COL)
    ax.set_ylabel("Count", color=TEXT_COL)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, fontsize=8)
    _style_ax(ax, "Confidence Change Distribution (Fused − Voice Only)")


def plot_emotion_averages(df, ax):
    """Side-by-side: mean individual probs voice vs fused."""
    v_cols = [f"v_{e}" for e in ALL_EMOTIONS]
    f_cols = [f"f_{e}" for e in ALL_EMOTIONS]
    v_means = df[v_cols].mean().values * 100
    f_means = df[f_cols].mean().values * 100

    x     = np.arange(len(ALL_EMOTIONS))
    width = 0.38
    ax.bar(x - width/2, v_means, width, color="#818cf8", alpha=0.8, label="Voice only")
    ax.bar(x + width/2, f_means, width, color="#22c55e", alpha=0.8, label="Fused")
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in ALL_EMOTIONS],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Probability (%)", color=TEXT_COL)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, fontsize=8)
    _style_ax(ax, "Avg Emotion Probabilities: Voice Only vs Fused")


def plot_face_top_emotion(df, ax):
    """What face detected most often (when face was present)."""
    detected = df[df["face_detected"]]
    if detected.empty:
        ax.text(0.5, 0.5, "No face detections in session",
                ha="center", va="center", transform=ax.transAxes,
                color=TEXT_COL, fontsize=10)
        _style_ax(ax, "Face Top Emotion Distribution")
        return
    counts = detected["face_top_emotion"].value_counts()
    colors = [EMOTION_COLOR.get(e.lower(), "#6366f1") for e in counts.index]
    ax.bar(counts.index, counts.values, color=colors, alpha=0.85)
    ax.set_ylabel("Count", color=TEXT_COL)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _style_ax(ax, "Face Top Emotion Distribution")


# ─────────────────────────────────────────────────────────────────────
# Master figure
# ─────────────────────────────────────────────────────────────────────

def generate_figure(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, :])

    plot_pair_frequency(df, ax1)
    plot_confidence_comparison(df, ax2)
    plot_face_impact(df, ax3)
    plot_confidence_delta_hist(df, ax4)
    plot_face_top_emotion(df, ax5)
    plot_emotion_averages(df, ax6)

    fig.suptitle("Multimodal Voice + Face Emotion Analysis — Research Report",
                 color=TEXT_COL, fontsize=14, fontweight="bold", y=0.99)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Export functions
# ─────────────────────────────────────────────────────────────────────

def export_csv(recorder: SessionRecorder, path: Path) -> Path:
    recorder.get_dataframe().to_csv(path, index=False)
    print(f"✅  CSV → {path}")
    return path


def export_excel(recorder: SessionRecorder, path: Path) -> Path:
    df      = recorder.get_dataframe()
    summary = recorder.get_summary()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)

        pd.DataFrame(list(summary.items()),
                     columns=["Metric", "Value"]).to_excel(
            writer, sheet_name="Summary", index=False)

        # Fused pair frequency
        freq = df["fused_pair"].value_counts().reset_index()
        freq.columns = ["Pair", "Count"]
        freq["% of Session"] = (freq["Count"] / len(df) * 100).round(1)
        freq.to_excel(writer, sheet_name="Fused Pair Frequency", index=False)

        # Voice-only pair frequency
        vfreq = df["voice_pair"].value_counts().reset_index()
        vfreq.columns = ["Pair", "Count"]
        vfreq["% of Session"] = (vfreq["Count"] / len(df) * 100).round(1)
        vfreq.to_excel(writer, sheet_name="Voice-Only Pair Freq", index=False)

        # Face impact
        impact = pd.DataFrame({
            "Metric": ["Face detected", "Face NOT detected",
                       "Pair changed by face", "Pair unchanged"],
            "Count": [
                df["face_detected"].sum(),
                (~df["face_detected"]).sum(),
                df["pair_changed_by_face"].sum(),
                (~df["pair_changed_by_face"]).sum(),
            ]
        })
        impact["% of Total"] = (impact["Count"] / len(df) * 100).round(1)
        impact.to_excel(writer, sheet_name="Face Impact", index=False)

        # Confidence comparison
        conf = pd.DataFrame({
            "Metric":     ["Mean voice confidence", "Mean fused confidence",
                           "Mean delta", "Std delta"],
            "Value (%)":  [
                round(df["voice_score"].mean() * 100, 2),
                round(df["fused_score"].mean() * 100, 2),
                round(df["confidence_delta"].mean() * 100, 2),
                round(df["confidence_delta"].std() * 100, 2),
            ]
        })
        conf.to_excel(writer, sheet_name="Confidence Comparison", index=False)

    print(f"✅  Excel → {path}")
    return path


def export_pdf(recorder: SessionRecorder, path: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage,
                                    HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER

    df      = recorder.get_dataframe()
    summary = recorder.get_summary()

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    title_s = ParagraphStyle("t", fontSize=20, fontName="Helvetica-Bold",
                              alignment=TA_CENTER,
                              textColor=colors.HexColor("#6366f1"), spaceAfter=6)
    sub_s   = ParagraphStyle("s", fontSize=11, fontName="Helvetica",
                              alignment=TA_CENTER,
                              textColor=colors.HexColor("#64748b"), spaceAfter=20)
    h2_s    = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1e293b"),
                              spaceAfter=8, spaceBefore=14)
    foot_s  = ParagraphStyle("f", fontSize=8, alignment=TA_CENTER,
                              textColor=colors.HexColor("#94a3b8"))

    def make_table(data, col_widths):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#334155")),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f8fafc"),
                                               colors.HexColor("#f1f5f9")]),
            ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWHEIGHT",     (0,0), (-1,-1), 18),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        return t

    story = []

    # Cover
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("🎙️👁️ Multimodal Emotion Analysis", title_s),
        Paragraph("Voice + Face Research Report", sub_s),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 0.5*cm),
    ]

    # Session summary
    story.append(Paragraph("Session Summary", h2_s))
    sdata = [["Metric", "Value"]] + [[k, v] for k, v in summary.items()]
    story.append(make_table(sdata, [9*cm, 7*cm]))
    story.append(Spacer(1, 0.5*cm))

    # Key research finding
    story.append(Paragraph("Key Research Finding: Face Impact", h2_s))
    n = len(df)
    changed_n = df["pair_changed_by_face"].sum()
    face_rate = df["face_detected"].mean() * 100
    mean_delta = df["confidence_delta"].mean() * 100
    finding_data = [
        ["Metric", "Value", "Interpretation"],
        ["Face detection rate",
         f"{face_rate:.1f}%",
         "How often face was visible during session"],
        ["Pairs changed by face",
         f"{changed_n}/{n} ({changed_n/n*100:.1f}%)",
         "Predictions where face altered the top pair"],
        ["Mean confidence delta",
         f"{mean_delta:+.2f}%",
         "Average confidence change from fusion"],
        ["Voice–face agreement",
         summary.get("voice_face_agreement","—"),
         "How often V and F agreed on top pair"],
    ]
    story.append(make_table(finding_data, [5.5*cm, 4*cm, 7*cm]))
    story.append(PageBreak())

    # Charts
    story.append(Paragraph("Visualisations", h2_s))
    fig = generate_figure(df)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    buf.seek(0)
    plt.close(fig)
    story.append(RLImage(buf, width=16*cm, height=14*cm))

    # Footer
    story += [
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 0.2*cm),
        Paragraph(
            f"Generated by Multimodal Emotion Analyser · "
            f"Birzeit University · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            foot_s
        ),
    ]

    doc.build(story)
    print(f"✅  PDF → {path}")
    return path