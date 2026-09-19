"""
realtime_app.py — Multimodal Voice + Face Emotion Pair Analyser
Three tabs: 🎤 Live Mic (voice+webcam fused) | 📁 File | 📊 Report
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib, json, threading, time, collections
import gradio as gr
import sounddevice as sd
from pathlib import Path
from itertools import combinations
from datetime import datetime

from config import (MODEL_DIR, OUTPUTS_DIR, EMOTION_LABELS,
                    EMOTION_EMOJI, EMOTION_COLOR, SAMPLE_RATE, DURATION_SEC)
from src.feature_extractor import EnhancedAudioFeatureExtractor
from src.preprocessor      import AudioPreprocessor
from src.face_analyser     import FaceEmotionAnalyser
from src.fusion            import fuse
from src.reporter          import (SessionRecorder, generate_figure,
                                   export_csv, export_excel, export_pdf)


# ─────────────────────────────────────────────────────────────────────
# Pair definitions
# ─────────────────────────────────────────────────────────────────────

ALL_PAIRS   = list(combinations(
    ['neutral','calm','happy','sad','angry','fear','disgust','surprise'], 2))
PAIR_LOOKUP = {frozenset(p): p for p in ALL_PAIRS}


# ─────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────

def load_artifacts():
    required = ["model.pkl", "scaler.pkl", "emotion_labels.json"]
    missing  = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}")
    model    = joblib.load(MODEL_DIR / "model.pkl")
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    sel_path = MODEL_DIR / "selector.pkl"
    selector = joblib.load(sel_path) if sel_path.exists() else None
    with open(MODEL_DIR / "emotion_labels.json") as f:
        labels = json.load(f)
    print(f"✅  Model → {type(model).__name__}  |  Emotions → {labels}")
    return model, scaler, selector, labels


# ─────────────────────────────────────────────────────────────────────
# Pair scoring from a prob dict
# ─────────────────────────────────────────────────────────────────────

def get_top_pair(prob_dict: dict):
    ranked     = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    e1, _      = ranked[0]
    e2, _      = ranked[1]
    pair       = PAIR_LOOKUP.get(frozenset([e1, e2]), (e1, e2))
    pair_probs = {(pa, pb): prob_dict.get(pa,0)+prob_dict.get(pb,0)
                  for pa, pb in ALL_PAIRS}
    total      = sum(pair_probs.values()) or 1
    pair_probs = {k: v/total for k, v in pair_probs.items()}
    return pair, pair_probs[pair], pair_probs


# ─────────────────────────────────────────────────────────────────────
# Voice inference
# ─────────────────────────────────────────────────────────────────────

_extractor    = EnhancedAudioFeatureExtractor(sr=SAMPLE_RATE, duration=DURATION_SEC)
_preprocessor = AudioPreprocessor(sr=SAMPLE_RATE)


def run_voice_inference(audio, model, scaler, selector, labels, preprocess=True):
    """Returns raw prob_dict or None if silence."""
    if preprocess:
        audio, is_speech, _ = _preprocessor.process(audio)
        if not is_speech:
            return None
    features = _extractor.extract_features(audio, SAMPLE_RATE).reshape(1, -1)
    if selector is not None:
        features = selector.transform(features)
    fs    = scaler.transform(features)
    probs = model.predict_proba(fs)[0]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


# ─────────────────────────────────────────────────────────────────────
# Continuous Listener
# ─────────────────────────────────────────────────────────────────────

class ContinuousListener:
    PREDICT_EVERY_SEC = 1.0
    BUFFER_SEC        = DURATION_SEC

    def __init__(self, model, scaler, selector, labels,
                 face_analyser: FaceEmotionAnalyser,
                 recorder: SessionRecorder):
        self.model = model; self.scaler = scaler
        self.selector = selector; self.labels = labels
        self.face_analyser = face_analyser
        self.recorder = recorder
        self.buffer_size = int(self.BUFFER_SEC * SAMPLE_RATE)
        self.buffer   = collections.deque(maxlen=self.buffer_size)
        self._lock    = threading.Lock()
        self._result  = None   # (voice_pair, v_score, v_pair_probs,
                               #  fused_pair, f_score, f_pair_probs, f_raw, source)
        self._status  = "idle"
        self._error   = ""
        self._stream  = None
        self._thread  = None
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self.buffer.extend(mono.tolist())

    def _predict_loop(self):
        while self._running:
            time.sleep(self.PREDICT_EVERY_SEC)
            if not self._running or len(self.buffer) < int(SAMPLE_RATE * 1.0):
                continue
            audio = np.array(list(self.buffer), dtype=np.float32)
            try:
                # Voice
                voice_raw = run_voice_inference(
                    audio, self.model, self.scaler,
                    self.selector, self.labels, preprocess=True)

                if voice_raw is None:
                    with self._lock:
                        self._status = "silence"
                    print("  🔇  silence")
                    continue

                # Face
                face_res    = self.face_analyser.get_result()
                face_raw    = face_res[2] if face_res else None   # prob_dict
                face_top    = face_res[0] if face_res else None
                face_score  = face_res[1] if face_res else None
                face_detected = face_raw is not None

                # Fusion
                fusion_out = fuse(voice_raw, face_raw)
                fused_raw  = fusion_out['fused_probs']
                source     = fusion_out['source']

                # Pairs
                v_pair, v_score, v_pair_probs = get_top_pair(voice_raw)
                f_pair, f_score, f_pair_probs = get_top_pair(fused_raw)

                # Record
                self.recorder.record(
                    voice_pair=v_pair, voice_score=v_score, voice_probs=voice_raw,
                    fused_pair=f_pair, fused_score=f_score, fused_probs=fused_raw,
                    face_detected=face_detected, face_top=face_top,
                    face_score=face_score, source=source
                )

                with self._lock:
                    self._result = (v_pair, v_score, v_pair_probs,
                                    f_pair, f_score, f_pair_probs,
                                    fused_raw, source)
                    self._status = "listening"

                print(f"  🎤 {v_pair[0]}-{v_pair[1]}  👁️ {face_top or '—'}  "
                      f"→ {f_pair[0]}-{f_pair[1]}  [{source}]")

            except Exception as e:
                with self._lock:
                    self._status = "error"; self._error = str(e)

    def start(self):
        if self._running: return
        self.buffer.clear(); self._result = None
        self._running = True; self._status = "listening"; self._error = ""
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", blocksize=int(SAMPLE_RATE * 0.1),
            callback=self._audio_callback)
        self._stream.start()
        self._thread = threading.Thread(target=self._predict_loop, daemon=True)
        self._thread.start()
        print("🎤  Audio stream started")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None
        self._status = "idle"; self._result = None
        print("⏹️  Stream stopped")

    def get_result(self):
        with self._lock:
            return self._status, self._result, self._error


# ─────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────

PULSE_CSS = """<style>
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}
</style>"""

def _pair_color(e1, e2):
    def h2r(h):
        h=h.lstrip('#'); return tuple(int(h[i:i+2],16) for i in(0,2,4))
    c1=h2r(EMOTION_COLOR.get(e1,"#6366f1")); c2=h2r(EMOTION_COLOR.get(e2,"#818cf8"))
    return "#{:02x}{:02x}{:02x}".format(*((a+b)//2 for a,b in zip(c1,c2)))

def _pair_bars(pair_probs, top_pair, max_show=8):
    bars=""
    for (pa,pb),p in sorted(pair_probs.items(), key=lambda x:x[1], reverse=True)[:max_show]:
        bold="700" if (pa,pb)==top_pair else "400"
        col=_pair_color(pa,pb)
        label=f"{EMOTION_EMOJI.get(pa,'')} {pa.capitalize()} – {EMOTION_EMOJI.get(pb,'')} {pb.capitalize()}"
        bars+=f"""
        <div style="margin-bottom:7px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;
                      font-weight:{bold};color:#e2e8f0;">
            <span>{label}</span><span>{p*100:.1f}%</span>
          </div>
          <div style="background:#1e293b;border-radius:5px;height:8px;margin-top:3px;overflow:hidden;">
            <div style="width:{p*100:.1f}%;background:{col};height:8px;
                        border-radius:5px;transition:width 0.6s ease;"></div>
          </div>
        </div>"""
    return bars

def _source_badge(source):
    badges = {
        "voice+face": ('<span style="background:#14532d;color:#86efac;font-size:11px;'
                       'font-weight:700;padding:2px 9px;border-radius:20px;">🎤+👁️ FUSED</span>'),
        "voice_only": ('<span style="background:#1e3a5f;color:#93c5fd;font-size:11px;'
                       'font-weight:700;padding:2px 9px;border-radius:20px;">🎤 VOICE ONLY</span>'),
        "face_only":  ('<span style="background:#2d1f5e;color:#c4b5fd;font-size:11px;'
                       'font-weight:700;padding:2px 9px;border-radius:20px;">👁️ FACE ONLY</span>'),
    }
    return badges.get(source, "")

def idle_html():
    return f"""{PULSE_CSS}
    <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:18px;
                padding:40px 30px;color:#94a3b8;text-align:center;box-shadow:0 8px 40px rgba(0,0,0,.5);">
      <div style="font-size:52px;margin-bottom:14px;">🎙️👁️</div>
      <div style="font-size:18px;font-weight:600;color:#cbd5e1;">
        Press <span style="color:#818cf8;">▶ Start</span> to begin
      </div>
      <div style="font-size:13px;margin-top:10px;color:#475569;">
        Voice + Face fusion · 28 emotion pairs · session recorded for research export
      </div>
    </div>"""

def warming_html():
    return f"""{PULSE_CSS}
    <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:18px;
                padding:40px 30px;color:#94a3b8;text-align:center;box-shadow:0 8px 40px rgba(0,0,0,.5);">
      <div style="font-size:52px;margin-bottom:14px;animation:pulse 1.2s infinite;">🎤</div>
      <div style="font-size:18px;font-weight:600;color:#cbd5e1;">Listening…</div>
      <div style="font-size:13px;margin-top:10px;color:#475569;">Speak and face the webcam · first result in ~3s</div>
    </div>"""

def silence_html(last_result=None):
    stale=""
    if last_result:
        e1,e2=last_result[3]
        stale=(f'<div style="font-size:12px;color:#475569;margin-top:8px;">'
               f'Last: {EMOTION_EMOJI.get(e1,"")} {e1} – {EMOTION_EMOJI.get(e2,"")} {e2}</div>')
    return f"""{PULSE_CSS}
    <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:18px;
                padding:40px 30px;color:#94a3b8;text-align:center;box-shadow:0 8px 40px rgba(0,0,0,.5);">
      <div style="font-size:52px;margin-bottom:14px;animation:pulse 2s infinite;">🔇</div>
      <div style="font-size:18px;font-weight:600;color:#cbd5e1;">No speech detected</div>
      {stale}
    </div>"""

def waiting_html():
    return f"""{PULSE_CSS}
    <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:18px;
                padding:40px 30px;color:#94a3b8;text-align:center;box-shadow:0 8px 40px rgba(0,0,0,.5);">
      <div style="font-size:52px;margin-bottom:14px;">📂</div>
      <div style="font-size:18px;font-weight:600;color:#cbd5e1;">Upload an audio file to analyse</div>
      <div style="font-size:13px;margin-top:8px;color:#475569;">.wav · .mp3 · .flac · .ogg</div>
    </div>"""

def error_html(msg):
    return f"""{PULSE_CSS}
    <div style="font-family:monospace;background:#0f172a;border-radius:18px;
                padding:24px;color:#f87171;box-shadow:0 8px 40px rgba(0,0,0,.5);">❌ {msg}</div>"""

def result_html(v_pair, v_score, v_pair_probs,
                f_pair, f_score, f_pair_probs,
                fused_raw, source, is_live=False):
    e1,e2   = f_pair
    color   = _pair_color(e1,e2)
    c1,c2   = EMOTION_COLOR.get(e1,"#6366f1"), EMOTION_COLOR.get(e2,"#818cf8")

    live_dot = (
        '<span style="display:inline-flex;align-items:center;gap:5px;color:#22c55e;'
        'font-weight:700;font-size:11px;"><span style="width:8px;height:8px;border-radius:50%;'
        'background:#22c55e;display:inline-block;animation:pulse 1s infinite;"></span>LIVE</span>'
    ) if is_live else '<span style="color:#818cf8;font-size:11px;font-weight:700;">📁 FILE</span>'

    # Voice-only mini card
    ve1,ve2 = v_pair
    vc1,vc2 = EMOTION_COLOR.get(ve1,"#6366f1"), EMOTION_COLOR.get(ve2,"#818cf8")
    changed_badge = ""
    if v_pair != f_pair:
        changed_badge = ('<span style="background:#7c3aed;color:#ddd6fe;font-size:10px;'
                         'padding:1px 7px;border-radius:10px;margin-left:6px;">changed by face</span>')

    # Top-4 individual probs
    top_indiv = sorted(fused_raw.items(), key=lambda x:x[1], reverse=True)[:4]
    indiv=""
    for emo,p in top_indiv:
        bc=EMOTION_COLOR.get(emo,"#6366f1")
        bold="700" if emo in(e1,e2) else "400"
        indiv+=f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
          <span style="font-size:12px;font-weight:{bold};color:#e2e8f0;width:82px;">
            {EMOTION_EMOJI.get(emo,'')} {emo.capitalize()}</span>
          <div style="flex:1;background:#1e293b;border-radius:4px;height:7px;overflow:hidden;">
            <div style="width:{p*100:.1f}%;background:{bc};height:7px;border-radius:4px;
                        transition:width 0.5s ease;"></div>
          </div>
          <span style="font-size:11px;color:#64748b;width:34px;text-align:right;">{p*100:.0f}%</span>
        </div>"""

    return f"""{PULSE_CSS}
    <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:18px;
                padding:24px 26px;color:#f1f5f9;box-shadow:0 8px 40px rgba(0,0,0,.5);animation:fadeIn 0.3s ease;">

      <!-- Fused result -->
      <div style="text-align:center;margin-bottom:16px;">
        <div style="font-size:56px;line-height:1;letter-spacing:8px;">
          {EMOTION_EMOJI.get(e1,'')}{EMOTION_EMOJI.get(e2,'')}</div>
        <div style="margin-top:10px;display:flex;align-items:center;
                    justify-content:center;gap:10px;flex-wrap:wrap;">
          <span style="font-size:20px;font-weight:800;color:{c1};
                       text-transform:uppercase;letter-spacing:2px;">{e1.capitalize()}</span>
          <span style="font-size:16px;color:#475569;">–</span>
          <span style="font-size:20px;font-weight:800;color:{c2};
                       text-transform:uppercase;letter-spacing:2px;">{e2.capitalize()}</span>
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-top:7px;display:flex;
                    align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;">
          <span>Confidence <strong style="color:{color};font-size:14px;">{f_score*100:.1f}%</strong></span>
          {_source_badge(source)} {live_dot}
        </div>
      </div>

      <!-- Voice-only vs Face comparison -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px;">
        <div style="background:#0d1f35;border-radius:10px;padding:10px 12px;">
          <div style="font-size:10px;color:#475569;margin-bottom:6px;text-transform:uppercase;
                      letter-spacing:1px;font-weight:600;">🎤 Voice only</div>
          <div style="font-size:13px;font-weight:700;color:#93c5fd;">
            {EMOTION_EMOJI.get(ve1,'')} {ve1.capitalize()} – {EMOTION_EMOJI.get(ve2,'')} {ve2.capitalize()}
            {changed_badge}
          </div>
          <div style="font-size:11px;color:#475569;margin-top:3px;">
            {v_score*100:.1f}% confidence</div>
        </div>
        <div style="background:#0d1f35;border-radius:10px;padding:10px 12px;">
          <div style="font-size:10px;color:#475569;margin-bottom:6px;text-transform:uppercase;
                      letter-spacing:1px;font-weight:600;">👁️ Face (fused weight 45%)</div>
          <div style="font-size:13px;font-weight:700;color:#a78bfa;">
            {'Detected · contributed' if source=='voice+face' else 'Not detected'}
          </div>
          <div style="font-size:11px;color:#475569;margin-top:3px;">
            {'Δ ' + f'{(f_score-v_score)*100:+.1f}% vs voice-only' if source=='voice+face' else 'Voice-only result used'}
          </div>
        </div>
      </div>

      <!-- Individual fused scores -->
      <div style="background:#0d1f35;border-radius:10px;padding:10px 12px;margin-bottom:12px;">
        <div style="font-size:10px;color:#475569;margin-bottom:8px;text-transform:uppercase;
                    letter-spacing:1px;font-weight:600;">Fused individual scores (top 4)</div>
        {indiv}
      </div>

      <!-- Pair breakdown -->
      <div style="font-size:10px;color:#475569;margin-bottom:8px;text-transform:uppercase;
                  letter-spacing:1px;font-weight:600;">Top 8 pairs</div>
      {_pair_bars(f_pair_probs, f_pair)}
    </div>"""


def live_stats_html(recorder: SessionRecorder) -> str:
    s = recorder.get_summary()
    if not s:
        return f"""{PULSE_CSS}
        <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:14px;
                    padding:28px;color:#64748b;text-align:center;">
          No session data yet — start a Live session first.
        </div>"""
    rows="".join(
        f'<tr><td style="color:#94a3b8;padding:5px 10px;">{k}</td>'
        f'<td style="color:#e2e8f0;font-weight:600;padding:5px 10px;text-align:right;">{v}</td></tr>'
        for k,v in s.items()
    )
    return f"""{PULSE_CSS}
    <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:14px;padding:18px;">
      <div style="font-size:12px;font-weight:700;color:#818cf8;margin-bottom:10px;
                  text-transform:uppercase;letter-spacing:1px;">Session Summary</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;">{rows}</table>
    </div>"""


# ─────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────
# Video processor (real implementation)
# ─────────────────────────────────────────────────────────────────────

def _extract_audio_wav(video_path: str) -> str:
    """Extract audio to temp WAV via ffmpeg or moviepy fallback."""
    import tempfile, subprocess, shutil
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    if shutil.which("ffmpeg"):
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE), "-ac", "1", tmp.name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        # moviepy fallback (supports both v1 and v2)
        try:
            from moviepy.editor import VideoFileClip   # moviepy 1.x
        except ImportError:
            from moviepy import VideoFileClip          # moviepy 2.x
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(tmp.name, fps=SAMPLE_RATE)
        clip.close()
    return tmp.name


def _do_analyse_video(video_path, recorder, model, scaler, selector, labels):
    """
    Analyse a video file second-by-second.
    Returns (status_str, html_timeline).
    """
    if video_path is None:
        return "No video uploaded.", ""

    import cv2, librosa, os, tempfile

    status_msgs = []

    # ── 1. Extract audio ─────────────────────────────────────────────
    try:
        status_msgs.append("🔊 Extracting audio…")
        wav_path = _extract_audio_wav(video_path)
        full_audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        os.unlink(wav_path)
        status_msgs.append(f"   Audio: {len(full_audio)/SAMPLE_RATE:.1f}s loaded")
    except Exception as e:
        return f"❌ Audio extraction failed: {e}", ""

    # ── 2. Open video for frames ─────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "❌ Could not open video file.", ""

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    n_seconds  = max(1, int(duration_s))

    status_msgs.append(f"🎬 Video: {duration_s:.1f}s · {fps:.0f}fps · {n_seconds} segments to process")

    # ── 3. Process each second ───────────────────────────────────────
    results = []   # list of (t, voice_pair, v_score, fused_pair, f_score, fused_raw, source, face_top)
    half_win = int(SAMPLE_RATE * DURATION_SEC / 2)

    recorder.start()   # reset + start logging

    for t in range(n_seconds):
        # -- Audio window (3 sec centred at t) --
        centre  = int(t * SAMPLE_RATE)
        a_start = max(0, centre - half_win)
        a_end   = min(len(full_audio), centre + half_win)
        segment = full_audio[a_start:a_end]

        # pad if too short
        needed = int(SAMPLE_RATE * DURATION_SEC)
        if len(segment) < needed:
            segment = np.pad(segment, (0, needed - len(segment)))

        voice_raw = run_voice_inference(
            segment, model, scaler, selector, labels, preprocess=True)

        # -- Frame at time t --
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame_bgr = cap.read()
        face_raw   = None
        face_top   = None
        face_score = None

        if ok and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_res  = FaceEmotionAnalyser.analyse_image_file(frame_rgb)
            if face_res:
                face_top, face_score, face_raw = face_res

        # -- Fuse --
        if voice_raw is None and face_raw is None:
            continue   # silence + no face — skip this second

        fusion_out = fuse(voice_raw, face_raw)
        fused_raw  = fusion_out["fused_probs"]
        source     = fusion_out["source"]

        v_pair, v_score, _ = get_top_pair(voice_raw) if voice_raw else (("—","—"), 0.0, {})
        f_pair, f_score, f_pprobs = get_top_pair(fused_raw)

        recorder.record(
            voice_pair=v_pair, voice_score=v_score,
            voice_probs=voice_raw or {e:0.0 for e in EMOTION_LABELS},
            fused_pair=f_pair, fused_score=f_score, fused_probs=fused_raw,
            face_detected=(face_raw is not None),
            face_top=face_top, face_score=face_score, source=source
        )

        results.append((t, v_pair, v_score, f_pair, f_score, fused_raw, source, face_top))

    cap.release()
    recorder.stop()

    status_msgs.append(f"✅ Done — {len(results)} segments analysed, {n_seconds - len(results)} skipped (silence/no-face)")

    if not results:
        return "\n".join(status_msgs), "<p style='color:#f87171'>No speech or faces detected in video.</p>"

    # ── 4. Build HTML timeline ───────────────────────────────────────
    rows = ""
    for (t, v_pair, v_score, f_pair, f_score, fused_raw, source, face_top) in results:
        e1, e2  = f_pair
        color   = _pair_color(e1, e2)
        c1, c2  = EMOTION_COLOR.get(e1,"#6366f1"), EMOTION_COLOR.get(e2,"#818cf8")
        changed = (v_pair != f_pair) and source == "voice+face"
        src_icon = {"voice+face":"🎤👁️","voice_only":"🎤","face_only":"👁️"}.get(source,"?")
        face_str = f"👁️ {face_top}" if face_top else "👁️ —"

        mm, ss = divmod(t, 60)
        time_str = f"{mm:02d}:{ss:02d}"

        changed_pill = (
            '<span style="background:#7c3aed;color:#ddd6fe;font-size:10px;' +
            'padding:1px 6px;border-radius:10px;margin-left:5px;">changed</span>'
            if changed else ""
        )

        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;
                    margin-bottom:4px;background:#0d1f35;border-radius:10px;
                    border-left:3px solid {color};">
          <span style="font-size:12px;color:#475569;font-weight:600;
                       min-width:38px;">{time_str}</span>
          <span style="font-size:14px;font-weight:700;color:{c1};">{EMOTION_EMOJI.get(e1,"")} {e1.capitalize()}</span>
          <span style="color:#475569;">–</span>
          <span style="font-size:14px;font-weight:700;color:{c2};">{EMOTION_EMOJI.get(e2,"")} {e2.capitalize()}</span>
          {changed_pill}
          <span style="margin-left:auto;font-size:11px;color:#64748b;">{src_icon} · {f_score*100:.0f}% · {face_str}</span>
        </div>"""

    # Summary banner
    n         = len(results)
    face_n    = sum(1 for r in results if r[6]=="voice+face")
    changed_n = sum(1 for r in results if r[1]!=r[3] and r[6]=="voice+face")
    from collections import Counter
    top_pair  = Counter(r[3] for r in results).most_common(1)[0][0]
    top_pair_str = f"{EMOTION_EMOJI.get(top_pair[0],"")} {top_pair[0].capitalize()} – {EMOTION_EMOJI.get(top_pair[1],"")} {top_pair[1].capitalize()}"

    banner = f"""
    <div style="background:#1e293b;border-radius:12px;padding:14px 18px;
                margin-bottom:14px;display:flex;gap:24px;flex-wrap:wrap;">
      <div style="text-align:center;">
        <div style="font-size:20px;font-weight:800;color:#e2e8f0;">{n}</div>
        <div style="font-size:11px;color:#64748b;">segments</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:20px;font-weight:800;color:#22c55e;">{face_n}</div>
        <div style="font-size:11px;color:#64748b;">face detected</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:20px;font-weight:800;color:#f472b6;">{changed_n}</div>
        <div style="font-size:11px;color:#64748b;">pair changed by face</div>
      </div>
      <div style="text-align:center;flex:1;min-width:160px;">
        <div style="font-size:14px;font-weight:700;color:#e2e8f0;">{top_pair_str}</div>
        <div style="font-size:11px;color:#64748b;">dominant pair</div>
      </div>
    </div>"""

    html = f"""{PULSE_CSS}
    <div style="font-family:'Inter',system-ui,sans-serif;background:#0f172a;border-radius:18px;
                padding:20px;box-shadow:0 8px 40px rgba(0,0,0,.5);">
      <div style="font-size:13px;font-weight:700;color:#818cf8;margin-bottom:12px;
                  text-transform:uppercase;letter-spacing:1px;">📊 Video Analysis Timeline</div>
      {banner}
      <div style="max-height:420px;overflow-y:auto;padding-right:4px;">
        {rows}
      </div>
      <div style="font-size:11px;color:#475569;margin-top:10px;">
        Go to <strong>📊 Report</strong> tab to export full CSV / Excel / PDF
      </div>
    </div>"""

    return "\n".join(status_msgs), html



def build_ui(listener: ContinuousListener, face_analyser: FaceEmotionAnalyser,
             recorder: SessionRecorder, model, scaler, selector, labels):

    # ── Tab 1 callbacks ──────────────────────────────────────────────

    def toggle(btn_label):
        if "Start" in btn_label:
            recorder.start()
            listener.start()
            face_analyser.start()
            return gr.update(value="⏹️  Stop", variant="stop"), warming_html()
        else:
            listener.stop()
            face_analyser.stop()
            recorder.stop()
            return gr.update(value="▶  Start Listening", variant="primary"), idle_html()

    def process_webcam(frame):
        if frame is None: return None
        face_analyser.update_frame(frame)
        return face_analyser.get_annotated_frame(frame)

    def tick():
        status, res, err = listener.get_result()
        if status == "idle":    return idle_html()
        if status == "error":   return error_html(err)
        if status == "silence": return silence_html(res)
        if res is None:         return warming_html()
        return result_html(*res, is_live=True)

    # ── Tab 2 callbacks ──────────────────────────────────────────────

    def analyse_file(audio_path):
        if audio_path is None: return waiting_html()
        try:
            import librosa
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION_SEC)
            if len(audio) < int(SAMPLE_RATE*0.5):
                return error_html("Too short — need 0.5s minimum.")
            voice_raw = run_voice_inference(audio, model, scaler, selector, labels, preprocess=True)
            if voice_raw is None:
                return error_html("No speech detected in the file.")
            fusion_out = fuse(voice_raw, None)
            fused_raw  = fusion_out['fused_probs']
            source     = fusion_out['source']
            v_pair, v_score, v_pprobs = get_top_pair(voice_raw)
            f_pair, f_score, f_pprobs = get_top_pair(fused_raw)
            recorder.record(
                voice_pair=v_pair, voice_score=v_score, voice_probs=voice_raw,
                fused_pair=f_pair, fused_score=f_score, fused_probs=fused_raw,
                face_detected=False, face_top=None, face_score=None, source=source
            )
            return result_html(v_pair, v_score, v_pprobs,
                                f_pair, f_score, f_pprobs, fused_raw, source, is_live=False)
        except Exception as e:
            return error_html(str(e))

    # ── Tab 3 callbacks ──────────────────────────────────────────────

    def refresh_stats(): return live_stats_html(recorder)

    def do_csv():
        if recorder.is_empty(): return None, "⚠️ No data yet."
        p = OUTPUTS_DIR / f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_csv(recorder, p); return str(p), f"✅ Saved: {p.name}"

    def do_excel():
        if recorder.is_empty(): return None, "⚠️ No data yet."
        p = OUTPUTS_DIR / f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        export_excel(recorder, p); return str(p), f"✅ Saved: {p.name}"

    def do_pdf():
        if recorder.is_empty(): return None, "⚠️ No data yet."
        p = OUTPUTS_DIR / f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        export_pdf(recorder, p); return str(p), f"✅ Saved: {p.name}"

    def do_clear():
        recorder.start(); recorder.stop()
        return live_stats_html(recorder), "🗑️ Cleared."

    # ── Layout ───────────────────────────────────────────────────────

    with gr.Blocks(title="Multimodal Emotion Analyser") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:14px 0 4px;">
          <h1 style="margin:0;font-size:24px;">🎙️👁️ Multimodal Emotion Pair Analyser</h1>
          <p style="color:#64748b;margin:5px 0 0;font-size:13px;">
            Voice + Face fusion · 28 emotion pairs · research-grade reports
          </p>
        </div>""")

        with gr.Tabs():

            # ══ TAB 1 — LIVE ══════════════════════════════════════════
            with gr.Tab("🎤  Live"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=180):
                        toggle_btn = gr.Button("▶  Start Listening",
                                               variant="primary", size="lg")
                        gr.Markdown("""
---
**Steps:**
1. Click **▶ Start**
2. Face the webcam + speak
3. V+F fused every second
4. **⏹️ Stop** → go to Report

**Fusion:** 🎤 55% · 👁️ 45%

**Preprocessing:**
🔇 Noise gate
📈 Pre-emphasis
🗣️ VAD
                        """)

                    with gr.Column(scale=2, min_width=280):
                        webcam_in  = gr.Image(sources=["webcam"], streaming=True,
                                              type="numpy", label="📷 Webcam")
                        webcam_out = gr.Image(type="numpy", label="Face Detection")

                    with gr.Column(scale=2, min_width=340):
                        live_out = gr.HTML(value=idle_html())

                timer = gr.Timer(value=1.0)
                timer.tick(fn=tick, outputs=live_out)
                webcam_in.stream(fn=process_webcam,
                                 inputs=webcam_in, outputs=webcam_out)
                toggle_btn.click(fn=toggle, inputs=toggle_btn,
                                 outputs=[toggle_btn, live_out])

            # ══ TAB 2 — FILE ══════════════════════════════════════════
            with gr.Tab("📁  File"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        upload_audio = gr.Audio(sources=["upload"], type="filepath",
                                                label=".wav / .mp3 / .flac / .ogg")
                        analyse_btn  = gr.Button("🔍  Analyse", variant="primary")
                        gr.Markdown("Results are logged to the session for export.")
                    with gr.Column(scale=2, min_width=360):
                        file_out = gr.HTML(value=waiting_html())

                analyse_btn.click(fn=analyse_file, inputs=upload_audio, outputs=file_out)
                upload_audio.change(fn=analyse_file, inputs=upload_audio, outputs=file_out)

            # ══ TAB 3 — REPORT ════════════════════════════════════════
            with gr.Tab("📊  Report"):
                gr.Markdown("### Research Report\n"
                            "Logs every prediction with voice-only vs fused comparison. "
                            "The PDF includes charts on **how much face improved accuracy**.")
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        gr.Markdown("#### Export")
                        csv_btn   = gr.Button("⬇️ CSV   (raw data)",       variant="secondary")
                        excel_btn = gr.Button("⬇️ Excel (6 sheets)",       variant="secondary")
                        pdf_btn   = gr.Button("⬇️ PDF   (full report)",    variant="primary")
                        clear_btn = gr.Button("🗑️ Clear session data",      variant="stop")
                        status_box = gr.Textbox(label="Status", interactive=False)
                        gr.Markdown("""
---
**PDF contains:**
- Session summary table
- Face impact table (key research metric)
- 6 charts:
  - Fused pair frequency
  - Voice vs fused confidence over time
  - Face detection impact bar chart
  - Confidence delta distribution
  - Face top emotion distribution
  - Avg probabilities: voice vs fused
                        """)
                    with gr.Column(scale=2, min_width=340):
                        refresh_btn = gr.Button("🔄 Refresh Stats", size="sm")
                        stats_out   = gr.HTML(value=live_stats_html(recorder))
                        csv_file    = gr.File(label="CSV")
                        excel_file  = gr.File(label="Excel")
                        pdf_file    = gr.File(label="PDF")

                refresh_btn.click(fn=refresh_stats, outputs=stats_out)
                csv_btn.click(fn=do_csv,     outputs=[csv_file,   status_box])
                excel_btn.click(fn=do_excel, outputs=[excel_file, status_box])
                pdf_btn.click(fn=do_pdf,     outputs=[pdf_file,   status_box])
                clear_btn.click(fn=do_clear, outputs=[stats_out,  status_box])

            # ══ TAB 4 — VIDEO ═════════════════════════════════════════
            with gr.Tab("🎬  Video"):
                gr.Markdown(
                    "### Analyse a video file\n"
                    "Extracts audio + face frames second by second, fuses them, "
                    "and logs every result to the session recorder for export."
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        video_upload = gr.Video(label="Upload video (.mp4 / .avi / .mov / .mkv)")
                        video_btn    = gr.Button("▶  Analyse Video", variant="primary", size="lg")
                        gr.Markdown("""
---
**What it does:**
- Extracts audio → voice inference per 3-sec sliding window
- Extracts 1 frame/sec → DeepFace per frame
- Fuses voice + face every second
- Logs all results to session

**After analysis → go to 📊 Report** to export CSV / Excel / PDF with full modality comparison.

---
**Formats:** `.mp4` · `.avi` · `.mov` · `.mkv` · `.webm`
                        """)
                    with gr.Column(scale=2, min_width=380):
                        video_status = gr.Textbox(label="Progress", interactive=False,
                                                  value="Upload a video and click Analyse.")
                        video_out    = gr.HTML()

                def _analyse_video_closure(video_path):
                    return _do_analyse_video(
                        video_path, recorder, model, scaler, selector, labels)

                video_btn.click(
                    fn=_analyse_video_closure,
                    inputs=video_upload,
                    outputs=[video_status, video_out]
                )

    return demo


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Multimodal Emotion Pair Analyser — starting …")
    print("="*55)

    model, scaler, selector, labels = load_artifacts()
    recorder      = SessionRecorder()
    face_analyser = FaceEmotionAnalyser()
    listener      = ContinuousListener(model, scaler, selector, labels,
                                       face_analyser, recorder)
    demo          = build_ui(listener, face_analyser, recorder,
                             model, scaler, selector, labels)

    demo.launch(
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )