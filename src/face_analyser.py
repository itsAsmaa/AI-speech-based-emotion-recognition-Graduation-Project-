"""
src/face_analyser.py
Real-time facial emotion analyser using DeepFace + OpenCV.

Architecture:
  - update_frame()     → called every webcam frame (fast, just stores frame)
  - get_annotated_frame() → draws bounding box + cached label (fast)
  - background thread  → runs DeepFace.analyze() every ~1 second (slow part)
  - get_result()       → returns latest (label, conf, prob_dict) or None
"""

import cv2
import numpy as np
import threading
import time

# DeepFace is imported lazily so startup is fast
_deepface = None

def _get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


# Emotions DeepFace returns (no "calm")
FACE_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Colour map for annotation overlay (BGR for OpenCV)
FACE_COLOR_BGR = {
    'neutral':  (148, 163, 148),
    'happy':    ( 34, 191, 251),
    'sad':      (250, 165,  96),
    'angry':    (113, 113, 248),
    'fear':     (204, 132, 192),
    'disgust':  ( 78, 222, 128),
    'surprise': ( 51, 147, 251),
}


class FaceEmotionAnalyser:
    """
    Continuous facial emotion analyser.
    Call start() / stop() in sync with the audio listener.
    Call update_frame(rgb_array) on every Gradio webcam frame.
    Call get_annotated_frame(rgb_array) to get the display frame.
    Call get_result() from the gr.Timer tick to read latest emotion.
    """

    ANALYSE_EVERY_SEC = 1.0

    def __init__(self):
        self._lock       = threading.Lock()
        self._result     = None      # (label, conf, prob_dict_face)
        self._face_box   = None      # (x, y, w, h) — cached for annotation
        self._latest_rgb = None      # most recent frame

        self._running = False
        self._thread  = None

        # Fast Haar-cascade for real-time annotation (no DL inference needed)
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── Called on every Gradio webcam frame (must be fast) ──────────

    def update_frame(self, frame_rgb: np.ndarray):
        """Store the latest frame for the background inference thread."""
        with self._lock:
            self._latest_rgb = frame_rgb.copy()

    def get_annotated_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Return the frame with a bounding box and emotion label drawn on it.
        Uses cached face box from last DeepFace inference — no new inference here.
        """
        frame = frame_rgb.copy()

        with self._lock:
            box    = self._face_box
            result = self._result

        if box is not None:
            x, y, w, h = box
            label  = result[0] if result else ""
            color  = FACE_COLOR_BGR.get(label, (200, 200, 200))

            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label badge
            if label:
                badge_txt = label.upper()
                (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
                cv2.putText(frame, badge_txt, (x + 5, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 23, 42), 2)
        else:
            # No cached box — run fast Haar detector just for annotation
            gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (100, 100, 100), 2)

        return frame

    # ── Background DeepFace inference loop ──────────────────────────

    def _analyse_loop(self):
        DeepFace = _get_deepface()
        print("👁️  Face analyser ready (DeepFace loaded)")

        while self._running:
            frame = None
            with self._lock:
                if self._latest_rgb is not None:
                    frame = self._latest_rgb.copy()

            if frame is not None:
                try:
                    results = DeepFace.analyze(
                        img_path        = frame,
                        actions         = ["emotion"],
                        enforce_detection = False,
                        detector_backend = "opencv",
                        silent           = True,
                    )

                    r = results[0] if isinstance(results, list) else results
                    raw_emotions  = r.get("emotion", {})
                    face_region   = r.get("region", {})

                    if raw_emotions:
                        # Normalise to 0–1 probabilities
                        total = sum(raw_emotions.values()) or 1
                        prob_dict = {k.lower(): v / total for k, v in raw_emotions.items()}

                        label = max(prob_dict, key=prob_dict.get)
                        conf  = float(prob_dict[label])

                        box = None
                        if face_region and face_region.get("w", 0) > 0:
                            box = (
                                int(face_region.get("x", 0)),
                                int(face_region.get("y", 0)),
                                int(face_region.get("w", 80)),
                                int(face_region.get("h", 80)),
                            )

                        with self._lock:
                            self._result   = (label, conf, prob_dict)
                            self._face_box = box

                        print(f"  👁️  {label:8s}  {conf*100:.0f}%")

                except Exception:
                    pass    # face not detected — keep last result

            time.sleep(self.ANALYSE_EVERY_SEC)

    # ── Public start / stop ─────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running    = True
        self._result     = None
        self._face_box   = None
        self._latest_rgb = None
        self._thread = threading.Thread(target=self._analyse_loop, daemon=True)
        self._thread.start()
        print("👁️  Face analyser started")

    def stop(self):
        self._running  = False
        self._result   = None
        self._face_box = None
        print("⏹️  Face analyser stopped")

    def get_result(self):
        """Returns (label, conf, prob_dict) or None."""
        with self._lock:
            return self._result

    # ── Static: analyse a single image file (for the File tab) ──────

    @staticmethod
    def analyse_image_file(image_rgb: np.ndarray):
        """
        Run DeepFace on a single image array (RGB).
        Returns (label, conf, prob_dict) or None.
        """
        DeepFace = _get_deepface()
        try:
            results = DeepFace.analyze(
                img_path         = image_rgb,
                actions          = ["emotion"],
                enforce_detection = False,
                detector_backend = "opencv",
                silent           = True,
            )
            r = results[0] if isinstance(results, list) else results
            raw = r.get("emotion", {})
            if not raw:
                return None
            total    = sum(raw.values()) or 1
            prob_dict = {k.lower(): v / total for k, v in raw.items()}
            label = max(prob_dict, key=prob_dict.get)
            conf  = float(prob_dict[label])
            return label, conf, prob_dict
        except Exception:
            return None
