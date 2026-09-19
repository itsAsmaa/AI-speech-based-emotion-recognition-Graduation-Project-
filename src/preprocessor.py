"""
src/preprocessor.py
Audio preprocessing pipeline applied before feature extraction.

Steps (in order):
  1. Noise Gate       — skip/silence frames below RMS energy threshold
  2. Pre-emphasis     — boost high frequencies (standard in speech processing)
  3. VAD              — detect and keep only speech frames using WebRTC VAD
                        falls back to energy-based VAD if webrtcvad not installed

Usage:
    preprocessor = AudioPreprocessor(sr=44100)
    clean_audio, is_speech = preprocessor.process(audio_array)
    if is_speech:
        features = extractor.extract_features(clean_audio, sr)
"""

import numpy as np
import warnings

# Try to import webrtcvad — falls back gracefully if not installed
try:
    import webrtcvad
    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False
    warnings.warn(
        "webrtcvad not installed — using energy-based VAD fallback.\n"
        "For better VAD: pip install webrtcvad",
        stacklevel=2
    )


class AudioPreprocessor:
    """
    Full preprocessing pipeline for speech emotion recognition.

    Parameters
    ----------
    sr                  : int   — sample rate (must match model training, default 44100)
    noise_gate_rms      : float — RMS threshold below which audio is considered silence
                                  (0.01 works well for typical laptop mics)
    pre_emphasis_coeff  : float — pre-emphasis filter coefficient (0.95–0.97 standard)
    vad_aggressiveness  : int   — WebRTC VAD aggressiveness 0–3 (3 = most aggressive)
    vad_frame_ms        : int   — VAD frame size in ms (10, 20, or 30 — WebRTC requirement)
    speech_ratio_thresh : float — min fraction of frames that must be speech (0.0–1.0)
    """

    # WebRTC VAD requires 8000, 16000, 32000, or 48000 Hz
    # We use 16000 Hz internally for VAD, then work at original sr for features
    VAD_SR = 16000

    def __init__(
        self,
        sr                  : int   = 44100,
        noise_gate_rms      : float = 0.01,
        pre_emphasis_coeff  : float = 0.97,
        vad_aggressiveness  : int   = 2,
        vad_frame_ms        : int   = 30,
        speech_ratio_thresh : float = 0.25,
    ):
        self.sr                   = sr
        self.noise_gate_rms       = noise_gate_rms
        self.pre_emphasis_coeff   = pre_emphasis_coeff
        self.vad_aggressiveness   = vad_aggressiveness
        self.vad_frame_ms         = vad_frame_ms
        self.speech_ratio_thresh  = speech_ratio_thresh

        # Initialise WebRTC VAD if available
        self._vad_engine = None
        if _WEBRTC_AVAILABLE:
            self._vad_engine = webrtcvad.Vad(vad_aggressiveness)

    # ── Step 1: Noise Gate ───────────────────────────────────────────

    def _noise_gate(self, audio: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Returns (audio, is_loud_enough).
        If overall RMS is below threshold → treat as silence.
        """
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.noise_gate_rms:
            return audio, False   # too quiet — silence / background noise
        return audio, True

    # ── Step 2: Pre-emphasis ─────────────────────────────────────────

    def _pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply first-order high-pass filter:  y[n] = x[n] - coeff * x[n-1]

        Boosts high-frequency components (consonants, fricatives) that carry
        emotional cues but are often under-represented in raw waveforms.
        """
        return np.append(audio[0], audio[1:] - self.pre_emphasis_coeff * audio[:-1])

    # ── Step 3: VAD ──────────────────────────────────────────────────

    def _vad_webrtc(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        WebRTC VAD — industry-standard, frame-based speech detector.
        Returns (is_speech, speech_ratio).
        """
        import librosa

        # Resample to VAD_SR for WebRTC
        audio_16k = librosa.resample(audio, orig_sr=self.sr, target_sr=self.VAD_SR)

        # Convert to 16-bit PCM (WebRTC requirement)
        pcm = (audio_16k * 32767).astype(np.int16)

        frame_len  = int(self.VAD_SR * self.vad_frame_ms / 1000)
        n_frames   = len(pcm) // frame_len
        if n_frames == 0:
            return False, 0.0

        speech_frames = 0
        for i in range(n_frames):
            frame = pcm[i * frame_len : (i + 1) * frame_len].tobytes()
            try:
                if self._vad_engine.is_speech(frame, self.VAD_SR):
                    speech_frames += 1
            except Exception:
                pass

        speech_ratio = speech_frames / n_frames
        return speech_ratio >= self.speech_ratio_thresh, speech_ratio

    def _vad_energy(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Energy-based VAD fallback — splits audio into frames, counts how many
        frames exceed a local energy threshold.
        """
        frame_len  = int(self.sr * self.vad_frame_ms / 1000)
        n_frames   = len(audio) // frame_len
        if n_frames == 0:
            return False, 0.0

        energies = []
        for i in range(n_frames):
            frame = audio[i * frame_len : (i + 1) * frame_len]
            energies.append(float(np.sqrt(np.mean(frame ** 2))))

        energies    = np.array(energies)
        threshold   = np.mean(energies) * 0.5   # adaptive: half of mean energy
        speech_frames = int(np.sum(energies > threshold))
        speech_ratio  = speech_frames / n_frames
        return speech_ratio >= self.speech_ratio_thresh, speech_ratio

    def _run_vad(self, audio: np.ndarray) -> tuple[bool, float]:
        if self._vad_engine is not None:
            return self._vad_webrtc(audio)
        return self._vad_energy(audio)

    # ── Public API ───────────────────────────────────────────────────

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        """
        Run the full preprocessing pipeline.

        Returns
        -------
        clean_audio : np.ndarray  — preprocessed audio (same sr as input)
        is_speech   : bool        — True if speech was detected, False = skip inference
        info        : dict        — diagnostic info for debugging
            {
              'rms'          : float,
              'passed_gate'  : bool,
              'speech_ratio' : float,
              'vad_backend'  : str,
            }
        """
        info = {}

        # 1. Noise gate
        audio, passed_gate = self._noise_gate(audio)
        info['rms']         = float(np.sqrt(np.mean(audio ** 2)))
        info['passed_gate'] = passed_gate

        if not passed_gate:
            info['speech_ratio'] = 0.0
            info['vad_backend']  = 'skipped (noise gate)'
            return audio, False, info

        # 2. Pre-emphasis (always applied if audio passes noise gate)
        audio = self._pre_emphasis(audio)

        # 3. VAD
        is_speech, speech_ratio = self._run_vad(audio)
        info['speech_ratio'] = speech_ratio
        info['vad_backend']  = 'webrtcvad' if _WEBRTC_AVAILABLE else 'energy'

        return audio, is_speech, info

    def __repr__(self):
        backend = 'webrtcvad' if _WEBRTC_AVAILABLE else 'energy-VAD'
        return (f"AudioPreprocessor(sr={self.sr}, "
                f"noise_gate={self.noise_gate_rms}, "
                f"pre_emphasis={self.pre_emphasis_coeff}, "
                f"vad={backend}, aggressiveness={self.vad_aggressiveness})")