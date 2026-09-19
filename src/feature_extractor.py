"""
feature_extractor.py
Exact replica of the EnhancedAudioFeatureExtractor from your notebook.
Imported by both train.py and realtime_app.py so the pipeline is IDENTICAL.
"""

import numpy as np
import librosa
from config import SAMPLE_RATE, DURATION_SEC, N_MFCC, N_MELS


class EnhancedAudioFeatureExtractor:
    """
    Extracts a 1-D feature vector from a .wav file.
    Features: 40 MFCCs (mean+std) · 12 Chroma · 64 Mel-spectrogram ·
              ZCR (mean+std) · RMS (mean+std)
    Output size: 40*2 + 12 + 64 + 2 + 2 = 162 features
    """

    def __init__(self, sr: int = SAMPLE_RATE, duration: float = DURATION_SEC):
        self.sr       = sr
        self.duration = duration

    # ── Public API ───────────────────────────────────────────────────

    def process_audio_file(self, path: str) -> np.ndarray:
        """Load from disk → extract features."""
        audio, sr = librosa.load(str(path), sr=self.sr, duration=self.duration)
        return self.extract_features(audio, sr)

    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract features from an in-memory audio array."""
        if audio is None or len(audio) < 100:
            raise ValueError("Audio too short — record at least 1 second of speech.")

        # MFCC (mean + std of each coefficient)
        mfcc     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_m   = np.mean(mfcc, axis=1)   # shape (40,)
        mfcc_s   = np.std(mfcc,  axis=1)   # shape (40,)

        # Chroma (12 pitch classes)
        chroma   = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_m = np.mean(chroma, axis=1)  # shape (12,)

        # Mel-spectrogram in dB
        mel      = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_db   = librosa.power_to_db(mel)
        mel_m    = np.mean(mel_db, axis=1)  # shape (64,)

        # Zero Crossing Rate
        zcr      = librosa.feature.zero_crossing_rate(audio)
        zcr_feat = np.array([np.mean(zcr), np.std(zcr)])

        # Root Mean Square Energy
        rms      = librosa.feature.rms(y=audio)
        rms_feat = np.array([np.mean(rms), np.std(rms)])

        return np.concatenate(
            [mfcc_m, mfcc_s, chroma_m, mel_m, zcr_feat, rms_feat]
        ).astype(np.float32)
