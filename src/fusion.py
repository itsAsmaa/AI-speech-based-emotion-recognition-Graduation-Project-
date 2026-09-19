"""
src/fusion.py
Fuses voice and face emotion probability dicts into a single 8-emotion dict.

Returns richer info than before so the reporter can log per-modality data.
"""

VOICE_EMOTIONS  = ['neutral','calm','happy','sad','angry','fear','disgust','surprise']
VOICE_W_DEFAULT = 0.55   # voice weight (custom trained model → slightly preferred)


def _face_to_voice_space(face_probs: dict) -> dict:
    """Map DeepFace 7-emotion probs → 8-emotion voice space ('calm' split from 'neutral')."""
    fn = face_probs.get('neutral', 0.0)
    return {
        'neutral':  fn * 0.60,
        'calm':     fn * 0.40,
        'happy':    face_probs.get('happy',    0.0),
        'sad':      face_probs.get('sad',      0.0),
        'angry':    face_probs.get('angry',    0.0),
        'fear':     face_probs.get('fear',     0.0),
        'disgust':  face_probs.get('disgust',  0.0),
        'surprise': face_probs.get('surprise', 0.0),
    }


def fuse(voice_raw: dict | None,
         face_raw:  dict | None,
         voice_weight: float = VOICE_W_DEFAULT) -> dict:
    """
    Parameters
    ----------
    voice_raw    : {emotion: prob}  from voice model  (8 emotions) | None
    face_raw     : {emotion: prob}  from DeepFace     (7 emotions) | None
    voice_weight : float in [0,1]

    Returns
    -------
    {
      'fused_probs'       : dict   — 8-emotion normalised probs after fusion
      'face_in_voice'     : dict   — face probs mapped to voice space (for logging)
      'source'            : str    — 'voice+face' | 'voice_only' | 'face_only'
      'face_detected'     : bool
    }
    """
    face_weight = 1.0 - voice_weight

    # ── No data at all ───────────────────────────────────────────────
    if voice_raw is None and face_raw is None:
        return None

    face_in_voice = _face_to_voice_space(face_raw) if face_raw else None
    face_detected = face_raw is not None

    # ── Voice only ───────────────────────────────────────────────────
    if face_raw is None:
        return {
            'fused_probs':   voice_raw,
            'face_in_voice': None,
            'source':        'voice_only',
            'face_detected': False,
        }

    # ── Face only ────────────────────────────────────────────────────
    if voice_raw is None:
        total = sum(face_in_voice.values()) or 1
        norm  = {k: v / total for k, v in face_in_voice.items()}
        return {
            'fused_probs':   norm,
            'face_in_voice': face_in_voice,
            'source':        'face_only',
            'face_detected': True,
        }

    # ── Both available ────────────────────────────────────────────────
    fused = {
        emo: voice_weight * voice_raw.get(emo, 0.0)
           + face_weight  * face_in_voice.get(emo, 0.0)
        for emo in VOICE_EMOTIONS
    }
    total = sum(fused.values()) or 1
    fused = {k: v / total for k, v in fused.items()}

    return {
        'fused_probs':   fused,
        'face_in_voice': face_in_voice,
        'source':        'voice+face',
        'face_detected': True,
    }
