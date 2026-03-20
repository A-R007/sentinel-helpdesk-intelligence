# app/core/audio.py
# ═══════════════════════════════════════════════════════════
# Audio loading, validation, and pre-processing.
# ═══════════════════════════════════════════════════════════

import os
import numpy as np
from pathlib import Path
from typing  import Optional, Tuple

from app.utils.logger import get_logger
from app.config       import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB

log = get_logger(__name__)

# Target sample rate — Whisper requires 16 kHz mono float32
TARGET_SR = 16_000


def validate_audio_file(path: str | Path) -> Tuple[bool, str]:
    """
    Check a file is safe to process before loading it.

    Returns (ok: bool, reason: str).
    reason is empty string when ok=True.
    """
    path = Path(path)

    if not path.exists():
        return False, f"File not found: {path}"

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported format '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return False, (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum: {MAX_UPLOAD_MB} MB"
        )

    return True, ""


def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR
) -> Tuple[Optional[np.ndarray], int, float]:
    """
    Load any audio file into a float32 mono numpy array at target_sr.

    Returns (audio_array, sample_rate, duration_seconds).
    Returns (None, 0, 0.0) on failure — caller checks for None.

    Uses librosa which handles format conversion internally,
    including MP3, FLAC, M4A and more.
    """
    import librosa

    path = Path(path)
    ok, reason = validate_audio_file(path)
    if not ok:
        log.error("Audio validation failed: %s", reason)
        return None, 0, 0.0

    try:
        log.debug("Loading audio: %s", path.name)
        # librosa.load handles resampling and mono conversion
        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
        duration  = len(audio) / sr

        # Normalise amplitude to [-1, 1] to prevent clipping artefacts
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95

        log.info(
            "Audio loaded: %s | %.1fs | %d Hz | %.0f KB",
            path.name, duration, sr, path.stat().st_size / 1024
        )
        return audio.astype(np.float32), sr, round(duration, 2)

    except Exception as exc:
        log.error("Failed to load audio '%s': %s", path.name, exc)
        return None, 0, 0.0


def compute_speech_ratio(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    """
    Estimate what fraction of the audio contains actual speech
    (vs silence/noise). Used for quality checks.

    Returns a float 0.0–1.0.
    """
    if audio is None or len(audio) == 0:
        return 0.0

    # Energy-based VAD: split into 20ms frames, mark as speech
    # if RMS energy exceeds a noise floor estimate
    frame_len  = int(sr * 0.02)   # 20 ms
    frames     = [audio[i:i+frame_len] for i in range(0, len(audio)-frame_len, frame_len)]
    rms_values = [np.sqrt(np.mean(f**2)) for f in frames if len(f) == frame_len]

    if not rms_values:
        return 0.0

    noise_floor = np.percentile(rms_values, 15)   # bottom 15% ≈ silence
    threshold   = noise_floor * 3.5               # speech > 3.5× noise floor
    speech_frames = sum(1 for r in rms_values if r > threshold)

    return round(speech_frames / len(rms_values), 3)