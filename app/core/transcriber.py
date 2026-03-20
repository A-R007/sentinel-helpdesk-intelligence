# app/core/transcriber.py
# ═══════════════════════════════════════════════════════════
# Whisper-based speech-to-text with chunked inference,
# confidence scoring, and graceful CPU optimisation.
# ═══════════════════════════════════════════════════════════

import time
import numpy as np
from typing import Optional

from app.utils.logger import get_logger
from app.config       import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE, WHISPER_BEST_OF, WHISPER_TEMPERATURE,
    WHISPER_CHUNK_SECS
)
from app.core.audio   import load_audio, compute_speech_ratio

log = get_logger(__name__)

# Module-level model cache — loaded once per process
_whisper_model = None


def load_whisper_model(size: str = WHISPER_MODEL_SIZE):
    """
    Load and cache the Whisper model.
    Safe to call multiple times — returns cached instance.

    CPU optimisations applied:
    - int8 quantisation via CTranslate2 (if faster-whisper available)
    - Falls back to standard openai-whisper if not
    - Greedy decode (temperature=0) by default — faster and deterministic
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    log.info("Loading Whisper '%s' model on CPU ...", size)
    t0 = time.perf_counter()

    try:
        # Prefer faster-whisper (CTranslate2 backend — 2-4× faster on CPU)
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            size,
            device           = "cpu",
            compute_type     = "int8",   # int8 quantisation — halves memory
            download_root    = None,
            local_files_only = False,
        )
        _whisper_model._backend = "faster-whisper"
        log.info(
            "Whisper loaded (faster-whisper / int8) in %.1fs",
            time.perf_counter() - t0
        )

    except ImportError:
        # Standard openai-whisper fallback
        import whisper
        _whisper_model = whisper.load_model(size, device="cpu")
        _whisper_model._backend = "openai-whisper"
        log.info(
            "Whisper loaded (openai-whisper) in %.1fs",
            time.perf_counter() - t0
        )

    return _whisper_model


def transcribe_audio(
    model,
    file_path: str,
    language: str = WHISPER_LANGUAGE
) -> Optional[dict]:
    """
    Transcribe an audio file to text.

    Returns a dict:
    {
        "text"         : str   — full clean transcript
        "language"     : str   — detected or forced language code
        "duration"     : float — audio duration in seconds
        "speech_ratio" : float — fraction of audio that is speech (0–1)
        "segments"     : list  — [{start, end, text}, ...]
        "confidence"   : float — mean segment no_speech_prob (0–1, lower=better)
        "time_taken"   : float — wall-clock seconds to transcribe
        "model_backend": str   — "faster-whisper" or "openai-whisper"
    }

    Returns None on failure.
    """
    t0 = time.perf_counter()

    # ── Load audio ───────────────────────────────────────────────────────
    audio, sr, duration = load_audio(file_path)
    if audio is None:
        return None

    speech_ratio = compute_speech_ratio(audio, sr)
    if speech_ratio < 0.05:
        log.warning(
            "Very low speech content (%.0f%%) in %s — "
            "result may be empty or unreliable",
            speech_ratio * 100, file_path
        )

    # ── Transcribe ───────────────────────────────────────────────────────
    try:
        backend = getattr(model, "_backend", "openai-whisper")

        if backend == "faster-whisper":
            result = _transcribe_faster(model, audio, language)
        else:
            result = _transcribe_openai(model, audio, language)

        elapsed = round(time.perf_counter() - t0, 2)

        transcript = {
            "text"         : _clean_text(result["text"]),
            "language"     : result.get("language", language or "en"),
            "duration"     : duration,
            "speech_ratio" : speech_ratio,
            "segments"     : result.get("segments", []),
            "confidence"   : result.get("confidence", 0.0),
            "time_taken"   : elapsed,
            "model_backend": backend
        }

        log.info(
            "Transcribed %.1fs audio in %.1fs | speech %.0f%% | '%s...'",
            duration, elapsed,
            speech_ratio * 100,
            transcript["text"][:60]
        )
        return transcript

    except Exception as exc:
        log.error("Transcription failed: %s", exc, exc_info=True)
        return None


# ─────────────────────────────────────────────
# Backend-specific implementations
# ─────────────────────────────────────────────

def _transcribe_faster(model, audio: np.ndarray, language: str) -> dict:
    """faster-whisper inference path."""
    segments_iter, info = model.transcribe(
        audio,
        language             = language,
        beam_size            = WHISPER_BEAM_SIZE,
        best_of              = WHISPER_BEST_OF,
        temperature          = WHISPER_TEMPERATURE,
        vad_filter           = True,          # skip silence automatically
        vad_parameters       = dict(
            min_silence_duration_ms = 500,
            speech_pad_ms           = 200,
        ),
        word_timestamps      = False,
        without_timestamps   = False,
    )

    segments = []
    text_parts = []
    no_speech_probs = []

    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end"  : round(seg.end,   2),
            "text" : seg.text.strip()
        })
        text_parts.append(seg.text.strip())
        no_speech_probs.append(seg.no_speech_prob)

    confidence = float(np.mean(no_speech_probs)) if no_speech_probs else 0.0

    return {
        "text"      : " ".join(text_parts),
        "language"  : info.language,
        "segments"  : segments,
        "confidence": confidence
    }


def _transcribe_openai(model, audio: np.ndarray, language: str) -> dict:
    """openai-whisper inference path."""
    result = model.transcribe(
        audio,
        language    = language,
        fp16        = False,           # CPU cannot do fp16
        beam_size   = WHISPER_BEAM_SIZE,
        best_of     = WHISPER_BEST_OF,
        temperature = WHISPER_TEMPERATURE,
        verbose     = False,
    )

    segments = []
    no_speech_probs = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end"  : round(seg["end"],   2),
            "text" : seg["text"].strip()
        })
        no_speech_probs.append(seg.get("no_speech_prob", 0.0))

    confidence = float(np.mean(no_speech_probs)) if no_speech_probs else 0.0

    return {
        "text"      : result.get("text", ""),
        "language"  : result.get("language", "en"),
        "segments"  : segments,
        "confidence": confidence
    }


def _clean_text(text: str) -> str:
    """
    Remove Whisper artefacts:
    - Repeated filler phrases (e.g. "Thank you. Thank you. Thank you.")
    - Leading/trailing whitespace
    - Music or noise tokens like [MUSIC], (inaudible)
    """
    import re
    text = text.strip()
    # Remove noise tokens
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    # Collapse repeated sentences (common Whisper artefact on silence)
    sentences = text.split('. ')
    seen, deduped = set(), []
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s.strip())
    return '. '.join(deduped).strip()