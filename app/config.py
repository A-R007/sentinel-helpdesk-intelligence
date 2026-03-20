# app/config.py
# ═══════════════════════════════════════════════════════════
# Central configuration — every tunable value lives here.
# Change behaviour by editing this file only.
# ═══════════════════════════════════════════════════════════

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT_DIR / "helpdesk.db"
LOG_DIR    = ROOT_DIR / "logs"
LOG_PATH   = LOG_DIR  / "app.log"
UPLOAD_DIR = ROOT_DIR / "temp_uploads"

# Ensure directories exist at import time
LOG_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ── API ──────────────────────────────────────────────────────
API_TITLE       = "Helpdesk Sentiment Intelligence Platform"
API_VERSION     = "3.0.0"
API_DESCRIPTION = (
    "Real-time sentiment analysis and priority routing "
    "for helpdesk call centers. CPU-only, production-grade."
)
MAX_UPLOAD_MB      = 50
CALLS_PAGE_SIZE    = 25
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}

# ── Whisper STT ──────────────────────────────────────────────
# "tiny"  → 75MB,  fastest,  good accuracy  ← recommended
# "base"  → 145MB, fast,     better accuracy
# "small" → 461MB, moderate, best CPU accuracy
WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE      = "cpu"
WHISPER_LANGUAGE    = "en"          # None = auto-detect (slower)
WHISPER_BEAM_SIZE   = 3             # higher = more accurate, slower
WHISPER_BEST_OF     = 3             # candidates evaluated per segment
WHISPER_TEMPERATURE = 0.0           # 0 = greedy decode (fastest, deterministic)
WHISPER_CHUNK_SECS  = 30            # process audio in 30s windows

# ── Sentiment model ──────────────────────────────────────────
# Primary: DistilBERT fine-tuned on SST-2 (67MB, POSITIVE/NEGATIVE)
SENTIMENT_MODEL  = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_DEVICE = -1               # -1 = CPU

# ── Advanced neutral detection ───────────────────────────────
#
# Problem: DistilBERT was trained on SST-2 which only has
# POSITIVE / NEGATIVE labels. Helpdesk enquiries ("What is
# my balance?", "Can I change my address?") often score as
# weakly NEGATIVE because they share vocabulary with complaints.
#
# Solution: We apply a multi-signal neutral classifier:
#
#   Signal 1 — Polarity confidence band
#     If the model's raw softmax confidence is below
#     NEUTRAL_CONFIDENCE_THRESHOLD, the model isn't sure
#     → treat as neutral.
#
#   Signal 2 — Polarity magnitude band
#     If absolute polarity < NEUTRAL_POLARITY_BAND, the
#     signal is too weak to be a genuine sentiment.
#
#   Signal 3 — Question / enquiry detection
#     Count interrogative markers. If the enquiry ratio
#     exceeds ENQUIRY_RATIO_THRESHOLD, override to neutral
#     regardless of model output.
#
#   Signal 4 — Vocabulary context check
#     Presence of purely factual / administrative language
#     with NO emotional intensifiers → neutral override.
#
# All four signals feed a weighted vote.
# ─────────────────────────────────────────────────────────────

# Raw model confidence below this → neutral candidate
NEUTRAL_CONFIDENCE_THRESHOLD = 0.72

# Absolute polarity below this → neutral (even if confident)
NEUTRAL_POLARITY_BAND = 0.20

# If enquiry word count / total word count exceeds this → neutral
ENQUIRY_RATIO_THRESHOLD = 0.08

# Minimum enquiry words to trigger ratio check
ENQUIRY_WORD_MIN = 2

# Weights for the four neutral signals (must sum to 1.0)
NEUTRAL_WEIGHT_CONFIDENCE = 0.30
NEUTRAL_WEIGHT_POLARITY   = 0.25
NEUTRAL_WEIGHT_ENQUIRY    = 0.30
NEUTRAL_WEIGHT_VOCAB      = 0.15

# Score above this → classify as neutral
NEUTRAL_VOTE_THRESHOLD = 0.45

# ── Urgency scoring weights ──────────────────────────────────
WEIGHT_ML         = 0.45
WEIGHT_KEYWORDS   = 0.30
WEIGHT_METADATA   = 0.15
WEIGHT_EMOTION    = 0.10    # acoustic/linguistic emotion intensity

WEIGHT_ML_NO_META       = 0.55
WEIGHT_KEYWORDS_NO_META = 0.35
WEIGHT_EMOTION_NO_META  = 0.10

# ── Priority thresholds ──────────────────────────────────────
PRIORITY_P1 = 0.82    # Critical  — escalate now
PRIORITY_P2 = 0.60    # High      — senior agent, < 5 min
PRIORITY_P3 = 0.40    # Medium    — standard queue
              #           below → P4 Low

# ── Kaggle dataset ───────────────────────────────────────────
KAGGLE_DATASET_ID = "oleksiymaliovanyy/call-center-transcripts-dataset"
KAGGLE_CACHE_DIR  = Path.home() / ".cache" / "kagglehub" / "datasets"