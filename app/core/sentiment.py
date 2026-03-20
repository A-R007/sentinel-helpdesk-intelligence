# app/core/sentiment.py
# ═══════════════════════════════════════════════════════════
# Advanced sentiment analysis with multi-signal neutral
# detection — solving the enquiry/question misclassification
# problem that plagues standard SST-2 models.
#
# THE CORE PROBLEM:
#   DistilBERT-SST2 was trained on movie reviews.
#   "Can I change my password?" has no emotional content
#   but the word "change" appears frequently in negative
#   movie reviews → model returns NEGATIVE with 0.65 confidence.
#
# OUR SOLUTION — four independent signals, weighted vote:
#
#  Signal 1: Confidence band
#    The model's softmax output is a probability. When the
#    model outputs NEGATIVE with only 0.65 confidence (vs
#    genuine anger at 0.99), that low confidence IS information.
#    Threshold: if max(softmax) < 0.72 → neutral candidate.
#
#  Signal 2: Polarity magnitude
#    We convert confidence to polarity: POSITIVE 0.65 → +0.30,
#    NEGATIVE 0.65 → -0.30. Small absolute values mean weak
#    signal → neutral candidate.
#    Threshold: |polarity| < 0.20 → neutral candidate.
#
#  Signal 3: Enquiry/question detection
#    Count interrogative markers (question words, "?", modal
#    verbs in question patterns). Compute enquiry_ratio =
#    enquiry_tokens / total_tokens. High ratio = factual
#    question → neutral override.
#    Threshold: ratio > 0.08 AND count >= 2 → neutral candidate.
#
#  Signal 4: Vocabulary context
#    Check for ABSENCE of emotional intensifiers alongside
#    PRESENCE of administrative vocabulary.
#    (e.g. "account", "balance", "invoice", "update", "check")
#    No intensifiers + admin vocab → neutral candidate.
#
#  Weighted vote:
#    neutral_score = (s1 × 0.30) + (s2 × 0.25) + (s3 × 0.30) + (s4 × 0.15)
#    if neutral_score ≥ 0.45 → classify as NEUTRAL
#
# SENTENCE-LEVEL ESCALATION DETECTION:
#   We also score each sentence independently and track the
#   TRAJECTORY: a call that starts neutral and becomes
#   increasingly negative is more urgent than one that stays
#   consistently negative (caller has exhausted patience).
# ═══════════════════════════════════════════════════════════

import re
import time
import numpy as np
from typing import Optional

from app.utils.logger import get_logger
from app.config import (
    SENTIMENT_MODEL, SENTIMENT_DEVICE,
    NEUTRAL_CONFIDENCE_THRESHOLD, NEUTRAL_POLARITY_BAND,
    ENQUIRY_RATIO_THRESHOLD, ENQUIRY_WORD_MIN,
    NEUTRAL_WEIGHT_CONFIDENCE, NEUTRAL_WEIGHT_POLARITY,
    NEUTRAL_WEIGHT_ENQUIRY, NEUTRAL_WEIGHT_VOCAB,
    NEUTRAL_VOTE_THRESHOLD
)

log = get_logger(__name__)

# Module-level model cache
_sentiment_pipeline = None

# ─────────────────────────────────────────────
# Vocabulary sets for neutral detection
# ─────────────────────────────────────────────

# Interrogative question words and modal patterns
QUESTION_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose",
    "how", "why", "can", "could", "would", "should", "will",
    "is", "are", "was", "were", "do", "does", "did",
    "may", "might", "shall"
}

# Administrative / neutral vocabulary
ADMIN_VOCAB = {
    "account", "balance", "invoice", "payment", "bill", "statement",
    "address", "email", "phone", "number", "name", "details",
    "update", "change", "check", "status", "order", "reference",
    "confirm", "verify", "information", "query", "enquiry", "request",
    "schedule", "appointment", "date", "time", "open", "close",
    "hours", "location", "branch", "office", "department",
    "password", "login", "access", "register", "sign", "subscription",
    "plan", "package", "service", "product", "price", "cost", "fee"
}

# Emotional intensifiers — presence suggests genuine sentiment
INTENSIFIERS = {
    "very", "extremely", "absolutely", "completely", "totally",
    "utterly", "incredibly", "furious", "furiously", "disgusting",
    "terrible", "horrible", "awful", "amazing", "wonderful",
    "outraged", "livid", "devastated", "thrilled", "delighted",
    "never", "always", "worst", "best", "love", "hate",
    "unacceptable", "disgraceful", "excellent", "perfect",
    "appalling", "ridiculous", "incompetent", "fantastic"
}

# Negation words — important for polarity reversal detection
NEGATIONS = {
    "not", "no", "never", "none", "nothing", "neither",
    "nor", "nobody", "nowhere", "cannot", "can't", "won't",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "shouldn't",
    "wouldn't", "couldn't"
}


def load_sentiment_model():
    """
    Load and cache the HuggingFace sentiment pipeline.
    Uses torch.jit compilation hint for marginal CPU speedup.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    from transformers import pipeline
    log.info("Loading DistilBERT sentiment model (%s) ...", SENTIMENT_MODEL)
    t0 = time.perf_counter()

    _sentiment_pipeline = pipeline(
        task        = "text-classification",
        model       = SENTIMENT_MODEL,
        device      = SENTIMENT_DEVICE,
        top_k       = None,     # return ALL label scores, not just top-1
                                # this gives us both POSITIVE and NEGATIVE
                                # probabilities for better calibration
    )

    log.info("Sentiment model loaded in %.1fs", time.perf_counter() - t0)
    return _sentiment_pipeline


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def analyze_sentiment(pipeline_model, text: str) -> dict:
    """
    Full sentiment analysis with neutral detection.

    Returns:
    {
        "label"           : "positive" | "neutral" | "negative"
        "polarity"        : float  -1.0 to +1.0
        "confidence"      : float  0.0 to 1.0
        "neutral_score"   : float  0.0 to 1.0  (how neutral the text is)
        "neutral_signals" : dict   breakdown of four neutral signals
        "raw_positive"    : float  raw model P(POSITIVE)
        "raw_negative"    : float  raw model P(NEGATIVE)
        "sentence_scores" : list   per-sentence breakdown
        "trajectory"      : str    "stable" | "escalating" | "de-escalating"
        "negation_count"  : int    number of negation words detected
        "intensifier_count": int   number of emotional intensifiers
        "enquiry_ratio"   : float  fraction of text that is interrogative
        "word_count"      : int
        "chunks_processed": int
    }
    """
    if not text or not text.strip():
        return _empty_result()

    text = text.strip()
    word_count = len(text.split())

    # ── Step 1: Get raw model probabilities ─────────────────────────────
    raw_pos, raw_neg = _get_raw_scores(pipeline_model, text)

    # ── Step 2: Compute base polarity ────────────────────────────────────
    # polarity = P(positive) - P(negative)  →  range [-1, +1]
    # This is more informative than just taking the argmax label.
    base_polarity = raw_pos - raw_neg

    # ── Step 3: Linguistic features ──────────────────────────────────────
    words     = text.lower().split()
    neg_count = sum(1 for w in words if re.sub(r"[^a-z']", "", w) in NEGATIONS)
    int_count = sum(1 for w in words if re.sub(r"[^a-z]", "",  w) in INTENSIFIERS)
    adm_count = sum(1 for w in words if re.sub(r"[^a-z]", "",  w) in ADMIN_VOCAB)

    enq_tokens = _count_enquiry_tokens(text, words)
    enq_ratio  = enq_tokens / max(len(words), 1)

    # ── Step 4: Four-signal neutral vote ─────────────────────────────────
    neutral_signals, neutral_score = _compute_neutral_vote(
        raw_confidence = max(raw_pos, raw_neg),
        polarity       = base_polarity,
        enq_ratio      = enq_ratio,
        enq_count      = enq_tokens,
        int_count      = int_count,
        adm_count      = adm_count,
        word_count     = word_count
    )

    # ── Step 5: Apply negation correction to polarity ────────────────────
    # Many negation words in a POSITIVE context → reduce polarity
    # e.g. "it's not bad, not a problem" — model says positive,
    # but negations soften the signal
    corrected_polarity = _apply_negation_correction(
        base_polarity, neg_count, int_count, word_count
    )

    # ── Step 6: Final label decision ─────────────────────────────────────
    label = _decide_label(corrected_polarity, neutral_score)

    # ── Step 7: Sentence-level trajectory ────────────────────────────────
    sentence_scores, trajectory = _analyze_trajectory(pipeline_model, text)

    # ── Step 8: Final confidence ──────────────────────────────────────────
    # Reduce confidence when neutral signals are strong
    raw_confidence = max(raw_pos, raw_neg)
    confidence     = raw_confidence * (1.0 - neutral_score * 0.5)

    log.debug(
        "Sentiment | label=%s polarity=%.3f neutral_score=%.3f "
        "enq_ratio=%.2f intensifiers=%d negations=%d",
        label, corrected_polarity, neutral_score,
        enq_ratio, int_count, neg_count
    )

    return {
        "label"            : label,
        "polarity"         : round(corrected_polarity, 4),
        "confidence"       : round(confidence, 4),
        "neutral_score"    : round(neutral_score, 4),
        "neutral_signals"  : neutral_signals,
        "raw_positive"     : round(raw_pos, 4),
        "raw_negative"     : round(raw_neg, 4),
        "sentence_scores"  : sentence_scores,
        "trajectory"       : trajectory,
        "negation_count"   : neg_count,
        "intensifier_count": int_count,
        "enquiry_ratio"    : round(enq_ratio, 4),
        "word_count"       : word_count,
        "chunks_processed" : 1 if word_count <= 380 else
                             -(-word_count // 380)  # ceiling division
    }


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _get_raw_scores(pipeline_model, text: str):
    """
    Run DistilBERT and return (p_positive, p_negative).

    For long texts we chunk and average — DistilBERT has a
    512-token limit (~380 words). We use overlapping windows
    to avoid losing context at chunk boundaries.
    """
    words  = text.split()
    chunks = _chunk_text(words, max_words=380, overlap=20)

    pos_scores, neg_scores = [], []

    for chunk in chunks:
        try:
            result = pipeline_model(
                chunk, truncation=True, max_length=512
            )
            # top_k=None returns all labels
            score_map = {r["label"]: r["score"] for r in result[0]}
            pos_scores.append(score_map.get("POSITIVE", 0.5))
            neg_scores.append(score_map.get("NEGATIVE", 0.5))
        except Exception as exc:
            log.warning("Chunk inference failed: %s", exc)

    if not pos_scores:
        return 0.5, 0.5

    return float(np.mean(pos_scores)), float(np.mean(neg_scores))


def _chunk_text(words: list, max_words: int = 380, overlap: int = 20) -> list:
    """
    Split word list into overlapping chunks.
    Overlap prevents losing context at boundaries.
    """
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    step   = max_words - overlap
    for i in range(0, len(words), step):
        chunk = words[i : i + max_words]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def _count_enquiry_tokens(text: str, words: list) -> int:
    """
    Count interrogative markers in text.

    Markers counted:
    - Question marks
    - Question words at sentence start
    - Modal verbs at sentence start (Can you...? Could I...?)
    - "I would like to" / "I want to" patterns (polite requests)
    """
    count = 0

    # Question marks
    count += text.count("?")

    # Question words / modals at sentence start
    sentences = re.split(r'[.!?]+', text.lower())
    for sent in sentences:
        sent_words = sent.strip().split()
        if sent_words and sent_words[0] in QUESTION_WORDS:
            count += 1

    # Polite request patterns
    polite_patterns = [
        r'\bi would like\b', r'\bi want to\b', r'\bcould you\b',
        r'\bcan you\b', r'\bwould you\b', r'\bplease\b',
        r'\bjust (calling|checking|wondering)\b',
        r'\bi need to (check|know|find|update|change)\b'
    ]
    for pat in polite_patterns:
        if re.search(pat, text.lower()):
            count += 1

    return count


def _compute_neutral_vote(
    raw_confidence: float,
    polarity: float,
    enq_ratio: float,
    enq_count: int,
    int_count: int,
    adm_count: int,
    word_count: int
) -> tuple[dict, float]:
    """
    Compute the four neutral signals and their weighted vote.

    Each signal returns a score 0.0–1.0 where 1.0 = strongly neutral.

    Returns (signals_dict, weighted_neutral_score).
    """

    # ── Signal 1: Confidence band ────────────────────────────────────────
    # Low confidence → the model is unsure → more likely neutral
    # Score goes from 0 (very confident = not neutral) to 1 (low confidence)
    if raw_confidence >= NEUTRAL_CONFIDENCE_THRESHOLD:
        s1 = 0.0
    else:
        # Linear interpolation: confidence 0.50 → score 1.0
        #                        confidence 0.72 → score 0.0
        s1 = 1.0 - (raw_confidence - 0.50) / (NEUTRAL_CONFIDENCE_THRESHOLD - 0.50)
        s1 = max(0.0, min(1.0, s1))

    # ── Signal 2: Polarity magnitude ─────────────────────────────────────
    # Small absolute polarity = weak sentiment = more likely neutral
    abs_pol = abs(polarity)
    if abs_pol <= NEUTRAL_POLARITY_BAND:
        s2 = 1.0
    elif abs_pol >= 0.50:
        s2 = 0.0
    else:
        s2 = 1.0 - (abs_pol - NEUTRAL_POLARITY_BAND) / (0.50 - NEUTRAL_POLARITY_BAND)

    # ── Signal 3: Enquiry/question ratio ─────────────────────────────────
    # High interrogative density → factual question → neutral
    if enq_count >= ENQUIRY_WORD_MIN and enq_ratio >= ENQUIRY_RATIO_THRESHOLD:
        # Scale: ratio 0.08 → s3 0.5,  ratio 0.25+ → s3 1.0
        s3 = min(1.0, (enq_ratio - ENQUIRY_RATIO_THRESHOLD) /
                 (0.25 - ENQUIRY_RATIO_THRESHOLD) * 0.5 + 0.5)
    elif enq_count >= ENQUIRY_WORD_MIN:
        s3 = 0.3   # some questions but not dominant
    else:
        s3 = 0.0

    # ── Signal 4: Vocabulary context ─────────────────────────────────────
    # Admin vocab WITHOUT emotional intensifiers → neutral
    # Normalise by word count to make it length-independent
    admin_density = adm_count / max(word_count, 1)

    if int_count == 0 and admin_density > 0.05:
        # Pure administrative content with no emotional language
        s4 = min(1.0, admin_density / 0.15)
    elif int_count == 0 and admin_density > 0:
        s4 = 0.3
    else:
        # Intensifiers present — not neutral regardless of admin vocab
        s4 = max(0.0, 0.3 - int_count * 0.15)

    # ── Weighted vote ─────────────────────────────────────────────────────
    neutral_score = (
        s1 * NEUTRAL_WEIGHT_CONFIDENCE +
        s2 * NEUTRAL_WEIGHT_POLARITY   +
        s3 * NEUTRAL_WEIGHT_ENQUIRY    +
        s4 * NEUTRAL_WEIGHT_VOCAB
    )

    signals = {
        "confidence_signal" : round(s1, 3),
        "polarity_signal"   : round(s2, 3),
        "enquiry_signal"    : round(s3, 3),
        "vocabulary_signal" : round(s4, 3),
        "weighted_score"    : round(neutral_score, 3)
    }

    return signals, round(neutral_score, 4)


def _apply_negation_correction(
    polarity: float,
    neg_count: int,
    int_count: int,
    word_count: int
) -> float:
    """
    Correct polarity for negation patterns.

    High negation density in otherwise POSITIVE text:
    "It's not too bad, not a huge issue" — positive polarity
    but softened by negations. We pull polarity toward centre.

    High negation in NEGATIVE text with intensifiers:
    "Never works, absolutely not acceptable" — double-down,
    polarity already negative, no correction needed.
    """
    if word_count == 0:
        return polarity

    neg_density = neg_count / word_count

    # Only correct when negations are notable (>3% of words)
    # and no intensifiers are counteracting
    if neg_density > 0.03 and int_count == 0:
        correction = neg_density * 0.4 * (-1 if polarity > 0 else 1)
        polarity   = polarity + correction

    return max(-1.0, min(1.0, polarity))


def _decide_label(polarity: float, neutral_score: float) -> str:
    """
    Final three-way label decision.

    Decision logic:
    1. If neutral_score ≥ NEUTRAL_VOTE_THRESHOLD → "neutral"
    2. Otherwise use polarity sign
    """
    if neutral_score >= NEUTRAL_VOTE_THRESHOLD:
        return "neutral"
    return "positive" if polarity >= 0 else "negative"


def _analyze_trajectory(pipeline_model, text: str) -> tuple[list, str]:
    """
    Score each sentence independently and determine the
    emotional trajectory of the call.

    Trajectory:
    - "stable"         → sentiment roughly constant throughout
    - "escalating"     → call starts neutral/positive, ends negative
                          (IMPORTANT: more urgent than flat negative)
    - "de-escalating"  → call starts negative, ends positive
                          (agent is resolving the issue)

    This is a key insight: an "escalating" call needs immediate
    attention even if the overall average sentiment is only medium.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 8]

    if len(sentences) < 2:
        return [], "stable"

    scores = []
    for sent in sentences[:10]:     # cap at 10 to avoid slow processing
        try:
            result   = pipeline_model(sent, truncation=True, max_length=128)
            score_map = {r["label"]: r["score"] for r in result[0]}
            raw_pos  = score_map.get("POSITIVE", 0.5)
            raw_neg  = score_map.get("NEGATIVE", 0.5)
            polarity = raw_pos - raw_neg
            scores.append({
                "sentence": sent[:80],
                "polarity": round(polarity, 3),
                "label"   : "positive" if polarity > 0.15 else
                            "negative" if polarity < -0.15 else "neutral"
            })
        except Exception:
            pass

    if len(scores) < 2:
        return scores, "stable"

    # Compute trajectory: compare first third vs last third
    third = max(1, len(scores) // 3)
    first_avg = np.mean([s["polarity"] for s in scores[:third]])
    last_avg  = np.mean([s["polarity"] for s in scores[-third:]])
    delta     = last_avg - first_avg

    if delta < -0.25:
        trajectory = "escalating"      # getting worse
    elif delta > 0.25:
        trajectory = "de-escalating"   # getting better
    else:
        trajectory = "stable"

    return scores, trajectory


def _empty_result() -> dict:
    return {
        "label"            : "neutral",
        "polarity"         : 0.0,
        "confidence"       : 0.0,
        "neutral_score"    : 1.0,
        "neutral_signals"  : {},
        "raw_positive"     : 0.5,
        "raw_negative"     : 0.5,
        "sentence_scores"  : [],
        "trajectory"       : "stable",
        "negation_count"   : 0,
        "intensifier_count": 0,
        "enquiry_ratio"    : 0.0,
        "word_count"       : 0,
        "chunks_processed" : 0
    }