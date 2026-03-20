# app/core/urgency.py
# ═══════════════════════════════════════════════════════════
# Urgency scoring: combines ML sentiment + keyword rules
# + metadata + trajectory into a single 0–1 score mapped
# to P1/P2/P3/P4 priority with detailed action guidance.
# ═══════════════════════════════════════════════════════════

import re
from app.utils.logger import get_logger
from app.config import (
    WEIGHT_ML, WEIGHT_KEYWORDS, WEIGHT_METADATA, WEIGHT_EMOTION,
    WEIGHT_ML_NO_META, WEIGHT_KEYWORDS_NO_META, WEIGHT_EMOTION_NO_META,
    PRIORITY_P1, PRIORITY_P2, PRIORITY_P3
)

log = get_logger(__name__)

# ─────────────────────────────────────────────
# Keyword dictionaries (weighted by severity)
# ─────────────────────────────────────────────

KEYWORD_GROUPS = {
    "critical": {
        "weight": 0.45,
        "words": [
            "lawyer", "lawsuit", "sue", "legal action", "solicitor",
            "court", "police", "fraud", "scam", "stolen", "hacked",
            "data breach", "identity theft", "emergency", "dangerous",
            "injury", "fire", "immediately or else", "final warning",
            "completely broken", "totally down", "cannot work at all",
            "threatening to", "take action against"
        ]
    },
    "high": {
        "weight": 0.25,
        "words": [
            "urgent", "urgently", "critical", "immediately", "asap",
            "right now", "cannot wait", "losing money", "losing business",
            "affecting my business", "deadline", "losing clients",
            "escalate", "escalation", "supervisor", "manager", "complaint",
            "three days", "four days", "five days", "a week",
            "still broken", "still not fixed", "never resolved",
            "second time calling", "third time", "multiple times",
            "already called", "called before", "been waiting days"
        ]
    },
    "medium": {
        "weight": 0.12,
        "words": [
            "frustrated", "annoyed", "unacceptable", "disappointed",
            "not working", "broken", "issue", "problem", "error",
            "incorrect", "wrong", "missing", "not received",
            "waiting", "delayed", "slow", "no response",
            "two days", "yesterday", "last week", "still waiting",
            "unhappy", "not satisfied", "need help", "please fix"
        ]
    },
    "cancellation": {
        "weight": 0.30,
        "words": [
            "cancel", "cancellation", "terminate", "close account",
            "leave", "switch provider", "going elsewhere", "competitor",
            "done with you", "fed up", "had enough", "last chance",
            "final straw", "moving on", "cancelling subscription",
            "port my number", "transfer my account"
        ]
    },
    "positive_reducers": {
        "weight": -0.12,
        "words": [
            "thank you", "thanks", "great", "excellent", "happy",
            "resolved", "fixed", "working now", "perfect", "satisfied",
            "appreciate", "helpful", "wonderful", "brilliant", "amazing",
            "no problem", "all good", "sorted", "you helped me"
        ]
    }
}


def compute_keyword_score(text: str) -> dict:
    """
    Scan transcript for urgency keywords.

    Returns:
    {
        "score"       : float 0.0–1.0
        "matched"     : list of (keyword, category, weight)
        "by_category" : dict category → [matched words]
    }
    """
    text_lower = text.lower()
    score      = 0.0
    matched    = []
    by_category = {cat: [] for cat in KEYWORD_GROUPS}

    for category, config in KEYWORD_GROUPS.items():
        for kw in config["words"]:
            if kw in text_lower:
                score += config["weight"]
                matched.append((kw, category, config["weight"]))
                by_category[category].append(kw)

    # Cap and floor
    score = max(0.0, min(1.0, score))

    return {
        "score"      : round(score, 4),
        "matched"    : matched,
        "by_category": by_category
    }


def compute_metadata_score(metadata: dict) -> float:
    """
    Score urgency from call metadata signals.
    All fields optional — missing fields safely ignored.
    """
    if not metadata:
        return 0.0

    score = 0.0

    # Call duration (long = complex problem)
    duration = metadata.get("duration_seconds", 0)
    if   duration > 600: score += 0.30
    elif duration > 300: score += 0.18
    elif duration > 120: score += 0.08

    # Repeat caller (unresolved issue)
    if metadata.get("repeat_caller", False):
        score += 0.28

    # Wait time (already frustrated before speaking)
    wait = metadata.get("wait_time_seconds", 0)
    if   wait > 600: score += 0.25
    elif wait > 300: score += 0.18
    elif wait > 120: score += 0.10

    # Multiple calls today (persistent issue)
    calls_today = metadata.get("call_number_today", 1)
    if   calls_today >= 3: score += 0.28
    elif calls_today == 2: score += 0.14

    return round(max(0.0, min(1.0, score)), 4)


def compute_emotion_intensity(sentiment_result: dict) -> float:
    """
    Derive an emotion intensity signal from sentiment analysis features.

    This is the WEIGHT_EMOTION component — it captures things
    that neither pure polarity nor keywords catch alone:
    - Escalating trajectory is more urgent than flat negative
    - High intensifier count = heightened emotion
    - High negation density in negative text = emphatic rejection
    """
    score = 0.0

    # Trajectory escalation bonus
    trajectory = sentiment_result.get("trajectory", "stable")
    if   trajectory == "escalating":    score += 0.40
    elif trajectory == "de-escalating": score -= 0.20    # being resolved

    # Intensifier density
    wc        = max(sentiment_result.get("word_count", 1), 1)
    int_count = sentiment_result.get("intensifier_count", 0)
    int_density = int_count / wc
    score += min(0.30, int_density * 3.0)

    # High negation in negative text = emphatic refusal/complaint
    if sentiment_result.get("label") == "negative":
        neg_count   = sentiment_result.get("negation_count", 0)
        neg_density = neg_count / wc
        score += min(0.20, neg_density * 2.0)

    # Very low neutral_score in a confirmed negative call
    # means the negativity is unambiguous
    if (sentiment_result.get("label") == "negative" and
            sentiment_result.get("neutral_score", 1.0) < 0.15):
        score += 0.15

    return round(max(0.0, min(1.0, score)), 4)


def score_urgency(
    text: str,
    sentiment_result: dict,
    metadata: dict = None
) -> dict:
    """
    Combine all signals into a final urgency score and priority.

    Returns:
    {
        "urgency_score"    : float 0.0–1.0
        "priority"         : "P1" | "P2" | "P3" | "P4"
        "priority_label"   : "Critical" | "High" | "Medium" | "Low"
        "action"           : str  recommended helpdesk action
        "action_detail"    : str  fuller explanation
        "components"       : dict  per-signal breakdown
        "matched_keywords" : list  urgency keywords found
        "keyword_categories": dict  keywords by category
        "escalation_flag"  : bool  True if P1 or escalating trajectory
    }
    """
    # ── Individual component scores ──────────────────────────────────────
    ml_score       = _polarity_to_ml_score(
        sentiment_result.get("polarity", 0.0),
        sentiment_result.get("label", "neutral")
    )
    kw_result      = compute_keyword_score(text)
    kw_score       = kw_result["score"]
    meta_score     = compute_metadata_score(metadata)
    emotion_score  = compute_emotion_intensity(sentiment_result)

    # ── Weighted combination ──────────────────────────────────────────────
    if metadata:
        final = (
            ml_score      * WEIGHT_ML       +
            kw_score      * WEIGHT_KEYWORDS +
            meta_score    * WEIGHT_METADATA +
            emotion_score * WEIGHT_EMOTION
        )
        weights_used = {
            "ml"      : WEIGHT_ML,
            "keywords": WEIGHT_KEYWORDS,
            "metadata": WEIGHT_METADATA,
            "emotion" : WEIGHT_EMOTION
        }
    else:
        final = (
            ml_score      * WEIGHT_ML_NO_META       +
            kw_score      * WEIGHT_KEYWORDS_NO_META +
            emotion_score * WEIGHT_EMOTION_NO_META
        )
        weights_used = {
            "ml"      : WEIGHT_ML_NO_META,
            "keywords": WEIGHT_KEYWORDS_NO_META,
            "metadata": 0.0,
            "emotion" : WEIGHT_EMOTION_NO_META
        }

    urgency_score = round(max(0.0, min(1.0, final)), 4)

    # ── Priority mapping ──────────────────────────────────────────────────
    priority, label, action, detail = _map_priority(
        urgency_score,
        sentiment_result.get("trajectory", "stable"),
        kw_result["by_category"].get("cancellation", [])
    )

    escalation_flag = (
        priority == "P1" or
        sentiment_result.get("trajectory") == "escalating"
    )

    components = {
        "ml_score"      : round(ml_score      * weights_used["ml"],       4),
        "keyword_score" : round(kw_score       * weights_used["keywords"], 4),
        "metadata_score": round(meta_score     * weights_used["metadata"], 4),
        "emotion_score" : round(emotion_score  * weights_used["emotion"],  4),
        "raw_ml"        : round(ml_score, 4),
        "raw_keywords"  : round(kw_score, 4),
        "raw_metadata"  : round(meta_score, 4),
        "raw_emotion"   : round(emotion_score, 4),
        "total"         : urgency_score
    }

    log.debug(
        "Urgency | %s %.2f | ml=%.2f kw=%.2f meta=%.2f emo=%.2f",
        priority, urgency_score,
        ml_score, kw_score, meta_score, emotion_score
    )

    return {
        "urgency_score"     : urgency_score,
        "priority"          : priority,
        "priority_label"    : label,
        "action"            : action,
        "action_detail"     : detail,
        "components"        : components,
        "matched_keywords"  : [m[0] for m in kw_result["matched"]],
        "keyword_categories": kw_result["by_category"],
        "escalation_flag"   : escalation_flag
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _polarity_to_ml_score(polarity: float, label: str) -> float:
    """
    Map polarity [-1, +1] to urgency contribution [0, 1].

    Neutral calls: mapped to 0.3 (not zero — they still need handling).
    Negative calls: urgency scales from 0.5 → 1.0.
    Positive calls: urgency scales from 0.0 → 0.3.
    """
    if label == "neutral":
        # Neutral: low-medium urgency
        return 0.30

    if label == "negative":
        # polarity -0.20 → 0.50,  polarity -1.0 → 1.0
        return round(0.50 + abs(polarity) * 0.50, 4)

    # positive: polarity +0.20 → 0.25,  polarity +1.0 → 0.0
    return round(max(0.0, 0.25 - polarity * 0.25), 4)


def _map_priority(
    score: float,
    trajectory: str,
    cancel_keywords: list
) -> tuple:
    """
    Map urgency score + context signals to a priority level
    with specific helpdesk action guidance.
    """
    # Trajectory override: escalating calls jump one tier up
    effective_score = score
    if trajectory == "escalating":
        effective_score = min(1.0, score + 0.08)

    # Cancellation intent: always at least P2
    if cancel_keywords and effective_score < PRIORITY_P2:
        effective_score = PRIORITY_P2

    if effective_score >= PRIORITY_P1:
        return (
            "P1",
            "Critical",
            "Escalate immediately to senior agent",
            (
                "This call requires IMMEDIATE attention. Assign the most senior "
                "available agent. Log as a formal complaint. Alert supervisor "
                "on duty. Do not place caller on hold. If legal threats were "
                "made, escalate to customer relations team and document verbatim."
            )
        )
    elif effective_score >= PRIORITY_P2:
        return (
            "P2",
            "High",
            "Route to senior agent within 5 minutes",
            (
                "This caller is significantly distressed or has a complex issue. "
                "Assign an experienced agent within 5 minutes. Acknowledge the "
                "wait and apologise proactively. "
                + ("Caller has expressed cancellation intent — retention protocol applies. " if cancel_keywords else "")
                + "Log all actions taken."
            )
        )
    elif effective_score >= PRIORITY_P3:
        return (
            "P3",
            "Medium",
            "Standard queue — assign within 15 minutes",
            (
                "Routine issue requiring a human agent. No escalation needed yet. "
                "Standard response time applies. If wait exceeds 15 minutes, "
                "upgrade to P2."
            )
        )
    else:
        return (
            "P4",
            "Low",
            "Self-service or scheduled callback",
            (
                "Low-complexity enquiry. Offer self-service resources first "
                "(FAQ, knowledge base, automated systems). If caller prefers "
                "human assistance, schedule a callback within 24 hours."
            )
        )