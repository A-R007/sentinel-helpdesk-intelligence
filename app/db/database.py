# app/db/database.py
# ═══════════════════════════════════════════════════════════
# SQLite database — schema, CRUD, pagination, search, export.
# ═══════════════════════════════════════════════════════════

import sqlite3
import json
import csv
import io
from datetime  import datetime
from pathlib   import Path
from contextlib import contextmanager
from typing    import Optional

from app.utils.logger import get_logger
from app.config       import DB_PATH, CALLS_PAGE_SIZE

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Connection context manager
# ─────────────────────────────────────────────

@contextmanager
def get_connection():
    """
    Yield a SQLite connection with row_factory set.
    Always closes the connection, even on error.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")    # better concurrent read perf
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    source              TEXT    NOT NULL DEFAULT 'upload',  -- 'upload'|'text'|'dataset'
    filename            TEXT,
    duration            REAL    DEFAULT 0,
    transcript          TEXT    NOT NULL DEFAULT '',
    language            TEXT    DEFAULT 'en',
    speech_ratio        REAL    DEFAULT 1.0,
    sentiment_label     TEXT    NOT NULL DEFAULT 'neutral',
    sentiment_score     REAL    DEFAULT 0,
    polarity            REAL    DEFAULT 0,
    neutral_score       REAL    DEFAULT 0,
    trajectory          TEXT    DEFAULT 'stable',
    intensifier_count   INTEGER DEFAULT 0,
    negation_count      INTEGER DEFAULT 0,
    enquiry_ratio       REAL    DEFAULT 0,
    urgency_score       REAL    DEFAULT 0,
    priority            TEXT    DEFAULT 'P4',
    priority_label      TEXT    DEFAULT 'Low',
    action              TEXT    DEFAULT '',
    action_detail       TEXT    DEFAULT '',
    escalation_flag     INTEGER DEFAULT 0,
    matched_keywords    TEXT    DEFAULT '[]',
    metadata            TEXT    DEFAULT '{}',
    notes               TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_calls_created_at     ON calls(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_sentiment_label ON calls(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_calls_priority        ON calls(priority);
CREATE INDEX IF NOT EXISTS idx_calls_urgency_score   ON calls(urgency_score DESC);
CREATE INDEX IF NOT EXISTS idx_calls_escalation      ON calls(escalation_flag);
"""


def init_db():
    """Create database schema if it doesn't exist yet."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    log.info("Database initialised: %s", DB_PATH)


# ─────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────

def save_call(
    transcript_result: dict,
    sentiment_result:  dict,
    urgency_result:    dict,
    filename:          str  = "unknown",
    metadata:          dict = None,
    source:            str  = "upload",
    notes:             str  = ""
) -> int:
    """
    Insert a fully analysed call into the database.
    Returns the new row ID.
    """
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO calls (
                source, filename, duration, transcript, language, speech_ratio,
                sentiment_label, sentiment_score, polarity, neutral_score,
                trajectory, intensifier_count, negation_count, enquiry_ratio,
                urgency_score, priority, priority_label,
                action, action_detail, escalation_flag,
                matched_keywords, metadata, notes
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            source,
            filename,
            transcript_result.get("duration", 0),
            transcript_result.get("text", ""),
            transcript_result.get("language", "en"),
            transcript_result.get("speech_ratio", 1.0),
            sentiment_result.get("label", "neutral"),
            sentiment_result.get("confidence", 0),
            sentiment_result.get("polarity", 0),
            sentiment_result.get("neutral_score", 0),
            sentiment_result.get("trajectory", "stable"),
            sentiment_result.get("intensifier_count", 0),
            sentiment_result.get("negation_count", 0),
            sentiment_result.get("enquiry_ratio", 0),
            urgency_result.get("urgency_score", 0),
            urgency_result.get("priority", "P4"),
            urgency_result.get("priority_label", "Low"),
            urgency_result.get("action", ""),
            urgency_result.get("action_detail", ""),
            int(urgency_result.get("escalation_flag", False)),
            json.dumps(urgency_result.get("matched_keywords", [])),
            json.dumps(metadata or {}),
            notes
        ))
        call_id = cursor.lastrowid

    log.info(
        "Saved call #%d | %s | %s | urgency %.2f",
        call_id, source,
        urgency_result.get("priority"),
        urgency_result.get("urgency_score", 0)
    )
    return call_id


def delete_call(call_id: int) -> bool:
    """Delete a call by ID. Returns True if a row was deleted."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM calls WHERE id=?", (call_id,))
        deleted = cursor.rowcount > 0
    if deleted:
        log.info("Deleted call #%d", call_id)
    else:
        log.warning("Delete: call #%d not found", call_id)
    return deleted


def update_notes(call_id: int, notes: str) -> bool:
    """Update the free-text notes field on a call."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE calls SET notes=? WHERE id=?", (notes, call_id)
        )
        return cursor.rowcount > 0


# ─────────────────────────────────────────────
# Read operations
# ─────────────────────────────────────────────

def get_call_by_id(call_id: int) -> Optional[dict]:
    """Fetch a single call. Returns None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM calls WHERE id=?", (call_id,)
        ).fetchone()

    if row is None:
        return None
    return _deserialize(dict(row))


def get_all_calls(
    page:      int  = 1,
    limit:     int  = CALLS_PAGE_SIZE,
    sentiment: str  = None,
    priority:  str  = None,
    search:    str  = None,
    source:    str  = None,
    escalation_only: bool = False
) -> dict:
    """
    Paginated, filtered call list.

    Returns:
    {
        "calls"    : list of call dicts
        "total"    : int  total matching records
        "page_info": { current_page, total_pages, has_more, ... }
    }
    """
    conditions, params = [], []

    if sentiment:
        conditions.append("LOWER(sentiment_label)=?")
        params.append(sentiment.lower())
    if priority:
        conditions.append("UPPER(priority)=?")
        params.append(priority.upper())
    if source:
        conditions.append("source=?")
        params.append(source)
    if escalation_only:
        conditions.append("escalation_flag=1")
    if search:
        conditions.append("(LOWER(transcript) LIKE ? OR LOWER(filename) LIKE ?)")
        params += [f"%{search.lower()}%", f"%{search.lower()}%"]

    where  = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (max(1, page) - 1) * limit

    with get_connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM calls {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT * FROM calls {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()

    calls = [_deserialize(dict(r)) for r in rows]

    return {
        "calls"    : calls,
        "total"    : total,
        "page_info": {
            "current_page": page,
            "total_pages" : max(1, -(-total // limit)),
            "has_more"    : (offset + limit) < total,
            "limit"       : limit,
            "offset"      : offset
        }
    }


def get_stats() -> dict:
    """Aggregate statistics for the dashboard."""
    with get_connection() as conn:

        total = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        if total == 0:
            return _empty_stats()

        # Sentiment distribution
        sent_rows = conn.execute(
            "SELECT sentiment_label, COUNT(*) FROM calls GROUP BY sentiment_label"
        ).fetchall()
        sentiment_counts = {r[0]: r[1] for r in sent_rows}

        # Priority distribution
        pri_rows = conn.execute(
            "SELECT priority, COUNT(*) FROM calls GROUP BY priority"
        ).fetchall()
        priority_counts = {r[0]: r[1] for r in pri_rows}

        # Trajectory distribution
        traj_rows = conn.execute(
            "SELECT trajectory, COUNT(*) FROM calls GROUP BY trajectory"
        ).fetchall()
        trajectory_counts = {r[0]: r[1] for r in traj_rows}

        # Averages
        avgs = conn.execute("""
            SELECT
                ROUND(AVG(polarity),      4) as avg_polarity,
                ROUND(AVG(urgency_score), 4) as avg_urgency,
                ROUND(AVG(duration),      1) as avg_duration,
                ROUND(AVG(neutral_score), 4) as avg_neutral_score,
                SUM(escalation_flag)         as escalation_count
            FROM calls
        """).fetchone()

        # Recent trend (last 20 calls)
        trend_rows = conn.execute("""
            SELECT polarity, urgency_score, priority, created_at
            FROM calls ORDER BY id DESC LIMIT 20
        """).fetchall()

    return {
        "total_calls"       : total,
        "sentiment_counts"  : sentiment_counts,
        "priority_counts"   : priority_counts,
        "trajectory_counts" : trajectory_counts,
        "avg_polarity"      : avgs[0] or 0.0,
        "avg_urgency"       : avgs[1] or 0.0,
        "avg_duration_secs" : avgs[2] or 0.0,
        "avg_neutral_score" : avgs[3] or 0.0,
        "escalation_count"  : avgs[4] or 0,
        "recent_trend"      : [dict(r) for r in trend_rows]
    }


def export_csv() -> str:
    """Return all calls serialised as a CSV string."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM calls ORDER BY id DESC"
        ).fetchall()

    buf = io.StringIO()
    if not rows:
        return ""

    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))

    return buf.getvalue()


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _deserialize(row: dict) -> dict:
    """Decode JSON fields back to Python objects."""
    row["matched_keywords"] = json.loads(row.get("matched_keywords") or "[]")
    row["metadata"]         = json.loads(row.get("metadata")         or "{}")
    return row


def _empty_stats() -> dict:
    return {
        "total_calls": 0, "sentiment_counts": {},
        "priority_counts": {}, "trajectory_counts": {},
        "avg_polarity": 0.0, "avg_urgency": 0.0,
        "avg_duration_secs": 0.0, "avg_neutral_score": 0.0,
        "escalation_count": 0, "recent_trend": []
    }