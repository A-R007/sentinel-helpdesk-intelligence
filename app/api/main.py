# app/api/main.py
# ═══════════════════════════════════════════════════════════
# FastAPI application — all routes, startup, error handling.
# ═══════════════════════════════════════════════════════════

import os
import io
import uuid
import shutil
import time
from pathlib import Path
from typing  import Optional

from fastapi               import FastAPI, File, UploadFile, HTTPException, Request, Form, Query
from fastapi.responses     import JSONResponse, StreamingResponse
from fastapi.staticfiles   import StaticFiles
from fastapi.templating    import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config      import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, UPLOAD_DIR
)
from app.utils.logger import get_logger
from app.core.transcriber import load_whisper_model, transcribe_audio
from app.core.sentiment   import load_sentiment_model, analyze_sentiment
from app.core.urgency     import score_urgency
from app.db.database      import (
    init_db, save_call, get_call_by_id, get_all_calls,
    get_stats, delete_call, update_notes, export_csv
)

log = get_logger(__name__)

# ─────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────

app = FastAPI(
    title       = API_TITLE,
    version     = API_VERSION,
    description = API_DESCRIPTION,
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Static files and templates (relative to project root)
_ROOT = Path(__file__).resolve().parent.parent.parent
app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(_ROOT / "templates"))


# ─────────────────────────────────────────────
# Startup — load models once
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    log.info("=" * 56)
    log.info("  %s v%s", API_TITLE, API_VERSION)
    log.info("=" * 56)

    init_db()

    log.info("Loading Whisper STT model ...")
    app.state.whisper = load_whisper_model()

    log.info("Loading DistilBERT sentiment model ...")
    app.state.sentiment = load_sentiment_model()

    log.info("All models ready. Server accepting requests.")


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def _build_response(
    call_id: int,
    transcript: dict,
    sentiment:  dict,
    urgency:    dict,
    elapsed:    float
) -> dict:
    """Consistent response structure for all analysis endpoints."""
    return {
        "call_id"    : call_id,
        "status"     : "success",
        "time_taken" : elapsed,
        "transcript" : {
            "text"        : transcript["text"],
            "language"    : transcript.get("language", "en"),
            "duration"    : transcript.get("duration", 0),
            "speech_ratio": transcript.get("speech_ratio", 1.0)
        },
        "sentiment"  : {
            "label"             : sentiment["label"],
            "polarity"          : sentiment["polarity"],
            "confidence"        : sentiment["confidence"],
            "neutral_score"     : sentiment["neutral_score"],
            "trajectory"        : sentiment["trajectory"],
            "neutral_signals"   : sentiment.get("neutral_signals", {}),
            "intensifier_count" : sentiment.get("intensifier_count", 0),
            "negation_count"    : sentiment.get("negation_count", 0),
            "enquiry_ratio"     : sentiment.get("enquiry_ratio", 0)
        },
        "urgency"    : {
            "score"            : urgency["urgency_score"],
            "priority"         : urgency["priority"],
            "priority_label"   : urgency["priority_label"],
            "action"           : urgency["action"],
            "action_detail"    : urgency["action_detail"],
            "components"       : urgency["components"],
            "keywords_found"   : urgency["matched_keywords"],
            "escalation_flag"  : urgency["escalation_flag"]
        }
    }


# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────

@app.get("/")
@app.get("/dashboard")
async def dashboard(request: Request):
    stats = get_stats()
    calls = get_all_calls(page=1, limit=25)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats"  : stats,
        "calls"  : calls["calls"]
    })


# ─────────────────────────────────────────────
# POST /api/analyze — audio file upload
# ─────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_audio(
    file              : UploadFile = File(...),
    repeat_caller     : bool = Form(False),
    wait_time_seconds : int  = Form(0),
    call_number_today : int  = Form(1),
    notes             : str  = Form("")
):
    """
    Upload an audio file (.wav/.mp3/.flac/.m4a/.ogg).
    Runs full pipeline: STT → Sentiment → Urgency.
    """
    t0  = time.perf_counter()
    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Save upload to temp file
    tmp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        size_mb = tmp.stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(
                413, f"File too large ({size_mb:.1f} MB). Max: {MAX_UPLOAD_MB} MB"
            )

        # ── Pipeline ────────────────────────────────────────────────────
        transcript = transcribe_audio(app.state.whisper, str(tmp))
        if not transcript or not transcript["text"].strip():
            raise HTTPException(
                422, "No speech detected. Check audio quality and try again."
            )

        sentiment = analyze_sentiment(app.state.sentiment, transcript["text"])

        metadata  = {
            "duration_seconds" : transcript["duration"],
            "repeat_caller"    : repeat_caller,
            "wait_time_seconds": wait_time_seconds,
            "call_number_today": call_number_today
        }
        urgency   = score_urgency(transcript["text"], sentiment, metadata)

        call_id   = save_call(
            transcript, sentiment, urgency,
            filename=file.filename, metadata=metadata,
            source="upload", notes=notes
        )

        elapsed = round(time.perf_counter() - t0, 2)
        log.info(
            "Analyzed upload '%s' → call#%d %s %.2f (%ss)",
            file.filename, call_id, urgency["priority"],
            urgency["urgency_score"], elapsed
        )

        return JSONResponse(_build_response(call_id, transcript, sentiment, urgency, elapsed))

    finally:
        if tmp.exists():
            tmp.unlink()


# ─────────────────────────────────────────────
# POST /api/analyze/text — paste transcript
# ─────────────────────────────────────────────

@app.post("/api/analyze/text")
async def analyze_text(
    transcript        : str  = Form(...),
    caller_name       : str  = Form(""),
    repeat_caller     : bool = Form(False),
    wait_time_seconds : int  = Form(0),
    call_number_today : int  = Form(1),
    notes             : str  = Form("")
):
    """
    Analyse a typed or pasted transcript directly.
    Skips Whisper — sentiment + urgency only.
    """
    if not transcript or len(transcript.strip()) < 5:
        raise HTTPException(400, "Transcript must be at least 5 characters.")

    t0       = time.perf_counter()
    text     = transcript.strip()
    wc       = len(text.split())
    duration = round(wc * 0.4, 1)    # ~150 wpm estimate

    sentiment = analyze_sentiment(app.state.sentiment, text)

    metadata  = {
        "duration_seconds" : duration,
        "repeat_caller"    : repeat_caller,
        "wait_time_seconds": wait_time_seconds,
        "call_number_today": call_number_today,
        "caller_name"      : caller_name
    }
    urgency   = score_urgency(text, sentiment, metadata)

    tr = {"text": text, "language": "en", "duration": duration, "speech_ratio": 1.0}

    call_id = save_call(
        tr, sentiment, urgency,
        filename=f"text_{caller_name or 'entry'}",
        metadata=metadata, source="text", notes=notes
    )

    elapsed = round(time.perf_counter() - t0, 2)
    log.info(
        "Analyzed text entry → call#%d %s %.2f (%ss)",
        call_id, urgency["priority"], urgency["urgency_score"], elapsed
    )

    return JSONResponse(_build_response(call_id, tr, sentiment, urgency, elapsed))


# ─────────────────────────────────────────────
# GET /api/calls
# ─────────────────────────────────────────────

@app.get("/api/calls")
async def list_calls(
    page           : int  = Query(1,   ge=1),
    limit          : int  = Query(25,  ge=1, le=100),
    sentiment      : Optional[str]  = None,
    priority       : Optional[str]  = None,
    search         : Optional[str]  = None,
    source         : Optional[str]  = None,
    escalation_only: bool           = False
):
    result = get_all_calls(
        page=page, limit=limit, sentiment=sentiment,
        priority=priority, search=search, source=source,
        escalation_only=escalation_only
    )
    return JSONResponse(result)


# ─────────────────────────────────────────────
# GET /api/calls/{id}
# ─────────────────────────────────────────────

@app.get("/api/calls/{call_id}")
async def get_call(call_id: int):
    call = get_call_by_id(call_id)
    if call is None:
        raise HTTPException(404, f"Call #{call_id} not found.")
    return JSONResponse(call)


# ─────────────────────────────────────────────
# PATCH /api/calls/{id}/notes
# ─────────────────────────────────────────────

@app.patch("/api/calls/{call_id}/notes")
async def patch_notes(call_id: int, notes: str = Form(...)):
    ok = update_notes(call_id, notes)
    if not ok:
        raise HTTPException(404, f"Call #{call_id} not found.")
    return JSONResponse({"status": "updated", "call_id": call_id})


# ─────────────────────────────────────────────
# DELETE /api/calls/{id}
# ─────────────────────────────────────────────

@app.delete("/api/calls/{call_id}")
async def remove_call(call_id: int):
    ok = delete_call(call_id)
    if not ok:
        raise HTTPException(404, f"Call #{call_id} not found.")
    return JSONResponse({"status": "deleted", "call_id": call_id})


# ─────────────────────────────────────────────
# GET /api/stats
# ─────────────────────────────────────────────

@app.get("/api/stats")
async def stats():
    return JSONResponse(get_stats())


# ─────────────────────────────────────────────
# GET /api/export/csv
# ─────────────────────────────────────────────

@app.get("/api/export/csv")
async def export():
    content = export_csv()
    return StreamingResponse(
        io.StringIO(content),
        media_type = "text/csv",
        headers    = {"Content-Disposition": "attachment; filename=calls_export.csv"}
    )


# ─────────────────────────────────────────────
# GET /api/health
# ─────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status"       : "ok",
        "version"      : API_VERSION,
        "models_loaded": True
    }