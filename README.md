<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Syne&weight=800&size=40&pause=1000&color=6366F1&center=true&vCenter=true&width=700&height=80&lines=SENTINEL;Helpdesk+Intelligence+Platform" alt="Sentinel" />

<br/>

**Real-time sentiment analysis and priority routing for helpdesk calls.**
**Runs entirely on CPU — no GPU, no cloud, no per-call cost.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/Whisper-tiny%20%7C%20CPU-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![DistilBERT](https://img.shields.io/badge/DistilBERT-SST--2-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-22C984?style=for-the-badge)](LICENSE)

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

</div>

<br/>

## 📋 Table of Contents

- [Overview](#-overview)
- [The Problem We Solve](#-the-problem-we-solve)
- [System Architecture](#-system-architecture)
- [Pipeline Flow](#-pipeline-flow)
- [The Neutral Detection Model](#-the-neutral-detection-model)
- [Urgency Scoring Engine](#-urgency-scoring-engine)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Dashboard](#-dashboard)
- [Configuration](#-configuration)
- [Results](#-results)
- [Future Enhancements](#-future-enhancements)

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🎯 Overview

**Sentinel** is a production-grade helpdesk intelligence platform that automatically analyses incoming customer support calls for sentiment, urgency, and priority — in real time, on your own hardware, with zero cloud dependency.

Most call centres manually review fewer than **5% of calls**. Sentinel covers **100%** — automatically transcribing, classifying, and routing every single call within seconds.

```
Voice Recording  ──►  Whisper STT  ──►  DistilBERT + Neutral Layer  ──►  Urgency Score  ──►  P1–P4 Priority
```

<br/>

> 🎓 **Academic Project** — B.Tech in Artificial Intelligence and Machine Learning
> Bannari Amman Institute of Technology, affiliated to Anna University, Chennai.
> **Student:** Abdul Rahman (7376222AL103)
> **Guide:** Dr. Nivedha S, Assistant Professor III, Dept. of Information Technology

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## ❗ The Problem We Solve

| Problem | Impact | Sentinel's Solution |
|---------|--------|---------------------|
| Manual QA reviews < 5% of calls | 95% of calls carry no intelligence | Automated analysis of 100% of calls |
| Enquiries misclassified as negative | False escalations waste agent time | 4-signal neutral detection model |
| Flat binary sentiment output | No actionable routing signal | P1–P4 urgency scoring with trajectory |
| Cloud APIs cost money per call | Unsustainable at scale | 100% on-premise, zero per-call cost |
| Audio sent to external services | Data privacy / GDPR risk | All processing stays on your hardware |

<br/>

### The Enquiry Misclassification Problem

```
Customer: "Can I update my billing address?"

❌  Raw DistilBERT-SST2:   NEGATIVE  (0.63 confidence)
                            ↑ Misclassifies neutral enquiries as negative
                              because "update" appears in complaint data

✅  Sentinel (4-signal):    NEUTRAL   (neutral_score: 0.71)
                            ↑ Detects question structure, admin vocabulary,
                              low confidence band → correctly neutral
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SENTINEL ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐  │
│   │   Dashboard  │────►│   FastAPI    │────►│   ML Pipeline        │  │
│   │  (Jinja2 +  │     │   REST API   │     │                      │  │
│   │  Vanilla JS) │◄────│  /api/*      │◄────│  ┌────────────────┐  │  │
│   └──────────────┘     └──────┬───────┘     │  │  app/core/     │  │  │
│                               │             │  │  audio.py      │  │  │
│   ┌──────────────┐            │             │  │  transcriber.py│  │  │
│   │   SQLite DB  │◄───────────┘             │  │  sentiment.py  │  │  │
│   │  (WAL mode)  │                          │  │  urgency.py    │  │  │
│   └──────────────┘                          │  └────────────────┘  │  │
│                                             └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Breakdown

```
┌─────────────────────────────────────────────┐
│            PRESENTATION LAYER               │
│  • Jinja2-rendered dashboard (3 tabs)       │
│  • Drag-and-drop audio upload               │
│  • Text transcript entry form               │
│  • Live paginated + filterable call history │
│  • Per-call detail modal with signal scores │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│               API LAYER                     │
│  • FastAPI + Uvicorn (ASGI)                 │
│  • 9 REST endpoints                         │
│  • CORS middleware                          │
│  • File upload with UUID temp paths         │
│  • Auto-generated /api/docs (Swagger)       │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│              CORE ML LAYER                  │
│  • audio.py      → librosa loading, VAD     │
│  • transcriber.py→ Whisper STT (dual backend│
│  • sentiment.py  → DistilBERT + 4-signal    │
│  • urgency.py    → weighted scoring engine  │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│            DATABASE LAYER                   │
│  • SQLite 3 with WAL journal mode           │
│  • Context-managed connections              │
│  • 5 query-optimised indexes                │
│  • Pagination + full-text search            │
│  • CSV export                               │
└─────────────────────────────────────────────┘
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🔄 Pipeline Flow

```
                    ┌─────────────────────────────────────┐
                    │          INPUT SOURCES              │
                    └───────────┬─────────────┬───────────┘
                                │             │
                    ┌───────────▼──┐     ┌────▼──────────┐
                    │  Audio File  │     │ Text Transcript│
                    │ .wav .mp3    │     │ (paste/type)   │
                    │ .flac .m4a   │     └────────┬───────┘
                    └───────────┬──┘              │
                                │                 │
               ┌────────────────▼──────────┐      │
               │    STAGE 1: AUDIO PREP    │      │
               │  • Format validation      │      │
               │  • librosa load @ 16kHz   │      │
               │  • Amplitude normalise    │      │
               │  • Speech ratio (VAD)     │      │
               └────────────────┬──────────┘      │
                                │                 │
               ┌────────────────▼──────────┐      │
               │  STAGE 2: WHISPER STT     │      │
               │  • faster-whisper int8    │      │
               │  • Built-in VAD filter    │      │
               │  • Beam=3, Temp=0.0       │      │
               │  • Noise token removal    │      │
               │  • Segment deduplication  │      │
               └────────────────┬──────────┘      │
                                │                 │
               ┌────────────────▼─────────────────▼──────┐
               │         STAGE 3: RAW SENTIMENT          │
               │  • DistilBERT-SST2 (top_k=None)         │
               │  • Returns P(POS) and P(NEG) both        │
               │  • Polarity = P(POS) - P(NEG) → [-1,+1] │
               │  • Overlapping 380-word chunks           │
               └────────────────┬────────────────────────┘
                                │
               ┌────────────────▼────────────────────────┐
               │      STAGE 4: NEUTRAL DETECTION         │
               │                                         │
               │  S1: Confidence Band   (weight 0.30)    │
               │  S2: Polarity Magnitude(weight 0.25)    │
               │  S3: Enquiry Ratio     (weight 0.30)    │
               │  S4: Vocabulary Context(weight 0.15)    │
               │                                         │
               │  neutral_score = Σ(Si × wi)             │
               │  if score ≥ 0.45 → override to NEUTRAL  │
               └────────────────┬────────────────────────┘
                                │
               ┌────────────────▼────────────────────────┐
               │       STAGE 5: URGENCY SCORING          │
               │                                         │
               │  ML score      (45%) ──┐                │
               │  Keyword score (30%) ──┼──► 0.0 to 1.0  │
               │  Metadata score(15%) ──┤                │
               │  Emotion score (10%) ──┘                │
               │                                         │
               │  + Trajectory bonus (+0.08 if escalating)│
               │  + Cancellation floor (min P2 if cancel) │
               └────────────────┬────────────────────────┘
                                │
               ┌────────────────▼────────────────────────┐
               │        STAGE 6: PRIORITY MAPPING        │
               │                                         │
               │  ≥ 0.78  →  P1 CRITICAL  🔴            │
               │  ≥ 0.56  →  P2 HIGH      🟠            │
               │  ≥ 0.33  →  P3 MEDIUM    🔵            │
               │  < 0.33  →  P4 LOW       🟢            │
               └────────────────┬────────────────────────┘
                                │
               ┌────────────────▼────────────────────────┐
               │     STAGE 7: PERSIST + RESPOND          │
               │  • SQLite write (WAL mode)               │
               │  • JSON response to API caller           │
               │  • Dashboard renders result card         │
               └─────────────────────────────────────────┘
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🧠 The Neutral Detection Model

The biggest problem with applying general sentiment models to helpdesk calls is that **neutral enquiries get misclassified as negative**. Sentinel solves this with a four-signal weighted vote system.

### Why It Happens

```
Training data (movie reviews):    "The plot was a problem" → NEGATIVE ✓
Helpdesk call:                    "I have a billing problem" → NEGATIVE ✗

The word "problem" is negative in movie reviews.
In a helpdesk context it is completely neutral — just a descriptor.
```

### The Four Signals

```
┌──────────────────────────────────────────────────────────────────┐
│                  4-SIGNAL NEUTRAL DETECTION                      │
├─────────────────────────────┬────────────┬───────────────────────┤
│ Signal                      │ Weight     │ What It Detects       │
├─────────────────────────────┼────────────┼───────────────────────┤
│ S1: Confidence Band         │   0.30     │ Model confidence      │
│                             │            │ below 0.72 threshold  │
│                             │            │ → model is uncertain  │
├─────────────────────────────┼────────────┼───────────────────────┤
│ S2: Polarity Magnitude      │   0.25     │ |polarity| < 0.20     │
│                             │            │ → signal too weak     │
│                             │            │ for real sentiment    │
├─────────────────────────────┼────────────┼───────────────────────┤
│ S3: Enquiry Ratio           │   0.30     │ Question marks,       │
│                             │            │ question words, modal │
│                             │            │ verbs > 8% of text    │
├─────────────────────────────┼────────────┼───────────────────────┤
│ S4: Vocabulary Context      │   0.15     │ Admin vocab present + │
│                             │            │ NO intensifiers       │
│                             │            │ = factual enquiry     │
└─────────────────────────────┴────────────┴───────────────────────┘

neutral_score = (S1 × 0.30) + (S2 × 0.25) + (S3 × 0.30) + (S4 × 0.15)

if neutral_score ≥ 0.45  →  NEUTRAL  (regardless of model output)
```

### Accuracy Improvement

```
                    WITHOUT Sentinel          WITH Sentinel
                   ┌──────────────┐          ┌──────────────┐
  Neutral calls    │  61% correct │    ──►   │  93% correct │  +32 pts
  Negative recall  │ 100% correct │    ──►   │  97% correct │  -3 pts ✓
  Positive recall  │ 100% correct │    ──►   │ 100% correct │  no change
                   └──────────────┘          └──────────────┘
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## ⚡ Urgency Scoring Engine

```
┌──────────────────────────────────────────────────────────────────┐
│                    URGENCY FORMULA                               │
│                                                                  │
│  urgency = (ML × 0.45) + (keywords × 0.30)                      │
│          + (metadata × 0.15) + (emotion × 0.10)                 │
│                                                                  │
│  Without metadata: ML × 0.55 + keywords × 0.35 + emotion × 0.10 │
└──────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

```
ML SCORE  (0.45 weight)
├── Negative call:  0.50 + |polarity| × 0.50   → 0.50 to 1.00
├── Neutral call:   fixed 0.30                  → 0.30
└── Positive call:  max(0, 0.25 - polarity × 0.25) → 0.00 to 0.25

KEYWORD SCORE  (0.30 weight)
├── 🔴 Critical  (+0.45): lawyer, lawsuit, fraud, emergency, hacked
├── 🟠 High      (+0.25): urgent, escalate, supervisor, already called
├── 🟡 Medium    (+0.12): frustrated, not working, delayed, waiting
├── 🚨 Cancel    (+0.30): cancel, terminate, switching provider
└── 🟢 Positive  (-0.12): thank you, resolved, satisfied, sorted

METADATA SCORE  (0.15 weight)
├── Duration > 10min:      +0.30
├── Repeat caller:         +0.28
├── Hold time > 10min:     +0.25
└── 3+ calls today:        +0.28

EMOTION INTENSITY  (0.10 weight)
├── Escalating trajectory: +0.40  ← biggest urgency signal
├── Intensifier density:   up to +0.30
└── Emphatic negation:     up to +0.20
```

### Priority Tiers

```
Score  │  Priority  │  Response Target  │  Action
───────┼────────────┼───────────────────┼─────────────────────────────────
≥ 0.78 │  🔴 P1     │  Immediate        │  Senior agent now. Alert
       │  Critical  │                   │  supervisor. Log as complaint.
───────┼────────────┼───────────────────┼─────────────────────────────────
≥ 0.56 │  🟠 P2     │  < 5 minutes      │  Experienced agent. Retention
       │  High      │                   │  protocol if cancel detected.
───────┼────────────┼───────────────────┼─────────────────────────────────
≥ 0.33 │  🔵 P3     │  < 15 minutes     │  Standard queue. Upgrade to
       │  Medium    │                   │  P2 if wait exceeds 15 min.
───────┼────────────┼───────────────────┼─────────────────────────────────
< 0.33 │  🟢 P4     │  24 hours / self  │  Self-service resources or
       │  Low       │  service          │  scheduled callback.
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## ✨ Features

```
CORE ANALYSIS                       DASHBOARD & UI
─────────────────────────────       ──────────────────────────────
✅ Speech-to-text (Whisper tiny)    ✅ Dark theme dashboard
✅ CPU-only, no GPU needed          ✅ 3-tab layout (Overview / Add / History)
✅ 10x real-time transcription      ✅ Drag-and-drop audio upload
✅ 4-signal neutral detection       ✅ Text transcript entry (no audio needed)
✅ Sentence-level trajectory        ✅ Live result card after analysis
✅ P1–P4 urgency scoring            ✅ Per-call detail modal
✅ Keyword category detection       ✅ Neutral signal breakdown bars
✅ Call metadata integration        ✅ Agent notes (editable, persistent)
                                    ✅ Paginated + searchable history
API & DATA                          ✅ Filter by sentiment / priority
─────────────────────────────       ✅ CSV export
✅ 9 FastAPI REST endpoints         ✅ Auto-refresh every 30 seconds
✅ Auto Swagger docs at /api/docs
✅ SQLite WAL persistence           DATA
✅ Full-text search                 ──────────────────────────────
✅ Pagination (configurable)        ✅ Kaggle dataset importer
✅ Structured rotating logs         ✅ Sample data loader
✅ Centralised config.py            ✅ 24-column normalised schema
✅ Context-managed DB connections   ✅ 5 query-optimised indexes
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **STT** | OpenAI Whisper tiny + faster-whisper | CPU inference, VAD, open-source |
| **Quantisation** | CTranslate2 int8 | Halves memory, 2-4x faster on CPU |
| **Sentiment** | DistilBERT-SST2 (HuggingFace) | 67MB, CPU-friendly, strong accuracy |
| **ML Framework** | PyTorch (CPU build) | Required by both Whisper and transformers |
| **Audio** | librosa + soundfile | Resampling, normalisation, all formats |
| **API** | FastAPI + Uvicorn | Async, type-safe, auto-generates docs |
| **Database** | SQLite 3 (WAL mode) | Zero-config, built into Python |
| **Templates** | Jinja2 | Server-side dashboard rendering |
| **Logging** | Python logging + RotatingFileHandler | Standard library, no extra deps |

### Resource Profile

```
Component                     RAM Usage
─────────────────────────────────────────
FastAPI + SQLite (idle)         ~91 MB
+ Whisper tiny int8            ~301 MB   (+210 MB)
+ DistilBERT-SST2              ~486 MB   (+185 MB)
Peak during 3-min inference    ~623 MB

✅ Total: well under 1 GB
✅ Works on any 4GB+ RAM machine
✅ No GPU required
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📁 Project Structure

```
sentinel-helpdesk-intelligence/
│
├── app/
│   ├── __init__.py
│   ├── config.py                    # All tunable parameters
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                  # FastAPI app, all 9 endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio.py                 # Loading, VAD, validation
│   │   ├── transcriber.py           # Whisper STT, dual backend
│   │   ├── sentiment.py             # DistilBERT + 4-signal neutral
│   │   └── urgency.py               # Scoring, P1–P4 mapping
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py              # SQLite WAL, CRUD, pagination
│   │   └── dataset_loader.py        # Kaggle dataset importer
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py                # Rotating file + console logger
│
├── templates/
│   └── dashboard.html               # Full interactive dashboard
│
├── static/                          # CSS / JS assets
├── logs/                            # Auto-created, rotating
├── temp_uploads/                    # Temporary audio (auto-cleaned)
│
├── requirements.txt
├── run.py                           # Entry point: python run.py
└── README.md
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or above
- 4 GB RAM minimum (8 GB recommended)
- No GPU required

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sentinel-helpdesk-intelligence.git
cd sentinel-helpdesk-intelligence
```

### 2 — Create and activate virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ First install downloads Whisper (~75 MB) and DistilBERT (~248 MB) model weights. This is a one-time download — they are cached for all future runs.

### 4 — Start the server

```bash
python run.py
```

```
════════════════════════════════════════════════════════
  Sentinel Helpdesk Intelligence Platform v3.0.0
════════════════════════════════════════════════════════
Database ready: helpdesk.db
Loading Whisper 'tiny' model ...
Loading DistilBERT sentiment model ...
All models ready. Server accepting requests.

INFO: Uvicorn running on http://0.0.0.0:8000
```

### 5 — Open the dashboard

```
http://localhost:8000
```

### 6 — (Optional) Import sample data from Kaggle

```bash
pip install kagglehub
python -m app.db.dataset_loader
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload audio file → full pipeline analysis |
| `POST` | `/api/analyze/text` | Submit transcript text directly |
| `GET` | `/api/calls` | List calls (pagination, filter, search) |
| `GET` | `/api/calls/{id}` | Full detail for a single call |
| `PATCH` | `/api/calls/{id}/notes` | Update agent notes |
| `DELETE` | `/api/calls/{id}` | Delete a call record |
| `GET` | `/api/stats` | Aggregate statistics |
| `GET` | `/api/export/csv` | Download all calls as CSV |
| `GET` | `/api/health` | Server health check |

### Example: Analyze a call

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@call_recording.wav" \
  -F "repeat_caller=false" \
  -F "wait_time_seconds=120" \
  -F "call_number_today=1"
```

```json
{
  "call_id": 42,
  "status": "success",
  "time_taken": 13.7,
  "transcript": {
    "text": "My internet has been down for three days...",
    "language": "en",
    "duration": 142.3,
    "speech_ratio": 0.81
  },
  "sentiment": {
    "label": "negative",
    "polarity": -0.94,
    "confidence": 0.97,
    "neutral_score": 0.08,
    "trajectory": "escalating"
  },
  "urgency": {
    "score": 0.82,
    "priority": "P1",
    "priority_label": "Critical",
    "action": "Escalate immediately to senior agent",
    "keywords_found": ["three days", "still broken"],
    "escalation_flag": true
  }
}
```

### Example: Filter call history

```bash
# All P1 critical calls
GET /api/calls?priority=P1&limit=10

# Search for calls mentioning "cancel"
GET /api/calls?search=cancel&sentiment=negative

# Escalated calls only, page 2
GET /api/calls?escalation_only=true&page=2
```

> 📖 Full interactive API docs: `http://localhost:8000/api/docs`

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🖥️ Dashboard

The dashboard is designed for non-technical helpdesk supervisors.

### Overview Tab
- 5 live stat cards (total calls, avg sentiment, P1 count, escalations, avg urgency)
- Sentiment distribution bar (positive / neutral / negative split)
- Priority queue breakdown (P1–P4 proportions)
- Emotional trajectory distribution (stable / escalating / de-escalating)

### Add Call Tab
- **Audio mode**: Drag-and-drop recording upload with metadata form
- **Text mode**: Paste transcript directly for instant analysis
- Live result card showing all scores after submission

### Call History Tab
- Paginated table (25 per page, configurable)
- Real-time search with 350ms debounce
- Filter by sentiment, priority, escalation status
- Click any row → detail modal with:
  - Full transcript
  - All four neutral signal scores with visual bars
  - Urgency component breakdown
  - Recommended action with extended guidance
  - Urgency keywords detected
  - Editable agent notes
  - Delete button

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## ⚙️ Configuration

All parameters live in `app/config.py`. No code changes needed — just edit the values.

```python
# ── Whisper ─────────────────────────────────────────
WHISPER_MODEL_SIZE  = "tiny"    # "tiny" | "base" | "small"
WHISPER_BEAM_SIZE   = 3         # Higher = more accurate, slower
WHISPER_LANGUAGE    = "en"      # None = auto-detect

# ── Neutral Detection ────────────────────────────────
NEUTRAL_CONFIDENCE_THRESHOLD = 0.72  # Signal 1: max confidence for neutral candidate
NEUTRAL_POLARITY_BAND        = 0.20  # Signal 2: max |polarity| for neutral candidate
ENQUIRY_RATIO_THRESHOLD      = 0.08  # Signal 3: min interrogative ratio
NEUTRAL_VOTE_THRESHOLD       = 0.45  # Final: min vote score to override to NEUTRAL

# ── Priority Thresholds ──────────────────────────────
PRIORITY_P1 = 0.78   # Critical
PRIORITY_P2 = 0.56   # High
PRIORITY_P3 = 0.33   # Medium

# ── Urgency Weights ──────────────────────────────────
WEIGHT_ML         = 0.45
WEIGHT_KEYWORDS   = 0.30
WEIGHT_METADATA   = 0.15
WEIGHT_EMOTION    = 0.10
```

> 💡 **Tip:** After deploying, run 100–200 real calls through the system, label them manually, then adjust `NEUTRAL_CONFIDENCE_THRESHOLD` and `NEUTRAL_VOTE_THRESHOLD` to match your organisation's call language patterns.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📊 Results

### Transcription Performance

| Audio Duration | Backend | Processing Time | Real-Time Factor |
|---------------|---------|----------------|-----------------|
| 30 seconds | faster-whisper int8 | 2.8s | **10.8x** real time |
| 60 seconds | faster-whisper int8 | 5.9s | **10.2x** real time |
| 120 seconds | faster-whisper int8 | 11.3s | **10.6x** real time |
| 120 seconds | openai-whisper fp32 | 18.7s | 6.4x real time |

### Neutral Classification

| Test Group | Baseline | With Sentinel | Improvement |
|-----------|---------|---------------|-------------|
| All neutral samples (60) | 61% | **93%** | +32 pts |
| Admin enquiries (30) | 57% | **90%** | +33 pts |
| Info questions (30) | 67% | **97%** | +30 pts |
| Negative recall (30) | 100% | **97%** | -3 pts ✓ |

### Urgency Validation (7 Scenarios)

| Scenario | Urgency Score | Priority | Correct? |
|----------|--------------|---------|---------|
| Legal threat + 5-day outage + repeat caller | 0.89 | P1 | ✅ |
| Cancellation intent + supervisor request | 0.74 | P2 | ✅ |
| Incorrect billing + escalating trajectory | 0.64 | P2 | ✅ |
| Mild complaint, first contact | 0.42 | P3 | ✅ |
| Password reset question (neutral) | 0.21 | P4 | ✅ |
| Address update enquiry (neutral) | 0.18 | P4 | ✅ |
| Post-resolution positive feedback | 0.06 | P4 | ✅ |

**7/7 correct priority assignments. Strict monotonic ordering maintained.**

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🔮 Future Enhancements

```
┌─────────────────────────────────────────────────────────────────┐
│  ROADMAP                                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [ ] Real-time streaming analysis (WebSocket + chunked STT)     │
│  [ ] Fine-tuned 3-class sentiment model on helpdesk data        │
│  [ ] Speaker diarisation (agent vs customer separation)         │
│  [ ] Acoustic feature integration (pitch, energy, speech rate)  │
│  [ ] Multi-language support (Tamil, Hindi, + 90 more via Whisper│
│  [ ] Zendesk / Freshdesk / ServiceNow API integration           │
│  [ ] Feedback learning loop (agent notes → auto-retuning)       │
│  [ ] PostgreSQL migration for high-volume deployments           │
│  [ ] Batch file upload for historical call analysis             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

<br/>


<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

<div align="center">

**Built with ❤️ by Abdul Rahman**

<br/>

*If this project helped you, consider giving it a ⭐*

</div>