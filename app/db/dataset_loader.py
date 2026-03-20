# app/db/dataset_loader.py
# ═══════════════════════════════════════════════════════════
# Import Kaggle call center transcript dataset into SQLite.
# Run: python -m app.db.dataset_loader
# ═══════════════════════════════════════════════════════════

import os, csv, glob, json, random
from pathlib import Path
from app.utils.logger import get_logger
from app.config       import KAGGLE_DATASET_ID, KAGGLE_CACHE_DIR
from app.db.database  import init_db, save_call

log = get_logger(__name__)

COLUMN_ALIASES = {
    "text"      : ["transcript","text","call_text","content","conversation",
                   "dialogue","utterance","message","body"],
    "sentiment" : ["sentiment","label","emotion","category","class"],
    "duration"  : ["duration","call_duration","length","seconds","time"],
}

def _find_col(headers, aliases):
    for a in aliases:
        if a in headers: return a
    return None

def _safe_float(v, default=0.0):
    try:    return float(str(v).strip())
    except: return default

def find_csvs(path):
    return glob.glob(os.path.join(path,"**","*.csv"), recursive=True)

def read_transcripts(csv_path):
    records = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader  = csv.DictReader(f)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]
        log.info("Columns: %s", headers)

        tc = _find_col(headers, COLUMN_ALIASES["text"])
        sc = _find_col(headers, COLUMN_ALIASES["sentiment"])
        dc = _find_col(headers, COLUMN_ALIASES["duration"])

        if not tc:
            log.error("No transcript column found in %s", csv_path)
            return []

        for i, row in enumerate(reader):
            row = {k.lower().strip(): v for k,v in row.items()}
            text = row.get(tc,"").strip()
            if not text or len(text) < 10: continue
            records.append({
                "text"     : text,
                "sentiment": row.get(sc,"").strip().lower() if sc else "",
                "duration" : _safe_float(row.get(dc,"0")) if dc else 0.0,
                "index"    : i
            })
    log.info("Read %d valid records from %s", len(records), csv_path)
    return records

def process_and_import(records, limit=None):
    from app.core.sentiment import load_sentiment_model, analyze_sentiment
    from app.core.urgency   import score_urgency

    model = load_sentiment_model()
    if limit: records = records[:limit]

    imported = skipped = 0
    for i, rec in enumerate(records):
        try:
            sentiment = analyze_sentiment(model, rec["text"])
            duration  = rec["duration"] if rec["duration"] > 0 else random.uniform(30,240)
            metadata  = {
                "duration_seconds":duration,"repeat_caller":False,
                "wait_time_seconds":0,"call_number_today":1,"source":"kaggle"
            }
            urgency   = score_urgency(rec["text"], sentiment, metadata)
            tr        = {"text":rec["text"],"language":"en","duration":round(duration,1),"speech_ratio":1.0}
            save_call(tr, sentiment, urgency,
                      filename=f"kaggle_{i}.csv", metadata=metadata, source="dataset")
            imported += 1
            if imported % 25 == 0:
                log.info("  Imported %d/%d ...", imported, len(records))
        except Exception as exc:
            log.warning("Skipped row %d: %s", i, exc)
            skipped += 1

    log.info("Import complete — %d imported, %d skipped", imported, skipped)
    return imported, skipped

if __name__ == "__main__":
    import kagglehub
    print("Downloading dataset...")
    path     = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    csvs     = find_csvs(path)
    if not csvs: raise SystemExit(f"No CSVs found in {path}")
    csv_path = max(csvs, key=os.path.getsize)
    records  = read_transcripts(csv_path)
    ans      = input(f"Found {len(records)} records. How many to import? (number or 'all'): ").strip()
    limit    = None if ans.lower()=="all" else int(ans)
    init_db()
    n, s     = process_and_import(records, limit)
    print(f"\nDone — {n} imported, {s} skipped.")
    print("Start server: python -m uvicorn app.api.main:app --reload")