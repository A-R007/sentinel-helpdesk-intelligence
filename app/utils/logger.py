# app/utils/logger.py
# ═══════════════════════════════════════════════════════════
# Structured rotating logger — replaces all print() calls.
# ═══════════════════════════════════════════════════════════

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Avoid circular import — resolve path directly
_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "app.log"
_LOG_PATH.parent.mkdir(exist_ok=True)

_FMT_CONSOLE = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
_FMT_FILE    = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT    = "%H:%M:%S"

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger writing to console (INFO+) and
    a rotating file (DEBUG+, max 5 MB × 3 backups).

    Usage:
        from app.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("ready")
        log.warning("low confidence: %.3f", score)
        log.error("file not found: %s", path)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT_CONSOLE, _DATE_FMT))

    # Rotating file
    fh = RotatingFileHandler(
        _LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT_FILE))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger