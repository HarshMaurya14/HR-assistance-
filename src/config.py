"""Central configuration. Everything environment-specific is read here."""

import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = "openai/gpt-oss-20b"

# --- Leave rules (assumptions are documented in README) ---------------------
SICK_LEAVE_BACKDATE_DAYS = 3   # sick leave may be reported up to N days late
MAX_LEAVE_SPAN_DAYS = 90       # sanity limit on a single request
ANNUAL_LEAVE_NOTICE_DAYS = 7   # leave_policy.md: "at least 7 days in advance"
CASUAL_LEAVE_NOTICE_DAYS = 1   # leave_policy.md: "at least 1 day in advance"


def db_path() -> Path:
    """Read at call time so tests (and deployments) can override it."""
    return Path(os.getenv("HR_DB_PATH", BASE_DIR / "data" / "hr_database.db"))


def documents_dir() -> Path:
    return Path(os.getenv("HR_DOCS_DIR", BASE_DIR / "data" / "hr_documents"))


def holidays_file() -> Path:
    return Path(os.getenv("HR_HOLIDAYS_FILE", BASE_DIR / "data" / "holidays.json"))


def groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY") or None


def groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def demo_mode() -> bool:
    return os.getenv("SEED_DEMO_DATA", "1").strip().lower() in {"1", "true", "yes"}


def load_holidays() -> dict[date, str]:
    """Company holidays that do not count as working days."""
    path = holidays_file()
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {date.fromisoformat(e["date"]): e.get("name", "Holiday") for e in entries}
    except (ValueError, KeyError, TypeError):
        return {}
