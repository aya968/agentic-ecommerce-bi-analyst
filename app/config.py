import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _configured_path(name, default):
    value = os.getenv(name)
    path = Path(value) if value else Path(default)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


DATA_DIR = _configured_path("DATA_DIR", BASE_DIR / "data")
UPLOAD_DIR = _configured_path("UPLOAD_DIR", DATA_DIR / "uploads")
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = str(_configured_path("DB_PATH", DATA_DIR / "ecommerce.db"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_DELAY_SECONDS = float(
    os.getenv("GEMINI_RETRY_DELAY_SECONDS", "1")
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")
API_TIMEOUT_SECONDS = float(
    os.getenv("API_TIMEOUT_SECONDS", "300")
)

MAX_ROWS_PREVIEW = 10
MAX_SQL_ROWS = 500