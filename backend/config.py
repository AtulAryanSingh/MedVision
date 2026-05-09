import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Prefer backend/.env when present, then fall back to repository-root .env
load_dotenv(BASE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _csv_to_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


_default_origins = ",".join(
    [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5500",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",
        "http://frontend:5173",
        "null",
    ]
)


ALLOWED_ORIGINS = _csv_to_list(os.environ.get("ALLOWED_ORIGINS", _default_origins))
CORS_ALLOW_CREDENTIALS = os.environ.get("CORS_ALLOW_CREDENTIALS", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
CACHE_DIR = os.environ.get("CACHE_DIR", str(BASE_DIR / "data" / "cache"))
USERS_DB_PATH = os.environ.get("USERS_DB_PATH", str(BASE_DIR / "data" / "users.db"))

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))
BCRYPT_ROUNDS = int(os.environ.get("BCRYPT_ROUNDS", "12"))
