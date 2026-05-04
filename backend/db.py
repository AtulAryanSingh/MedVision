"""
db.py

What this module does:
  Initialises a minimal SQLite database that stores application users.
  Provides helpers for creating the table, looking up users, and
  creating new user accounts with bcrypt-hashed passwords.

  The database file is stored at  backend/data/users.db  so it is
  persisted across restarts but excluded from source control via .gitignore.

  The table schema is intentionally minimal:
    users(id INTEGER PK, username TEXT UNIQUE, hashed_password TEXT)
"""

import os
import sqlite3
from typing import Optional

from passlib.context import CryptContext

_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "users.db")
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _get_conn() -> sqlite3.Connection:
    """Return a connection to the users database (thread-check disabled for uvicorn workers)."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the users table if it does not already exist and seed a default
    admin account when the table is first created.

    The default credentials (admin / changeme) are intended for first-boot
    only – operators should change the password immediately after deployment.
    """
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                username         TEXT    NOT NULL UNIQUE,
                hashed_password  TEXT    NOT NULL
            )
            """
        )
        conn.commit()

        # Seed a default admin user if the table is empty
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        if row[0] == 0:
            hashed = _pwd_ctx.hash("changeme")
            conn.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                ("admin", hashed),
            )
            conn.commit()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* password."""
    return _pwd_ctx.verify(plain, hashed)


def get_user(username: str) -> Optional[sqlite3.Row]:
    """Return the users row for *username*, or None if not found."""
    with _get_conn() as conn:
        return conn.execute(
            "SELECT id, username, hashed_password FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def create_user(username: str, password: str) -> None:
    """
    Insert a new user with a bcrypt-hashed password.

    Raises sqlite3.IntegrityError if *username* already exists.
    """
    hashed = _pwd_ctx.hash(password)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (username, hashed),
        )
        conn.commit()
