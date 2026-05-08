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

Thread safety:
  Each OS thread gets its own SQLite connection via threading.local() so that
  concurrent uvicorn workers never share a connection handle.
"""

import logging
import os
import secrets
import sqlite3
import threading
from typing import Optional

from passlib.context import CryptContext
import config

logger = logging.getLogger(__name__)

_DB_PATH = config.USERS_DB_PATH
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """
    Return a per-thread SQLite connection to the users database.

    Using threading.local() gives every OS thread its own connection, which
    is the recommended approach when running under a multi-threaded ASGI
    server (e.g. uvicorn with multiple workers or threads).
    """
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db() -> None:
    """
    Create the users table if it does not already exist and seed a default
    admin account when the table is first created.

    On first boot a cryptographically random password is generated for the
    admin account and printed to stdout so the operator can retrieve it.
    Operators should change this password (or provision their own user) as
    soon as possible after deployment.
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
            initial_password = secrets.token_urlsafe(16)
            hashed = _pwd_ctx.hash(initial_password)
            conn.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                ("admin", hashed),
            )
            conn.commit()
            # Print to stdout so the operator can capture it from server logs
            print(
                f"\n[MedVision] First-boot admin credentials:\n"
                f"  username: admin\n"
                f"  password: {initial_password}\n"
                f"Change this password immediately after first login.\n",
                flush=True,
            )


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* password."""
    return _pwd_ctx.verify(plain, hashed)


def get_user(username: str) -> Optional[sqlite3.Row]:
    """Return the users row for *username*, or None if not found."""
    return _get_conn().execute(
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
