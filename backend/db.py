"""
db.py

What this module does:
  Initialises a minimal SQLite database that stores application users.
  Provides helpers for creating the table, looking up users, and
  creating new user accounts with bcrypt-hashed passwords.

  The database file is stored at  backend/data/users.db  so it is
  persisted across restarts but excluded from source control via .gitignore.

  The table schema is intentionally minimal:
    users(
      id INTEGER PK,
      username TEXT UNIQUE,
      email TEXT UNIQUE,
      hashed_password TEXT,
      role TEXT
    )

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
_pwd_ctx = CryptContext(
    schemes=["bcrypt"],
    bcrypt__rounds=config.BCRYPT_ROUNDS,
    deprecated="auto",
)
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
                email            TEXT    NOT NULL UNIQUE,
                role             TEXT    NOT NULL DEFAULT 'guest',
                hashed_password  TEXT    NOT NULL
            )
            """
        )
        _migrate_users_table(conn)
        conn.commit()

        # Seed a default admin user if the table is empty
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        if row[0] == 0:
            initial_password = secrets.token_urlsafe(16)
            hashed = _pwd_ctx.hash(initial_password)
            conn.execute(
                "INSERT INTO users (username, email, role, hashed_password) VALUES (?, ?, ?, ?)",
                ("admin", "admin@medvision.local", "admin", hashed),
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


def _migrate_users_table(conn: sqlite3.Connection) -> None:
    """Perform additive schema migrations for legacy users tables."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }

    if "email" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "role" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'guest'")

    conn.execute(
        "UPDATE users SET email = username || '@legacy.local' WHERE email IS NULL OR trim(email) = ''"
    )
    conn.execute("UPDATE users SET role = 'guest' WHERE role IS NULL OR trim(role) = ''")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email)"
    )


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* password."""
    return _pwd_ctx.verify(plain, hashed)


def get_user(username: str) -> Optional[sqlite3.Row]:
    """Return the users row for *username*, or None if not found."""
    return _get_conn().execute(
        "SELECT id, username, email, role, hashed_password FROM users WHERE username = ?",
        (username,),
    ).fetchone()


def get_user_by_username_or_email(username: str, email: str) -> Optional[sqlite3.Row]:
    """Return a user row matching *username* or *email* (case-insensitive email)."""
    return _get_conn().execute(
        """
        SELECT id, username, email, role, hashed_password
        FROM users
        WHERE username = ? OR lower(email) = lower(?)
        """,
        (username, email),
    ).fetchone()


def create_user(username: str, email: str, password: str, role: str = "guest") -> sqlite3.Row:
    """
    Insert a new user with a bcrypt-hashed password and return the created row.

    Raises sqlite3.IntegrityError if *username* or *email* already exists.
    """
    hashed = _pwd_ctx.hash(password)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, email, role, hashed_password) VALUES (?, ?, ?, ?)",
            (username, email, role, hashed),
        )
        conn.commit()
        return conn.execute(
            "SELECT id, username, email, role, hashed_password FROM users WHERE username = ?",
            (username,),
        ).fetchone()
