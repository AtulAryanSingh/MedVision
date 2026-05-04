"""
api/deps.py

What this module does:
  Provides FastAPI dependency functions for JWT-based authentication.

  get_current_user(token) → dict
    Validates the Bearer token from the Authorization header and returns the
    decoded payload.  Raises HTTP 401 on any failure so that protected routes
    automatically reject unauthenticated requests.

  The JWT secret is read from the JWT_SECRET environment variable.  A random
  256-bit default is generated at import time so the server starts without
  manual configuration in development, but tokens are invalidated on restart
  unless the variable is set explicitly.

Configuration
-------------
  JWT_SECRET   – HMAC-SHA256 signing secret (required in production).
  JWT_ALGORITHM – defaults to HS256.
  JWT_EXPIRE_MINUTES – access-token lifetime, defaults to 60 minutes.
"""

import os
import secrets

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ── Configuration ─────────────────────────────────────────────────────────────

JWT_SECRET: str = os.environ.get("JWT_SECRET") or secrets.token_hex(32)
JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

# ── OAuth2 scheme ─────────────────────────────────────────────────────────────
# tokenUrl points to the login endpoint so Swagger UI's "Authorize" button works.

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ── Dependency ────────────────────────────────────────────────────────────────

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency that validates a JWT Bearer token.

    What it does:
      1. Decodes the token using the configured secret and algorithm.
      2. Verifies the 'sub' (subject / username) claim is present.
      3. Returns the decoded payload dict on success.
      4. Raises HTTP 401 for any validation failure (expired, tampered, missing).

    Usage
    -----
    Add as a router dependency so every endpoint in the router is protected::

        router = APIRouter(dependencies=[Depends(get_current_user)])
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise _CREDENTIALS_EXCEPTION

    if not payload.get("sub"):
        raise _CREDENTIALS_EXCEPTION

    return payload
