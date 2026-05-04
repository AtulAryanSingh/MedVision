"""
api/auth.py

What this module does:
  POST /api/auth/login  – validate username + password against the SQLite
                          users table and issue a signed JWT access token.

  This endpoint is intentionally left unprotected (no auth dependency) so
  clients can obtain their first token without a chicken-and-egg problem.

Why it exists:
  Centralising credential exchange here keeps all JWT-issuance logic in one
  place and separates it from the get_current_user dependency in deps.py.
"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends

import db
from api.deps import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET

router = APIRouter()


@router.post(
    "/auth/login",
    summary="Obtain a JWT access token",
    tags=["Auth"],
)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Exchange username + password for a signed JWT access token.

    What it does:
      1. Looks up the user record in the SQLite users table.
      2. Verifies the supplied password against the stored bcrypt hash.
      3. Issues a HS256-signed JWT with 'sub' = username and an 'exp' claim.
      4. Returns {"access_token": "<jwt>", "token_type": "bearer"}.

    Raises HTTP 401 if the username does not exist or the password is wrong.
    The error message is intentionally generic to avoid username enumeration.
    """
    user = db.get_user(form.username)
    if user is None or not db.verify_password(form.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": user["username"], "exp": expire}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {"access_token": token, "token_type": "bearer"}
