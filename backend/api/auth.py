"""
api/auth.py

What this module does:
  POST /api/auth/register – create a new user account with a bcrypt-hashed
                            password.
  POST /api/auth/login  – validate username + password against the SQLite
                          users table and issue a signed JWT access token.

  This endpoint is intentionally left unprotected (no auth dependency) so
  clients can obtain their first token without a chicken-and-egg problem.

Why it exists:
  Centralising credential exchange here keeps all JWT-issuance logic in one
  place and separates it from the get_current_user dependency in deps.py.
"""

from datetime import datetime, timedelta, timezone
import sqlite3

import jwt
from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from pydantic import BaseModel, EmailStr, Field

import db
from api.deps import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


@router.post(
    "/auth/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    tags=["Auth"],
)
def register(payload: RegisterRequest):
    existing = db.get_user_by_username_or_email(payload.username, payload.email)
    if existing:
        if existing["username"].lower() == payload.username.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists.",
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists.",
        )

    try:
        created = db.create_user(
            username=payload.username,
            email=payload.email,
            password=payload.password,
            role=db.DEFAULT_USER_ROLE,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists.",
        ) from exc

    return {
        "id": created["id"],
        "username": created["username"],
        "email": created["email"],
        "role": created["role"],
    }


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
