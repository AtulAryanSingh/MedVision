"""
tests/integration/test_auth.py

Integration tests for auth endpoints.

Covers
------
• Correct credentials → 200 + JWT token returned
• Wrong password → 401
• Non-existent user → 401
• Missing fields → 422 (FastAPI form validation)
• Registration success and duplicate checks
• Protected endpoint without token → 401
• Protected endpoint with invalid token → 401
• Protected endpoint with valid token → passes auth (delegates to health)
"""

import sqlite3


class TestRegister:
    def test_register_returns_201_with_guest_role(self, anon_client, monkeypatch):
        import db

        created_user = {
            "id": 7,
            "username": "newuser",
            "email": "newuser@example.com",
            "role": "guest",
        }
        monkeypatch.setattr(db, "get_user_by_username_or_email", lambda *_: None)
        monkeypatch.setattr(db, "create_user", lambda **_: created_user)

        response = anon_client.post(
            "/api/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "StrongPass123!",
            },
        )

        assert response.status_code == 201
        assert response.json() == created_user

    def test_register_duplicate_username_returns_409(self, anon_client, monkeypatch):
        import db

        monkeypatch.setattr(
            db,
            "get_user_by_username_or_email",
            lambda *_: {"username": "taken", "email": "taken@example.com"},
        )

        response = anon_client.post(
            "/api/auth/register",
            json={
                "username": "taken",
                "email": "new@example.com",
                "password": "StrongPass123!",
            },
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Username already exists."

    def test_register_duplicate_username_case_insensitive_returns_409(self, anon_client, monkeypatch):
        import db

        monkeypatch.setattr(
            db,
            "get_user_by_username_or_email",
            lambda *_: {"username": "Alice", "email": "alice@example.com"},
        )

        response = anon_client.post(
            "/api/auth/register",
            json={
                "username": "alice",
                "email": "new@example.com",
                "password": "StrongPass123!",
            },
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Username already exists."

    def test_register_duplicate_email_returns_409(self, anon_client, monkeypatch):
        import db

        monkeypatch.setattr(
            db,
            "get_user_by_username_or_email",
            lambda *_: {"username": "existing", "email": "taken@example.com"},
        )

        response = anon_client.post(
            "/api/auth/register",
            json={
                "username": "newuser",
                "email": "taken@example.com",
                "password": "StrongPass123!",
            },
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Email already exists."

    def test_register_integrity_error_returns_409(self, anon_client, monkeypatch):
        import db

        monkeypatch.setattr(db, "get_user_by_username_or_email", lambda *_: None)

        def _raise_integrity_error(**_):
            raise sqlite3.IntegrityError("duplicate")

        monkeypatch.setattr(db, "create_user", _raise_integrity_error)

        response = anon_client.post(
            "/api/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "StrongPass123!",
            },
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Username or email already exists."


class TestLogin:
    def test_valid_credentials_return_200(self, anon_client, monkeypatch):
        """
        Monkeypatch db.get_user and db.verify_password so the test does not
        depend on the real SQLite database or a pre-existing user.
        """
        import db

        fake_user = {"username": "testuser", "hashed_password": "hashed"}
        monkeypatch.setattr(db, "get_user", lambda _: fake_user)
        monkeypatch.setattr(db, "verify_password", lambda _plain, _hashed: True)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "correctpassword"},
        )
        assert response.status_code == 200

    def test_valid_credentials_return_token(self, anon_client, monkeypatch):
        import db

        fake_user = {"username": "testuser", "hashed_password": "hashed"}
        monkeypatch.setattr(db, "get_user", lambda _: fake_user)
        monkeypatch.setattr(db, "verify_password", lambda _plain, _hashed: True)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "correctpassword"},
        )
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 10  # a non-trivial JWT

    def test_wrong_password_returns_401(self, anon_client, monkeypatch):
        import db

        fake_user = {"username": "testuser", "hashed_password": "hashed"}
        monkeypatch.setattr(db, "get_user", lambda _: fake_user)
        monkeypatch.setattr(db, "verify_password", lambda _plain, _hashed: False)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_unknown_user_returns_401(self, anon_client, monkeypatch):
        import db

        monkeypatch.setattr(db, "get_user", lambda _: None)

        response = anon_client.post(
            "/api/auth/login",
            data={"username": "ghost", "password": "anything"},
        )
        assert response.status_code == 401

    def test_missing_username_returns_422(self, anon_client):
        response = anon_client.post(
            "/api/auth/login",
            data={"password": "nouser"},
        )
        assert response.status_code == 422

    def test_missing_password_returns_422(self, anon_client):
        response = anon_client.post(
            "/api/auth/login",
            data={"username": "testuser"},
        )
        assert response.status_code == 422


class TestProtectedEndpointAuth:
    def test_no_token_returns_401(self, anon_client):
        """Calling a protected endpoint without a token should yield 401."""
        response = anon_client.get("/api/preview/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, anon_client):
        response = anon_client.get(
            "/api/preview/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": "Bearer this.is.not.valid"},
        )
        assert response.status_code == 401

    def test_valid_token_passes_auth(self, anon_client, auth_headers):
        """
        A correctly signed token should pass auth (endpoint may still return
        404 because no image exists, but must NOT return 401 or 403).
        """
        response = anon_client.get(
            "/api/preview/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert response.status_code != 401
        assert response.status_code != 403
