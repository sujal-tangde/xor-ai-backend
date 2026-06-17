"""Supabase Auth wrappers: email/password, Google OAuth, token validation."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

from supabase import Client, create_client

from src.core.config import (
    AUTH_REDIRECT_URL,
    SUPABASE_ANON_KEY,
    SUPABASE_URL,
)

_client: Client | None = None

# Short-lived cache of validated tokens so we don't hit Supabase on every request.
_TOKEN_TTL_SECONDS = 300
_token_cache: dict[str, tuple[dict[str, Any], float]] = {}


class AuthError(Exception):
    """Raised when an auth operation fails (bad credentials, expired token, etc.)."""


def _auth():
    """Return the GoTrue auth sub-client (created from the anon key)."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise AuthError("Supabase auth is not configured (missing URL or anon key).")
        _client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _client.auth


def _user_dict(user: Any) -> dict[str, Any]:
    return {
        "id": str(user.id),
        "email": getattr(user, "email", None),
        "name": (user.user_metadata or {}).get("name")
        or (user.user_metadata or {}).get("full_name"),
    }


def _session_dict(response: Any) -> dict[str, Any]:
    session = getattr(response, "session", None)
    user = getattr(response, "user", None)
    result: dict[str, Any] = {
        "user": _user_dict(user) if user else None,
    }
    if session:
        result["access_token"] = session.access_token
        result["refresh_token"] = session.refresh_token
        result["expires_at"] = session.expires_at
    return result


def sign_up(email: str, password: str) -> dict[str, Any]:
    try:
        response = _auth().sign_up({"email": email, "password": password})
    except Exception as exc:  # supabase raises AuthApiError / AuthError
        raise AuthError(str(exc)) from exc
    data = _session_dict(response)
    # When email confirmation is enabled, no session is returned yet.
    data["needs_confirmation"] = data.get("access_token") is None
    return data


def sign_in(email: str, password: str) -> dict[str, Any]:
    try:
        response = _auth().sign_in_with_password(
            {"email": email, "password": password}
        )
    except Exception as exc:
        raise AuthError(str(exc)) from exc
    data = _session_dict(response)
    if not data.get("access_token"):
        raise AuthError("Invalid email or password.")
    return data


def get_oauth_url(provider: str = "google") -> str:
    """Build the Supabase implicit-flow authorize URL.

    We construct this manually (rather than via the SDK's ``sign_in_with_oauth``)
    so the provider uses the implicit grant and returns the session tokens in the
    redirect URL hash. The PKCE flow would store a verifier on the backend client
    that the frontend callback cannot access.
    """
    if not SUPABASE_URL:
        raise AuthError("Supabase auth is not configured (missing URL).")
    base = SUPABASE_URL.rstrip("/")
    return (
        f"{base}/auth/v1/authorize"
        f"?provider={quote(provider)}"
        f"&redirect_to={quote(AUTH_REDIRECT_URL, safe='')}"
    )


def refresh(refresh_token: str) -> dict[str, Any]:
    try:
        response = _auth().refresh_session(refresh_token)
    except Exception as exc:
        raise AuthError(str(exc)) from exc
    data = _session_dict(response)
    if not data.get("access_token"):
        raise AuthError("Could not refresh session.")
    return data


def validate_token(token: str) -> dict[str, Any] | None:
    """Return the user dict for a valid access token, else None."""
    if not token:
        return None

    cached = _token_cache.get(token)
    now = time.monotonic()
    if cached and cached[1] > now:
        return cached[0]

    try:
        response = _auth().get_user(token)
    except Exception:
        return None

    user = getattr(response, "user", None)
    if not user:
        return None

    user_dict = _user_dict(user)
    _token_cache[token] = (user_dict, now + _TOKEN_TTL_SECONDS)
    return user_dict
