"""Authentication endpoints backed by Supabase Auth."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.core.auth import get_current_user
from src.services.auth_service import (
    AuthError,
    get_oauth_url,
    refresh,
    sign_in,
    sign_up,
)

router = APIRouter(tags=["auth"])


class Credentials(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/signup")
def signup(body: Credentials) -> dict[str, Any]:
    try:
        return sign_up(body.email, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/login")
def login(body: Credentials) -> dict[str, Any]:
    try:
        return sign_in(body.email, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/refresh")
def refresh_token(body: RefreshRequest) -> dict[str, Any]:
    try:
        return refresh(body.refresh_token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/google")
def google_oauth() -> dict[str, str]:
    try:
        return {"url": get_oauth_url("google")}
    except AuthError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/me")
def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return user
