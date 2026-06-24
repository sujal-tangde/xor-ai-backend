"""Parts database API: exact-MPN existence + pricing lookups."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.core.auth import get_current_user
from src.services import parts_db

router = APIRouter(tags=["parts"])

# Cap the batch so a single request can't fan out into a huge IN-list.
_MAX_MPNS = 300


class MpnLookupRequest(BaseModel):
    mpns: list[str] = Field(default_factory=list)


@router.get("/health")
async def health(user=Depends(get_current_user)):
    if not parts_db.is_enabled():
        raise HTTPException(status_code=503, detail="Parts database is not enabled.")
    try:
        return await asyncio.to_thread(parts_db.ping)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Parts DB unreachable: {exc}") from exc


@router.post("/lookup")
async def lookup(payload: MpnLookupRequest, user=Depends(get_current_user)):
    """Check whether each MPN exists in the parts DB and return its detail.

    Body: ``{"mpns": ["0805B101K500CT", ...]}``. Matching is exact and
    case-insensitive. Returns one entry per requested MPN.
    """
    if not parts_db.is_enabled():
        raise HTTPException(status_code=503, detail="Parts database is not enabled.")

    mpns = [m for m in payload.mpns if m and m.strip()]
    if not mpns:
        raise HTTPException(status_code=400, detail="Provide at least one MPN in 'mpns'.")
    if len(mpns) > _MAX_MPNS:
        raise HTTPException(
            status_code=400, detail=f"Too many MPNs (max {_MAX_MPNS} per request)."
        )

    try:
        results = await asyncio.to_thread(parts_db.lookup_mpns, mpns)
    except parts_db.PartsDBError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {exc}") from exc

    found = sum(1 for r in results if r["found"])
    return {"count": len(results), "found": found, "results": results}
