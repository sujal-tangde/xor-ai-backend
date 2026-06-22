"""HTTP endpoints for generated should-cost reports (preview + PDF download)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from src.core.auth import get_current_user
from src.services import reports as reports_service

router = APIRouter(tags=["reports"])


def _safe_filename(title: str | None) -> str:
    base = (title or "should-cost-report").strip() or "should-cost-report"
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in base)
    return keep.replace(" ", "_")[:80] + ".pdf"


@router.get("/{report_id}")
async def get_report(report_id: str, user=Depends(get_current_user)):
    """Return report metadata + rendered HTML + structured JSON for preview."""
    record = reports_service.get_report(report_id, user["id"])
    if record is None:
        raise HTTPException(status_code=404, detail="Report not found")
    report_json = record.get("report_json") or {}
    return {
        "id": record["id"],
        "title": record.get("title"),
        "volume": record.get("volume"),
        "html": record.get("html"),
        "report_json": report_json,
        "pdf_url": record.get("pdf_url"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        # Legacy markdown reports (pre-migration) still render in the panel.
        "markdown": reports_service.normalize_report_markdown(record.get("markdown") or "")
        if record.get("markdown")
        else None,
    }


@router.get("/{report_id}/download")
async def download_report(report_id: str, user=Depends(get_current_user)):
    """Stream the report PDF as an attachment (from the bucket, else re-rendered)."""
    record = reports_service.get_report(report_id, user["id"])
    if record is None:
        raise HTTPException(status_code=404, detail="Report not found")
    pdf_bytes = reports_service.get_report_pdf_bytes(report_id, user["id"])
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="Report PDF not available")
    filename = _safe_filename(record.get("title"))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
