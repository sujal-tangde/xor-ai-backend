"""HTTP endpoints for generated should-cost reports (preview + PDF download)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from src.core.auth import get_current_user
from src.services import currency, reports as reports_service

router = APIRouter(tags=["reports"])


def _safe_filename(title: str | None) -> str:
    base = (title or "should-cost-report").strip() or "should-cost-report"
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in base)
    return keep.replace(" ", "_")[:80] + ".pdf"


@router.get("/{report_id}")
async def get_report(report_id: str, user=Depends(get_current_user)):
    """Return report metadata + markdown (no PDF bytes) for preview."""
    record = reports_service.get_report(report_id, user["id"])
    if record is None:
        raise HTTPException(status_code=404, detail="Report not found")
    markdown = reports_service.normalize_report_markdown(record.get("markdown") or "")
    costs = currency.costs_for_display(record.get("costs"))
    if costs.get("currency") == "INR":
        markdown = currency.enforce_inr_markdown(markdown, costs)
    return {
        "id": record["id"],
        "title": record.get("title"),
        "markdown": markdown,
        "volume": record.get("volume"),
        "costs": costs,
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


@router.get("/{report_id}/download")
async def download_report(report_id: str, user=Depends(get_current_user)):
    """Stream the report PDF as an attachment."""
    record = reports_service.get_report(report_id, user["id"])
    if record is None:
        raise HTTPException(status_code=404, detail="Report not found")
    markdown = reports_service.normalize_report_markdown(record.get("markdown") or "")
    costs = currency.costs_for_display(record.get("costs"))
    if costs.get("currency") == "INR":
        markdown = currency.enforce_inr_markdown(markdown, costs)
    if markdown:
        subtitle = currency.report_subtitle(record.get("volume"), costs)
        pdf_bytes = reports_service.render_pdf(
            markdown, title=record.get("title"), subtitle=subtitle
        )
    else:
        pdf_bytes = reports_service.get_report_pdf_bytes(report_id, user["id"])
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="Report PDF not available")
    filename = _safe_filename(record.get("title"))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
