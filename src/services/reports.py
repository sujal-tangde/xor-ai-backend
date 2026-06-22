"""Persistence + PDF rendering for generated should-cost reports.

A report is stored as three things on the ``reports`` row:
  - ``report_json`` — the aggregated structured JSON (the single source of truth;
    later edits operate on this, never on a fresh teardown run).
  - ``html`` — the rendered locked-template HTML (used for the live preview panel).
  - ``pdf_path`` / ``pdf_url`` — the rendered PDF is uploaded to the ``reports``
    storage bucket; only the path/URL is kept on the row (no base64 inline).

PDFs are rendered from the template HTML with headless Chromium (Playwright) for
full-fidelity CSS, falling back to xhtml2pdf if Playwright is unavailable so the
report still renders. The HILT questions + answers live in ``report_questions``.
"""

from __future__ import annotations

import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from src.core.config import REPORTS_BUCKET, SUPABASE_URL
from src.services.file_storage import get_supabase

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL | re.IGNORECASE)


def normalize_report_markdown(text: str) -> str:
    """Strip accidental ```markdown fences (kept for legacy markdown reports)."""
    if not text:
        return text
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


# --------------------------------------------------------------------------- #
# PDF rendering — Playwright (Chromium) with an xhtml2pdf fallback.
# --------------------------------------------------------------------------- #
def _render_with_playwright(html: str) -> bytes:
    """Render HTML → PDF via headless Chromium.

    Runs in a fresh thread (no event loop) so the Playwright *sync* API is safe
    regardless of whether the caller is inside an asyncio loop.
    """
    def _work() -> bytes:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                pdf = page.pdf(
                    format="A4",
                    print_background=True,
                    margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
                )
            finally:
                browser.close()
        return pdf

    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_work).result()


def _render_with_xhtml2pdf(html: str) -> bytes:
    """Fallback renderer. Lower CSS fidelity but no native browser dependency."""
    from xhtml2pdf import pisa

    buf = io.BytesIO()
    result = pisa.CreatePDF(src=html, dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError("xhtml2pdf failed to render report PDF")
    return buf.getvalue()


def render_pdf_from_html(html: str) -> bytes:
    """Render the report HTML to PDF bytes, preferring Playwright."""
    try:
        return _render_with_playwright(html)
    except Exception as exc:  # noqa: BLE001 - fall back so a report always renders
        logger.warning("Playwright PDF render failed (%s); falling back to xhtml2pdf", exc)
        return _render_with_xhtml2pdf(html)


# --------------------------------------------------------------------------- #
# Storage bucket for the rendered PDFs.
# --------------------------------------------------------------------------- #
def ensure_reports_bucket() -> None:
    """Create the reports storage bucket if it doesn't already exist (best effort)."""
    try:
        client = get_supabase()
        client.storage.create_bucket(REPORTS_BUCKET, options={"public": True})
        logger.info("Created reports storage bucket '%s'", REPORTS_BUCKET)
    except Exception:
        # Already exists (or transient) — uploads use upsert so this is non-fatal.
        pass


def _public_url(path: str) -> str:
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/storage/v1/object/public/{REPORTS_BUCKET}/{path}"


def upload_report_pdf(report_id: str, pdf_bytes: bytes) -> tuple[str, str]:
    """Upload (overwrite) the report PDF to the bucket. Returns ``(path, url)``."""
    path = f"{report_id}.pdf"
    client = get_supabase()
    try:
        client.storage.from_(REPORTS_BUCKET).upload(
            path,
            pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )
    except Exception:
        # Bucket may not exist yet on first ever run — create and retry once.
        ensure_reports_bucket()
        client.storage.from_(REPORTS_BUCKET).upload(
            path,
            pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )
    return path, _public_url(path)


def download_report_pdf_bytes(path: str) -> bytes | None:
    """Fetch a stored PDF's bytes from the bucket (used by the download route)."""
    if not path:
        return None
    try:
        return get_supabase().storage.from_(REPORTS_BUCKET).download(path)
    except Exception:
        logger.warning("Could not download report PDF from bucket: %s", path, exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# reports table
# --------------------------------------------------------------------------- #
def create_report(
    project_id: str,
    conversation_id: str | None,
    user_id: str | None,
    *,
    title: str | None,
    volume: int | None,
    report_json: dict[str, Any],
    html: str,
    pdf_path: str | None,
    pdf_url: str | None,
    status: str = "ready",
) -> dict[str, Any] | None:
    row = {
        "project_id": project_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "title": title,
        "volume": volume,
        "report_json": report_json,
        "html": html,
        "pdf_path": pdf_path,
        "pdf_url": pdf_url,
        "status": status,
    }
    result = get_supabase().table("reports").insert(row).execute()
    return result.data[0] if result.data else None


def update_report(
    report_id: str,
    *,
    title: str | None = None,
    volume: int | None = None,
    report_json: dict[str, Any] | None = None,
    html: str | None = None,
    pdf_path: str | None = None,
    pdf_url: str | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"updated_at": _now()}
    if title is not None:
        payload["title"] = title
    if volume is not None:
        payload["volume"] = volume
    if report_json is not None:
        payload["report_json"] = report_json
    if html is not None:
        payload["html"] = html
    if pdf_path is not None:
        payload["pdf_path"] = pdf_path
    if pdf_url is not None:
        payload["pdf_url"] = pdf_url
    if status is not None:
        payload["status"] = status
    result = get_supabase().table("reports").update(payload).eq("id", report_id).execute()
    return result.data[0] if result.data else None


def get_report(report_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    query = get_supabase().table("reports").select("*").eq("id", report_id)
    if user_id is not None:
        query = query.eq("user_id", user_id)
    result = query.limit(1).execute()
    return result.data[0] if result.data else None


def get_report_pdf_bytes(report_id: str, user_id: str | None = None) -> bytes | None:
    """Best-effort PDF bytes: from the bucket, else re-rendered from stored HTML."""
    record = get_report(report_id, user_id)
    if not record:
        return None
    if record.get("pdf_path"):
        data = download_report_pdf_bytes(record["pdf_path"])
        if data:
            return data
    html = record.get("html")
    if html:
        try:
            return render_pdf_from_html(html)
        except Exception:
            logger.warning("Re-render of report %s failed", report_id, exc_info=True)
    return None


def latest_report_for_conversation(conversation_id: str) -> dict[str, Any] | None:
    result = (
        get_supabase()
        .table("reports")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# --------------------------------------------------------------------------- #
# report_questions table
# --------------------------------------------------------------------------- #
def save_question_answer(
    project_id: str,
    conversation_id: str | None,
    user_id: str | None,
    *,
    question: str,
    kind: str,
    answer: str | None,
    file_ids: list[str] | None,
    status: str,
) -> dict[str, Any] | None:
    row = {
        "project_id": project_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "question": question,
        "kind": kind,
        "answer": answer,
        "file_ids": file_ids or None,
        "status": status,
        "answered_at": _now() if status != "pending" else None,
    }
    result = get_supabase().table("report_questions").insert(row).execute()
    return result.data[0] if result.data else None
