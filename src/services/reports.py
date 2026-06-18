"""Persistence + PDF rendering for generated should-cost reports.

Reports are stored in the ``reports`` table with the rendered PDF inlined as
base64 (reports are small). The HILT questions the report tool asked, and their
answers, are stored in ``report_questions``.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from datetime import datetime, timezone
from typing import Any

import markdown as md_lib
from xhtml2pdf import pisa

from src.services.file_storage import get_supabase

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_FENCE_RE = re.compile(
    r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def normalize_report_markdown(text: str) -> str:
    """Strip accidental ```markdown fences wrapping the entire report."""
    if not text:
        return text
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


# --------------------------------------------------------------------------- #
# PDF rendering
# --------------------------------------------------------------------------- #
_PDF_CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 20pt; color: #0f172a; margin: 0 0 4pt 0; }
h2 { font-size: 14pt; color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 3pt; margin-top: 16pt; }
h3 { font-size: 12pt; color: #334155; margin-top: 12pt; }
p { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
th { background-color: #0f172a; color: #ffffff; text-align: left; padding: 5pt 6pt; }
td { border: 1px solid #cbd5e1; padding: 4pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background-color: #f1f5f9; }
code { font-family: "Courier New", monospace; background-color: #f1f5f9; padding: 1pt 2pt; }
ul, ol { margin: 4pt 0 4pt 14pt; }
.report-meta { color: #64748b; font-size: 9pt; margin-bottom: 10pt; }
"""


def render_pdf(markdown_text: str, *, title: str | None = None, subtitle: str | None = None) -> bytes:
    """Render report markdown (GitHub-flavored) to PDF bytes."""
    body_html = md_lib.markdown(
        normalize_report_markdown(markdown_text or ""),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    header = ""
    if title:
        header += f"<h1>{title}</h1>"
    if subtitle:
        header += f'<div class="report-meta">{subtitle}</div>'
    html = (
        f"<html><head><meta charset='utf-8'><style>{_PDF_CSS}</style></head>"
        f"<body>{header}{body_html}</body></html>"
    )
    buf = io.BytesIO()
    result = pisa.CreatePDF(src=html, dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError("Failed to render report PDF")
    return buf.getvalue()


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
    markdown_text: str,
    costs: dict[str, Any] | None,
    pdf_bytes: bytes,
    status: str = "ready",
) -> dict[str, Any] | None:
    row = {
        "project_id": project_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "title": title,
        "volume": volume,
        "markdown": normalize_report_markdown(markdown_text),
        "costs": costs,
        "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "status": status,
    }
    result = get_supabase().table("reports").insert(row).execute()
    return result.data[0] if result.data else None


def update_report(
    report_id: str,
    *,
    title: str | None = None,
    markdown_text: str | None = None,
    costs: dict[str, Any] | None = None,
    pdf_bytes: bytes | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"updated_at": _now()}
    if title is not None:
        payload["title"] = title
    if markdown_text is not None:
        payload["markdown"] = normalize_report_markdown(markdown_text)
    if costs is not None:
        payload["costs"] = costs
    if pdf_bytes is not None:
        payload["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
    if status is not None:
        payload["status"] = status
    result = (
        get_supabase().table("reports").update(payload).eq("id", report_id).execute()
    )
    return result.data[0] if result.data else None


def get_report(report_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    query = get_supabase().table("reports").select("*").eq("id", report_id)
    if user_id is not None:
        query = query.eq("user_id", user_id)
    result = query.limit(1).execute()
    return result.data[0] if result.data else None


def get_report_pdf_bytes(report_id: str, user_id: str | None = None) -> bytes | None:
    record = get_report(report_id, user_id)
    if not record or not record.get("pdf_base64"):
        return None
    return base64.b64decode(record["pdf_base64"])


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
