"""The ``report_generation`` agent tool.

Generates (or edits) a professional should-cost PDF report for a project. It:

1.  Reads the project knowledge base (vision-derived components/pcb/enclosure/…).
2.  Uses HILT (``interrupt()``) to ask a SMALL number of clarifying questions
    when material data is missing; answers are persisted + folded back into KB.
3.  Resolves MPNs, then in parallel prices the BOM (Mouser), quotes the PCB
    (PCBWay), estimates non-quotable blocks, and gathers market context (Tavily).
4.  Applies FX (PCBWay only), customs duty, the assembly model and the volume
    curve, aggregates one structured JSON, fills the locked HTML template,
    renders a PDF, uploads it to the ``reports`` bucket, and streams the result.

Every stage emits a ``report_progress`` event (start → done/warning) and degrades
gracefully — the report always generates; any gap is disclosed in plain language.

The governing principle: **the LLM narrates, the code computes.** All numbers are
computed in code with a ``Live``/``Est`` source tag; nothing is hallucinated.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agent.tools.validation import invalid_project_id_message, is_uuid
from src.core.config import REPORT_DEFAULT_VOLUME
from src.services import report_builder, report_edit, report_pipeline, report_template, reports
from src.services.failure_log import record_failure
from src.services.projects_service import get_project_knowledge_base, get_project_name
from src.services.queue import enqueue_qa_insight

logger = logging.getLogger(__name__)

TOOL_NAME = "report_generation"
TOOL_LABEL = "report generation"

# HILT is implemented with LangGraph ``interrupt()``, which re-runs the whole tool
# from the top on resume — so ``assess_missing`` (an LLM call) would run twice and
# could return DIFFERENT questions the second time, breaking the answer↔question
# matching. We cache the first assessment per thread_id (stable across a turn and
# its resume) so the resume reuses the exact same questions. Cleared on completion.
_ASSESS_CACHE: dict[str, list[dict[str, Any]]] = {}
_ASSESS_CACHE_MAX = 256


def _assess_questions(thread_id: str | None, theory: str, structured: dict, request: str) -> list:
    if thread_id and thread_id in _ASSESS_CACHE:
        return _ASSESS_CACHE[thread_id]
    questions = report_builder.assess_missing(theory, structured, request)
    if thread_id:
        if len(_ASSESS_CACHE) >= _ASSESS_CACHE_MAX:
            _ASSESS_CACHE.clear()
        _ASSESS_CACHE[thread_id] = questions
    return questions

# Cumulative (start, end) progress fraction per stage for the overall bar. The
# tool computes the emitted ``progress`` from this table so the pipeline only has
# to name the stage + status (+ optional meta for interpolation within a stage).
_STAGE_RANGE: dict[str, tuple[float, float]] = {
    "reading_kb": (0.00, 0.04),
    "hilt": (0.04, 0.08),
    "resolving_mpns": (0.08, 0.20),
    "pricing": (0.20, 0.45),
    "fab_quote": (0.45, 0.50),
    "non_quotable": (0.50, 0.55),
    "market_context": (0.55, 0.62),
    "fx": (0.62, 0.66),
    "duty": (0.66, 0.74),
    "assembly": (0.74, 0.80),
    "volume_curve": (0.80, 0.88),
    "rendering": (0.88, 0.94),
    "storing": (0.94, 0.99),
    "report_ready": (1.0, 1.0),
}


def _writer():
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # pragma: no cover - no streaming context
        return lambda _payload: None


def _make_emit():
    """Capture the stream writer once (in the node context) and return an emit fn.

    The returned ``emit`` is safe to call from worker threads (it closes over the
    captured writer rather than re-reading a contextvar that wouldn't propagate).
    """
    writer = _writer()

    def emit(stage: str, status: str, message: str, meta: dict[str, Any] | None = None) -> None:
        start, end = _STAGE_RANGE.get(stage, (0.0, 1.0))
        if status == "in_progress" and meta and meta.get("total"):
            try:
                frac = float(meta["current"]) / float(meta["total"])
            except (TypeError, ValueError, ZeroDivisionError):
                frac = 0.0
            progress = round(start + (end - start) * max(0.0, min(1.0, frac)), 3)
        elif status in ("started",):
            progress = round(start, 3)
        else:  # done | warning | error
            progress = round(end, 3)
        try:
            writer({
                "kind": "report_progress",
                "stage": stage,
                "status": status,
                "message": message,
                "progress": progress,
                "meta": meta or {},
            })
        except Exception:  # pragma: no cover - streaming best-effort
            pass

    return emit, writer


def _cfg(config: RunnableConfig | None) -> dict[str, Any]:
    return (config or {}).get("configurable", {}) or {}


def _parse_volume(text: str) -> int | None:
    """Pull a production volume ONLY when the user states it explicitly.

    Requires an explicit quantity context — a unit word, a 'k' suffix, or a
    'volume/qty of N' phrase — so a stray number in the message (e.g. "9") is
    never mistaken for a production volume. Otherwise returns None (the caller
    defaults to a per-unit basis).
    """
    if not text:
        return None
    patterns = (
        r"(\d[\d,\.]*)\s*([kK])\b",  # "10k"
        r"(\d[\d,\.]*)\s*(?:units?|pcs|pieces|qty|quantity)\b",  # "5000 units"
        r"(?:volume|qty|quantity)\s*(?:of|=|:|is)?\s*(\d[\d,\.]*)\s*([kK])?",  # "volume of 5000"
        r"\bfor\s+(\d[\d,\.]*)\s*([kK])?\s*(?:units?|pcs|pieces)\b",  # "for 1000 units"
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except (ValueError, IndexError):
            continue
        groups = match.groups()
        if len(groups) > 1 and groups[1] and groups[1].lower() == "k":
            value *= 1000
        if value > 0:
            return int(value)
    return None


def _normalize_answer(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        answer = (raw.get("answer") or "").strip() if raw.get("answer") else ""
        file_ids = raw.get("file_ids") or None
        status = raw.get("status") or ("skipped" if not answer and not file_ids else "answered")
        return {"answer": answer, "file_ids": file_ids, "status": status}
    if isinstance(raw, str):
        text = raw.strip()
        return {"answer": text, "file_ids": None, "status": "answered" if text else "skipped"}
    return {"answer": "", "file_ids": None, "status": "skipped"}


def _merge_qa_into_product(
    structured: dict[str, Any],
    theory: str,
    qa_pairs: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Fold HILT answers into THIS report's data (not just the async KB).

    Additive only — it can fill/sharpen fields (pcb layers, dimensions, enclosure
    material/process, a confirmed IC marking/MPN) and append new components, but
    never drops anything already known. Returns the enriched (structured, theory).
    """
    from src.services import report_qa_ingest

    qa_theory, patch = report_qa_ingest.extract_qa_facts(qa_pairs)
    if not qa_theory and not patch:
        return structured, theory

    merged = dict(structured)
    if isinstance(patch, dict):
        # Shallow-fill dict sections (only set fields that are currently empty/missing).
        for section in ("product", "pcb", "enclosure"):
            incoming = patch.get(section)
            if isinstance(incoming, dict):
                base = dict(merged.get(section) or {})
                for k, v in incoming.items():
                    if v in (None, "", []):
                        continue
                    if base.get(k) in (None, "", [], {}):
                        base[k] = v
                merged[section] = base

        # Append any components the answers revealed (match by ref_des to update mpn/top_mark).
        incoming_components = patch.get("components")
        if isinstance(incoming_components, list) and incoming_components:
            existing = list(merged.get("components") or [])
            by_ref = {
                str(c.get("ref_des")).lower(): c
                for c in existing
                if isinstance(c, dict) and c.get("ref_des")
            }
            for comp in incoming_components:
                if not isinstance(comp, dict):
                    continue
                ref = str(comp.get("ref_des") or "").lower()
                target = by_ref.get(ref)
                if target is not None:
                    for k in ("mpn", "top_mark", "value", "manufacturer", "package"):
                        if comp.get(k) and not target.get(k):
                            target[k] = comp[k]
                else:
                    existing.append(comp)
            merged["components"] = existing

        # Carry assumptions through.
        if isinstance(patch.get("assumptions"), list) and patch["assumptions"]:
            merged["assumptions"] = list(merged.get("assumptions") or []) + patch["assumptions"]

    new_theory = (theory + "\n\n" + qa_theory).strip() if qa_theory else theory
    return merged, new_theory


def _structured_from_kb(kb: dict[str, Any]) -> dict[str, Any]:
    structured = kb.get("structured_context")
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError:
            structured = None
    return structured if isinstance(structured, dict) else {}


def _render_store_stream(
    *,
    report_json: dict[str, Any],
    project_id: str,
    conversation_id: str | None,
    user_id: str | None,
    title: str,
    volume: int,
    emit,
    writer,
    existing_report_id: str | None = None,
) -> str | None:
    """Render HTML→PDF, upload to the bucket, persist JSON, and stream report_ready."""
    emit("rendering", "in_progress", "Building the report document…")
    # HTML backs the downloadable PDF (full-fidelity CSS); markdown backs the
    # in-app preview panel (no CSS quirks).
    html = report_template.render_html(report_json)
    markdown_text = report_template.render_markdown(report_json)

    pdf_url = None
    report_id = existing_report_id
    try:
        pdf_bytes = reports.render_pdf_from_html(html)
    except Exception as exc:
        logger.warning("PDF render failed", exc_info=True)
        record_failure(
            "rendering", title or "report",
            "Could not render the report PDF — the report was saved without a PDF",
            error=exc, context={"project_id": project_id, "report_id": existing_report_id},
        )
        pdf_bytes = None

    emit("storing", "started", "Saving your report…")
    try:
        if existing_report_id:
            pdf_path = None
            if pdf_bytes is not None:
                pdf_path, pdf_url = reports.upload_report_pdf(existing_report_id, pdf_bytes)
            reports.update_report(
                existing_report_id,
                title=title,
                volume=volume,
                report_json=report_json,
                html=html,
                markdown_text=markdown_text,
                pdf_path=pdf_path,
                pdf_url=pdf_url,
            )
            report_id = existing_report_id
        else:
            record = reports.create_report(
                project_id, conversation_id, user_id,
                title=title, volume=volume, report_json=report_json,
                html=html, markdown_text=markdown_text, pdf_path=None, pdf_url=None,
            )
            report_id = record["id"] if record else None
            if report_id and pdf_bytes is not None:
                pdf_path, pdf_url = reports.upload_report_pdf(report_id, pdf_bytes)
                reports.update_report(report_id, pdf_path=pdf_path, pdf_url=pdf_url)
        emit("storing", "done", "Report saved.")
    except Exception as exc:
        logger.warning("Report storage failed", exc_info=True)
        record_failure(
            "storing", title or "report",
            "Could not finish saving the report (upload/persist failed)",
            error=exc, context={"project_id": project_id, "report_id": report_id},
        )
        emit("storing", "error", "We couldn't finish saving the report — please try again.")
        return report_id

    emit("report_ready", "done", "Report ready.")
    writer({
        "kind": "report_ready",
        "report_id": report_id,
        "title": title,
        "markdown": markdown_text,
        "pdf_url": pdf_url,
        "volume": volume,
        "fx_rate": (report_json.get("fx") or {}).get("rate"),
    })
    return report_id


def _resolve_image_refs(file_ids: list[str] | None) -> list[dict[str, Any]]:
    """Resolve attached image file IDs into ``[{"url","name"}]`` references."""
    if not file_ids:
        return []
    try:
        from src.services.file_storage import get_image_refs_by_ids

        return get_image_refs_by_ids([str(fid) for fid in file_ids])
    except Exception:
        logger.warning("Failed to resolve attached images for the report", exc_info=True)
        return []


# Lenient: the user attached an image and referred to it ("this image", "the
# photo", "attach it below the summary"). We don't depend on the model's
# paraphrase being precise — any image noun or attach verb is enough.
_IMAGE_INTENT_RE = re.compile(
    r"\b(?:image|images|photo|photos|picture|pictures|pic|pics|screenshot|figure|"
    r"\.heic|\.jpe?g|\.png)\b|\b(?:attach|embed|insert)\b",
    re.IGNORECASE,
)


def _wants_image_attached(*texts: str | None) -> bool:
    return any(t and _IMAGE_INTENT_RE.search(t) for t in texts)


def _image_position_from_text(*texts: str | None) -> str:
    blob = " ".join(t for t in texts if t).lower()
    if "executive" in blob or "summary" in blob:
        return "after_executive"
    return "end"


def _generate_modification(
    project_id: str,
    conversation_id: str | None,
    user_id: str | None,
    request: str,
    modification_request: str,
    file_ids: list[str] | None = None,
) -> str:
    """Edit the latest report for this conversation, operating on its saved JSON."""
    emit, writer = _make_emit()
    base = reports.latest_report_for_conversation(conversation_id) if conversation_id else None
    # Reports belong to a project, not just one chat. If this conversation has no
    # report of its own yet, fall back to the project's latest so the user can edit
    # a report they generated in an earlier conversation.
    if base is None or not base.get("report_json"):
        base = reports.latest_report_for_project(project_id, user_id) or base
    if base is None or not base.get("report_json"):
        return (
            "There's no existing report for this product yet. "
            "Ask me to generate a report first, then I can tweak it."
        )

    report_json = base["report_json"]
    if isinstance(report_json, str):
        try:
            report_json = json.loads(report_json)
        except json.JSONDecodeError:
            return "The saved report couldn't be read for editing — please regenerate it."

    image_refs = _resolve_image_refs(file_ids)
    images_before = len(report_json.get("images") or [])

    emit("rendering", "started", "Applying your requested changes…")
    report_json, summary = report_edit.apply_edit(
        report_json, modification_request, emit, image_refs=image_refs,
    )

    # Fallback: if the user attached an image and clearly wants it embedded but the
    # edit classifier didn't emit an add_image op, attach it here. Intent is read
    # from the VERBATIM request too — the model's paraphrase often drops it.
    image_intended = _wants_image_attached(request, modification_request)
    if (
        image_refs
        and len(report_json.get("images") or []) == images_before
        and image_intended
    ):
        position = _image_position_from_text(request, modification_request)
        added = report_edit._add_images(report_json, image_refs, position, None)
        if added:
            summary = (summary + f"; attached {added} image(s)").strip("; ")

    images_added = len(report_json.get("images") or []) - images_before

    title = (report_json.get("meta") or {}).get("title") or base.get("title") or "Should-Cost Report"
    volume = (report_json.get("meta") or {}).get("volume") or base.get("volume") or REPORT_DEFAULT_VOLUME

    _render_store_stream(
        report_json=report_json,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        title=title,
        volume=int(volume),
        emit=emit,
        writer=writer,
        existing_report_id=base["id"],
    )

    # Honest note about the image outcome so the assistant doesn't claim an attach
    # that didn't happen.
    image_note = ""
    if image_intended and images_added <= 0:
        if not image_refs:
            image_note = (
                " NOTE: I could not embed an image because no image file was attached to this "
                "message (or it wasn't an image). Tell the user to attach the image to their "
                "message and ask again — do NOT claim the image was added."
            )
        else:
            image_note = " NOTE: the image could not be embedded — do NOT claim it was added."
    return (
        f'Updated the report — {summary}. The revised version is shown on the right and the PDF is '
        f"ready to download. Let me know if you'd like any other changes.{image_note}"
    )


@tool(TOOL_NAME)
def report_generation_tool(
    request: str,
    config: RunnableConfig = None,  # type: ignore[assignment]
    modification_request: str | None = None,
    file_ids: list[str] | None = None,
) -> str:
    """Generate a NEW professional should-cost PDF report for the product.

    Call this whenever the user asks to generate a report, a should-cost report,
    a BOM cost report, a cost breakdown PDF, or similar. It reads the project's
    knowledge base, asks a few clarifying questions only if needed, prices the BOM
    (Mouser), quotes the PCB (PCBWay), applies duty and assembly costs across a
    volume curve, renders a PDF in the locked report format, and streams it.

    To CHANGE a report that already exists in this conversation, use the dedicated
    ``report_edit`` tool instead — it is much faster (no KB re-read, no new
    questions, no full re-pricing). ``modification_request`` here is retained only
    as a backward-compatible fallback that delegates to the same edit logic.

    Args:
        request: The user's request, verbatim (used to infer intent/volume).
        modification_request: DEPRECATED entry point — prefer the ``report_edit``
            tool. If set (e.g. "change the target volume to 5000", "remove the LED
            line", "shorten the executive summary"), the existing report is edited
            in place via the same logic ``report_edit`` uses.
        file_ids: The IDs of any images the user ATTACHED to this message that they
            want embedded in the report. Pass the attached file IDs here whenever
            the user asks to attach/add/embed an image or photo.
    """
    cfg = _cfg(config)
    project_id = cfg.get("project_id")
    user_id = cfg.get("user_id")
    conversation_id = cfg.get("conversation_id")
    thread_id = cfg.get("thread_id")
    # Attached image IDs can arrive two ways: threaded through the request config
    # (reliable, set by chat_stream) or passed by the model as a tool arg. Merge
    # both so an attach works regardless of which path delivered them.
    cfg_file_ids = [str(x) for x in (cfg.get("file_ids") or [])]
    arg_file_ids = [str(x) for x in (file_ids or [])]
    file_ids = list(dict.fromkeys(cfg_file_ids + arg_file_ids))

    if not project_id:
        return "No project is associated with this conversation, so I can't build a report."
    if not is_uuid(str(project_id)):
        return invalid_project_id_message(str(project_id))

    # ---- Edit path: revise the existing report instead of rebuilding. ------- #
    if modification_request and modification_request.strip():
        return _generate_modification(
            str(project_id), conversation_id, user_id,
            request, modification_request.strip(), file_ids=file_ids,
        )

    emit, writer = _make_emit()

    # ---- Step 1: read KB ---------------------------------------------------- #
    emit("reading_kb", "started", "Reading what we know about the product…")
    kb = get_project_knowledge_base(str(project_id))
    if not kb or (not (kb.get("theory_context") or "").strip() and not kb.get("structured_context")):
        return (
            "I don't have enough analyzed information about this product yet to build a "
            "report. Please upload photos of the PCB and product (and any datasheets) "
            "so I can analyze them first, then ask me to generate the report."
        )
    theory = (kb.get("theory_context") or "").strip()
    structured = _structured_from_kb(kb)
    project_name = get_project_name(str(project_id)) or "this product"
    product_label = ((structured.get("product") or {}).get("name")) or project_name
    emit("reading_kb", "done", "Product knowledge loaded.")

    # ---- Step 2: HILT gap questions ----------------------------------------- #
    emit("hilt", "started", "Checking whether I need any details from you…")
    # Cached per thread_id so the resume re-run yields the SAME questions.
    questions = _assess_questions(thread_id, theory, structured, request)
    answers_by_id: dict[str, dict[str, Any]] = {}
    if questions:
        from langgraph.types import interrupt

        resumed = interrupt({"type": "report_questions", "questions": questions})
        raw_answers = resumed.get("answers") if isinstance(resumed, dict) else resumed
        if isinstance(raw_answers, dict):
            answers_by_id = {k: _normalize_answer(v) for k, v in raw_answers.items()}
        elif isinstance(raw_answers, list):
            for q, a in zip(questions, raw_answers):
                answers_by_id[q["id"]] = _normalize_answer(a)

        qa_pairs: list[dict[str, Any]] = []
        for q in questions:
            ans = answers_by_id.get(q["id"], {"answer": "", "file_ids": None, "status": "skipped"})
            try:
                reports.save_question_answer(
                    str(project_id), conversation_id, user_id,
                    question=q["prompt"], kind=q["kind"],
                    answer=ans["answer"] or None, file_ids=ans["file_ids"], status=ans["status"],
                )
            except Exception:
                logger.warning("Failed to persist report question", exc_info=True)
            qa_pairs.append({"question": q["prompt"], "answer": ans["answer"], "file_ids": ans["file_ids"]})

        # Fold the answers into THIS report run (sharpen MPNs/specs), then queue
        # the same facts for the async KB recompute so future reports benefit too.
        if any(p.get("answer") or p.get("file_ids") for p in qa_pairs):
            try:
                structured, theory = _merge_qa_into_product(structured, theory, qa_pairs)
                product_label = ((structured.get("product") or {}).get("name")) or product_label
            except Exception:
                logger.warning("Failed to fold Q&A answers into the report", exc_info=True)
        try:
            enqueue_qa_insight(str(project_id), user_id, qa_pairs)
        except Exception:
            logger.warning("Failed to enqueue Q&A insight ingestion", exc_info=True)
    # Resume consumed the cached assessment — release it.
    if thread_id:
        _ASSESS_CACHE.pop(thread_id, None)
    emit("hilt", "done", "Thanks — using your details." if questions else "No extra details needed.")

    # ---- Determine production volume ---------------------------------------- #
    # The report is always for a SINGLE unit. We do not parse a production volume
    # from the request — costs are the recurring per-unit cost, with one-time NRE
    # reported separately (never amortized over a quantity).
    volume = 1

    # ---- Steps 3–9: run the pipeline ---------------------------------------- #
    report_json = report_pipeline.run_pipeline(
        structured=structured,
        theory=theory,
        product_label=product_label,
        volume=volume,
        emit=emit,
    )
    # The report is always single-unit: per-unit recurring cost, with one-time NRE
    # (tooling/firmware/line setup) reported separately — never amortized in.
    report_json.setdefault("dataQuality", []).append(
        "Figures are the recurring cost to build a single unit. One-time tooling, firmware "
        "and line-setup (NRE) is listed separately and is not included in the per-unit cost."
    )

    # Embed any attached images when the user asked to include one in the report.
    if file_ids and _wants_image_attached(request):
        image_refs = _resolve_image_refs(file_ids)
        if image_refs:
            report_edit._add_images(
                report_json, image_refs, _image_position_from_text(request), None
            )

    # ---- Steps 10–11: render, store, stream --------------------------------- #
    title = report_json.get("meta", {}).get("title") or f"Should-Cost Report — {product_label}"
    _render_store_stream(
        report_json=report_json,
        project_id=str(project_id),
        conversation_id=conversation_id,
        user_id=user_id,
        title=title,
        volume=volume,
        emit=emit,
        writer=writer,
    )

    live = report_json.get("bom", {}).get("live_count", 0)
    total_lines = live + report_json.get("bom", {}).get("est_count", 0)
    note = "" if not report_json.get("dataQuality") else " Some figures use industry estimates where live data wasn't available — see the Data Confidence notes."
    return (
        f'I generated the single-unit should-cost report "{title}" '
        f"({live} of {total_lines} BOM lines priced live).{note} "
        "It's the recurring cost to build one unit, with one-time tooling/NRE listed separately. "
        "It's shown on the right and the PDF is ready to download — want a line removed or more detail in a section?"
    )
