"""Free-form HTML report edit engine.

This REPLACES the old closed-operation edit system (``services/report_edit.py``'s
``classify_edit`` + the operation appliers). A should-cost report is stored as
free-form HTML, so editing it works the way a person edits text: read the
request, change exactly what was asked, leave everything else untouched.

The engine asks a capable LLM to return the COMPLETE modified HTML, then VERIFIES
the result deterministically in code (not via the LLM):

  1. The returned HTML must actually differ from the original. If the model
     claims a change but the HTML is unchanged, that's a fabrication — we reject
     it and report an honest failure (no ``report_ready`` is ever emitted).
  2. Existing ``<img>`` tags must survive unless the user asked to remove one.
     This stops "add an image" from silently dropping existing images.
  3. The fraction of the document that changed is measured; an edit far larger
     than a normal targeted change is retried with a stricter prompt and, if it
     stays huge, surfaced as a warning.

Crucially, the human-facing summary is built from the COMPUTED diff, never from
the model's self-described ``changes`` — so the assistant can never claim a change
that did not happen.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from markdownify import markdownify

from src.core.config import LLM_MODEL
from src.services.llm_analysis import invoke_llm

logger = logging.getLogger(__name__)

# Fraction-of-document-changed above which an edit is considered suspiciously
# broad for a targeted request (the "I asked for one thing and it rewrote
# everything" guard). Tuned loosely — legitimate big edits still go through with
# a warning rather than a hard block.
_SCOPE_WARN_THRESHOLD = 0.6

# Output limit for the editor call. The model returns the whole document, so this
# must comfortably exceed the rendered report size.
_EDIT_MAX_TOKENS = 24000

# --------------------------------------------------------------------------- #
# Edit prompt (the rules are authoritative; only the OUTPUT FORMAT differs from
# pure JSON — a delimiter block is far more robust than JSON-escaping a whole
# HTML document with embedded quotes/newlines).
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are an HTML report editor. You receive a report as HTML and a user's edit request. You return the COMPLETE modified HTML with ONLY the requested change applied.

ABSOLUTE RULES:

1. SURGICAL EDITS ONLY. Change ONLY what the user explicitly asked for. Every other character of the HTML must remain identical. If the user asks to change the title, change ONLY the title element. Do not reformat, reindent, "improve," or touch anything else.

2. PRESERVE EVERYTHING NOT MENTIONED. Existing images, tables, sections, numbers, styling — all of it stays exactly as-is unless the user specifically asked to change that thing.

3. ADDING != REPLACING. If the user asks to ADD an image, APPEND it. Do NOT remove or replace any existing image. If the user asks to ADD a row, table, or section, the existing ones stay. Only remove/replace when the user explicitly uses words like "remove", "replace", "delete", "get rid of".

4. NO UNREQUESTED SIDE EFFECTS. If the user says "attach this image," you change ONLY the image situation — you do NOT touch the title, descriptions, numbers, or anything else. If you find yourself changing something the user didn't mention, STOP and don't change it.

5. NUMBERS ARE EDITABLE BUT NOT INVENTED. The user may change cost numbers, quantities, prices — when they explicitly give you a value, apply it exactly. Never invent, recalculate, or "correct" numbers on your own initiative. If the user says "change the unit price of U1 to $2.50," set exactly that. If they don't mention a number, leave it.

6. IF YOU CANNOT DO SOMETHING, SAY SO. If part of the request is impossible or ambiguous, list it under UNABLE and do not guess.

OUTPUT FORMAT — return EXACTLY these three delimited sections and nothing else:
===CHANGES===
(one bullet per change you made, or "none")
===UNABLE===
(one bullet per thing you could not do, or "none")
===HTML===
(the complete modified HTML document, raw — no code fences)

EXAMPLES OF CORRECT BEHAVIOR:

Request: "Change the title to Q3 Analysis"
-> Find the title element, change its text to "Q3 Analysis", return the full HTML otherwise identical. CHANGES: Changed title to 'Q3 Analysis'.

Request: "Attach this image" (image URL provided), report already has 2 images
-> Append a new <img> with the provided URL at the appropriate location. Both existing images remain untouched. CHANGES: Added the provided image.

Request: "Make the executive summary shorter"
-> Rewrite ONLY the executive summary prose, more concise. Title, numbers, other sections, images — all untouched. CHANGES: Shortened the executive summary.

Request: "Change the unit price of the STM32 to $4.20 and center the title"
-> Two changes: update that one price value, add text-align:center (or equivalent) to the title element only. Nothing else changes. CHANGES: Set STM32 unit price to $4.20; Centered the report title."""


@dataclass
class EditResult:
    """Outcome of an edit attempt.

    ``applied`` is True ONLY when the HTML genuinely changed and passed the
    image-preservation check. When False, ``failure_message`` explains why and no
    report should be re-rendered or marked ready.
    """

    applied: bool
    html: str | None = None
    markdown: str | None = None
    summary: str = ""
    failure_message: str = ""
    warnings: list[str] = field(default_factory=list)
    unable_to_do: list[str] = field(default_factory=list)
    changes_claimed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Deterministic verification helpers (no LLM).
# --------------------------------------------------------------------------- #
_IMG_SRC_RE = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
# Words that mean the user wants an existing image gone (so its disappearance is
# expected, not a violation). Used only to RELAX the preservation guard.
_IMAGE_REMOVAL_RE = re.compile(
    r"\b(remove|delete|drop|get\s+rid\s+of|take\s+out|replace|swap|clear)\b",
    re.IGNORECASE,
)


def _normalize(html: str | None) -> str:
    """Collapse insignificant whitespace so reindentation alone reads as no-op."""
    return re.sub(r"\s+", " ", html or "").strip()


# The report's <style> block is a large, static stylesheet. Sending it to the
# editor and having the model re-emit it verbatim wastes a lot of input AND output
# tokens (output dominates latency). We swap it for a tiny placeholder before the
# call and restore it after, so the model never regenerates the CSS. Edits that
# target styling still work via inline styles on the element (e.g. text-align).
_STYLE_RE = re.compile(r"(<style\b[^>]*>)(.*?)(</style>)", re.IGNORECASE | re.DOTALL)
_CSS_TOKEN = "/*__REPORT_CSS_OMITTED__*/"


def _strip_css(html: str) -> tuple[str, str | None]:
    """Replace the first <style> block's body with a placeholder. Returns (html, css)."""
    saved: list[str] = []

    def repl(m: "re.Match[str]") -> str:
        saved.append(m.group(2))
        return m.group(1) + _CSS_TOKEN + m.group(3)

    stripped = _STYLE_RE.sub(repl, html, count=1)
    return stripped, (saved[0] if saved else None)


def _restore_css(html: str, css: str | None) -> str:
    """Put the original CSS back where the placeholder is."""
    if css is None or not html:
        return html
    if _CSS_TOKEN in html:
        return html.replace(_CSS_TOKEN, css, 1)
    # Model dropped the placeholder — reinsert a style block so styling survives.
    style = f"<style>{css}</style>"
    if "</head>" in html:
        return html.replace("</head>", style + "</head>", 1)
    return html.replace("<body", style + "<body", 1) if "<body" in html else html


def image_srcs(html: str | None) -> list[str]:
    return _IMG_SRC_RE.findall(html or "")


def removal_requested(request: str) -> bool:
    """True when the request explicitly asks to remove/replace something."""
    return bool(_IMAGE_REMOVAL_RE.search(request or ""))


def compute_diff(old_html: str, new_html: str) -> dict:
    """Structured, deterministic comparison of two HTML documents."""
    old_words = _normalize(old_html).split()
    new_words = _normalize(new_html).split()
    matcher = difflib.SequenceMatcher(None, old_words, new_words)
    old_imgs = image_srcs(old_html)
    new_imgs = image_srcs(new_html)
    return {
        "ratio": matcher.ratio(),
        "scope": round(1.0 - matcher.ratio(), 4),
        "empty": _normalize(old_html) == _normalize(new_html),
        "old_images": old_imgs,
        "new_images": new_imgs,
        "removed_images": [s for s in old_imgs if s not in set(new_imgs)],
        "added_images": [s for s in new_imgs if s not in set(old_imgs)],
        "old_len": len(old_html or ""),
        "new_len": len(new_html or ""),
    }


def diff_is_empty(diff: dict) -> bool:
    return bool(diff.get("empty"))


def existing_images_preserved(
    old_html: str, new_html: str, removal_ok: bool
) -> bool:
    """Every image present before is still present after — unless removal asked."""
    if removal_ok:
        return True
    new_set = set(image_srcs(new_html))
    return all(src in new_set for src in image_srcs(old_html))


def change_scope_ratio(old_html: str, new_html: str) -> float:
    """Fraction of the document (token-level) that changed, in [0, 1]."""
    return compute_diff(old_html, new_html)["scope"]


def _title_of(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
    except Exception:
        return None
    node = soup.select_one(".cover h1") or soup.find("h1")
    if node and node.get_text(strip=True):
        return node.get_text(strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    return None


def _visible_lines(html: str) -> list[str]:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
    except Exception:
        return []
    for tag in soup(["style", "script"]):
        tag.decompose()
    text = soup.get_text("\n")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def build_summary_from_diff(old_html: str, new_html: str) -> str:
    """Human-readable summary derived from what ACTUALLY changed (never claims)."""
    diff = compute_diff(old_html, new_html)
    parts: list[str] = []

    old_title, new_title = _title_of(old_html), _title_of(new_html)
    title_changed = bool(new_title) and old_title != new_title
    if title_changed:
        parts.append(f'changed the title to "{new_title}"')

    if diff["added_images"]:
        n = len(diff["added_images"])
        parts.append(f"added {n} image{'s' if n != 1 else ''}")
    if diff["removed_images"]:
        n = len(diff["removed_images"])
        parts.append(f"removed {n} image{'s' if n != 1 else ''}")

    # Did text content change beyond the title line itself?
    changed_lines = sum(
        1
        for ln in difflib.ndiff(_visible_lines(old_html), _visible_lines(new_html))
        if ln[:1] in ("+", "-")
    )
    text_beyond_title = changed_lines > (2 if title_changed else 0)
    if text_beyond_title:
        parts.append("revised the report content")

    if not parts:
        # The HTML changed (verified upstream) but in a way not captured above —
        # e.g. styling/layout/alignment only. Say so honestly without specifics.
        parts.append("applied your requested formatting/layout change")

    return "I've " + _join(parts) + ". The updated report is on the right and the refreshed PDF is ready to download."


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------- #
# Model call + parsing.
# --------------------------------------------------------------------------- #
def _build_user_prompt(
    html: str, request: str, image_urls: list[str], extra: str, css_stripped: bool
) -> str:
    img_block = ""
    if image_urls:
        listed = "\n".join(f"- {u}" for u in image_urls)
        img_block = (
            "\n\nThe user attached the following image URL(s). If the request asks "
            "to add/attach/insert an image, embed them with <img> tags at the "
            "appropriate place (append; do not replace existing images unless the "
            f"user said to):\n{listed}"
        )
    css_block = ""
    if css_stripped:
        css_block = (
            f"\n\nNOTE: the document's <style> contents have been replaced with the "
            f"placeholder `{_CSS_TOKEN}` to save space. Leave that placeholder "
            "EXACTLY as-is in your output (do not expand or remove it). To change "
            "styling, add an inline style attribute on the specific element instead."
        )
    extra_block = f"\n\n{extra}" if extra else ""
    return (
        f"USER EDIT REQUEST:\n{request}{img_block}{css_block}{extra_block}\n\n"
        f"CURRENT REPORT HTML:\n{html}"
    )


def _parse_output(raw: str) -> tuple[list[str], list[str], str | None]:
    """Parse the delimited editor output into (changes, unable, html)."""
    if not raw:
        return [], [], None

    def _section(name: str, nxt: list[str]) -> str:
        start = re.search(rf"={{2,}}\s*{name}\s*={{2,}}", raw, re.IGNORECASE)
        if not start:
            return ""
        rest = raw[start.end():]
        end_pos = len(rest)
        for other in nxt:
            m = re.search(rf"={{2,}}\s*{other}\s*={{2,}}", rest, re.IGNORECASE)
            if m and m.start() < end_pos:
                end_pos = m.start()
        return rest[:end_pos].strip()

    changes_txt = _section("CHANGES", ["UNABLE", "HTML", "END"])
    unable_txt = _section("UNABLE", ["HTML", "END"])
    html_txt = _section("HTML", ["END"])

    # Strip an accidental code fence around the HTML.
    fence = re.match(r"^```(?:html)?\s*(.*?)\s*```$", html_txt, re.DOTALL)
    if fence:
        html_txt = fence.group(1).strip()

    # If the model ignored the format and just returned a document, salvage it.
    if not html_txt:
        doc = re.search(r"(<!DOCTYPE html.*?</html\s*>)", raw, re.IGNORECASE | re.DOTALL)
        if doc:
            html_txt = doc.group(1).strip()

    def _bullets(text: str) -> list[str]:
        items = []
        for line in text.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if line and line.lower() != "none":
                items.append(line)
        return items

    return _bullets(changes_txt), _bullets(unable_txt), (html_txt or None)


def _call_model(
    html: str, request: str, image_urls: list[str], extra: str = "",
    css_stripped: bool = False,
) -> tuple[list[str], list[str], str | None]:
    raw = invoke_llm(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(html, request, image_urls, extra, css_stripped)},
        ],
        max_tokens=_EDIT_MAX_TOKENS,
        model=LLM_MODEL,
    )
    return _parse_output(raw)


def _looks_like_document(html: str | None) -> bool:
    return bool(html) and "<" in html and ">" in html and len(html) > 50


def html_to_markdown(html: str) -> str:
    """Best-effort HTML -> Markdown for the preview-panel/markdown column fallback.

    The panel renders the HTML itself; this is a degraded fallback for any
    consumer that only has the markdown column.
    """
    try:
        md = markdownify(html, heading_style="ATX", strip=["style", "script"])
        return re.sub(r"\n{3,}", "\n\n", md).strip()
    except Exception:
        logger.warning("HTML->markdown conversion failed", exc_info=True)
        return ""


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def edit(html: str, request: str, image_urls: list[str] | None = None) -> EditResult:
    """Apply ``request`` to ``html`` and verify the result deterministically.

    Returns an :class:`EditResult`. ``applied`` is True only when the HTML
    genuinely changed and survived verification; otherwise ``failure_message``
    holds an honest explanation and the caller MUST NOT emit ``report_ready``.
    """
    image_urls = [u for u in (image_urls or []) if u]
    removal_ok = removal_requested(request)

    # Send the model the document WITHOUT the big static CSS block (placeholder
    # swapped in) to cut input+output tokens; restore the real CSS on every
    # returned candidate so all verification runs on the full document.
    stripped_input, css = _strip_css(html)

    def _attempt(extra: str = "") -> tuple[list[str], list[str], str | None]:
        c, u, h = _call_model(
            stripped_input, request, image_urls, extra, css_stripped=css is not None
        )
        return c, u, (_restore_css(h, css) if h else h)

    # First attempt.
    changes, unable, new_html = _attempt()

    # Retry once if the model returned no usable document at all.
    if not _looks_like_document(new_html):
        logger.info("Editor returned no usable HTML; retrying once.")
        changes, unable, new_html = _attempt(
            "Your previous reply did not contain the complete HTML document. "
            "Return the FULL modified HTML under ===HTML===."
        )
    if not _looks_like_document(new_html):
        return EditResult(
            applied=False,
            unable_to_do=unable,
            changes_claimed=changes,
            failure_message=(
                "I wasn't able to edit the report just now — the editor didn't return "
                "a usable document, so nothing was changed. Please try again, or "
                "rephrase what you'd like changed."
            ),
        )

    diff = compute_diff(html, new_html)

    # Guard 1: empty diff. If nothing changed, the report is unchanged — report
    # honestly. If the model also listed things it couldn't do, surface those.
    if diff_is_empty(diff):
        logger.info("Editor produced an empty diff; retrying once with a stricter nudge.")
        changes, unable, retry_html = _attempt(
            "Your previous reply returned the HTML UNCHANGED. Actually apply "
            "the requested change to the HTML this time, or, if it truly cannot "
            "be done, list it under UNABLE and explain."
        )
        if _looks_like_document(retry_html):
            new_html = retry_html
            diff = compute_diff(html, new_html)

    if diff_is_empty(diff):
        if unable:
            return EditResult(
                applied=False,
                unable_to_do=unable,
                changes_claimed=changes,
                failure_message=(
                    "I couldn't make that change, so the report is unchanged: "
                    + "; ".join(unable)
                    + ". Could you clarify exactly what to change?"
                ),
            )
        return EditResult(
            applied=False,
            changes_claimed=changes,
            failure_message=(
                "That didn't end up changing anything in the report — it's unchanged. "
                "Could you say a bit more specifically what you'd like changed (which "
                "text, section, number, or image)?"
            ),
        )

    # Guard 2: existing images must survive unless removal was requested.
    if not existing_images_preserved(html, new_html, removal_ok):
        logger.info("Edit dropped an existing image without being asked; retrying.")
        changes, unable, retry_html = _attempt(
            "Your previous reply REMOVED one or more existing <img> tags that "
            "the user did not ask to remove. Redo the edit and keep EVERY "
            "existing image exactly as it was — only add what was requested."
        )
        if _looks_like_document(retry_html):
            new_html, diff = retry_html, compute_diff(html, retry_html)
        if diff_is_empty(diff) or not existing_images_preserved(html, new_html, removal_ok):
            return EditResult(
                applied=False,
                changes_claimed=changes,
                unable_to_do=unable,
                failure_message=(
                    "I couldn't apply that without dropping an image already in the "
                    "report, so I left the report unchanged. Tell me explicitly if you "
                    "want an existing image replaced or removed."
                ),
            )

    # Guard 3: scope. An edit far larger than a targeted change is retried once,
    # then allowed through with a warning (some edits really are large).
    warnings: list[str] = []
    if change_scope_ratio(html, new_html) > _SCOPE_WARN_THRESHOLD:
        logger.info(
            "Edit scope %.2f exceeds threshold; retrying with a stricter nudge.",
            change_scope_ratio(html, new_html),
        )
        changes2, unable2, retry_html = _attempt(
            "Your previous reply changed far more of the document than the "
            "request warranted. Apply ONLY the specific change requested and "
            "keep every other part of the HTML byte-for-byte identical."
        )
        if _looks_like_document(retry_html):
            retry_diff = compute_diff(html, retry_html)
            if not retry_diff["empty"] and existing_images_preserved(
                html, retry_html, removal_ok
            ):
                # Prefer the tighter result if it's smaller in scope.
                if retry_diff["scope"] < diff["scope"]:
                    new_html, diff, changes, unable = (
                        retry_html, retry_diff, changes2, unable2,
                    )
        if change_scope_ratio(html, new_html) > _SCOPE_WARN_THRESHOLD:
            warnings.append(
                "This edit changed a large portion of the report — please review it "
                "to make sure only what you intended was altered."
            )

    summary = build_summary_from_diff(html, new_html)
    if unable:
        summary += " I couldn't do the following: " + "; ".join(unable) + "."
    if warnings:
        summary += " " + " ".join(warnings)

    return EditResult(
        applied=True,
        html=new_html,
        markdown=html_to_markdown(new_html),
        summary=summary,
        warnings=warnings,
        unable_to_do=unable,
        changes_claimed=changes,
    )
