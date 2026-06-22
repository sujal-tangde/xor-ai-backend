"""Per-failure debug logging for report generation.

Whenever a report stage can't return live data (a Mouser/PCBWay/Frankfurter call
fails or returns nothing usable, an LLM estimator falls back, etc.), we drop a
small JSON file — named by timestamp — into a dedicated folder so the failure can
be debugged later without trawling application logs. The report itself still
generates from the stage's fallback; this is purely a debugging breadcrumb.

Best-effort by design: writing a breadcrumb must NEVER raise into the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import REPORT_FAILURE_LOG_DIR, REPORT_FAILURE_LOG_ENABLED

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def record_failure(
    stage: str,
    name: str,
    reason: str,
    *,
    error: BaseException | None = None,
    context: dict[str, Any] | None = None,
) -> str | None:
    """Write one timestamped JSON breadcrumb for a failed stage. Returns its path.

    Args:
        stage: pipeline stage key (e.g. "pricing", "fab_quote", "fx", "market_context").
        name: short subject of the failure (e.g. the MPN, "PCBWay quote", project id).
        reason: brief plain-language reason ("Mouser API request failed", "no price break").
        error: the caught exception, if any (its type + message are recorded, briefly).
        context: small dict of extra debugging fields (qty, volume, project_id, …).
    """
    if not REPORT_FAILURE_LOG_ENABLED:
        return None
    try:
        ts = datetime.now(timezone.utc)
        folder = Path(REPORT_FAILURE_LOG_DIR)
        with _lock:
            folder.mkdir(parents=True, exist_ok=True)
        safe_stage = _SAFE.sub("-", str(stage or "stage")).strip("-") or "stage"
        fname = f"{ts.strftime('%Y%m%dT%H%M%S_%f')}_{safe_stage}_{uuid.uuid4().hex[:6]}.json"

        payload: dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "stage": stage,
            "name": name,
            "reason": reason,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error)[:500] if error is not None else None,
            "context": _slim(context or {}),
        }
        path = folder / fname
        with _lock:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        logger.info("Recorded report failure breadcrumb: %s (%s)", fname, reason)
        return str(path)
    except Exception:  # pragma: no cover - logging must never crash the pipeline
        logger.warning("Could not write report failure breadcrumb", exc_info=True)
        return None


def _slim(context: dict[str, Any]) -> dict[str, Any]:
    """Keep the breadcrumb brief: truncate long strings, cap list/dict sizes."""
    out: dict[str, Any] = {}
    for key, value in list(context.items())[:25]:
        if isinstance(value, str):
            out[key] = value[:300]
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)[:300]
    return out
