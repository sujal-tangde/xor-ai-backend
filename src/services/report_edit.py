"""Report image embedding for the GENERATION pipeline.

Historically this module also classified free-text edit requests into a closed
set of operations and applied them to the saved ``report_json``. That whole
system was replaced: should-cost reports are free-form HTML and edits now go
through the deterministic HTML edit engine (``src/agent/html_editor.py``), which
modifies the report HTML directly and verifies the change with a computed diff.

All that remains here is :func:`_add_images`, used by ``report_generation`` to
embed images the user attached at generation time into the structured
``report_json`` (the generation path still computes numbers in code and fills the
locked template). Edits never touch this file.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _add_images(
    report_json: dict[str, Any],
    image_refs: list[dict[str, Any]],
    position: str,
    caption: str | None,
) -> int:
    """Append attached images to ``report_json['images']`` (additive, de-duped).

    Returns the number of images actually added. Existing images are never
    removed; a URL already present is skipped so a re-run can't duplicate it.
    """
    if not image_refs:
        return 0
    images = report_json.setdefault("images", [])
    existing_urls = {img.get("url") for img in images if isinstance(img, dict)}
    pos = "after_executive" if str(position).lower() == "after_executive" else "end"
    added = 0
    for ref in image_refs:
        url = ref.get("url")
        if not url or url in existing_urls:
            continue
        images.append({"url": url, "caption": caption or ref.get("name") or "", "position": pos})
        existing_urls.add(url)
        added += 1
    return added
