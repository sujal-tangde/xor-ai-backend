"""Shared input validation for agent tools.

Tool arguments are produced by the LLM and are frequently wrong — most commonly
a file *name* (e.g. ``all components.HEIC``) handed to a tool that expects a UUID.
Passing those straight into UUID/uuid-typed Postgres columns raises ``22P02``
(invalid input syntax for type uuid), which previously bubbled all the way up and
crashed the whole chat turn. These helpers let tools reject bad input gracefully
and tell the model how to recover instead.
"""

from __future__ import annotations

import uuid


def is_uuid(value: object) -> bool:
    """True if ``value`` is (parseable as) a UUID."""
    if not value:
        return False
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def invalid_project_id_message(project_id: object) -> str:
    return (
        f"'{project_id}' is not a valid project ID. The project UUID is provided "
        "in the message context — use that exact value."
    )


def invalid_file_id_message(file_id: object) -> str:
    return (
        f"'{file_id}' is not a valid file ID — it looks like a file name, not a "
        "UUID. Call list_project_uploads first to resolve the name to its file_id, "
        "then retry with that UUID."
    )
