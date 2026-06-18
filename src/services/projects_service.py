"""Postgres CRUD for projects, conversations, and persisted messages.

All reads/writes are scoped by ``user_id`` so a user can only ever touch their
own data (the Supabase service-key client bypasses RLS, so scoping is enforced
here in code).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.services.file_storage import get_supabase


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
def create_project(
    user_id: str, name: str, description: str | None = None
) -> dict[str, Any]:
    row = {
        "user_id": user_id,
        "name": name or "Untitled project",
        "description": description,
    }
    result = get_supabase().table("projects").insert(row).execute()
    return result.data[0] if result.data else row


def list_projects(user_id: str) -> list[dict[str, Any]]:
    result = (
        get_supabase()
        .table("projects")
        .select("id, name, description, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


def get_project(user_id: str, project_id: str) -> dict[str, Any] | None:
    result = (
        get_supabase()
        .table("projects")
        .select("*")
        .eq("id", project_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


def get_project_context(project_id: str) -> dict[str, Any] | None:
    """Fetch a project's accumulated theory + structured context by ID.

    Used by the agent tool, which receives a project_id already scoped to the
    authenticated user's conversation, so no user filter is applied here.
    """
    result = (
        get_supabase()
        .table("projects")
        .select("id, name, context, structured_context")
        .eq("id", project_id)
        .execute()
    )
    return result.data[0] if result.data else None


def get_project_name(project_id: str) -> str | None:
    """Fetch just a project's display name by ID (no user scoping).

    Used to label tool output with a human-readable name instead of the UUID.
    """
    result = (
        get_supabase()
        .table("projects")
        .select("name")
        .eq("id", project_id)
        .limit(1)
        .execute()
    )
    return (result.data[0].get("name") if result.data else None) or None


def get_project_knowledge_base(project_id: str) -> dict[str, Any] | None:
    """Fetch a project's recomputed whole-product knowledge base row.

    Replaces the old projects.context/structured_context read; the agent's
    get_project_context tool now sources from here.
    """
    result = (
        get_supabase()
        .table("project_knowledge_base")
        .select(
            "project_id, theory_context, structured_context, "
            "insights_total, insights_processed, status, updated_at"
        )
        .eq("project_id", project_id)
        .execute()
    )
    return result.data[0] if result.data else None


def list_uploads(project_id: str) -> list[dict[str, Any]]:
    """List all uploads for a project with their processing status."""
    result = (
        get_supabase()
        .table("uploaded_files")
        .select("id, name, file_type, processing_status, created_at")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def get_insight(project_id: str, file_id: str) -> dict[str, Any] | None:
    """Fetch a single upload's insight (theory + structured)."""
    result = (
        get_supabase()
        .table("project_insights")
        .select("file_id, media_kind, theory_context, structured_context, created_at")
        .eq("project_id", project_id)
        .eq("file_id", file_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def get_insights(project_id: str, file_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch insights for several file_ids in one call."""
    if not file_ids:
        return []
    result = (
        get_supabase()
        .table("project_insights")
        .select("file_id, media_kind, theory_context, structured_context")
        .eq("project_id", project_id)
        .in_("file_id", file_ids)
        .execute()
    )
    return result.data or []


def update_project(
    user_id: str, project_id: str, fields: dict[str, Any]
) -> dict[str, Any] | None:
    allowed = {"name", "description", "context", "structured_context"}
    payload = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not payload:
        return get_project(user_id, project_id)
    payload["updated_at"] = _now()
    result = (
        get_supabase()
        .table("projects")
        .update(payload)
        .eq("id", project_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


def delete_project(user_id: str, project_id: str) -> bool:
    result = (
        get_supabase()
        .table("projects")
        .delete()
        .eq("id", project_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(result.data)


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
def create_conversation(
    user_id: str, project_id: str, title: str | None = None
) -> dict[str, Any] | None:
    if get_project(user_id, project_id) is None:
        return None
    row = {
        "user_id": user_id,
        "project_id": project_id,
        "title": title or "New conversation",
    }
    result = get_supabase().table("conversations").insert(row).execute()
    return result.data[0] if result.data else row


def list_conversations(user_id: str, project_id: str) -> list[dict[str, Any]]:
    result = (
        get_supabase()
        .table("conversations")
        .select("id, project_id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .eq("project_id", project_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


def get_conversation(user_id: str, conversation_id: str) -> dict[str, Any] | None:
    result = (
        get_supabase()
        .table("conversations")
        .select("*")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


def update_conversation_title(
    user_id: str, conversation_id: str, title: str
) -> None:
    get_supabase().table("conversations").update(
        {"title": title[:120], "updated_at": _now()}
    ).eq("id", conversation_id).eq("user_id", user_id).execute()


def touch_conversation(user_id: str, conversation_id: str) -> None:
    get_supabase().table("conversations").update({"updated_at": _now()}).eq(
        "id", conversation_id
    ).eq("user_id", user_id).execute()


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    result = (
        get_supabase()
        .table("conversations")
        .delete()
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(result.data)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
def get_conversation_messages(
    conversation_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    """Return messages in chronological order (most recent ``limit``)."""
    result = (
        get_supabase()
        .table("messages")
        .select("role, content, file_ids, tools_used, seq")
        .eq("conversation_id", conversation_id)
        .order("seq", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    rows.reverse()
    out: list[dict[str, Any]] = []
    for row in rows:
        msg: dict[str, Any] = {"role": row["role"], "content": row.get("content") or ""}
        if row.get("file_ids"):
            msg["file_ids"] = row["file_ids"]
        if row.get("tools_used"):
            msg["tools_used"] = row["tools_used"]
        out.append(msg)
    return out


def save_messages(
    conversation_id: str, user_id: str, messages: list[dict[str, Any]]
) -> None:
    """Insert one or more messages. Safe to call as a fire-and-forget task."""
    if not messages:
        return
    rows = []
    for msg in messages:
        rows.append(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": msg.get("role", "user"),
                "content": msg.get("content") or "",
                "file_ids": msg.get("file_ids"),
                "tools_used": msg.get("tools_used"),
            }
        )
    get_supabase().table("messages").insert(rows).execute()
