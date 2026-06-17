"""Project and conversation management endpoints (all auth-protected)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.core.auth import get_current_user
from src.services import projects_service as ps

router = APIRouter(tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    context: str | None = None
    structured_context: str | None = None


class ConversationCreate(BaseModel):
    title: str | None = None


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
@router.post("/projects")
def create_project(body: ProjectCreate, user=Depends(get_current_user)) -> dict[str, Any]:
    return ps.create_project(user["id"], body.name, body.description)


@router.get("/projects")
def list_projects(user=Depends(get_current_user)) -> list[dict[str, Any]]:
    return ps.list_projects(user["id"])


@router.get("/projects/{project_id}")
def get_project(project_id: str, user=Depends(get_current_user)) -> dict[str, Any]:
    project = ps.get_project(user["id"], project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/projects/{project_id}")
def update_project(
    project_id: str, body: ProjectUpdate, user=Depends(get_current_user)
) -> dict[str, Any]:
    updated = ps.update_project(user["id"], project_id, body.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return updated


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, user=Depends(get_current_user)) -> dict[str, Any]:
    if not ps.delete_project(user["id"], project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True, "id": project_id}


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
@router.post("/projects/{project_id}/conversations")
def create_conversation(
    project_id: str, body: ConversationCreate, user=Depends(get_current_user)
) -> dict[str, Any]:
    conv = ps.create_conversation(user["id"], project_id, body.title)
    if conv is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return conv


@router.get("/projects/{project_id}/conversations")
def list_conversations(
    project_id: str, user=Depends(get_current_user)
) -> list[dict[str, Any]]:
    if ps.get_project(user["id"], project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return ps.list_conversations(user["id"], project_id)


@router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, user=Depends(get_current_user)) -> list[dict[str, Any]]:
    if ps.get_conversation(user["id"], conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ps.get_conversation_messages(conversation_id)


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str, user=Depends(get_current_user)
) -> dict[str, Any]:
    if not ps.delete_conversation(user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True, "id": conversation_id}
