"""WebSocket endpoint for real-time chat with the AI agent (auth-protected)."""

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.chat_agent import chat_stream
from src.core.auth import user_from_token
from src.services import projects_service as ps
from src.services.chat_history import (
    append_messages,
    count_messages,
    get_messages,
    seed_messages,
)

router = APIRouter(tags=["chat"])


async def _hydrate_if_needed(conversation_id: str) -> None:
    """Populate Redis from the DB the first time a conversation is opened."""
    if await count_messages(conversation_id) > 0:
        return
    db_messages = await asyncio.to_thread(ps.get_conversation_messages, conversation_id)
    await seed_messages(conversation_id, db_messages)


def _persist(conversation_id: str, user_id: str, messages: list[dict], title: str | None) -> None:
    """Fire-and-forget DB write so the user never waits on persistence."""

    async def _run() -> None:
        try:
            await asyncio.to_thread(ps.save_messages, conversation_id, user_id, messages)
            if title is not None:
                await asyncio.to_thread(
                    ps.update_conversation_title, user_id, conversation_id, title
                )
            else:
                await asyncio.to_thread(ps.touch_conversation, user_id, conversation_id)
        except Exception:
            # Persistence is best-effort; Redis already holds the live history.
            pass

    asyncio.create_task(_run())


@router.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    """Real-time chat with Redis-backed history per conversation.

    The socket authenticates via a ``token`` query param (Supabase access token).
    The ``chat_id`` field on each message is the conversation UUID and must belong
    to the authenticated user.
    """
    user = user_from_token(websocket.query_params.get("token"))
    if user is None:
        await websocket.close(code=4401)
        return

    user_id = user["id"]
    await websocket.accept()

    # Per-connection cache of conversations already verified as owned by this user.
    verified: dict[str, dict[str, Any]] = {}

    async def _resolve(conversation_id: str) -> dict[str, Any] | None:
        if conversation_id in verified:
            return verified[conversation_id]
        conv = await asyncio.to_thread(ps.get_conversation, user_id, conversation_id)
        if conv is not None:
            verified[conversation_id] = conv
        return conv

    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                await websocket.send_json(
                    {"role": "error", "content": "Expected a JSON object."}
                )
                continue

            chat_id = data.get("chat_id")
            if not chat_id:
                await websocket.send_json(
                    {"role": "error", "content": "Missing chat_id."}
                )
                continue

            chat_id = str(chat_id)

            conv = await _resolve(chat_id)
            if conv is None:
                await websocket.send_json(
                    {"role": "error", "content": "Conversation not found."}
                )
                continue

            if data.get("type") == "load_history":
                await _hydrate_if_needed(chat_id)
                history = await get_messages(chat_id)
                await websocket.send_json(
                    {
                        "type": "history",
                        "chat_id": chat_id,
                        "messages": history,
                    }
                )
                continue

            message = data.get("message")
            if not message:
                await websocket.send_json(
                    {"role": "error", "content": "Empty message."}
                )
                continue

            raw_file_ids = data.get("file_ids")
            file_ids: list[str] | None = None
            if raw_file_ids is not None:
                if not isinstance(raw_file_ids, list):
                    await websocket.send_json(
                        {"role": "error", "content": "file_ids must be an array."}
                    )
                    continue
                file_ids = [str(file_id) for file_id in raw_file_ids if file_id]

            try:
                await _hydrate_if_needed(chat_id)
                history = await get_messages(chat_id)
                was_empty = len(history) == 0
                user_message: dict = {"role": "user", "content": str(message)}
                if file_ids:
                    user_message["file_ids"] = file_ids
                agent_messages = [*history, user_message]

                await websocket.send_json(
                    {"type": "stream_start", "chat_id": chat_id}
                )

                reply_parts: list[str] = []
                tools_used: list[dict[str, str]] = []
                project_id = conv.get("project_id")
                async for event in chat_stream(agent_messages, project_id):
                    event_type = event.get("type")
                    if event_type == "reset":
                        reply_parts = []
                        await websocket.send_json(
                            {"type": "stream_reset", "chat_id": chat_id}
                        )
                        await asyncio.sleep(0)
                        continue
                    if event_type == "delta":
                        delta = event.get("text", "")
                        reply_parts.append(delta)
                        await websocket.send_json(
                            {
                                "type": "stream_delta",
                                "chat_id": chat_id,
                                "delta": delta,
                            }
                        )
                        await asyncio.sleep(0)
                        continue
                    if event_type == "tools_used":
                        tools_used = event.get("tools", [])
                        await websocket.send_json(
                            {
                                "type": "tools_used",
                                "chat_id": chat_id,
                                "tools": tools_used,
                            }
                        )
                        await asyncio.sleep(0)
                        continue
                    if event_type in {"tool_start", "tool_query", "tool_end"}:
                        payload = {k: v for k, v in event.items() if k != "type"}
                        await websocket.send_json(
                            {"type": event_type, "chat_id": chat_id, **payload}
                        )
                        await asyncio.sleep(0)

                reply = "".join(reply_parts)
                assistant_message: dict = {"role": "assistant", "content": reply}
                if tools_used:
                    assistant_message["tools_used"] = tools_used

                # Live cache (awaited, fast) then durable DB write (fire-and-forget).
                await append_messages(chat_id, user_message, assistant_message)
                new_title = str(message)[:60] if was_empty else None
                _persist(chat_id, user_id, [user_message, assistant_message], new_title)

                await websocket.send_json(
                    {
                        "type": "stream_end",
                        "chat_id": chat_id,
                        "content": reply,
                        "tools_used": tools_used,
                    }
                )
            except Exception as exc:
                await websocket.send_json({"role": "error", "content": str(exc)})
    except WebSocketDisconnect:
        return
