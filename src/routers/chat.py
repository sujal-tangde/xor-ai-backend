"""WebSocket endpoint for real-time chat with the AI agent (auth-protected)."""

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.chat_agent import chat_stream, resume_stream
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


def _slim_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lightweight report metadata to persist on a message (markdown lives in
    the ``reports`` table and is refetched by id on reload)."""
    if not report:
        return None
    return {
        "report_id": report.get("report_id"),
        "title": report.get("title"),
        "volume": report.get("volume"),
    }


def _new_acc() -> dict[str, Any]:
    """Mutable accumulator so a turn cancelled mid-stream keeps its partial text."""
    return {
        "reply_parts": [],
        "tools_used": [],
        "report": None,
        "questions": None,
        "thread_id": None,
    }


def _acc_result(acc: dict[str, Any]) -> dict[str, Any]:
    return {
        "reply": "".join(acc["reply_parts"]),
        "tools_used": acc["tools_used"],
        "report": acc["report"],
        "questions": acc["questions"],
        "thread_id": acc["thread_id"],
    }


async def _consume_stream(
    websocket: WebSocket, chat_id: str, events, acc: dict[str, Any]
) -> None:
    """Forward agent stream events to the socket, accumulating into ``acc``.

    ``acc`` is updated incrementally (rather than returned at the end) so that a
    turn cancelled mid-stream — e.g. when the user hits pause — still has the
    partial ``reply`` text and any ``tools_used`` / ``report`` collected so far.
    """
    async for event in events:
        event_type = event.get("type")
        if event_type == "reset":
            acc["reply_parts"].clear()
            await websocket.send_json({"type": "stream_reset", "chat_id": chat_id})
        elif event_type == "delta":
            delta = event.get("text", "")
            acc["reply_parts"].append(delta)
            await websocket.send_json(
                {"type": "stream_delta", "chat_id": chat_id, "delta": delta}
            )
        elif event_type == "tools_used":
            acc["tools_used"] = event.get("tools", [])
            await websocket.send_json(
                {"type": "tools_used", "chat_id": chat_id, "tools": acc["tools_used"]}
            )
        elif event_type == "report_ready":
            acc["report"] = {k: v for k, v in event.items() if k != "type"}
            await websocket.send_json(
                {"type": "report_ready", "chat_id": chat_id, **acc["report"]}
            )
        elif event_type == "questions":
            acc["questions"] = event.get("questions", [])
            acc["thread_id"] = event.get("thread_id")
            await websocket.send_json(
                {
                    "type": "report_questions",
                    "chat_id": chat_id,
                    "questions": acc["questions"],
                    "thread_id": acc["thread_id"],
                }
            )
        elif event_type in {"tool_start", "tool_query", "tool_end", "report_progress"}:
            payload = {k: v for k, v in event.items() if k != "type"}
            await websocket.send_json(
                {"type": event_type, "chat_id": chat_id, **payload}
            )
        await asyncio.sleep(0)


async def _run_stream(
    websocket: WebSocket, chat_id: str, events
) -> tuple[dict[str, Any], bool]:
    """Drive a stream to completion, tolerating cancellation (pause).

    Returns ``(result, stopped)`` where ``stopped`` is True when the turn was
    cancelled mid-stream. The underlying agent generator is always closed so the
    LLM call and any tool work are torn down promptly.
    """
    acc = _new_acc()
    stopped = False
    try:
        await _consume_stream(websocket, chat_id, events, acc)
    except asyncio.CancelledError:
        stopped = True
    finally:
        try:
            await events.aclose()
        except Exception:
            # Generator already exhausted or failed to unwind cleanly.
            pass
    return _acc_result(acc), stopped


@router.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    """Real-time chat with Redis-backed history per conversation.

    The socket authenticates via a ``token`` query param (Supabase access token).
    The ``chat_id`` field on each message is the conversation UUID and must belong
    to the authenticated user.

    Each streaming turn runs as a cancellable task so the receive loop stays free
    to accept a ``{"type": "stop"}`` message, which pauses the in-flight response.
    """
    user = user_from_token(websocket.query_params.get("token"))
    if user is None:
        await websocket.close(code=4401)
        return

    user_id = user["id"]
    await websocket.accept()

    # Register this socket so worker-published realtime events (insight counts)
    # can be pushed to it; unregistered in the finally below.
    from src.services.realtime_bridge import register, unregister

    register(user_id, websocket)

    # Per-connection cache of conversations already verified as owned by this user.
    verified: dict[str, dict[str, Any]] = {}
    # Turns paused for HILT report questions, keyed by chat_id, awaiting answers.
    pending: dict[str, dict[str, Any]] = {}
    # In-flight streaming turns, keyed by chat_id, so a `stop` message can cancel.
    streaming: dict[str, asyncio.Task] = {}

    async def _resolve(conversation_id: str) -> dict[str, Any] | None:
        if conversation_id in verified:
            return verified[conversation_id]
        conv = await asyncio.to_thread(ps.get_conversation, user_id, conversation_id)
        if conv is not None:
            verified[conversation_id] = conv
        return conv

    async def _handle_chat_turn(chat_id: str, data: dict) -> None:
        message = data.get("message")
        if not message:
            await websocket.send_json({"role": "error", "content": "Empty message."})
            return

        raw_file_ids = data.get("file_ids")
        file_ids: list[str] | None = None
        if raw_file_ids is not None:
            if not isinstance(raw_file_ids, list):
                await websocket.send_json(
                    {"role": "error", "content": "file_ids must be an array."}
                )
                return
            file_ids = [str(file_id) for file_id in raw_file_ids if file_id]

        try:
            raw_project_id = data.get("project_id")
            fast_new_chat = bool(raw_project_id)

            if not fast_new_chat:
                await _hydrate_if_needed(chat_id)

            history = await get_messages(chat_id)
            was_empty = len(history) == 0

            if fast_new_chat and was_empty:
                project_id = str(raw_project_id)
                verified[chat_id] = {"project_id": project_id}
            else:
                conv = await _resolve(chat_id)
                if conv is None:
                    await websocket.send_json(
                        {"role": "error", "content": "Conversation not found."}
                    )
                    return
                project_id = conv.get("project_id")

            user_message: dict = {"role": "user", "content": str(message)}
            if file_ids:
                user_message["file_ids"] = file_ids
            agent_messages = [*history, user_message]

            await websocket.send_json({"type": "stream_start", "chat_id": chat_id})

            result, stopped = await _run_stream(
                websocket,
                chat_id,
                chat_stream(
                    agent_messages,
                    project_id,
                    user_id=user_id,
                    conversation_id=chat_id,
                ),
            )

            # HILT pause: the report tool asked the user questions. Hold the
            # turn open and wait for a `report_answers` message to resume.
            if result["questions"] and not stopped:
                pending[chat_id] = {
                    "user_message": user_message,
                    "was_empty": was_empty,
                    "project_id": project_id,
                    "thread_id": result["thread_id"],
                }
                return

            reply = result["reply"]
            assistant_message: dict = {"role": "assistant", "content": reply}
            if result["tools_used"]:
                assistant_message["tools_used"] = result["tools_used"]
            slim = _slim_report(result["report"])
            if slim:
                assistant_message["report"] = slim

            # A turn paused before producing anything keeps only the user message
            # (no empty assistant bubble); otherwise persist the partial reply so
            # it survives a reload. Live cache first (awaited, fast), then the
            # durable DB write (fire-and-forget).
            has_reply = bool(reply.strip()) or bool(result["tools_used"]) or bool(slim)
            to_persist = [user_message, assistant_message] if has_reply else [user_message]
            await append_messages(chat_id, *to_persist)
            new_title = str(message)[:60] if was_empty else None
            _persist(chat_id, user_id, to_persist, new_title)

            await websocket.send_json(
                {
                    "type": "stream_end",
                    "chat_id": chat_id,
                    "content": reply,
                    "tools_used": result["tools_used"],
                    "report": result["report"],
                    "stopped": stopped,
                }
            )
        except Exception as exc:
            await websocket.send_json({"role": "error", "content": str(exc)})

    async def _handle_report_answers(chat_id: str, data: dict) -> None:
        pend = pending.pop(chat_id, None)
        if pend is None:
            await websocket.send_json(
                {"role": "error", "content": "No pending questions to answer."}
            )
            return
        answers = data.get("answers") or {}
        await websocket.send_json({"type": "stream_start", "chat_id": chat_id})
        try:
            result, stopped = await _run_stream(
                websocket,
                chat_id,
                resume_stream(
                    pend["thread_id"],
                    answers,
                    pend["project_id"],
                    user_id=user_id,
                    conversation_id=chat_id,
                ),
            )
        except Exception as exc:
            await websocket.send_json({"role": "error", "content": str(exc)})
            return

        # A second question round is possible though rare — re-pause.
        if result["questions"] and not stopped:
            pending[chat_id] = {
                "user_message": pend["user_message"],
                "was_empty": pend["was_empty"],
                "project_id": pend["project_id"],
                "thread_id": result["thread_id"],
            }
            return

        reply = result["reply"]
        assistant_message = {"role": "assistant", "content": reply}
        if result["tools_used"]:
            assistant_message["tools_used"] = result["tools_used"]
        slim = _slim_report(result["report"])
        if slim:
            assistant_message["report"] = slim

        await append_messages(chat_id, pend["user_message"], assistant_message)
        new_title = (
            str(pend["user_message"].get("content", ""))[:60]
            if pend["was_empty"]
            else None
        )
        _persist(
            chat_id, user_id, [pend["user_message"], assistant_message], new_title
        )
        await websocket.send_json(
            {
                "type": "stream_end",
                "chat_id": chat_id,
                "content": reply,
                "tools_used": result["tools_used"],
                "report": result["report"],
                "stopped": stopped,
            }
        )

    def _spawn_turn(chat_id: str, coro) -> None:
        """Run a streaming turn as a task tracked for cancellation."""
        existing = streaming.get(chat_id)
        if existing and not existing.done():
            # Frontend disables sending while streaming; ignore the duplicate.
            coro.close()
            return

        task = asyncio.create_task(coro)
        streaming[chat_id] = task
        task.add_done_callback(
            lambda t, cid=chat_id: streaming.pop(cid, None)
            if streaming.get(cid) is t
            else None
        )

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
            msg_type = data.get("type")

            # Pause the in-flight response for this conversation.
            if msg_type == "stop":
                task = streaming.get(chat_id)
                if task and not task.done():
                    task.cancel()
                continue

            if msg_type == "load_history":
                await _hydrate_if_needed(chat_id)
                history = await get_messages(chat_id)
                await websocket.send_json(
                    {"type": "history", "chat_id": chat_id, "messages": history}
                )
                continue

            # Resume a HILT-paused report turn with the user's answers.
            if msg_type == "report_answers":
                _spawn_turn(chat_id, _handle_report_answers(chat_id, data))
                continue

            # A normal chat turn.
            _spawn_turn(chat_id, _handle_chat_turn(chat_id, data))
    except WebSocketDisconnect:
        for task in streaming.values():
            if not task.done():
                task.cancel()
    finally:
        unregister(user_id, websocket)
