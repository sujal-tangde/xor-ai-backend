"""WebSocket endpoint for real-time chat with the AI agent."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.chat_agent import chat_stream
from src.services.chat_history import append_messages, get_messages

router = APIRouter(tags=["chat"])


@router.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    """Real-time chat with Redis-backed history per chat_id.

    Client -> server:
      {"chat_id": "...", "type": "load_history"}
      {"chat_id": "...", "message": "...", "file_ids": ["uuid", ...]}

    Server -> client:
      {"type": "history", "chat_id": "...", "messages": [...]}
      {"type": "stream_start", "chat_id": "..."}
      {"type": "tool_start", "chat_id": "...", "tool": "...", "label": "..."}
      {"type": "tool_query", "chat_id": "...", "tool": "...", "query": "..."}
      {"type": "tool_end", "chat_id": "...", "tool": "...", "label": "..."}
      {"type": "tools_used", "chat_id": "...", "tools": [...]}
      {"type": "stream_delta", "chat_id": "...", "delta": "..."}
      {"type": "stream_reset", "chat_id": "..."}
      {"type": "stream_end", "chat_id": "...", "content": "...", "tools_used": [...]}
      {"role": "error", "content": "..."}
    """
    await websocket.accept()
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

            if data.get("type") == "load_history":
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
                history = await get_messages(chat_id)
                user_message: dict = {"role": "user", "content": str(message)}
                if file_ids:
                    user_message["file_ids"] = file_ids
                agent_messages = [*history, user_message]

                await websocket.send_json(
                    {"type": "stream_start", "chat_id": chat_id}
                )

                reply_parts: list[str] = []
                tools_used: list[dict[str, str]] = []
                async for event in chat_stream(agent_messages):
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
                await append_messages(chat_id, user_message, assistant_message)
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
