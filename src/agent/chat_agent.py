"""Chat agent for XOR Chat: Bedrock LLM + Tavily search, streamed token-by-token."""

import json
import re
from collections.abc import AsyncIterator
from datetime import date
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_litellm import ChatLiteLLM
from langchain_tavily import TavilySearch
from deepagents import create_deep_agent

from src.agent.tools.image_analysis_tool import get_image_analysis
from src.core.config import (
    AWS_REGION,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TOOLS_ENABLED,
    TAVILY_API_KEY,
)

StreamEventType = Literal["delta", "reset", "tool_start", "tool_end", "tool_query", "tools_used"]


def _system_prompt() -> str:
    today = date.today().strftime("%B %d, %Y")
    return (
        f"Today's date is {today}. "
        "You are XOR, a helpful and friendly AI assistant for the XOR Chat app. "
        "Answer the user's questions clearly and concisely. "
        "You MUST use the web search tool for any question about current events, "
        "recent news, live data, prices, net worth, rankings, weather, or anything "
        "time-sensitive. Never answer from memory alone when fresh data may exist — "
        "search first, then answer using the search results. "
        "When search results are available, prefer the most recent figures, cite "
        "your sources, and mention when the data was published if known. "
        "When the user attaches image file IDs, you MUST call get_image_analysis "
        "with those IDs before answering questions about the hardware, PCB, "
        "components, enclosure, or teardown shown in those images. Combine image "
        "analysis with web search when the user asks for both hardware details and "
        "current external information (e.g. part datasheets, market prices). "
        "If you don't know something after searching, say so instead of guessing."
    )


def _build_model() -> ChatLiteLLM:
    kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY or None,
        "streaming": True,
        "model_kwargs": {"aws_region_name": AWS_REGION},
    }
    if LLM_API_BASE:
        kwargs["api_base"] = LLM_API_BASE
    return ChatLiteLLM(**kwargs)


@lru_cache(maxsize=1)
def get_agent():
    """Create (once) and return the compiled deep agent."""
    model = _build_model()
    tools = [get_image_analysis]
    if LLM_TOOLS_ENABLED and TAVILY_API_KEY:
        tools.append(
            TavilySearch(
                max_results=5,
                tavily_api_key=TAVILY_API_KEY,
                include_answer=True,
                auto_parameters=True,
            )
        )
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=_system_prompt(),
    )


def _file_ids_hint(file_ids: list[str]) -> str:
    ids_text = ", ".join(file_ids)
    return (
        f"\n\n[Attached image file IDs: {ids_text}. "
        "Use get_image_analysis with these IDs to load stored hardware analysis "
        "before answering questions about these images.]"
    )


def _to_lc_messages(messages: list[dict[str, Any]]) -> list[HumanMessage | AIMessage]:
    lc_messages: list[HumanMessage | AIMessage] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "user":
            file_ids = message.get("file_ids")
            if file_ids:
                content += _file_ids_hint([str(file_id) for file_id in file_ids])
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str)
            else str(block.get("text", "")) if isinstance(block, dict) and block.get("type") == "text"
            else ""
            for block in content
        ]
        return "".join(parts)
    return str(content) if content else ""


def _try_parse_tool_input(args_str: str) -> str:
    if not args_str:
        return ""
    try:
        data = json.loads(args_str)
        if isinstance(data.get("query"), str):
            return data["query"]
        file_ids = data.get("file_ids")
        if isinstance(file_ids, list) and file_ids:
            preview = ", ".join(str(file_id) for file_id in file_ids[:3])
            suffix = f" (+{len(file_ids) - 3} more)" if len(file_ids) > 3 else ""
            return f"{len(file_ids)} image(s): {preview}{suffix}"
    except json.JSONDecodeError:
        match = re.search(r'"query"\s*:\s*"([^"]*)', args_str)
        if match:
            return match.group(1)
        match = re.search(r'"file_ids"\s*:\s*\[([^\]]*)\]', args_str)
        if match:
            return f"images: {match.group(1).strip()}"
    return ""


def _tool_label(tool_name: str) -> str:
    if tool_name == "tavily_search":
        return "web search"
    if tool_name == "get_image_analysis":
        return "image analysis"
    return tool_name.replace("_", " ")


async def chat_stream(
    messages: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Stream assistant events for a conversation history.

    Yields dicts with a ``type`` key:
      - ``delta``: ``{"type": "delta", "text": "..."}``
      - ``reset``: ``{"type": "reset"}`` — new assistant message after a tool call
      - ``tool_start``: ``{"type": "tool_start", "tool": "...", "label": "..."}``
      - ``tool_query``: ``{"type": "tool_query", "tool": "...", "query": "..."}``
      - ``tool_end``: ``{"type": "tool_end", "tool": "...", "label": "..."}``
      - ``tools_used``: ``{"type": "tools_used", "tools": [...]}`` — final summary
    """
    agent = get_agent()
    lc_messages = _to_lc_messages(messages)
    current_id: str | None = None
    announced_tools: set[str] = set()
    pending_args: dict[str, str] = {}
    last_query_by_tool: dict[str, str] = {}
    used_tools: list[dict[str, str]] = []

    async for chunk, _metadata in agent.astream(
        {"messages": lc_messages},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            if chunk.tool_call_chunks:
                for tool_chunk in chunk.tool_call_chunks:
                    tool_name = tool_chunk.get("name")
                    chunk_id = tool_chunk.get("id") or "default"
                    args_piece = tool_chunk.get("args") or ""

                    if tool_name and tool_name not in announced_tools:
                        announced_tools.add(tool_name)
                        yield {
                            "type": "tool_start",
                            "tool": tool_name,
                            "label": _tool_label(tool_name),
                        }

                    if args_piece:
                        pending_args[chunk_id] = pending_args.get(chunk_id, "") + args_piece
                        query = _try_parse_tool_input(pending_args[chunk_id])
                        active_tool = tool_name or next(
                            (name for name in announced_tools if name),
                            "tavily_search",
                        )
                        if query and query != last_query_by_tool.get(active_tool):
                            last_query_by_tool[active_tool] = query
                            yield {
                                "type": "tool_query",
                                "tool": active_tool,
                                "query": query,
                            }

            if chunk.id and chunk.id != current_id:
                if current_id is not None:
                    yield {"type": "reset"}
                current_id = chunk.id

            delta = _text(chunk.content)
            if delta:
                yield {"type": "delta", "text": delta}
            continue

        if isinstance(chunk, ToolMessage):
            tool_name = chunk.name or "unknown"
            used_tools.append(
                {
                    "tool": tool_name,
                    "label": _tool_label(tool_name),
                    "query": last_query_by_tool.get(tool_name, ""),
                }
            )
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "label": _tool_label(tool_name),
            }
            pending_args.clear()

    if used_tools:
        yield {"type": "tools_used", "tools": used_tools}
