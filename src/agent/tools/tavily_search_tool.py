"""Tavily web search tool for the chat agent."""

from langchain_tavily import TavilySearch

from src.core.config import LLM_TOOLS_ENABLED, TAVILY_API_KEY

TOOL_NAME = "tavily_search"
TOOL_LABEL = "web search"


def get_tavily_search_tool() -> TavilySearch | None:
    """Return a configured Tavily search tool, or None if disabled/unconfigured."""
    if not LLM_TOOLS_ENABLED or not TAVILY_API_KEY:
        return None
    return TavilySearch(
        max_results=5,
        tavily_api_key=TAVILY_API_KEY,
        include_answer=True,
        auto_parameters=True,
    )
