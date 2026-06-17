"""Agent tools for XOR Chat."""

from src.agent.tools.image_analysis_tool import TOOL_LABEL as IMAGE_ANALYSIS_LABEL
from src.agent.tools.image_analysis_tool import TOOL_NAME as IMAGE_ANALYSIS_NAME
from src.agent.tools.image_analysis_tool import get_image_analysis
from src.agent.tools.tavily_search_tool import TOOL_LABEL as TAVILY_SEARCH_LABEL
from src.agent.tools.tavily_search_tool import TOOL_NAME as TAVILY_SEARCH_NAME
from src.agent.tools.tavily_search_tool import get_tavily_search_tool

TOOL_LABELS: dict[str, str] = {
    TAVILY_SEARCH_NAME: TAVILY_SEARCH_LABEL,
    IMAGE_ANALYSIS_NAME: IMAGE_ANALYSIS_LABEL,
}


def get_agent_tools() -> list:
    """Return all enabled agent tools."""
    tools = [get_image_analysis]
    tavily = get_tavily_search_tool()
    if tavily:
        tools.append(tavily)
    return tools


def tool_label(tool_name: str) -> str:
    """Human-readable label for a tool name."""
    return TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))
