"""Agent tools for XOR Chat."""

from src.agent.tools.project_context_tool import TOOL_LABEL as PROJECT_CONTEXT_LABEL
from src.agent.tools.project_context_tool import TOOL_NAME as PROJECT_CONTEXT_NAME
from src.agent.tools.project_context_tool import get_project_context_tool
from src.agent.tools.report_tool import TOOL_LABEL as REPORT_GENERATION_LABEL
from src.agent.tools.report_tool import TOOL_NAME as REPORT_GENERATION_NAME
from src.agent.tools.report_tool import report_generation_tool
from src.agent.tools.report_edit_tool import TOOL_LABEL as REPORT_EDIT_LABEL
from src.agent.tools.report_edit_tool import TOOL_NAME as REPORT_EDIT_NAME
from src.agent.tools.report_edit_tool import report_edit_tool
from src.agent.tools.rag_tools import (
    SEARCH_FILE_CHUNKS_LABEL,
    SEARCH_FILE_CHUNKS_NAME,
    SEARCH_IMAGE_CHUNKS_LABEL,
    SEARCH_IMAGE_CHUNKS_NAME,
    search_file_chunks,
    search_image_chunks,
)
from src.agent.tools.tavily_search_tool import TOOL_LABEL as TAVILY_SEARCH_LABEL
from src.agent.tools.tavily_search_tool import TOOL_NAME as TAVILY_SEARCH_NAME
from src.agent.tools.tavily_search_tool import get_tavily_search_tool
from src.agent.tools.uploads_tool import (
    GET_INSIGHT_LABEL,
    GET_INSIGHT_NAME,
    GET_INSIGHTS_BY_IDS_LABEL,
    GET_INSIGHTS_BY_IDS_NAME,
    LIST_UPLOADS_LABEL,
    LIST_UPLOADS_NAME,
    get_insights_by_file_ids,
    get_upload_insight,
    list_project_uploads,
)

TOOL_LABELS: dict[str, str] = {
    TAVILY_SEARCH_NAME: TAVILY_SEARCH_LABEL,
    PROJECT_CONTEXT_NAME: PROJECT_CONTEXT_LABEL,
    LIST_UPLOADS_NAME: LIST_UPLOADS_LABEL,
    GET_INSIGHT_NAME: GET_INSIGHT_LABEL,
    GET_INSIGHTS_BY_IDS_NAME: GET_INSIGHTS_BY_IDS_LABEL,
    SEARCH_IMAGE_CHUNKS_NAME: SEARCH_IMAGE_CHUNKS_LABEL,
    SEARCH_FILE_CHUNKS_NAME: SEARCH_FILE_CHUNKS_LABEL,
    REPORT_GENERATION_NAME: REPORT_GENERATION_LABEL,
    REPORT_EDIT_NAME: REPORT_EDIT_LABEL,
}


def get_agent_tools() -> list:
    """Return all enabled agent tools."""
    tools = [
        get_project_context_tool,
        list_project_uploads,
        get_upload_insight,
        get_insights_by_file_ids,
        search_image_chunks,
        search_file_chunks,
        report_generation_tool,
        report_edit_tool,
    ]
    tavily = get_tavily_search_tool()
    if tavily:
        tools.append(tavily)
    return tools


def tool_label(tool_name: str) -> str:
    """Human-readable label for a tool name."""
    return TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))


def is_known_tool(tool_name: str | None) -> bool:
    """True for the domain tools we surface in the UI.

    Filters out the deep-agent's internal bookkeeping tools (todo/task/file
    helpers) so the "tools used" display only ever shows the real product tools.
    """
    return bool(tool_name) and tool_name in TOOL_LABELS
