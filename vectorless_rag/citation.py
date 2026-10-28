"""A ``citation`` on every page ``get_page_content`` returns: the file and its sheet
for a spreadsheet, else the file and its page number.

Wraps PageIndex's ``get_page_content`` tool implementation, which every surface
(``client.chat``, MCP, ``langgraph_tools``) dispatches through ``call_tool``.
ponytail: the citations are added after the tool's size budget, so a full response
runs a few dozen characters per page over it.
"""
import functools

from .multimodal_api import SHEET_PREFIX


def page_citation(file_name: str, page: int, text: str) -> dict:
    if text.startswith(SHEET_PREFIX):
        return {"file": file_name, "sheet": text[len(SHEET_PREFIX):].split("\n", 1)[0]}
    return {"file": file_name, "page": page}


def install() -> None:
    """Add citations to ``get_page_content``; idempotent."""
    from pageindex import agent_tools
    implementation = agent_tools._IMPLEMENTATIONS["get_page_content"]
    if getattr(implementation, "multimodal", False):
        return

    # wraps: call_tool binds the arguments against the original's signature
    @functools.wraps(implementation)
    def get_page_content(*args, **kwargs):
        payload, is_error = implementation(*args, **kwargs)
        if not is_error:
            for entry in payload.get("content") or []:
                entry["citation"] = page_citation(payload["doc_name"], entry["page"], entry["text"])
        return payload, is_error

    get_page_content.multimodal = True
    agent_tools._IMPLEMENTATIONS["get_page_content"] = get_page_content
