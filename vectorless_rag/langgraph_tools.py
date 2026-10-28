"""PageIndex's read tools as LangGraph tools, for agents such as deepagents.

Each tool's description and argument schema are read from PageIndex at runtime
(``agent_tools._local_description`` / ``_local_schema``), so the model sees what
``client.chat``'s own agent sees, upstream changes included. Every call runs through
``agent_tools.call_tool``: validation, the JSON envelope, ``next_steps``, the
``doc_ids`` allowlist and the page ``citation`` are the SDK's own retrieval logic.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pageindex.agent_tools import _local_description, _local_schema, call_tool, tool_names


def pageindex_tools(client, doc_ids=None) -> list[BaseTool]:
    """The read tools bound to one client, scoped to ``doc_ids`` when given."""

    def make(name: str) -> BaseTool:
        def run(**arguments) -> str:
            envelope, _ = call_tool(client, name, arguments, doc_ids=doc_ids)
            return envelope

        return StructuredTool.from_function(run, name=name, description=_local_description(name),
                                            args_schema=_local_schema(name))

    return [make(name) for name in tool_names()]
