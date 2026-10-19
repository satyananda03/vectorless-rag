"""PageIndex's read tools as LangGraph tools, for agents such as deepagents.

Each docstring is the local tool description verbatim (``agent_tools._local_description``
and ``_local_schema``, rendered by ``_tool_docstring``), so the model sees what
``client.chat``'s own agent sees. Every call runs through ``agent_tools.call_tool``:
validation, the JSON envelope, ``next_steps``, the ``doc_ids`` allowlist and the page
``citation`` are the SDK's own retrieval logic, untouched.
"""
from __future__ import annotations
from langchain_core.tools import BaseTool, tool
from .agent_tools import call_tool

def pageindex_tools(client, doc_ids=None) -> list[BaseTool]:
    """The four read tools bound to one client, scoped to ``doc_ids`` when given."""

    def run(name: str, **arguments) -> str:
        envelope, _ = call_tool(client, name, arguments, doc_ids=doc_ids)
        return envelope

    @tool(parse_docstring=True)
    def browse_documents(offset: int = 0, limit: int = 10) -> str:
        """Primary document retrieval tool — first choice for any document-related question. Lists your documents newest first with names and descriptions; match them against the user's intent and page through with `offset: next_offset` (limit up to 50) while `has_more` is true. Folder browsing and semantic ranking (sort="relevance") are not supported in local mode yet — they work on PageIndex cloud.

        Args:
            offset: Zero-based pagination offset. Pass the value of `next_offset` from the previous response to fetch the next page.
            limit: Number of documents to return per page (1-50, default 10)
        """
        return run("browse_documents", offset=offset, limit=limit)

    @tool(parse_docstring=True)
    def get_document(doc_name: str, wait_for_completion: bool = False) -> str:
        """Check a document's processing status and metadata. `status` is one of "pending", "queued", "processing", "completed", or "failed" — call this before `get_document_structure()` or `get_page_content()` to confirm the document is ready.

        Args:
            doc_name: Copy the `name` field verbatim from a browse_documents() response (case-sensitive, include extension). Example: "Q3 Report.pdf". Document names are unique in a local library.
            wait_for_completion: If true and document is processing, automatically wait up to 3 minutes until completed. Reduces repeated tool calls.
        """
        return run("get_document", doc_name=doc_name, wait_for_completion=wait_for_completion)

    @tool(parse_docstring=True)
    def get_document_structure(doc_name: str, part: int = 1,
                               wait_for_completion: bool = False) -> str:
        """Extract a document's hierarchical outline (headers, sections, page references). REQUIRED for documents over 20 pages — call this first to locate relevant sections, then pass their page numbers to `get_page_content()`. Use the `part` parameter to iterate large outlines until `pagination.has_more` is false.

        Args:
            doc_name: Copy the `name` field verbatim from a browse_documents() response (case-sensitive, include extension). Example: "Q3 Report.pdf". Document names are unique in a local library.
            part: Part number for pagination (1-based, default 1). For large outlines, increment until the response's `pagination.has_more` becomes false.
            wait_for_completion: If true and document is processing, automatically wait up to 3 minutes until completed. Reduces repeated tool calls.
        """
        return run("get_document_structure", doc_name=doc_name, part=part,
                   wait_for_completion=wait_for_completion)

    @tool(parse_docstring=True)
    def get_page_content(doc_name: str, pages: str, wait_for_completion: bool = False) -> str:
        """Extract page content from a processed document. Use tight, targeted page ranges — never the whole document at once. For documents over 20 pages, call `get_document_structure()` first to pick relevant sections.

        Args:
            doc_name: Copy the `name` field verbatim from a browse_documents() response (case-sensitive, include extension). Example: "Q3 Report.pdf". Document names are unique in a local library.
            pages: Page specification: "5", "3,7,10", "5-10", or "1-3,7,9-12"
            wait_for_completion: If true and document is processing, automatically wait up to 3 minutes until completed. Reduces repeated tool calls.
        """
        return run("get_page_content", doc_name=doc_name, pages=pages,
                   wait_for_completion=wait_for_completion)

    return [browse_documents, get_document, get_document_structure, get_page_content]
