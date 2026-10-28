"""Offline checks for the agent's PageIndex tools. Run: uv run pytest app/agents/test_agent.py"""
import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from vectorless_rag import MultimodalClient
from pageindex.agent_tools import _local_description, _local_schema, call_tool
from vectorless_rag import pageindex_tools


def _library(tmp_path):
    """A local library holding one workbook, stored without any model call."""
    client = MultimodalClient(storage_path=str(tmp_path))
    meta = {"id": "pi-test", "name": "rekap.xlsx", "description": "Rekap layanan", "status": "completed",
            "createdAt": "2026-10-18T00:00:00.000Z", "pageNum": 2, "folderId": None,
            "metadata": None, "mode": "flash"}
    tree = [{"title": "Daftar", "node_id": "0000", "start_index": 1, "end_index": 1, "summary": "daftar"},
            {"title": "Rekap", "node_id": "0001", "start_index": 2, "end_index": 2, "summary": "rekap"}]
    pages = [{"page_index": 1, "markdown": "Sheet: Daftar\n\n| No | Layanan |"},
             {"page_index": 2, "markdown": "Sheet: Rekap\n\n| Metrik | Nilai |"}]
    client._api._store.save_document("pi-test", meta, tree, pages)
    return client


def test_tools_describe_themselves_as_the_sdk_does():
    for tool in pageindex_tools(client=None):
        schema = _local_schema(tool.name)["properties"]
        assert tool.description == _local_description(tool.name)
        assert {k: v.get("description") for k, v in tool.args.items()} == \
            {k: v["description"] for k, v in schema.items()}
        assert all(tool.args[k].get("default") == v["default"] for k, v in schema.items() if "default" in v)


def test_tools_return_what_call_tool_returns(tmp_path):
    client = _library(tmp_path)
    tools = {t.name: t for t in pageindex_tools(client)}
    for name, args in [("browse_documents", {}),
                       ("get_document", {"doc_name": "rekap.xlsx"}),
                       ("get_document_structure", {"doc_name": "rekap.xlsx"}),
                       ("get_page_content", {"doc_name": "rekap.xlsx", "pages": "1-2"}),
                       ("get_page_content", {"doc_name": "tidak-ada.pdf", "pages": "1"})]:
        assert tools[name].invoke(args) == call_tool(client, name, args)[0]
    pages = json.loads(tools["get_page_content"].invoke({"doc_name": "rekap.xlsx", "pages": "1-2"}))
    assert [p["citation"] for p in pages["content"]] == [
        {"file": "rekap.xlsx", "sheet": "Daftar"}, {"file": "rekap.xlsx", "sheet": "Rekap"}]


def test_doc_ids_scope_the_tools(tmp_path):
    client = _library(tmp_path)
    scoped = {t.name: t for t in pageindex_tools(client, doc_ids=["pi-lain"])}
    assert json.loads(scoped["get_page_content"].invoke({"doc_name": "rekap.xlsx", "pages": "1"})).get("error")


def test_agent_carries_the_pageindex_tools(tmp_path):
    from app.agents.loop import build_agent
    agent = build_agent(model=FakeMessagesListChatModel(responses=[AIMessage(content="ok")]),
                        storage_path=str(tmp_path))
    tool_names = set(agent.nodes["tools"].bound.tools_by_name)
    assert {"browse_documents", "get_document", "get_document_structure", "get_page_content"} <= tool_names
    assert "remove_document" not in tool_names
    assert not {"write_file", "edit_file", "execute"} & tool_names   # file tools stay read-only
