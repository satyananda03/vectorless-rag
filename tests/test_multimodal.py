"""Offline: the LLM is stubbed through litellm, Textract/VLM through MultimodalLocalAPI._parse.

The first test pins every upstream internal this package relies on; when it fails
after a `git pull` of PageIndex, the named hook point is what needs updating.
"""
import ast
import hashlib
import inspect
import json
import threading
from types import SimpleNamespace

import litellm
import pymupdf
import pytest
from openpyxl import Workbook
from PIL import Image

import vectorless_rag as pm
from pageindex.errors import PageIndexAPIError
from vectorless_rag import MultimodalClient, pageindex_tools
from vectorless_rag.language import LANGUAGE_RULE
from vectorless_rag.multimodal_api import MultimodalLocalAPI, Parsed, render_pages

# sha256 of upstream's generate_doc_description, which language.py copies
DOC_DESCRIPTION_SHA256 = "0da6d0020fd184073d2b46a4c6702b3dbac537033e267851177a5de0dd0f63f3"
LONG = " ".join(["anggaran"] * 200)  # past summary_max_words, so the leaf goes to the model


def test_upstream_contract():
    from pageindex import agent_tools, page_index_classic, utils
    from pageindex.flash import api as flash_api
    from pageindex.local_api import LocalAPI

    def params(func):
        return list(inspect.signature(func).parameters)

    assert params(LocalAPI.submit_document) == [
        "self", "file_path", "mode", "beta_headers", "folder_id", "metadata"]
    assert params(LocalAPI._extract_page_texts) == ["file_path"]
    assert params(LocalAPI._index_flash) == ["self", "file_path"]
    assert params(LocalAPI._unique_doc_name) == ["self", "name"]
    for source, needle in ((LocalAPI.submit_document, "self._extract_page_texts(file_path)"),
                           (LocalAPI.submit_document, "self._index_flash"),
                           (LocalAPI.submit_document, "self._unique_doc_name("),
                           (LocalAPI._index_flash, "generate_doc_description"),
                           (LocalAPI.__init__, "self._index_backend"),
                           (flash_api.page_index_flash, "extract_toc(_validate_pdf(pdf)"),
                           (flash_api.page_index_flash, 'result.pop("page_texts"'),
                           (flash_api.flash_rejection_reason, '"unreadable"'),
                           (utils.SummaryScheduler, "async def _ask(self, prompt, prio):")):
        assert needle in inspect.getsource(source), needle
    assert flash_api.extract_toc is pm.multimodal_api._extract_toc
    assert params(pm.multimodal_api._original_extract_toc)[0] == "doc_handle"
    assert utils.SummaryScheduler._ask.multimodal
    assert utils.generate_doc_description is pm.language.generate_doc_description
    assert page_index_classic.generate_doc_description is pm.language.generate_doc_description
    src = inspect.getsource(utils)
    upstream = next(node for node in ast.parse(src).body if isinstance(node, ast.FunctionDef)
                    and node.name == "generate_doc_description")
    assert hashlib.sha256(ast.get_source_segment(src, upstream).encode()).hexdigest() \
        == DOC_DESCRIPTION_SHA256, "upstream changed generate_doc_description: update language.py"
    page_content = agent_tools._IMPLEMENTATIONS["get_page_content"]
    assert page_content.multimodal
    assert params(page_content)[:3] == ["client", "doc_name", "pages"]
    for name in ("post_processing", "cover_subtree_ranges", "run_off_loop", "count_tokens"):
        assert callable(getattr(utils, name))


@pytest.fixture
def prompts(monkeypatch):
    """Every prompt sent to the model; summaries and description get canned replies."""
    seen = []

    def reply(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    async def acompletion(**kw):
        seen.append(json.dumps(kw["messages"], ensure_ascii=False))
        return reply('{"summary": "ringkasan", "subsections": []}')

    def completion(**kw):
        seen.append(json.dumps(kw["messages"], ensure_ascii=False))
        return reply("deskripsi")

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "completion", completion)
    return seen


@pytest.fixture
def client(tmp_path):
    return MultimodalClient(storage_path=str(tmp_path / "store"), index_model="gpt-test")


def stub_textract(monkeypatch, parsed_for):
    """Replace Textract + VLM: ``parsed_for(pages)`` builds the Parsed; returns the calls."""
    calls = []

    def parse(self, pdf, tmp, pages, image=None):
        calls.append(pages)
        return parsed_for(pages)

    monkeypatch.setattr(MultimodalLocalAPI, "_parse", parse)
    return calls


def write_pdf(path, texts):
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 770), text, fontsize=9)
        doc.save(path)
    return path


def tree(client, doc_id):
    return client.get_document_structure(doc_id)


def page_texts(client, doc_id, pages):
    return [p["markdown"] for p in client.get_page_content(doc_id, pages)]


def test_client_sets_output_cap_and_keeps_given_backend(tmp_path):
    api = MultimodalClient(storage_path=str(tmp_path), index_backend={"max_tokens": 5})._api
    assert isinstance(api, MultimodalLocalAPI) and api._index_backend == {"max_tokens": 5}
    assert MultimodalClient(storage_path=str(tmp_path))._api._index_backend == {"max_tokens": 16000}


def test_spreadsheet_sheets_are_pages(client, prompts, tmp_path):
    workbook = Workbook()
    workbook.active.title = "Anggaran"
    workbook.active.append(["Program", "Pagu"])
    workbook.active.append(["Smart Province", 1500000])
    sheet = workbook.create_sheet("Kinerja")
    sheet.append(["Indikator", "Target"])
    sheet.append(["Indeks SPBE", 3.5])
    workbook.save(tmp_path / "renja.xlsx")

    result = client.submit_document(str(tmp_path / "renja.xlsx"))
    assert result["name"] == "renja.xlsx" and "usage" in result
    doc = client.get_document(result["doc_id"])
    assert doc["pageNum"] == 2
    assert [node["title"] for node in tree(client, result["doc_id"])] == ["Anggaran", "Kinerja"]
    first, second = page_texts(client, result["doc_id"], "1-2")
    assert first.startswith("Sheet: Anggaran") and "Smart Province" in first
    assert second.startswith("Sheet: Kinerja") and "Indeks SPBE" in second

    tools = {tool.name: tool for tool in pageindex_tools(client)}
    envelope = json.loads(tools["get_page_content"].invoke({"doc_name": "renja.xlsx", "pages": "1-2"}))
    assert [entry["citation"] for entry in envelope["content"]] == [
        {"file": "renja.xlsx", "sheet": "Anggaran"}, {"file": "renja.xlsx", "sheet": "Kinerja"}]


def test_scanned_image_gets_its_tree_from_textract_headings(client, prompts, monkeypatch, tmp_path):
    Image.new("RGB", (1200, 1600), "white").save(tmp_path / "scan.png")
    content_list = [
        {"type": "text", "text": "BAB I PENDAHULUAN", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": f"Latar belakang {LONG}", "page_idx": 0},
        {"type": "text", "text": "Tujuan Renstra", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "Tujuan strategis dinas.", "page_idx": 0},
    ]

    def parsed_for(pages):
        markdown, plain = render_pages(content_list, {}, 1)
        return Parsed(content_list, markdown, plain, visual_pages=pages)

    calls = stub_textract(monkeypatch, parsed_for)
    result = client.submit_document(str(tmp_path / "scan.png"))
    assert calls == [[0]]
    assert result["name"] == "scan.png"
    titles = [node["title"] for node in tree(client, result["doc_id"])]
    assert titles == ["BAB I PENDAHULUAN"]
    assert page_texts(client, result["doc_id"], "1")[0].startswith("# BAB I PENDAHULUAN")
    summary_prompts = [p for p in prompts if "Latar belakang" in p]
    assert summary_prompts and all(LANGUAGE_RULE.strip() in p for p in summary_prompts)
    assert any("BAHASA INDONESIA" in p and "Document Structure" in p for p in prompts)


def test_digital_pdf_reads_textract_only_on_visual_pages(client, prompts, monkeypatch, tmp_path):
    pdf = write_pdf(tmp_path / "laporan.pdf", [f"Halaman satu {LONG}", f"Halaman dua {LONG}"])

    def parsed_for(pages):
        return Parsed([], ["| Program | Pagu |", f"Halaman dua {LONG}"],
                      [f"TABEL TEXTRACT {LONG}", f"Halaman dua {LONG}"], visual_pages=pages)

    monkeypatch.setattr(pm.multimodal_api, "textract_pages", lambda pdf: [0])
    stub_textract(monkeypatch, parsed_for)
    result = client.submit_document(str(pdf))
    assert result["name"] == "laporan.pdf"
    first, second = page_texts(client, result["doc_id"], "1-2")
    assert first == "| Program | Pagu |" and second.startswith("Halaman dua")
    assert any("TABEL TEXTRACT" in p for p in prompts)       # summaries read Textract's text
    assert not any("Halaman satu" in p for p in prompts)     # instead of pdfium's for page 1


def test_refused_submit_is_never_sent_to_textract(client, monkeypatch, tmp_path):
    pdf = write_pdf(tmp_path / "a.pdf", ["teks"])
    monkeypatch.setattr(pm.multimodal_api, "textract_pages", lambda pdf: [0])
    calls = stub_textract(monkeypatch, lambda pages: None)
    with pytest.raises(PageIndexAPIError, match="folders"):
        client.submit_document(str(pdf), folder_id="f")
    assert calls == []


def test_concurrent_submits_of_one_file_keep_their_own_parse(client, prompts, monkeypatch, tmp_path):
    pdf = write_pdf(tmp_path / "sama.pdf", [f"Asli {LONG}"])
    monkeypatch.setattr(pm.multimodal_api, "textract_pages", lambda pdf: [0])
    stub_textract(monkeypatch, lambda pages: Parsed([], [f"TEXTRACT {threading.get_ident()}"],
                                                    [f"TEXTRACT {LONG}"], visual_pages=pages))
    results = {}

    def submit():
        results[threading.get_ident()] = client.submit_document(str(pdf))

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(r["name"] for r in results.values()) == ["sama.pdf", "sama_1.pdf"]
    for ident, result in results.items():
        assert page_texts(client, result["doc_id"], "1") == [f"TEXTRACT {ident}"]


def test_non_multimodal_files_keep_upstream_errors(client, tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(PageIndexAPIError, match="only PDF"):
        client.submit_document(str(tmp_path / "notes.txt"))
    with pytest.raises(FileNotFoundError):
        client.submit_document(str(tmp_path / "missing.pdf"))
