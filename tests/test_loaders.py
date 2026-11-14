from pathlib import Path

import pymupdf
import pytest

from agentic_rag.ingestion.loaders import clean_text, load_document


def test_clean_text_joins_hyphenation() -> None:
    assert clean_text("retrie-\nval  augmented\ngeneration") == "retrieval augmented generation"


def test_markdown_sections_and_title(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text(
        "# Guide\n\nIntro para.\n\n## Setup\n\nInstall it.\n\n### Linux\n\nUse apt.\n\n"
        "## Usage\n\nRun it.\n\n```\n# not a heading\n```\n\n## References\n\n[1] skip me\n"
    )
    d = load_document(p)
    assert d.title == "Guide"
    paths = {b.text: b.section_path for b in d.blocks}
    assert paths["Use apt."] == ("Guide", "Setup", "Linux")
    assert paths["Run it."] == ("Guide", "Usage")
    assert any("# not a heading" in t for t in paths)
    assert not any("skip me" in t for t in paths)


def test_html_strips_scripts(tmp_path: Path) -> None:
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>T</title><script>evil()</script></head>"
        "<body><h1>Top</h1><p>Hello <b>world</b></p><h2>Sub</h2><p>Deep</p></body></html>"
    )
    d = load_document(p)
    assert d.title == "T"
    assert [(b.text, b.section_path) for b in d.blocks] == [
        ("Hello world", ("Top",)),
        ("Deep", ("Top", "Sub")),
    ]


def test_pdf_pages_headings_and_title(tmp_path: Path) -> None:
    p = tmp_path / "paper.pdf"
    pdf = pymupdf.open()
    body = "This is body text that explains the method in detail. " * 3
    for page_no, name in enumerate(["Intro", "Method", "Results"]):
        page = pdf.new_page()
        y = 60
        if page_no == 0:
            page.insert_text((50, y), "A Study of Things", fontsize=18, fontname="hebo")
            y += 40
        page.insert_text((50, y), f"{page_no + 1} {name}", fontsize=12, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(50, y + 20, 550, y + 200), body, fontsize=10)
    pdf.save(p)

    d = load_document(p)
    assert d.title == "A Study of Things"
    assert [b.page for b in d.blocks] == [1, 2, 3]
    assert d.blocks[2].section_path == ("3 Results",)


def test_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "x.xyz"
    p.write_text("hi")
    with pytest.raises(ValueError):
        load_document(p)


def test_doc_id_stable_and_hash_changes(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("one")
    a = load_document(p, source="a.txt")
    p.write_text("two")
    b = load_document(p, source="a.txt")
    assert a.doc_id == b.doc_id
    assert a.content_hash != b.content_hash
