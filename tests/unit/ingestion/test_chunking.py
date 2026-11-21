import itertools

from agentic_rag.domain.documents import Block, Document
from agentic_rag.ingestion.chunking import (
    ChunkingConfig,
    approx_tokens,
    chunk_document,
    embedding_text,
    split_sentences,
)


def make_doc(blocks: list[Block]) -> Document:
    return Document("d1", "src", "Title", "h", "txt", blocks)


def sentence(i: int) -> str:
    return f"Sentence number {i} describes an important fact about retrieval systems."


def test_split_sentences() -> None:
    assert split_sentences("One fact. Two facts! Three? (Four)") == [
        "One fact.",
        "Two facts!",
        "Three?",
        "(Four)",
    ]


def test_small_paragraphs_are_packed_together() -> None:
    doc = make_doc([Block(sentence(i), page=1) for i in range(3)])
    chunks = chunk_document(doc, ChunkingConfig(min_tokens=1))
    assert len(chunks) == 1
    assert all(sentence(i) in chunks[0].text for i in range(3))


def test_never_crosses_sections() -> None:
    doc = make_doc(
        [Block(sentence(1), section_path=("A",)), Block(sentence(2), section_path=("B",))]
    )
    chunks = chunk_document(doc, ChunkingConfig(min_tokens=1))
    assert [c.section_path for c in chunks] == [("A",), ("B",)]


def test_respects_max_and_overlaps() -> None:
    big = " ".join(sentence(i) for i in range(200))
    cfg = ChunkingConfig(target_tokens=120, max_tokens=150, overlap_tokens=30, min_tokens=1)
    chunks = chunk_document(make_doc([Block(big, page=4)]), cfg)
    assert len(chunks) > 5
    assert all(c.token_count <= cfg.max_tokens for c in chunks)
    # Chunk N+1 starts with trailing sentences of chunk N (overlap).
    for a, b in itertools.pairwise(chunks):
        a_sents, b_sents = split_sentences(a.text), split_sentences(b.text)
        assert b_sents[0] in a_sents[-3:]
        assert a_sents[-1] in b_sents
    assert chunks[0].page_start == 4
    # Every sentence survives chunking.
    joined = " ".join(c.text for c in chunks)
    assert all(sentence(i) in joined for i in range(200))


def test_drops_tiny_fragments_and_ids_are_sequential() -> None:
    doc = make_doc(
        [Block("Fig. 3", section_path=("X",)), Block(sentence(1) * 3, section_path=("Y",))]
    )
    chunks = chunk_document(doc, ChunkingConfig(min_tokens=10))
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_embedding_text_has_context_header() -> None:
    doc = make_doc([Block(sentence(1) * 3, section_path=("2 Method", "2.1 Retriever"))])
    c = chunk_document(doc)[0]
    assert embedding_text(doc.title, c).startswith("Title > 2 Method > 2.1 Retriever\n\n")


def test_approx_tokens_reasonable() -> None:
    assert 8 <= approx_tokens("Retrieval-augmented generation reduces hallucination.") <= 12


def test_short_document_is_kept_even_below_min_tokens() -> None:
    doc = Document(
        "d", "note.md", "Note", "h", "md", [Block("Codename is Bluebird.", None, ("Note",))]
    )
    chunks = chunk_document(doc, ChunkingConfig(min_tokens=20))
    assert [c.text for c in chunks] == ["Codename is Bluebird."]


def test_small_fragments_are_still_dropped_from_longer_documents() -> None:
    body = " ".join(["Retrieval augmented generation grounds answers in documents."] * 8)
    doc = Document(
        "d", "paper.md", "Paper", "h", "md",
        [Block(body, None, ("Intro",)), Block("Figure 1", None, ("Caption",))],
    )  # fmt: skip
    assert all("Figure 1" not in c.text for c in chunk_document(doc, ChunkingConfig(min_tokens=5)))
