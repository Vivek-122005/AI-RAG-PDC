from pathlib import Path

from fakes import FakeLLM

from agentic_rag.indexing.store import QdrantStore
from agentic_rag.ingestion.chunking import ChunkingConfig
from agentic_rag.ingestion.pipeline import ingest_path
from agentic_rag.pipelines import BaselineRAG, is_refusal
from agentic_rag.pipelines.prompts import REFUSAL
from agentic_rag.retrieval import NoopReranker, RetrievalConfig, Retriever


def setup(tmp_path: Path, store: QdrantStore, reply: str) -> tuple[BaselineRAG, FakeLLM]:
    (tmp_path / "tea.md").write_text("# Tea\n\nGreen tea is steamed or pan-fired after picking.\n")
    (tmp_path / "coffee.md").write_text("# Coffee\n\nCoffee beans are roasted seeds.\n")
    ingest_path(tmp_path, store, ChunkingConfig(min_tokens=1))
    llm = FakeLLM(reply)
    return BaselineRAG(Retriever(store, NoopReranker(), RetrievalConfig(top_k=2)), llm), llm


def test_answer_with_citations(tmp_path: Path, store: QdrantStore) -> None:
    rag, llm = setup(tmp_path, store, "Green tea is steamed [1]. Bogus [9].")
    ans = rag.answer("how is green tea processed")
    assert ans.contexts[0].source == "tea.md"
    assert ans.cited_ids == [1]  # out-of-range [9] ignored
    assert ans.cited_contexts[0].source == "tea.md"
    assert not ans.refused
    prompt = llm.calls[0][1].content
    assert '<evidence id="1"' in prompt and "Question: how is green tea processed" in prompt
    assert "total_s" in ans.timings


def test_refusal_detection(tmp_path: Path, store: QdrantStore) -> None:
    rag, _ = setup(tmp_path, store, REFUSAL)
    assert rag.answer("who won the 1998 world cup").refused
    assert is_refusal("I cannot answer this from the provided documents")
    assert not is_refusal("Tea is a drink [1].")
