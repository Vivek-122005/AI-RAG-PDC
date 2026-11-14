from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from agentic_rag.domain.search import SearchHit
from agentic_rag.evaluation.ablation import run_retrieval_ablation, to_markdown
from agentic_rag.evaluation.dataset import Evidence, GoldenItem
from agentic_rag.indexing.store import QdrantStore
from agentic_rag.ingestion.chunking import ChunkingConfig
from agentic_rag.ingestion.pipeline import ingest_path
from agentic_rag.retrieval import NoopReranker, RetrievalConfig, Retriever


class ReverseReranker:
    """Deterministic stand-in for a cross-encoder: reverses candidate order."""

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        return [replace(h, score=float(i)) for i, h in enumerate(reversed(hits))][:top_k]


def index(tmp_path: Path, store: QdrantStore) -> None:
    for name, text in {
        "tea.md": "# Tea\n\nGreen tea leaves are steamed after picking.",
        "coffee.md": "# Coffee\n\nCoffee beans are roasted before grinding.",
        "cocoa.md": "# Cocoa\n\nCocoa beans are fermented and dried.",
    }.items():
        (tmp_path / name).write_text(text)
    ingest_path(tmp_path, store, ChunkingConfig(min_tokens=1))


def test_two_stage_retrieval_respects_limits(tmp_path: Path, store: QdrantStore) -> None:
    index(tmp_path, store)
    r = Retriever(store, NoopReranker(), RetrievalConfig("hybrid", candidates=3, top_k=1))
    result = r.retrieve("green tea steamed")
    assert len(result.candidates) == 3
    assert [h.source for h in result.contexts] == ["tea.md"]


def test_reranker_controls_final_order(tmp_path: Path, store: QdrantStore) -> None:
    index(tmp_path, store)
    r = Retriever(store, ReverseReranker(), RetrievalConfig("hybrid", candidates=3, top_k=3))
    result = r.retrieve("green tea steamed")
    assert [h.source for h in result.contexts] == [h.source for h in reversed(result.candidates)]


def test_ablation_table(tmp_path: Path, store: QdrantStore) -> None:
    index(tmp_path, store)
    items = [
        GoldenItem(
            id="q",
            question="how are green tea leaves processed",
            type="single_hop",
            evidence=[
                Evidence(source="tea.md", quote="Green tea leaves are steamed after picking.")
            ],
        ),
        GoldenItem(id="u", question="unrelated", type="unanswerable", answerable=False),
    ]
    rows = run_retrieval_ablation(
        items,
        {
            "good": Retriever(store, NoopReranker(), RetrievalConfig(top_k=1)),
            "reversed": Retriever(store, ReverseReranker(), RetrievalConfig(candidates=3, top_k=1)),
        },
    )
    assert rows[0] == {
        "config": "good",
        "hit_rate@1": 1.0,
        "mrr@1": 1.0,
        "evidence_recall@1": 1.0,
    }
    assert rows[1]["hit_rate@1"] == 0.0
    assert to_markdown(rows).startswith("| config | hit_rate@1 |")
