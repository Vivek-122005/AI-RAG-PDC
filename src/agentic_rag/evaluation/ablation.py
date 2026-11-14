"""Retrieval-only ablations: compare search modes and rerankers without any LLM calls."""

from collections.abc import Mapping, Sequence
from typing import Any

from agentic_rag.evaluation.dataset import GoldenItem
from agentic_rag.evaluation.matching import evidence_recall, first_relevant_rank
from agentic_rag.evaluation.metrics import mean
from agentic_rag.retrieval.retriever import Retriever


def run_retrieval_ablation(
    items: Sequence[GoldenItem], retrievers: Mapping[str, Retriever]
) -> list[dict[str, Any]]:
    answerable = [i for i in items if i.answerable]
    rows = []
    for name, retriever in retrievers.items():
        k = retriever.config.top_k
        ranks, recalls = [], []
        for item in answerable:
            contexts = retriever.retrieve(item.question).contexts
            ranks.append(first_relevant_rank(contexts, item.evidence))
            recalls.append(evidence_recall(contexts, item.evidence))
        rows.append(
            {
                "config": name,
                f"hit_rate@{k}": mean(r is not None for r in ranks),
                "hit_rate@1": mean(r == 1 for r in ranks),
                f"mrr@{k}": mean(1 / r if r else 0.0 for r in ranks),
                f"evidence_recall@{k}": mean(recalls),
            }
        )
    return rows


def to_markdown(rows: Sequence[dict[str, Any]]) -> str:
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    return "\n".join(lines) + "\n"
