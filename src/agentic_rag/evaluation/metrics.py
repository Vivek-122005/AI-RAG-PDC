"""Metric definitions. Pure functions over run records: no I/O, no models."""

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.domain.search import SearchHit
from agentic_rag.evaluation import records
from agentic_rag.evaluation.dataset import GoldenItem
from agentic_rag.evaluation.matching import covers, evidence_recall, first_relevant_rank


@dataclass
class ItemResult:
    item: GoldenItem
    answer: dict[str, Any]
    judgment: dict[str, Any]

    @property
    def refused(self) -> bool:
        return bool(self.answer["refused"])

    @property
    def contexts(self) -> list[SearchHit]:
        return [records.hit_from_record(r) for r in self.answer["contexts"]]

    @property
    def candidates(self) -> list[SearchHit]:
        return [records.hit_from_record(r) for r in self.answer["candidates"]]

    @property
    def shown(self) -> list[SearchHit]:
        """Every passage the generator saw (all agent rounds); falls back to contexts."""
        rows = self.answer.get("shown") or self.answer["contexts"]
        return [records.hit_from_record(r) for r in rows]

    @property
    def cites_gold_evidence(self) -> bool:
        ctx = self.contexts
        cited = [ctx[i - 1] for i in self.answer["cited_ids"] if i <= len(ctx)]
        return any(covers(h.text, h.source, ev) for h in cited for ev in self.item.evidence)

    @property
    def is_failure(self) -> bool:
        if not self.item.answerable:
            return not self.refused
        return self.refused or self.judgment.get("correctness", 5) <= 2


def load_results(items: Sequence[GoldenItem], run_dir: Path) -> list[ItemResult]:
    answers = {a["id"]: a for a in records.read_jsonl(run_dir / records.ANSWERS)}
    judgments = {j["id"]: j for j in records.read_jsonl(run_dir / records.JUDGMENTS)}
    return [ItemResult(i, answers[i.id], judgments.get(i.id, {})) for i in items if i.id in answers]


def mean(values: Iterable[float]) -> float | None:
    xs = list(values)
    return round(statistics.fmean(xs), 3) if xs else None


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def compute_metrics(results: Sequence[ItemResult]) -> dict[str, Any]:
    answerable = [r for r in results if r.item.answerable]
    unanswerable = [r for r in results if not r.item.answerable]
    answered = [r for r in answerable if not r.refused]
    judged = [r for r in answered if "correctness" in r.judgment]
    ranks = [first_relevant_rank(r.contexts, r.item.evidence) for r in answerable]
    latencies = [r.answer["timings"].get("total_s", 0.0) for r in results]

    return {
        "n_items": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval": {
            "hit_rate@k": mean(rank is not None for rank in ranks),
            "mrr@k": mean(1 / rank if rank else 0.0 for rank in ranks),
            "evidence_recall@k": mean(
                evidence_recall(r.contexts, r.item.evidence) for r in answerable
            ),
            "candidate_evidence_recall": mean(
                evidence_recall(r.candidates, r.item.evidence) for r in answerable
            ),
        },
        "generation": {
            "answer_correctness_mean": mean(r.judgment["correctness"] for r in judged),
            # Refusing an answerable question counts as wrong.
            "accuracy(score>=4)": mean(r.judgment.get("correctness", 0) >= 4 for r in answerable),
            "faithfulness_mean": mean(r.judgment["faithfulness"] for r in judged),
            "fully_faithful_rate": mean(r.judgment["faithfulness"] == 1.0 for r in judged),
            "addresses_question_rate": mean(r.judgment["addresses_question"] for r in judged),
            "cites_gold_evidence_rate": mean(r.cites_gold_evidence for r in answered),
            "false_refusal_rate": mean(r.refused for r in answerable),
            "correct_refusal_rate": mean(r.refused for r in unanswerable),
        },
        "efficiency": {
            "latency_mean_s": mean(latencies),
            "latency_p50_s": percentile(latencies, 0.5),
            "latency_p95_s": percentile(latencies, 0.95),
            "prompt_tokens_mean": mean(r.answer["usage"]["prompt_tokens"] for r in results),
            "completion_tokens_mean": mean(r.answer["usage"]["completion_tokens"] for r in results),
        },
    }


def compute_breakdown(results: Sequence[ItemResult]) -> dict[str, Any]:
    types = sorted({r.item.type for r in results})
    return {
        "overall": compute_metrics(results),
        "by_type": {t: compute_metrics([r for r in results if r.item.type == t]) for t in types},
    }
