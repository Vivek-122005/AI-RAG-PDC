"""Failure taxonomy: which stage of the system a failed item broke in.

Run on the records of an eval run (answers with their agent traces, plus judgments), so
every bad answer gets a cause, not just a score:

  guardrail_block       a guard blocked or replaced the answer
  hallucinated_answer   answered a question the corpus cannot answer
  retrieval_miss        the gold evidence was never among the search candidates
  rerank_miss           it was a candidate, but never made it into the drafter's evidence
  critic_false_reject   a draft was given, then the critic's checks turned it into a refusal
  drafter_refusal       the gold evidence was in context, yet the drafter refused
  wrong_answer          answered, but the judge scored correctness ≤ 2
  unfaithful            answered with claims the judge found unsupported by the evidence
"""

from collections.abc import Sequence
from typing import Literal

from agentic_rag.evaluation.matching import evidence_recall
from agentic_rag.evaluation.metrics import ItemResult

Cause = Literal[
    "guardrail_block",
    "hallucinated_answer",
    "retrieval_miss",
    "rerank_miss",
    "critic_false_reject",
    "drafter_refusal",
    "wrong_answer",
    "unfaithful",
]

CAUSES: tuple[Cause, ...] = (
    "retrieval_miss",
    "rerank_miss",
    "drafter_refusal",
    "critic_false_reject",
    "wrong_answer",
    "unfaithful",
    "hallucinated_answer",
    "guardrail_block",
)


def classify(r: ItemResult) -> Cause | None:
    """The stage a failed item broke in, or None when the item succeeded."""
    if not r.item.answerable:
        return "hallucinated_answer" if not r.refused else None
    if r.answer.get("blocked_by") and r.refused:
        return "guardrail_block"
    if r.refused:
        evidence = r.item.evidence
        if evidence and evidence_recall(r.candidates, evidence) == 0:
            return "retrieval_miss"
        if evidence and evidence_recall(r.shown, evidence) == 0:  # never shown in any round
            return "rerank_miss"
        drafted = any(
            s.get("node") == "draft" and s.get("refused") is False
            for s in r.answer.get("trace", [])
        )
        return "critic_false_reject" if drafted else "drafter_refusal"
    if r.judgment.get("correctness", 5) <= 2:
        return "wrong_answer"
    if r.judgment.get("faithfulness", 1.0) < 1.0:
        return "unfaithful"
    return None


def taxonomy(results: Sequence[ItemResult]) -> dict[str, list[str]]:
    """Item ids per failure cause, in a fixed order (empty causes included)."""
    by_cause: dict[str, list[str]] = {c: [] for c in CAUSES}
    for r in results:
        if (cause := classify(r)) is not None:
            by_cause[cause].append(r.item.id)
    return by_cause


def trace_summary(trace: Sequence[dict[str, object]]) -> str:
    """One line per run: `plan → retrieve x3 → draft(refused) → critique(retry) → …`."""
    parts: list[str] = []
    for step in trace:
        node = str(step.get("node", "?"))
        if node == "retrieve":
            if parts and parts[-1].startswith("retrieve"):
                count = int(parts[-1].partition(" x")[2] or 1) + 1
                parts[-1] = f"retrieve x{count}"
                continue
            parts.append("retrieve")
        elif node == "draft":
            parts.append("draft(refused)" if step.get("refused") else "draft")
        elif node == "critique":
            uncovered = step.get("uncovered_subjects")
            extra = f", uncovered={uncovered}" if uncovered else ""
            parts.append(f"critique({step.get('decision')}{extra})")
        elif node == "replan":
            parts.append(f"replan({step.get('strategy')})")
        elif node in ("input_guard", "output_guard"):
            blocked = step.get("blocked") or step.get("refused")
            if blocked:
                parts.append(f"{node}({blocked})")
        else:
            parts.append(node)
    return " → ".join(parts)
