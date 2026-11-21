from typing import Any

from agentic_rag.evaluation.dataset import Evidence, GoldenItem
from agentic_rag.evaluation.metrics import ItemResult
from agentic_rag.evaluation.taxonomy import classify, taxonomy, trace_summary

EV = Evidence(source="a.pdf", quote="CRAG fine-tunes T5-large as the evaluator.")


def hit(text: str) -> dict[str, Any]:
    return {"chunk_id": text[:8], "score": 1.0, "text": text, "doc_id": "d", "title": "T",
            "source": "a.pdf", "section_path": [], "page_start": 1, "page_end": 1}  # fmt: skip


def result(
    *,
    refused: bool,
    answerable: bool = True,
    candidates: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    trace: tuple[dict[str, Any], ...] = (),
    judgment: dict[str, Any] | None = None,
    blocked_by: str | None = None,
) -> ItemResult:
    item = GoldenItem(
        id="x", question="q", type="single_hop" if answerable else "unanswerable",
        answerable=answerable, evidence=[EV] if answerable else [],
    )  # fmt: skip
    answer = {
        "refused": refused,
        "blocked_by": blocked_by,
        "candidates": [hit(t) for t in candidates],
        "contexts": [hit(t) for t in contexts],
        "trace": list(trace),
        "cited_ids": [],
    }
    return ItemResult(item, answer, judgment or {})


GOLD, NOISE = EV.quote, "Unrelated text about something else."


def test_classify_each_failure_stage() -> None:
    assert classify(result(refused=False, answerable=False)) == "hallucinated_answer"
    assert classify(result(refused=True, answerable=False)) is None
    assert classify(result(refused=True, blocked_by="input:prompt_injection")) == "guardrail_block"
    assert classify(result(refused=True, candidates=(NOISE,))) == "retrieval_miss"
    assert classify(result(refused=True, candidates=(GOLD,), contexts=(NOISE,))) == "rerank_miss"
    drafted = ({"node": "draft", "refused": False}, {"node": "critique", "decision": "refuse"})
    in_ctx = {"candidates": (GOLD,), "contexts": (GOLD,)}
    assert classify(result(refused=True, trace=drafted, **in_ctx)) == "critic_false_reject"  # type: ignore[arg-type]
    assert classify(result(refused=True, **in_ctx)) == "drafter_refusal"  # type: ignore[arg-type]
    # Shown in an earlier agent round but replaced later: still the drafter's refusal.
    earlier = result(refused=True, candidates=(GOLD,), contexts=(NOISE,))
    earlier.answer["shown"] = [hit(GOLD), hit(NOISE)]
    assert classify(earlier) == "drafter_refusal"
    assert classify(result(refused=False, judgment={"correctness": 2})) == "wrong_answer"
    assert (
        classify(result(refused=False, judgment={"correctness": 5, "faithfulness": 0.5}))
        == "unfaithful"
    )
    assert classify(result(refused=False, judgment={"correctness": 5, "faithfulness": 1.0})) is None


def test_taxonomy_groups_ids_and_keeps_every_cause() -> None:
    by_cause = taxonomy([result(refused=True, candidates=(NOISE,))])
    assert by_cause["retrieval_miss"] == ["x"]
    assert by_cause["wrong_answer"] == []


def test_trace_summary_is_one_readable_line() -> None:
    trace: list[dict[str, object]] = [
        {"node": "input_guard", "redacted": []},
        {"node": "plan"},
        {"node": "retrieve"}, {"node": "retrieve"}, {"node": "retrieve"},
        {"node": "draft", "refused": True},
        {"node": "critique", "decision": "retry", "uncovered_subjects": ["Ragas"]},
        {"node": "replan", "strategy": "search+widen"},
        {"node": "retrieve"},
        {"node": "draft", "refused": False},
        {"node": "critique", "decision": "accept", "uncovered_subjects": []},
        {"node": "output_guard", "refused": None},
    ]  # fmt: skip
    assert trace_summary(trace) == (
        "plan → retrieve x3 → draft(refused) → critique(retry, uncovered=['Ragas']) → "
        "replan(search+widen) → retrieve → draft → critique(accept)"
    )
