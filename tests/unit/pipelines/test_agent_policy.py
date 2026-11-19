from collections.abc import Sequence

from fakes import make_hit

from agentic_rag.domain.search import SearchHit
from agentic_rag.pipelines.agent import evidence as ev
from agentic_rag.pipelines.agent.policy import decide
from agentic_rag.pipelines.agent.schemas import Critique

GOOD = Critique.passing()


def critique(**changes: object) -> Critique:
    return GOOD.model_copy(update=changes)


def ids(hits: Sequence[SearchHit]) -> list[str]:
    return [h.chunk_id for h in hits]


# ── evidence selection ───────────────────────────────────────────────────────────


def test_interleave_takes_best_of_each_query_first_and_dedupes() -> None:
    a = [make_hit("a1"), make_hit("shared"), make_hit("a3")]
    b = [make_hit("b1"), make_hit("shared"), make_hit("b3")]
    assert ids(ev.interleave([a, b], limit=10)) == ["a1", "b1", "shared", "a3", "b3"]
    assert ids(ev.interleave([a, b], limit=2)) == ["a1", "b1"]
    assert ev.interleave([], limit=5) == []


def test_carry_forward_keeps_cited_then_new_then_rest() -> None:
    prev = [make_hit("p1"), make_hit("p2"), make_hit("p3")]
    new = [make_hit("n1"), make_hit("p2")]
    out = ev.carry_forward(prev, cited=[2], new=new, limit=4)
    assert ids(out) == ["p2", "n1", "p1", "p3"]


def test_unique_queries_drops_blanks_dupes_and_past() -> None:
    out = ev.unique_queries(["  CRAG ", "crag", "", "Self-RAG", "ragas"], exclude=["RAGAS"])
    assert out == ["CRAG", "Self-RAG"]


# ── routing policy ───────────────────────────────────────────────────────────────


def test_accepts_grounded_relevant_cited_draft() -> None:
    kw = {
        "refused": False,
        "has_citations": True,
        "subject_coverage": "all",
        "retries_left": 2,
        "can_search_more": True,
    }
    assert decide(GOOD, **kw) == "accept"  # type: ignore[arg-type]


def test_off_target_or_ungrounded_retries_then_refuses() -> None:
    off = critique(answers_question=False)
    assert (
        decide(
            off,
            refused=False,
            has_citations=True,
            subject_coverage="all",
            retries_left=1,
            can_search_more=True,
        )
        == "retry"
    )
    assert (
        decide(
            off,
            refused=False,
            has_citations=True,
            subject_coverage="all",
            retries_left=0,
            can_search_more=True,
        )
        == "refuse"
    )
    assert (
        decide(
            off,
            refused=False,
            has_citations=True,
            subject_coverage="all",
            retries_left=2,
            can_search_more=False,
        )
        == "refuse"
    )


def test_uncited_draft_counts_as_ungrounded() -> None:
    assert (
        decide(
            GOOD,
            refused=False,
            has_citations=False,
            subject_coverage="all",
            retries_left=0,
            can_search_more=False,
        )
        == "refuse"
    )


def test_refusal_is_retried_only_with_new_queries() -> None:
    assert (
        decide(
            GOOD,
            refused=True,
            has_citations=False,
            subject_coverage="all",
            retries_left=1,
            can_search_more=True,
        )
        == "retry"
    )
    assert (
        decide(
            GOOD,
            refused=True,
            has_citations=False,
            subject_coverage="all",
            retries_left=1,
            can_search_more=False,
        )
        == "refuse"
    )


def test_incomplete_but_grounded_is_retried_then_accepted() -> None:
    partial = critique(missing=["second paper"])
    assert (
        decide(
            partial,
            refused=False,
            has_citations=True,
            subject_coverage="all",
            retries_left=1,
            can_search_more=True,
        )
        == "retry"
    )
    assert (
        decide(
            partial,
            refused=False,
            has_citations=True,
            subject_coverage="all",
            retries_left=0,
            can_search_more=True,
        )
        == "accept"
    )


def test_critique_drops_empty_list_items() -> None:
    c = Critique.model_validate(
        {**GOOD.model_dump(), "unsupported_claims": [""], "follow_up_queries": [" q ", ""]}
    )
    assert c.unsupported_claims == [] and c.follow_up_queries == ["q"]


def test_draft_missing_a_named_subject_is_off_target() -> None:
    def run(retries_left: int) -> str:
        return decide(
            GOOD,
            refused=False,
            has_citations=True,
            subject_coverage="none",
            retries_left=retries_left,
            can_search_more=True,
        )

    assert run(1) == "retry"
    assert run(0) == "refuse"


def test_missing_subjects_ignores_case_and_punctuation() -> None:
    hits = [make_hit("a", "SELF-RAG trains on 4 A100 GPUs."), make_hit("b", "CRAG uses T5.")]
    assert ev.missing_subjects(["Self-RAG", "CRAG"], hits) == []
    assert ev.missing_subjects(["Ragas", "Self-RAG"], hits[1:]) == ["Ragas", "Self-RAG"]
    assert ev.missing_subjects([], hits) == []
    ragas = [make_hit("r", "Ragas: automated evaluation, on average.")]
    assert ev.missing_subjects(["RAG"], ragas) == ["RAG"]  # whole words only
    assert ev.missing_subjects(["Ragas"], ragas) == []
    token = [make_hit("t", "RAG-Token marginalizes over documents.")]
    assert ev.missing_subjects(["RAG paper", "null-document mechanisms"], token) == []


def test_listed_unsupported_claim_overrides_grounded_flag() -> None:
    contradictory = critique(grounded=True, unsupported_claims=["made-up number"])
    assert (
        decide(
            contradictory,
            refused=False,
            has_citations=True,
            subject_coverage="all",
            retries_left=0,
            can_search_more=False,
        )
        == "refuse"
    )


def test_partial_subject_coverage_is_incomplete_not_off_target() -> None:
    def run(retries_left: int) -> str:
        return decide(
            GOOD,
            refused=False,
            has_citations=True,
            subject_coverage="partial",
            retries_left=retries_left,
            can_search_more=True,
        )

    assert run(1) == "retry"
    assert run(0) == "accept"  # a grounded half-answer beats a refusal


def test_subject_coverage_levels() -> None:
    hits = [make_hit("c", "CRAG uses T5-large.")]
    assert ev.subject_coverage(["CRAG"], hits) == "all"
    assert ev.subject_coverage(["CRAG", "Ragas"], hits) == "partial"
    assert ev.subject_coverage(["Ragas"], hits) == "none"
    assert ev.subject_coverage(["the paper"], hits) == "all"  # nothing checkable
