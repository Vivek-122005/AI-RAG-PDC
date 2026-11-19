"""What to do after the critic has spoken. A pure function, so the policy is unit-tested."""

from typing import Literal

from agentic_rag.pipelines.agent.evidence import Coverage
from agentic_rag.pipelines.agent.schemas import Critique

Decision = Literal["accept", "retry", "refuse"]


def decide(
    critique: Critique,
    *,
    refused: bool,
    has_citations: bool,
    subject_coverage: Coverage,
    retries_left: int,
    can_search_more: bool,
) -> Decision:
    """Accept a grounded, relevant draft; otherwise search again while budget remains.

    - A refusal is retried while there is more to search, and stands otherwise.
    - An ungrounded or off-target draft is never returned: retry, else refuse.
    - A grounded but incomplete draft is retried if possible, else accepted as is.
    Whatever the critic says, a draft without any valid [n] citation, or with any claim the
    critic lists as unsupported, counts as ungrounded. If its citations mention none of the
    subjects the question names it is off-target, and if only some, it is incomplete.
    """
    can_retry = retries_left > 0 and can_search_more
    if refused:
        return "retry" if can_retry else "refuse"
    # Small critics often list an unsupported claim yet still mark the draft grounded.
    grounded = critique.grounded and not critique.unsupported_claims and has_citations
    on_target = critique.answers_question and subject_coverage != "none"
    if not (grounded and on_target):
        return "retry" if can_retry else "refuse"
    incomplete = bool(critique.missing) or subject_coverage == "partial"
    if incomplete and can_retry:
        return "retry"
    return "accept"
