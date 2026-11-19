"""Structured outputs the planner and critic must return (validated with Pydantic)."""

from pydantic import BaseModel, Field, field_validator


# Every field is required (no defaults): with a JSON-schema grammar, Ollama lets the model
# skip optional fields, and small models do. No max_length either: an over-long list is
# truncated in code rather than failing validation.
def _clean(items: list[str]) -> list[str]:
    """Small models pad lists with empty strings; drop them."""
    return [s.strip() for s in items if s.strip()]


class Plan(BaseModel):
    """Planner decision: answer directly, or split into self-contained search queries."""

    subjects: list[str] = Field(
        description="names of the papers, systems, methods or datasets the question asks about",
    )
    needs_decomposition: bool = Field(
        description="true if answering needs evidence about two or more different things"
    )
    sub_questions: list[str] = Field(
        description="at most 3 self-contained search queries, one per thing; empty if none",
    )

    _clean_lists = field_validator("subjects", "sub_questions")(_clean)


class Critique(BaseModel):
    """Critic verdict on a draft. Routing is decided in code from these fields."""

    grounded: bool = Field(description="every claim is supported by the evidence it cites")
    answers_question: bool = Field(
        description="the draft answers the question asked, about the same entities it names"
    )
    unsupported_claims: list[str] = Field(
        description="claims not supported by the evidence they cite"
    )
    missing: list[str] = Field(description="parts of the question the draft does not answer")
    follow_up_queries: list[str] = Field(
        description="at most 3 new search queries that could find the missing evidence",
    )

    _clean_lists = field_validator("unsupported_claims", "missing", "follow_up_queries")(_clean)

    @classmethod
    def passing(cls) -> "Critique":
        """A verdict with no objections (used when the critic is unavailable)."""
        return cls(
            grounded=True,
            answers_question=True,
            unsupported_claims=[],
            missing=[],
            follow_up_queries=[],
        )
