"""The result type every answer pipeline returns."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_rag.domain.search import SearchHit
from agentic_rag.llm.base import Usage
from agentic_rag.pipelines.prompts import REFUSAL

_CITATION_RE = re.compile(r"\[(\d+)\]")
_NON_WORD = re.compile(r"\W+")


def _norm(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def is_refusal(text: str) -> bool:
    norm = _norm(text)
    return _norm(REFUSAL) in norm or norm.startswith("i cannot answer")


def cited_ids(text: str, n_contexts: int) -> list[int]:
    """1-based evidence ids cited as [n] in `text` that refer to one of `n_contexts` passages."""
    ids = {int(m) for m in _CITATION_RE.findall(text)}
    return sorted(i for i in ids if 1 <= i <= n_contexts)


# Called with each pipeline step as it finishes (the same dicts that end up in Answer.trace),
# so callers such as the streaming API can report progress before the answer is ready.
StepCallback = Callable[[dict[str, Any]], None]


class AnswerPipeline(Protocol):
    """Anything that turns a question into an `Answer` (baseline, agent, guarded wrapper)."""

    def answer(self, question: str, on_step: StepCallback | None = None) -> "Answer": ...

    def unload(self) -> None:
        """Release model memory held by the pipeline's LLMs."""
        ...


@dataclass
class Answer:
    question: str
    text: str
    contexts: list[SearchHit]
    candidates: list[SearchHit] = field(default_factory=list)  # pre-rerank, for evaluation
    usage: Usage = field(default_factory=Usage)
    timings: dict[str, float] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)  # agent steps, for debugging
    blocked_by: str | None = None  # guardrail that stopped or replaced the answer
    # Every passage the drafter saw across retries (agent); empty means just `contexts`.
    shown: list[SearchHit] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.blocked_by is not None or is_refusal(self.text)

    @property
    def cited_ids(self) -> list[int]:
        """1-based evidence ids cited in the answer that refer to real contexts."""
        return cited_ids(self.text, len(self.contexts))

    @property
    def cited_contexts(self) -> list[SearchHit]:
        return [self.contexts[i - 1] for i in self.cited_ids]
