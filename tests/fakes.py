"""Test doubles for the LLM and retriever interfaces."""

from collections.abc import Callable, Iterator, Sequence

from pydantic import ValidationError

from agentic_rag.domain.search import SearchHit
from agentic_rag.llm.base import LLMError, LLMResult, Message, T, Usage
from agentic_rag.retrieval.retriever import RetrievalResult


class FakeLLM:
    """Returns canned text, or the result of a function of the last user message."""

    model = "fake"

    def __init__(self, reply: str | Callable[[str], str] = "ok") -> None:
        self.reply = reply
        self.calls: list[list[Message]] = []

    def _text(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        last = messages[-1].content
        return self.reply(last) if callable(self.reply) else self.reply

    def generate(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> LLMResult:
        return LLMResult(self._text(messages), self.model, Usage(10, 5, 0.01))

    def generate_structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[T, Usage]:
        try:  # mirror the real adapter: bad structured output surfaces as LLMError
            return schema.model_validate_json(self._text(messages)), Usage(10, 5, 0.01)
        except ValidationError as exc:
            raise LLMError(str(exc)) from exc

    def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]:
        yield from self._text(messages).split(" ")

    def unload(self) -> None:
        pass


def make_hit(chunk_id: str, text: str = "", source: str = "doc.md") -> SearchHit:
    return SearchHit(
        chunk_id, 1.0, text or f"text of {chunk_id}", source, source, source, [], None, None
    )


class FakeRetriever:
    """Returns preset hits per query (unknown queries get none) and records every query."""

    def __init__(self, hits: dict[str, list[SearchHit]]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        found = self.hits.get(query, [])
        return RetrievalResult(found, found)
