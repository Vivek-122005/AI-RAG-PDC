"""Test doubles for the LLM interface."""

from collections.abc import Callable, Iterator, Sequence

from agentic_rag.llm.base import LLMResult, Message, T, Usage


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
        return schema.model_validate_json(self._text(messages)), Usage(10, 5, 0.01)

    def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]:
        yield from self._text(messages).split(" ")

    def unload(self) -> None:
        pass
