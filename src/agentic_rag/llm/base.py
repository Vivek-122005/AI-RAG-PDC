"""The LLM interface every provider implements. Nodes depend on this, never on a vendor SDK."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.latency_s + other.latency_s,
        )


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    model: str

    def generate(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> LLMResult: ...

    def generate_structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[T, Usage]: ...

    def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]: ...

    def unload(self) -> None:
        """Release model memory if the provider holds any (no-op for hosted APIs)."""
        ...
