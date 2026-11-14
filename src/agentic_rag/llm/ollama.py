"""Ollama chat client (native HTTP API; no SDK needed).

Structured output uses Ollama's `format=<JSON schema>`, which constrains decoding
to the schema, then validates with Pydantic and retries once on failure.
"""

import contextlib
import json
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx
from pydantic import ValidationError

from agentic_rag.llm.base import LLMError, LLMResult, Message, T, Usage


class OllamaLLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        timeout_s: float = 120.0,
        keep_alive: str = "10m",
        think: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def _payload(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        stream: bool,
        fmt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        # Only send `think` when asked: non-thinking models reject the flag, and
        # thinking-only models ignore think=false (prefer an instruct variant instead).
        if self.think:
            payload["think"] = True
        if fmt is not None:
            payload["format"] = fmt
        return payload

    def _chat(self, payload: dict[str, Any]) -> LLMResult:
        start = time.perf_counter()
        try:
            r = self._client.post("/api/chat", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed ({self.model}): {exc}") from exc
        data = r.json()
        return LLMResult(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                latency_s=time.perf_counter() - start,
            ),
        )

    def generate(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> LLMResult:
        return self._chat(self._payload(messages, temperature, max_tokens, stream=False))

    def generate_structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[T, Usage]:
        fmt = schema.model_json_schema()
        usage = Usage()
        last_error: Exception | None = None
        for _ in range(2):
            res = self._chat(self._payload(messages, temperature, max_tokens, False, fmt))
            usage = usage + res.usage
            try:
                return schema.model_validate_json(res.text), usage
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
        raise LLMError(f"{self.model} returned invalid {schema.__name__}: {last_error}")

    def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Iterator[str]:
        payload = self._payload(messages, temperature, max_tokens, stream=True)
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if token := chunk.get("message", {}).get("content"):
                        yield token
                    if chunk.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama stream failed ({self.model}): {exc}") from exc

    def unload(self) -> None:
        """Free the model's memory now (useful on small machines between eval stages)."""
        with contextlib.suppress(httpx.HTTPError):
            self._client.post("/api/generate", json={"model": self.model, "keep_alive": 0})
