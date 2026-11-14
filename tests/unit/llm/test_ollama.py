import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from agentic_rag.llm.base import LLMError, Message
from agentic_rag.llm.ollama import OllamaLLM

Handler = Callable[[httpx.Request], httpx.Response]


def make(handler: Handler) -> tuple[OllamaLLM, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        response: httpx.Response = handler(req)
        return response

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(wrapped))
    return OllamaLLM("m", client=client), seen


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"message": {"content": content}, "prompt_eval_count": 7, "eval_count": 3},
    )


def test_generate_payload_and_usage() -> None:
    llm, seen = make(lambda r: chat_response("hi"))
    res = llm.generate([Message("system", "s"), Message("user", "u")], max_tokens=5)
    assert res.text == "hi"
    assert (res.usage.prompt_tokens, res.usage.completion_tokens) == (7, 3)
    body = seen[0]
    assert body["messages"][1] == {"role": "user", "content": "u"}
    assert body["options"]["num_predict"] == 5
    assert "think" not in body  # never sent unless requested


class Out(BaseModel):
    x: int


def test_structured_sends_schema_and_retries_once() -> None:
    replies = iter(["not json", '{"x": 3}'])
    llm, seen = make(lambda r: chat_response(next(replies)))
    out, usage = llm.generate_structured([Message("user", "q")], Out)
    assert out.x == 3
    assert seen[0]["format"]["properties"]["x"]["type"] == "integer"
    assert len(seen) == 2 and usage.prompt_tokens == 14


def test_structured_gives_up_after_two() -> None:
    llm, _ = make(lambda r: chat_response("nope"))
    with pytest.raises(LLMError):
        llm.generate_structured([Message("user", "q")], Out)


def test_http_error_wrapped() -> None:
    llm, _ = make(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError):
        llm.generate([Message("user", "q")])


def test_stream_yields_tokens() -> None:
    lines = "\n".join(
        json.dumps(x)
        for x in [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
    )
    llm, _ = make(lambda r: httpx.Response(200, text=lines))
    assert "".join(llm.stream([Message("user", "q")])) == "Hello"
