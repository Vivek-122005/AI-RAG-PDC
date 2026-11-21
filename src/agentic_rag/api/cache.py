"""Answer cache: an agent answer costs ~1 minute of local LLM time, a repeat costs nothing.

Keyed by tenant, pipeline and normalised question. Any document change clears it, since a
new or deleted document can change any answer (and public documents affect every tenant).
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def _normalise(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?!. ")


@dataclass
class AnswerCache:
    max_items: int = 256
    ttl_s: float = 3600.0
    _items: OrderedDict[tuple[str, str, str], tuple[float, dict[str, Any]]] = field(
        default_factory=OrderedDict
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, tenant: str, pipeline: str, question: str) -> dict[str, Any] | None:
        key = (tenant, pipeline, _normalise(question))
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.ttl_s:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def put(self, tenant: str, pipeline: str, question: str, value: dict[str, Any]) -> None:
        if self.max_items <= 0:
            return
        key = (tenant, pipeline, _normalise(question))
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
