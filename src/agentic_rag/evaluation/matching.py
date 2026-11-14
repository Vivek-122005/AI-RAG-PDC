"""Decide whether a retrieved chunk contains a piece of gold evidence.

A chunk "covers" an evidence quote if it is from the same source and either
contains the normalized quote, or shares >= `threshold` of the quote's tokens
(tolerates PDF extraction differences and quotes that straddle chunk boundaries).
"""

import re
from collections.abc import Sequence

from agentic_rag.domain.search import SearchHit
from agentic_rag.evaluation.dataset import Evidence

_WORD = re.compile(r"\w+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def covers(hit_text: str, hit_source: str, ev: Evidence, threshold: float = 0.8) -> bool:
    if hit_source != ev.source:
        return False
    quote, chunk = normalize(ev.quote), normalize(hit_text)
    if not quote:
        return False
    if quote in chunk:
        return True
    q_tokens = quote.split()
    chunk_tokens = set(chunk.split())
    return sum(t in chunk_tokens for t in q_tokens) / len(q_tokens) >= threshold


def first_relevant_rank(hits: Sequence[SearchHit], evidence: Sequence[Evidence]) -> int | None:
    """1-based rank of the first hit covering any evidence, or None."""
    for rank, h in enumerate(hits, 1):
        if any(covers(h.text, h.source, ev) for ev in evidence):
            return rank
    return None


def evidence_recall(hits: Sequence[SearchHit], evidence: Sequence[Evidence]) -> float:
    """Fraction of evidence quotes covered by at least one hit (matters for multi-hop)."""
    if not evidence:
        return 0.0
    found = sum(any(covers(h.text, h.source, ev) for h in hits) for ev in evidence)
    return found / len(evidence)
