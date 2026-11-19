"""Pure functions for combining evidence from several queries.

Kept free of LLM and graph code so the selection rules are easy to test and reason about.
"""

import re
from collections.abc import Sequence
from typing import Literal

from agentic_rag.domain.search import SearchHit

Coverage = Literal["all", "partial", "none"]  # how many named subjects passages mention


def interleave(ranked_lists: Sequence[Sequence[SearchHit]], limit: int) -> list[SearchHit]:
    """Round-robin merge of per-query rankings, dropping duplicate chunks.

    Taking the best hit of every query before the second-best of any guarantees that each
    sub-question (e.g. each paper in a comparison) is represented in the final evidence.
    """
    merged: list[SearchHit] = []
    seen: set[str] = set()
    depth = max((len(r) for r in ranked_lists), default=0)
    for rank in range(depth):
        for hits in ranked_lists:
            if rank < len(hits) and hits[rank].chunk_id not in seen:
                seen.add(hits[rank].chunk_id)
                merged.append(hits[rank])
                if len(merged) == limit:
                    return merged
    return merged


def carry_forward(
    previous: Sequence[SearchHit],
    cited: Sequence[int],
    new: Sequence[SearchHit],
    limit: int,
) -> list[SearchHit]:
    """Evidence for a retry: cited passages first, then new hits, then the rest of the old."""
    cited_hits = [previous[i - 1] for i in cited if 1 <= i <= len(previous)]
    uncited = [h for i, h in enumerate(previous, 1) if i not in set(cited)]
    return interleave([[*cited_hits, *new, *uncited]], limit)


def unique_queries(queries: Sequence[str], exclude: Sequence[str] = ()) -> list[str]:
    """Strip, drop empties and case-insensitive duplicates (including already-run queries)."""
    seen = {q.strip().lower() for q in exclude}
    out: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out


_JOINERS = re.compile(r"[-.]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NAME_LIKE = re.compile(r"[A-Z0-9]")


def _words(text: str, join: bool) -> str:
    """Lowercase words padded with spaces for whole-word search. With `join`, hyphens and
    dots are removed ("Self-RAG" → "selfrag"); without, they split words ("self rag")."""
    text = text.lower()
    text = _JOINERS.sub("" if join else " ", text)
    return f" {_NON_ALNUM.sub(' ', text).strip()} "


def _name_tokens(subject: str) -> list[str]:
    """The name-like words of a subject ("RAG paper" → ["rag"]); generic words are dropped."""
    return [_words(t, join=True).strip() for t in subject.split() if _NAME_LIKE.search(t)]


def missing_subjects(subjects: Sequence[str], hits: Sequence[SearchHit]) -> list[str]:
    """Subjects the question names that no given passage mentions (title or text).

    A cheap, deterministic guard against answering about the wrong entity, e.g. citing the
    RAG paper's GPUs for a question about Ragas. Only name-like words (with a capital or a
    digit) are matched, as whole words, ignoring case and hyphens: "Self-RAG" matches
    "SELF-RAG", "RAG paper" matches "RAG-Token", but "RAG" never matches "Ragas".
    Subjects without any name-like word cannot be checked and are skipped.
    """
    texts = [f"{h.title} {h.text}" for h in hits]
    haystack = "".join(_words(t, join=True) + _words(t, join=False) for t in texts)
    return [
        s
        for s in subjects
        if (names := _name_tokens(s)) and any(f" {n} " not in haystack for n in names)
    ]


def subject_coverage(subjects: Sequence[str], hits: Sequence[SearchHit]) -> Coverage:
    """ "all", "partial" or "none" of the checkable subjects are mentioned by `hits`."""
    checkable = [s for s in subjects if _name_tokens(s)]
    missing = missing_subjects(checkable, hits)
    if not missing:
        return "all"
    return "none" if len(missing) == len(checkable) else "partial"
