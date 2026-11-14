"""Synthesize candidate golden questions from indexed chunks.

Output is a *draft*: every item has verified=False and must be reviewed by a human
before it is trusted. Quotes are checked to be verbatim (after normalization).
"""

import random
from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from agentic_rag.evaluation.dataset import Evidence, GoldenItem
from agentic_rag.evaluation.matching import normalize
from agentic_rag.llm.base import LLM, LLMError, Message


@dataclass
class StoredChunk:
    source: str
    title: str
    section: str
    text: str


class SingleHopQA(BaseModel):
    question: str = Field(
        description="Standalone question; name the paper/system, never 'the paper'"
    )
    answer: str = Field(description="Short answer, 1-2 sentences")
    quote: str = Field(
        description="Exact sentence copied from the passage that supports the answer"
    )


class MultiHopQA(BaseModel):
    question: str = Field(description="Standalone question that needs BOTH passages to answer")
    answer: str
    quote_a: str = Field(description="Exact sentence copied from passage A")
    quote_b: str = Field(description="Exact sentence copied from passage B")


SINGLE_PROMPT = """You write evaluation questions for a document QA system.

Paper: {title}
Section: {section}
Passage:
\"\"\"{text}\"\"\"

Write ONE factual question that this passage answers precisely.
- The question must be understandable without seeing the passage: mention the paper or method by name.
- Prefer specific facts (numbers, components, definitions, design choices) over vague summaries.
- Do not ask about authors, affiliations, figures or table layout.
- "quote" must be copied character-for-character from the passage."""

MULTI_PROMPT = """You write evaluation questions for a document QA system.

Passage A (from "{title_a}", section "{section_a}"):
\"\"\"{text_a}\"\"\"

Passage B (from "{title_b}", section "{section_b}"):
\"\"\"{text_b}\"\"\"

Write ONE question whose answer requires combining a fact from passage A with a fact
from passage B (e.g. a comparison, or a two-step question).
- Mention papers/methods by name; the question must stand alone.
- quote_a / quote_b must be copied character-for-character from each passage."""


def load_chunks(client: QdrantClient, collection: str, min_tokens: int = 120) -> list[StoredChunk]:
    out: list[StoredChunk] = []
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True)
        for p in points:
            pl = p.payload or {}
            if int(pl.get("token_count", 0)) < min_tokens:
                continue
            section = " > ".join(pl.get("section_path") or [])
            if any(
                s in section.lower() for s in ("acknowledg", "appendix", "instruction", "prompt")
            ):
                continue
            out.append(StoredChunk(str(pl["source"]), str(pl["title"]), section, str(pl["text"])))
        if offset is None:
            return out


def _verbatim(quote: str, text: str) -> bool:
    q = normalize(quote)
    return len(q.split()) >= 4 and q in normalize(text)


def synthesize(
    llm: LLM,
    chunks: list[StoredChunk],
    n_single: int,
    n_multi: int,
    seed: int = 7,
    id_prefix: str = "syn",
) -> list[GoldenItem]:
    rng = random.Random(seed)
    by_source: dict[str, list[StoredChunk]] = defaultdict(list)
    for c in chunks:
        by_source[c.source].append(c)
    sources = sorted(by_source)
    items: list[GoldenItem] = []

    # Single-hop: round-robin over documents so each is represented.
    attempts = 0
    while sum(i.type == "single_hop" for i in items) < n_single and attempts < n_single * 3:
        attempts += 1
        c = rng.choice(by_source[sources[attempts % len(sources)]])
        prompt = SINGLE_PROMPT.format(title=c.title, section=c.section, text=c.text)
        try:
            qa, _ = llm.generate_structured([Message("user", prompt)], SingleHopQA, temperature=0.3)
        except LLMError:
            continue
        if not _verbatim(qa.quote, c.text):
            continue
        print(f"single {len(items) + 1}/{n_single}: {qa.question[:80]}", flush=True)
        items.append(
            GoldenItem(
                id=f"{id_prefix}-s{len(items) + 1:03d}",
                question=qa.question.strip(),
                reference_answer=qa.answer.strip(),
                type="single_hop",
                evidence=[Evidence(source=c.source, quote=qa.quote.strip())],
            )
        )

    # Multi-hop: pairs from different documents (comparisons) or sections of one doc.
    attempts = 0
    while sum(i.type != "single_hop" for i in items) < n_multi and attempts < n_multi * 4:
        attempts += 1
        a = rng.choice(chunks)
        cross = attempts % 2 == 0
        pool = [c for c in chunks if (c.source != a.source) == cross and c.section != a.section]
        if not pool:
            continue
        b = rng.choice(pool)
        prompt = MULTI_PROMPT.format(
            title_a=a.title, section_a=a.section, text_a=a.text,
            title_b=b.title, section_b=b.section, text_b=b.text,
        )  # fmt: skip
        try:
            qa2, _ = llm.generate_structured([Message("user", prompt)], MultiHopQA, temperature=0.3)
        except LLMError:
            continue
        if not (_verbatim(qa2.quote_a, a.text) and _verbatim(qa2.quote_b, b.text)):
            continue
        print(
            f"multi {sum(i.type != 'single_hop' for i in items) + 1}/{n_multi}: {qa2.question[:80]}",
            flush=True,
        )
        items.append(
            GoldenItem(
                id=f"{id_prefix}-m{len(items) + 1:03d}",
                question=qa2.question.strip(),
                reference_answer=qa2.answer.strip(),
                type="comparison" if cross else "multi_hop",
                evidence=[
                    Evidence(source=a.source, quote=qa2.quote_a.strip()),
                    Evidence(source=b.source, quote=qa2.quote_b.strip()),
                ],
            )
        )
    return items
