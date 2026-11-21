"""Structure-aware recursive chunking.

Strategy
--------
1. Never cross a section boundary: chunks inherit one clean `section_path`.
2. Pack whole paragraphs (Blocks) up to `target_tokens`.
3. Paragraphs larger than `max_tokens` are split on sentences, then on words.
4. Consecutive chunks in a section share ~`overlap_tokens` of trailing sentences,
   so a fact straddling a boundary is retrievable from either side.

Token counts are an approximation of the embedding model's WordPiece tokenizer
(bge-small truncates at 512 tokens, so chunks must stay well below that).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from agentic_rag.domain.documents import Block, Chunk, Document, chunk_id_for

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def approx_tokens(text: str) -> int:
    """Approximate WordPiece token count: words + punctuation, long words split."""
    return sum(1 + len(t) // 8 for t in _TOKEN_RE.findall(text))


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 350
    max_tokens: int = 450
    overlap_tokens: int = 50
    min_tokens: int = 20  # drop fragments smaller than this (stray labels, captions)


@dataclass
class _Unit:
    text: str
    tokens: int
    page: int | None


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split text that exceeds max_tokens: sentences first, then hard word windows."""
    out: list[str] = []
    for sent in split_sentences(text):
        if approx_tokens(sent) <= max_tokens:
            out.append(sent)
            continue
        words, buf = sent.split(), []
        for w in words:
            buf.append(w)
            if approx_tokens(" ".join(buf)) >= max_tokens:
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
    return out


def _units(blocks: Iterable[Block], cfg: ChunkingConfig) -> list[_Unit]:
    units: list[_Unit] = []
    for b in blocks:
        n = approx_tokens(b.text)
        if n <= cfg.target_tokens:
            units.append(_Unit(b.text, n, b.page))
        else:
            # Paragraph too big to pack whole: pack its sentences instead.
            units.extend(
                _Unit(s, approx_tokens(s), b.page) for s in _split_long(b.text, cfg.max_tokens)
            )
    return units


def _overlap_tail(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Trailing sentences of the previous chunk, up to overlap_tokens."""
    if overlap_tokens <= 0 or not units:
        return []
    sentences = [_Unit(s, approx_tokens(s), u.page) for u in units for s in split_sentences(u.text)]
    tail: list[_Unit] = []
    total = 0
    for s in reversed(sentences):
        if total + s.tokens > overlap_tokens:
            break
        tail.insert(0, s)
        total += s.tokens
    return tail


def _group_by_section(blocks: list[Block]) -> list[tuple[tuple[str, ...], list[Block]]]:
    groups: list[tuple[tuple[str, ...], list[Block]]] = []
    for b in blocks:
        if groups and groups[-1][0] == b.section_path:
            groups[-1][1].append(b)
        else:
            groups.append((b.section_path, [b]))
    return groups


def chunk_document(doc: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    chunks: list[Chunk] = []
    small: list[tuple[tuple[str, ...], list[_Unit]]] = []  # fragments under min_tokens

    def emit(section: tuple[str, ...], units: list[_Unit], keep_small: bool = False) -> None:
        text = " ".join(u.text for u in units).strip()
        tokens = approx_tokens(text)
        if not text or (tokens < cfg.min_tokens and not keep_small):
            small.append((section, units))
            return
        pages = [u.page for u in units if u.page is not None]
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(doc.doc_id, idx),
                doc_id=doc.doc_id,
                chunk_index=idx,
                text=text,
                token_count=tokens,
                section_path=section,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
            )
        )

    for section, blocks in _group_by_section(doc.blocks):
        current: list[_Unit] = []
        current_tokens = 0
        has_new = False  # avoid emitting a chunk that is only overlap
        for unit in _units(blocks, cfg):
            if current and current_tokens + unit.tokens > cfg.target_tokens and has_new:
                emit(section, current)
                current = _overlap_tail(current, cfg.overlap_tokens)
                current_tokens = sum(u.tokens for u in current)
                has_new = False
            current.append(unit)
            current_tokens += unit.tokens
            has_new = True
        if current and has_new:
            emit(section, current)
    if not chunks:  # a short document is all "fragments": keep them rather than lose it
        for section, units in list(small):
            emit(section, units, keep_small=True)
    return chunks


def embedding_text(doc_title: str, chunk: Chunk) -> str:
    """Text actually embedded: a short context header + the chunk.

    The header ("contextual chunk header") lets a chunk that says "our method"
    still match queries about the paper or section it belongs to.
    """
    header = " > ".join([doc_title, *chunk.section_path])
    return f"{header}\n\n{chunk.text}"
