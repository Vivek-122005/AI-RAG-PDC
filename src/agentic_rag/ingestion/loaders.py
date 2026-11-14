"""File loaders that turn PDF / Markdown / HTML / text into a list of Blocks.

Each loader preserves structure that matters for retrieval: paragraph boundaries,
page numbers and the heading path a paragraph lives under.
"""

import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from agentic_rag.domain.documents import Block, Document, doc_id_for, file_hash

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}

# Sections whose text is noise for QA (citations match almost every query lexically).
SKIP_SECTIONS = {"references", "bibliography"}

_NUMBERED_HEADING = re.compile(r"^(?P<num>(\d+(\.\d+)*|[A-Z](\.\d+)*))\.?\s+[A-Z]")
_KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
}


def clean_text(text: str) -> str:
    """Join hyphenated line breaks, collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("­", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _section_level(heading: str) -> int:
    m = _NUMBERED_HEADING.match(heading)
    return m.group("num").count(".") + 1 if m else 1


def _push_heading(path: list[str], heading: str) -> list[str]:
    level = _section_level(heading)
    return [*path[: level - 1], heading]


def _in_skipped_section(path: list[str]) -> bool:
    """True if any heading in the path is a references/bibliography section."""
    for heading in path:
        name = re.sub(r"^[\dA-Z](\.\d+)*\.?\s+(?=[A-Z])", "", heading).strip().lower()
        if name in SKIP_SECTIONS or heading.strip().lower() in SKIP_SECTIONS:
            return True
    return False


# ---------------------------------------------------------------- PDF


def _looks_like_pdf_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    words = text.split()
    if not words or len(words) > 12 or text.endswith((".", ",", ":", ";")):
        return False
    emphasized = bold or size > body_size + 0.8
    if not emphasized:
        return False
    # Only numbered ("3.2 Method") or well-known headings: unnumbered bold text is
    # too often a figure label or table header.
    return text.lower() in _KNOWN_HEADINGS or bool(_NUMBERED_HEADING.match(text))


def _normalize_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def load_pdf(path: Path) -> tuple[str, list[Block]]:
    raw: list[tuple[int, str, float, bool]] = []
    size_weights: Counter[float] = Counter()
    with pymupdf.open(path) as pdf:  # type: ignore[no-untyped-call]  # pymupdf is untyped
        n_pages = len(pdf)
        meta_title = ((pdf.metadata or {}).get("title") or "").strip()
        for page_no, page in enumerate(pdf, start=1):
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                # Skip rotated text (e.g. the vertical arXiv stamp).
                lines = [ln for ln in b["lines"] if abs(ln["dir"][0] - 1.0) < 1e-3]
                spans = [sp for ln in lines for sp in ln["spans"] if sp["text"].strip()]
                if not spans:
                    continue
                # Spans within a line are contiguous (small caps split into spans): join with "".
                text = clean_text(
                    "\n".join("".join(sp["text"] for sp in ln["spans"]) for ln in lines)
                )
                if not text or text.isdigit():
                    continue  # page numbers
                size = max(sp["size"] for sp in spans)
                bold = all(sp["flags"] & 16 for sp in spans)
                raw.append((page_no, text, size, bold))
                for sp in spans:
                    size_weights[round(sp["size"], 1)] += len(sp["text"])

    body_size = size_weights.most_common(1)[0][0] if size_weights else 10.0

    # Running headers/footers: short text repeated on many pages.
    page_counts: Counter[str] = Counter()
    for key in {(pg, _normalize_repeat(t)) for pg, t, _, _ in raw if len(t.split()) <= 10}:
        page_counts[key[1]] += 1
    repeated = {t for t, c in page_counts.items() if n_pages >= 3 and c >= max(3, n_pages * 0.3)}
    raw = [r for r in raw if _normalize_repeat(r[1]) not in repeated]

    # Title: the run of largest-font blocks at the top of page 1.
    title_idx: set[int] = set()
    first = [i for i, r in enumerate(raw) if r[0] == 1]
    if first:
        top = max(raw[i][2] for i in first)
        if top > body_size + 1.5:
            start = next(i for i in first if raw[i][2] == top)
            i = start
            while i < len(raw) and raw[i][0] == 1 and abs(raw[i][2] - top) < 0.5:
                title_idx.add(i)
                i += 1
    title = meta_title or " ".join(raw[i][1] for i in sorted(title_idx)) or path.stem

    blocks: list[Block] = []
    path_stack: list[str] = []
    for i, (page_no, text, size, bold) in enumerate(raw):
        if i in title_idx:
            continue
        if _looks_like_pdf_heading(text, size, bold, body_size):
            path_stack = _push_heading(path_stack, text)
            continue
        if _in_skipped_section(path_stack):
            continue
        blocks.append(Block(text=text, page=page_no, section_path=tuple(path_stack)))
    return title, blocks


# ---------------------------------------------------------------- Markdown

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def load_markdown(path: Path) -> tuple[str, list[Block]]:
    title = ""
    stack: list[str] = []
    blocks: list[Block] = []
    para: list[str] = []
    in_code = False

    def flush() -> None:
        if para:
            text = "\n".join(para).strip()
            if text and not _in_skipped_section(stack):
                blocks.append(Block(text=text, section_path=tuple(stack)))
            para.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            para.append(line)
            continue
        m = None if in_code else _MD_HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = heading
            stack = [*stack[: level - 1], heading]
        elif not in_code and not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    return title or path.stem, blocks


# ---------------------------------------------------------------- HTML


def load_html(path: Path) -> tuple[str, list[Block]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    stack: list[str] = []
    blocks: list[Block] = []
    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td"]):
        text = clean_text(el.get_text(" "))
        if not text:
            continue
        if el.name and el.name[0] == "h" and el.name[1:].isdigit():
            level = int(el.name[1:])
            title = title or (text if level == 1 else "")
            stack = [*stack[: level - 1], text]
        elif not _in_skipped_section(stack):
            blocks.append(Block(text=text, section_path=tuple(stack)))
    return title or path.stem, blocks


# ---------------------------------------------------------------- Plain text


def load_text(path: Path) -> tuple[str, list[Block]]:
    content = path.read_text(encoding="utf-8", errors="replace")
    paras = [clean_text(p) for p in re.split(r"\n\s*\n", content)]
    return path.stem, [Block(text=p) for p in paras if p]


_LOADERS: dict[str, Callable[[Path], tuple[str, list[Block]]]] = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".txt": load_text,
}


def load_document(path: Path, source: str | None = None) -> Document:
    """Load any supported file. `source` is the stable identifier (default: the path)."""
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"Unsupported file type: {path.name}")
    source = source or str(path)
    title, blocks = _LOADERS[suffix](path)
    return Document(
        doc_id=doc_id_for(source),
        source=source,
        title=title,
        content_hash=file_hash(path),
        file_type=suffix.lstrip("."),
        blocks=blocks,
    )
