"""On-disk records for an evaluation run (JSONL, append-only so stages can resume)."""

import json
from pathlib import Path
from typing import Any

from agentic_rag.domain.search import SearchHit

ANSWERS = "answers.jsonl"
JUDGMENTS = "judgments.jsonl"
CONFIG = "config.json"
METRICS = "metrics.json"
REPORT = "report.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hit_to_record(h: SearchHit) -> dict[str, Any]:
    return {"source": h.source, "text": h.text, "citation": h.citation, "score": h.score}


def hit_from_record(r: dict[str, Any]) -> SearchHit:
    return SearchHit(
        chunk_id="",
        score=float(r.get("score", 0.0)),
        text=r["text"],
        doc_id="",
        title="",
        source=r["source"],
        section_path=[],
        page_start=None,
        page_end=None,
    )
