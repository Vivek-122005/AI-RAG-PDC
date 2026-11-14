"""Golden dataset schema (JSONL, one item per line).

Evidence is stored as verbatim quotes + source rather than chunk IDs, so the
dataset stays valid when chunking parameters change.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["single_hop", "multi_hop", "comparison", "unanswerable"]


class Evidence(BaseModel):
    source: str
    quote: str = Field(description="Verbatim span from the document that supports the answer")


class GoldenItem(BaseModel):
    id: str
    question: str
    reference_answer: str | None = None
    type: QuestionType
    answerable: bool = True
    evidence: list[Evidence] = Field(default_factory=list)
    verified: bool = False  # set True only after a human has checked it
    origin: Literal["synthetic", "assistant", "human"] = "synthetic"
    notes: str = ""


def load_dataset(path: Path) -> list[GoldenItem]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                items.append(GoldenItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    return items


def save_dataset(items: list[GoldenItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
