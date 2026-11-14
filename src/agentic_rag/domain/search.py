"""Search result type shared by retrieval, pipelines and evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    text: str
    doc_id: str
    title: str
    source: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    payload: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def citation(self) -> str:
        pages = ""
        if self.page_start is not None:
            pages = (
                f" p.{self.page_start}"
                if self.page_start == self.page_end
                else f" pp.{self.page_start}-{self.page_end}"
            )
        section = f" § {self.section_path[-1]}" if self.section_path else ""
        return f"{self.title}{section}{pages}"
