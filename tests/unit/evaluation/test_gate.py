"""`rag eval gate`: passes or fails CI on retrieval floors."""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.cli import evaluate
from agentic_rag.core.config import Settings

DATASET = Path("eval/datasets/golden.jsonl")


@pytest.mark.parametrize(("hit", "expected"), [(0.9, 0), (0.5, 1)])
def test_gate_exit_code_follows_the_floors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hit: float, expected: int
) -> None:
    def fake_ablation(items: Any, retrievers: Any) -> list[dict[str, Any]]:
        return [{"config": "prod", "hit_rate@5": hit, "evidence_recall@5": 0.85}]

    monkeypatch.setattr("agentic_rag.evaluation.ablation.run_retrieval_ablation", fake_ablation)
    monkeypatch.setattr("agentic_rag.factory.make_retriever", lambda settings: None)
    out = tmp_path / "gate.json"
    args = argparse.Namespace(dataset=DATASET, min_hit_rate=0.8, min_recall=0.8, out=out)

    assert evaluate.gate(args, Settings(_env_file=None)) == expected
    assert json.loads(out.read_text())["passed"] is (expected == 0)
