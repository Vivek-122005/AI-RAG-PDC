import json
from pathlib import Path

from agentic_rag.evaluation.compare import compare_markdown, load_metrics


def metrics(accuracy: float, refusal: float) -> dict[str, object]:
    gen = {"accuracy(score>=4)": accuracy, "false_refusal_rate": 0.1}
    return {
        "overall": {"generation": gen, "efficiency": {"latency_p50_s": 120.0}},
        "by_type": {
            "single_hop": {"generation": gen},
            "unanswerable": {"generation": {"correct_refusal_rate": refusal}},
        },
    }


def test_compare_reads_runs_and_renders_side_by_side(tmp_path: Path) -> None:
    for name, acc, ref in [("base", 0.886, 0.9), ("agent", 0.914, 1.0)]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps(metrics(acc, ref)))
    md = compare_markdown(load_metrics([tmp_path / "base", tmp_path / "agent"]))
    assert "| metric | `base` | `agent` |" in md
    assert "| accuracy (correctness ≥ 4) | 0.886 | 0.914 |" in md
    assert "| latency p50 (s) | 120 | 120 |" in md
    assert "| unanswerable (correct refusal) | 0.9 | 1 |" in md
    assert "| evidence recall@k | - | - |" in md  # missing metrics render as "-"
