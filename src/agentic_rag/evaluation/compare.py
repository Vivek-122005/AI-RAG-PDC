"""Side-by-side comparison of evaluation runs, read from their metrics.json files."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentic_rag.evaluation import records

# (section, metric, label) rows of the comparison table, in reading order.
ROWS: tuple[tuple[str, str, str], ...] = (
    ("retrieval", "evidence_recall@k", "evidence recall@k"),
    ("generation", "accuracy(score>=4)", "accuracy (correctness ≥ 4)"),
    ("generation", "fully_faithful_rate", "fully faithful"),
    ("generation", "false_refusal_rate", "false refusal rate"),
    ("generation", "correct_refusal_rate", "correct refusal rate"),
    ("efficiency", "latency_p50_s", "latency p50 (s)"),
    ("efficiency", "latency_p95_s", "latency p95 (s)"),
    ("efficiency", "prompt_tokens_mean", "prompt tokens / question"),
)


def load_metrics(run_dirs: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {d.name: json.loads((d / records.METRICS).read_text()) for d in run_dirs}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0f}" if value >= 100 else f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _get(metrics: Mapping[str, Any], scope: str | None, section: str, name: str) -> Any:
    block = metrics["overall"] if scope is None else metrics["by_type"].get(scope, {})
    return block.get(section, {}).get(name)


def compare_markdown(runs: Mapping[str, Mapping[str, Any]]) -> str:
    """One overall table, then accuracy and refusal behaviour per question type."""
    names = list(runs)
    header = "| metric | " + " | ".join(f"`{n}`" for n in names) + " |"
    rule = "|---" * (len(names) + 1) + "|"

    lines = ["## Overall", "", header, rule]
    for section, metric, label in ROWS:
        cells = [_fmt(_get(runs[n], None, section, metric)) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    types = sorted({t for m in runs.values() for t in m["by_type"]})
    lines += ["", "## By question type", "", header, rule]
    for qtype in types:
        metric = "correct_refusal_rate" if qtype == "unanswerable" else "accuracy(score>=4)"
        label = "correct refusal" if qtype == "unanswerable" else "accuracy"
        cells = [_fmt(_get(runs[n], qtype, "generation", metric)) for n in names]
        lines.append(f"| {qtype} ({label}) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
