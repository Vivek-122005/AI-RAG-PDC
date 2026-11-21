"""Write metrics.json and a human-readable report.md for a run."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentic_rag.evaluation import records
from agentic_rag.evaluation.dataset import GoldenItem
from agentic_rag.evaluation.metrics import ItemResult, compute_breakdown, load_results
from agentic_rag.evaluation.taxonomy import classify, taxonomy, trace_summary

SECTIONS = ("retrieval", "generation", "efficiency")


def write_report(items: Sequence[GoldenItem], run_dir: Path) -> dict[str, Any]:
    results = load_results(items, run_dir)
    metrics = compute_breakdown(results)
    metrics["failure_taxonomy"] = taxonomy(results)
    (run_dir / records.METRICS).write_text(json.dumps(metrics, indent=2))
    (run_dir / records.REPORT).write_text(render_markdown(run_dir.name, metrics, results))
    return metrics


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _metrics_table(section: str, metrics: dict[str, Any]) -> list[str]:
    types = list(metrics["by_type"])
    lines = [
        f"## {section.title()}",
        "",
        "| metric | overall | " + " | ".join(types) + " |",
        "|---" * (2 + len(types)) + "|",
    ]
    for name, value in metrics["overall"][section].items():
        per_type = [_fmt(metrics["by_type"][t][section].get(name)) for t in types]
        lines.append(f"| {name} | {_fmt(value)} | " + " | ".join(per_type) + " |")
    return [*lines, ""]


def render_markdown(name: str, metrics: dict[str, Any], results: Sequence[ItemResult]) -> str:
    o = metrics["overall"]
    lines = [
        f"# Eval report: `{name}`",
        "",
        f"{o['n_items']} items: {o['n_answerable']} answerable, "
        f"{o['n_unanswerable']} unanswerable.",
        "",
    ]
    for section in SECTIONS:
        lines += _metrics_table(section, metrics)

    by_cause = metrics.get("failure_taxonomy") or taxonomy(results)
    lines += [
        "## Failure taxonomy",
        "",
        "Where each failed item broke (see `evaluation/taxonomy.py`).",
        "",
        "| cause | count | items |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {len(ids)} | {', '.join(ids)} |" for c, ids in by_cause.items() if ids]
    lines.append("")

    failures = [r for r in results if r.is_failure]
    lines += [f"## Failures ({len(failures)})", ""]
    for r in failures:
        lines += [
            f"### {r.item.id} ({r.item.type}): {classify(r) or 'low score'}",
            f"- **Question:** {r.item.question}",
            f"- **Reference:** {r.item.reference_answer or '(unanswerable)'}",
            f"- **Answer:** {r.answer['answer']}",
            f"- **Judge:** {r.judgment.get('judge_reasoning', '-')}",
            *(
                [f"- **Trace:** `{trace_summary(r.answer['trace'])}`"]
                if r.answer.get("trace")
                else []
            ),
            "",
        ]
    return "\n".join(lines)
