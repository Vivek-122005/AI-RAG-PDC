"""Evaluation stages. Each stage reads/writes JSONL under the run directory and resumes
where it stopped, so long local runs survive interruptions.

  generate → answers.jsonl    system under test (generator model loaded)
  judge    → judgments.jsonl  judge model loaded, generator unloaded (fits 8 GB RAM)

Metrics and reports are computed from these files (see metrics.py / report.py).
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import structlog

from agentic_rag.evaluation import records
from agentic_rag.evaluation.dataset import GoldenItem
from agentic_rag.evaluation.judge import judge_correctness, judge_faithfulness
from agentic_rag.llm.base import LLM, LLMError
from agentic_rag.pipelines.answer import Answer

log = structlog.get_logger()

AnswerFn = Callable[[str], Answer]


def run_generate(
    items: Sequence[GoldenItem], answer_fn: AnswerFn, run_dir: Path, config: dict[str, Any]
) -> None:
    path = run_dir / records.ANSWERS
    done = {r["id"] for r in records.read_jsonl(path)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / records.CONFIG).write_text(json.dumps(config, indent=2))

    todo = [i for i in items if i.id not in done]
    for n, item in enumerate(todo, 1):
        try:
            ans = answer_fn(item.question)
        except LLMError as exc:
            log.error("generate_failed", id=item.id, error=str(exc))
            continue
        records.append_jsonl(
            path,
            {
                "id": item.id,
                "answer": ans.text,
                "refused": ans.refused,
                "cited_ids": ans.cited_ids,
                "contexts": [records.hit_to_record(h) for h in ans.contexts],
                "candidates": [records.hit_to_record(h) for h in ans.candidates],
                "usage": vars(ans.usage),
                "timings": ans.timings,
            },
        )
        log.info(
            "generated",
            n=n,
            of=len(todo),
            id=item.id,
            seconds=round(ans.timings.get("total_s", 0), 1),
        )


def run_judge(items: Sequence[GoldenItem], judge: LLM, run_dir: Path) -> None:
    by_id = {i.id: i for i in items}
    path = run_dir / records.JUDGMENTS
    done = {r["id"] for r in records.read_jsonl(path)}
    pending = [a for a in records.read_jsonl(run_dir / records.ANSWERS) if a["id"] not in done]

    for n, answer in enumerate(pending, 1):
        item = by_id.get(answer["id"])
        if item is None:
            continue
        row: dict[str, Any] = {"id": item.id}
        # Only answered, answerable items need an LLM judge; refusals are scored exactly.
        if item.answerable and not answer["refused"]:
            try:
                score, claims, _ = judge_faithfulness(
                    judge, answer["answer"], [c["text"] for c in answer["contexts"]]
                )
                verdict, _ = judge_correctness(
                    judge, item.question, item.reference_answer or "", answer["answer"]
                )
            except LLMError as exc:
                log.error("judge_failed", id=item.id, error=str(exc))
                continue
            row |= {
                "faithfulness": score,
                "claims": [c.model_dump() for c in claims],
                "correctness": verdict.score,
                "addresses_question": verdict.addresses_question,
                "judge_reasoning": verdict.reasoning,
            }
        records.append_jsonl(path, row)
        log.info("judged", n=n, of=len(pending), id=item.id)
