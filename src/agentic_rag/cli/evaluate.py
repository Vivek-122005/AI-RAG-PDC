"""Evaluation commands: `rag eval {synth,generate,judge,report,all,retrieval,compare}`."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_rag.core.config import Settings

if TYPE_CHECKING:
    from agentic_rag.evaluation.dataset import GoldenItem
    from agentic_rag.pipelines.answer import AnswerPipeline
    from agentic_rag.retrieval.retriever import SearchMode

RESULTS_DIR = Path("eval/results")


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("eval", help="evaluation harness")
    stages = p.add_subparsers(dest="stage", required=True, metavar="<stage>")

    s = stages.add_parser("synth", help="draft golden questions from the index (needs review)")
    s.add_argument("--single", type=int, default=30)
    s.add_argument("--multi", type=int, default=16)
    s.add_argument("--model", help="defaults to GENERATOR_MODEL")
    s.add_argument("--out", type=Path, default=Path("eval/datasets/golden.draft.jsonl"))
    s.set_defaults(handler=synth)

    s = stages.add_parser("gate", help="CI quality gate: retrieval metrics vs thresholds (no LLM)")
    s.add_argument("--dataset", type=Path, default=Path("eval/datasets/golden.jsonl"))
    s.add_argument("--min-hit-rate", type=float, default=0.80, help="hit_rate@k floor")
    s.add_argument("--min-recall", type=float, default=0.80, help="evidence_recall@k floor")
    s.add_argument("--out", type=Path, help="also write the metrics as JSON")
    s.set_defaults(handler=gate)

    s = stages.add_parser("compare", help="compare finished runs side by side")
    s.add_argument("runs", nargs="+", help=f"run names under {RESULTS_DIR}/")
    s.add_argument("--out", type=Path, help="also write the table to this markdown file")
    s.set_defaults(handler=compare)

    for name, help_text in [
        ("generate", "answer every question with the system under test"),
        ("judge", "score answers with the judge model, then report"),
        ("report", "compute metrics and write report.md"),
        ("all", "generate, judge and report"),
        ("retrieval", "retrieval-only ablation (no LLM)"),
    ]:
        s = stages.add_parser(name, help=help_text)
        s.add_argument("--dataset", type=Path, default=Path("eval/datasets/golden.jsonl"))
        s.add_argument("--run", required=True, help=f"results go to {RESULTS_DIR}/<run>/")
        s.add_argument("--only-verified", action="store_true")
        if name in ("generate", "all"):
            s.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
            s.add_argument("--no-rerank", action="store_true")
            s.add_argument("--pipeline", choices=["baseline", "agent"], default="baseline")
            s.add_argument("--no-decompose", action="store_true", help="agent ablation")
            s.add_argument("--no-critic", action="store_true", help="agent ablation")
            s.add_argument("--max-retries", type=int, default=2, help="agent re-plan budget")
            s.add_argument("--no-guardrails", action="store_true", help="ablation")
        s.set_defaults(handler=run_stage)


def synth(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.evaluation.dataset import save_dataset
    from agentic_rag.evaluation.synth import load_chunks, synthesize
    from agentic_rag.factory import make_llm, make_qdrant_client

    chunks = load_chunks(make_qdrant_client(settings), settings.qdrant_collection)
    llm = make_llm(settings, args.model or settings.generator_model)
    items = synthesize(llm, chunks, args.single, args.multi)
    save_dataset(items, args.out)
    print(f"wrote {len(items)} draft items to {args.out} (review before use)")
    return 0


def compare(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.evaluation.compare import compare_markdown, load_metrics

    table = compare_markdown(load_metrics([RESULTS_DIR / r for r in args.runs]))
    print(table)
    if args.out:
        args.out.write_text(table)
    return 0


def gate(args: argparse.Namespace, settings: Settings) -> int:
    """Fail (exit 1) when the production retriever falls below the floors.

    Retrieval is deterministic and needs no LLM, so this runs in CI on every change;
    answer quality (judge-scored) stays a local, manual eval.
    """
    from agentic_rag.evaluation.ablation import run_retrieval_ablation
    from agentic_rag.evaluation.dataset import load_dataset
    from agentic_rag.factory import make_retriever

    row = run_retrieval_ablation(load_dataset(args.dataset), {"prod": make_retriever(settings)})[0]
    k = settings.retrieve_top_k
    checks = {f"hit_rate@{k}": args.min_hit_rate, f"evidence_recall@{k}": args.min_recall}
    failed = [m for m, floor in checks.items() if row[m] < floor]
    for metric, floor in checks.items():
        mark = "FAIL" if metric in failed else "ok"
        print(f"{mark:4}  {metric:20} {row[metric]:.3f}  (floor {floor:.2f})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row | {"floors": checks, "passed": not failed}, indent=2))
    return 1 if failed else 0


def run_stage(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.evaluation.runner import EvalAborted

    try:
        return _run_stage(args, settings)
    except EvalAborted as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_stage(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.evaluation.dataset import load_dataset

    items = load_dataset(args.dataset)
    if args.only_verified:
        items = [i for i in items if i.verified]
    run_dir = RESULTS_DIR / args.run

    if args.stage in ("generate", "all"):
        _generate(args, settings, items, run_dir)
    if args.stage in ("judge", "all"):
        _judge(settings, items, run_dir)
    if args.stage in ("report", "judge", "all"):
        from agentic_rag.evaluation.report import write_report

        metrics = write_report(items, run_dir)
        print(json.dumps(metrics["overall"], indent=2))
        print(f"report: {run_dir / 'report.md'}")
    if args.stage == "retrieval":
        _retrieval_ablation(settings, items, run_dir)
    return 0


def _generate(
    args: argparse.Namespace, settings: Settings, items: list["GoldenItem"], run_dir: Path
) -> None:
    from agentic_rag.evaluation.runner import run_generate
    from agentic_rag.factory import make_agent, make_baseline
    from agentic_rag.pipelines.agent import AgentConfig

    rerank = not args.no_rerank
    guardrails = settings.guardrails_enabled and not args.no_guardrails
    config: dict[str, Any] = {"pipeline": args.pipeline, "guardrails": guardrails}
    rag: AnswerPipeline
    if args.pipeline == "agent":
        agent_config = AgentConfig(
            decompose=not args.no_decompose,
            critic=not args.no_critic,
            max_retries=args.max_retries,
        )
        rag = make_agent(
            settings, agent_config, rerank=rerank, mode=args.mode, guardrails=guardrails
        )
        config |= {"agent": asdict(agent_config), "critic": settings.critic_model}
    else:
        rag = make_baseline(settings, rerank=rerank, mode=args.mode, guardrails=guardrails)
    config |= {
        "generator": settings.generator_model,
        "mode": args.mode,
        "reranker": None if args.no_rerank else settings.reranker_model,
        "candidates": settings.retrieve_candidates,
        "top_k": settings.retrieve_top_k,
        "dataset": str(args.dataset),
        "n_items": len(items),
    }
    run_generate(items, rag.answer, run_dir, config)
    rag.unload()  # free RAM before the judge model loads


def _judge(settings: Settings, items: list["GoldenItem"], run_dir: Path) -> None:
    from agentic_rag.evaluation.runner import run_judge
    from agentic_rag.factory import make_llm

    judge = make_llm(settings, settings.judge_model)
    run_judge(items, judge, run_dir)
    judge.unload()


def _retrieval_ablation(settings: Settings, items: list["GoldenItem"], run_dir: Path) -> None:
    from agentic_rag.evaluation.ablation import run_retrieval_ablation, to_markdown
    from agentic_rag.factory import make_reranker, make_store
    from agentic_rag.retrieval.retriever import RetrievalConfig, Retriever

    store = make_store(settings)
    n, k = settings.retrieve_candidates, settings.retrieve_top_k
    rerankers = {
        "none": make_reranker(settings, enabled=False),
        "minilm-l12": make_reranker(settings),
        "jina-tiny": make_reranker(settings, model="jinaai/jina-reranker-v1-tiny-en"),
    }
    grid: list[tuple[SearchMode, str]] = [
        ("dense", "none"), ("sparse", "none"), ("hybrid", "none"),
        ("dense", "minilm-l12"), ("hybrid", "minilm-l12"), ("hybrid", "jina-tiny"),
    ]  # fmt: skip
    retrievers = {
        f"{mode} / {rr}": Retriever(store, rerankers[rr], RetrievalConfig(mode, n, k))
        for mode, rr in grid
    }
    table = to_markdown(run_retrieval_ablation(items, retrievers))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "retrieval_ablation.md").write_text(table)
    print(table)
