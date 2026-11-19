"""Query commands: `rag search` (retrieval only) and `rag ask` (full answer)."""

import argparse
import json
import textwrap

from agentic_rag.core.config import Settings

MODES = ["hybrid", "dense", "sparse"]


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("search", help="raw retrieval results (no LLM)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.set_defaults(handler=search)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.add_argument("--pipeline", choices=["agent", "baseline"], default="agent")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--trace", action="store_true", help="print the agent's steps")
    p.set_defaults(handler=ask)


def search(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.factory import make_store

    for i, hit in enumerate(
        make_store(settings).search(args.query, limit=args.k, mode=args.mode), 1
    ):
        snippet = hit.text[:500] + ("…" if len(hit.text) > 500 else "")
        print(f"\n[{i}] score={hit.score:.4f}  {hit.citation}")
        print(textwrap.indent(textwrap.fill(snippet, 100), "    "))
    return 0


def ask(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.factory import make_agent, make_baseline

    rerank = not args.no_rerank
    rag = (
        make_agent(settings, rerank=rerank, mode=args.mode)
        if args.pipeline == "agent"
        else make_baseline(settings, rerank=rerank, mode=args.mode)
    )
    answer = rag.answer(args.question)
    if args.trace:
        for step in answer.trace:
            details = {k: v for k, v in step.items() if k not in ("node", "seconds")}
            print(f"- {step['node']:<9} {step.get('seconds', 0):5.1f}s  {json.dumps(details)}")
        print()
    print(textwrap.fill(answer.text, 100))
    print("\nSources:")
    for i, hit in enumerate(answer.contexts, 1):
        marker = "*" if i in answer.cited_ids else " "
        print(f" {marker}[{i}] {hit.citation}")
    t = answer.timings
    stages = ", ".join(f"{k.removesuffix('_s')} {v:.1f}s" for k, v in t.items() if k != "total_s")
    print(
        f"\n({t.get('total_s', 0):.1f}s: {stages}; "
        f"{answer.usage.prompt_tokens}+{answer.usage.completion_tokens} tokens)"
    )
    return 0
