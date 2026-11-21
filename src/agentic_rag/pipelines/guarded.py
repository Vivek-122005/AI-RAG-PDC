"""Wrap any answer pipeline with input and output guardrails (defence in depth).

Layers, from outside in:
1. InputGuard  — length limits, direct prompt injection, PII redacted before the LLM sees it.
2. Retriever   — passages that carry injected instructions are dropped (see factory).
3. Prompts     — evidence is spotlighted in <evidence> blocks the text cannot break out of.
4. OutputGuard — fabricated citations removed, uncited answers refused, no system-prompt
                 leak, PII redacted from the final text.
"""

import time
from dataclasses import replace

from agentic_rag.guardrails import InputGuard, OutputGuard
from agentic_rag.pipelines.answer import Answer, AnswerPipeline
from agentic_rag.pipelines.prompts import REFUSAL

BLOCKED_MESSAGES = {
    "too_short": "Please ask a complete question.",
    "too_long": "Please shorten your question.",
    "prompt_injection": (
        "I can only answer questions about the indexed documents, and I can't follow "
        "instructions that change how I work."
    ),
}


class GuardedPipeline:
    def __init__(
        self, inner: AnswerPipeline, input_guard: InputGuard, output_guard: OutputGuard
    ) -> None:
        self.inner = inner
        self.input_guard = input_guard
        self.output_guard = output_guard

    def unload(self) -> None:
        self.inner.unload()

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        decision = self.input_guard.check(question)
        if not decision.allowed:
            reason = decision.reason or "blocked"
            step = {
                "node": "input_guard",
                "blocked": reason,
                "rules": decision.injection_rules,
                "seconds": time.perf_counter() - t0,
            }
            return Answer(
                decision.question,
                BLOCKED_MESSAGES.get(reason, REFUSAL),
                [],
                timings={"total_s": time.perf_counter() - t0},
                trace=[step],
                blocked_by=f"input:{reason}",
            )

        ans = self.inner.answer(decision.question)
        out = self.output_guard.check(ans.text, len(ans.contexts), refused=ans.refused)
        trace = [
            {"node": "input_guard", "redacted": decision.redacted, "seconds": 0.0},
            *ans.trace,
            {
                "node": "output_guard",
                "refused": out.reason,
                "removed_citations": out.removed_citations,
                "redacted": out.redacted,
                "seconds": 0.0,
            },
        ]
        if out.refuse:
            return replace(ans, text=REFUSAL, trace=trace, blocked_by=f"output:{out.reason}")
        return replace(ans, text=out.text, trace=trace)
