"""Baseline ("naive") RAG: one retrieval pass, one generation pass, no self-check.

This is the control group the agentic pipeline must beat.
"""

import time
from collections.abc import Sequence

from agentic_rag.core.telemetry import traced
from agentic_rag.domain.search import SearchHit
from agentic_rag.llm.base import LLM, Message
from agentic_rag.pipelines.answer import Answer, StepCallback
from agentic_rag.pipelines.prompts import ANSWER_SYSTEM, REFUSAL, answer_user_prompt
from agentic_rag.retrieval.retriever import EvidenceRetriever


class BaselineRAG:
    def __init__(
        self, retriever: EvidenceRetriever, llm: LLM, max_answer_tokens: int = 400
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_answer_tokens = max_answer_tokens

    @staticmethod
    def messages(question: str, contexts: Sequence[SearchHit]) -> list[Message]:
        return [
            Message("system", ANSWER_SYSTEM),
            Message("user", answer_user_prompt(question, contexts)),
        ]

    def unload(self) -> None:
        self.llm.unload()

    @traced("baseline.answer")
    def answer(self, question: str, on_step: StepCallback | None = None) -> Answer:
        emit = on_step or (lambda _: None)
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question)
        t1 = time.perf_counter()
        emit({"node": "retrieve", "hits": len(retrieved.contexts), "seconds": t1 - t0})
        if not retrieved.contexts:
            return Answer(
                question, REFUSAL, [], retrieved.candidates, timings={"retrieve_s": t1 - t0}
            )

        res = self.llm.generate(
            self.messages(question, retrieved.contexts), max_tokens=self.max_answer_tokens
        )
        t2 = time.perf_counter()
        emit({"node": "generate", "seconds": t2 - t1})
        return Answer(
            question=question,
            text=res.text.strip(),
            contexts=retrieved.contexts,
            candidates=retrieved.candidates,
            usage=res.usage,
            timings={"retrieve_s": t1 - t0, "generate_s": t2 - t1, "total_s": t2 - t0},
        )
