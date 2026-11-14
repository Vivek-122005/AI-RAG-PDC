"""LLM-as-judge metrics, run with a *different, larger* model than the generator.

- Faithfulness: split the answer into atomic claims, check each against the
  contexts the generator saw. score = supported / total claims.
- Correctness: compare the answer with the reference answer (1-5 scale).
- Relevance: does the answer address the question that was asked?

Judges are themselves fallible; Phase 3 measures judge/critic agreement with
human labels before trusting them for decisions.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from agentic_rag.llm.base import LLM, Message, Usage


class Claim(BaseModel):
    claim: str
    supported: bool = Field(description="True only if the evidence explicitly states this")


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim]


class CorrectnessVerdict(BaseModel):
    reasoning: str = Field(description="One or two sentences")
    score: int = Field(ge=1, le=5, description="5 = fully correct and complete, 1 = wrong")
    addresses_question: bool = Field(description="Does the answer respond to what was asked?")


FAITHFULNESS_PROMPT = """You check whether an answer is supported by evidence.

Evidence:
{evidence}

Answer:
\"\"\"{answer}\"\"\"

Split the answer into short atomic factual claims (ignore citation markers like [1]).
For each claim, set supported=true ONLY if the evidence explicitly states it.
Claims using outside knowledge, even if true, are NOT supported."""

CORRECTNESS_PROMPT = """You grade an answer against a reference answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Score 1-5: 5 = same meaning as the reference and complete; 4 = correct but missing a
minor detail; 3 = partially correct; 2 = mostly wrong; 1 = wrong or about something else.
Also say whether the candidate addresses the question that was asked (not a different entity or topic)."""


def judge_faithfulness(
    llm: LLM, answer: str, contexts: Sequence[str]
) -> tuple[float, list[Claim], Usage]:
    evidence = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    verdict, usage = llm.generate_structured(
        [Message("user", FAITHFULNESS_PROMPT.format(evidence=evidence, answer=answer))],
        FaithfulnessVerdict,
        max_tokens=800,
    )
    claims = verdict.claims
    score = sum(c.supported for c in claims) / len(claims) if claims else 0.0
    return score, claims, usage


def judge_correctness(
    llm: LLM, question: str, reference: str, answer: str
) -> tuple[CorrectnessVerdict, Usage]:
    return llm.generate_structured(
        [
            Message(
                "user",
                CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer),
            )
        ],
        CorrectnessVerdict,
        max_tokens=300,
    )
