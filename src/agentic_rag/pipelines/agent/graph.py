"""Self-correcting RAG agent as a LangGraph state machine.

    plan ──► retrieve (one per query, fanned out) ──► gather ──► draft ──► critique
               ▲                                        ▲                   │
               │ new queries                     widen  │                   │ retry
               └───────────────────── replan ───────────┴───────────────────┘
                                                          accept / refuse ──► END

The planner splits multi-part questions into self-contained queries, each query is
retrieved separately so every part gets evidence, and the critic checks the draft for
groundedness and relevance. On a retry the agent searches the critic's follow-up queries,
or, if it proposed none, "widens": it goes deeper into the rankings it already has, since
the reranker often places the passage that holds a specific fact below generic ones.
Routing after the critic lives in `policy.decide`.
"""

import operator
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Annotated, Any, TypedDict, cast

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from agentic_rag.domain.search import SearchHit
from agentic_rag.llm.base import LLM, LLMError, Message, Usage
from agentic_rag.pipelines.agent import evidence as ev
from agentic_rag.pipelines.agent.policy import Decision, decide
from agentic_rag.pipelines.agent.prompts import (
    CRITIC_SYSTEM,
    DRAFT_SYSTEM,
    PLANNER_SYSTEM,
    critic_user_prompt,
    planner_user_prompt,
)
from agentic_rag.pipelines.agent.schemas import Critique, Plan
from agentic_rag.pipelines.answer import Answer, cited_ids, is_refusal
from agentic_rag.pipelines.prompts import REFUSAL, answer_user_prompt
from agentic_rag.retrieval.retriever import EvidenceRetriever

log = structlog.get_logger()


@dataclass(frozen=True)
class AgentConfig:
    decompose: bool = True  # ablation switch: planner off → retrieve the question as-is
    critic: bool = True  # ablation switch: critic off → return the first draft
    max_retries: int = 2  # re-plan/re-retrieve rounds after the first draft
    max_sub_questions: int = 3
    max_evidence: int = 10  # passages shown to the drafter (~5k tokens, fits an 8k context)
    retrieve_depth: int = 15  # reranked hits kept per query, so a retry can widen
    max_answer_tokens: int = 400


@dataclass(frozen=True)
class QueryHits:
    iteration: int
    index: int  # position of the query in its round, to keep a stable order
    query: str
    contexts: list[SearchHit]
    candidates: list[SearchHit]


class RetrieveTask(TypedDict):
    query: str
    index: int
    iteration: int


class AgentState(TypedDict, total=False):
    question: str
    subjects: list[str]  # entities the question names (from the planner)
    iteration: int  # 0 for the first pass, +1 per retry
    queries: list[str]  # queries for the current round
    past_queries: list[str]
    retrieved: Annotated[list[QueryHits], operator.add]  # written by parallel retrieve nodes
    evidence: list[SearchHit]
    shown: list[str]  # chunk ids already given to the drafter in any round
    candidates: list[SearchHit]
    draft: str
    critique: Critique | None
    next_queries: list[str]
    decision: Decision
    usage: Usage
    trace: Annotated[list[dict[str, Any]], operator.add]


class AgenticRAG:
    def __init__(
        self,
        retriever: EvidenceRetriever,
        llm: LLM,
        critic_llm: LLM | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.critic_llm = critic_llm or llm
        self.config = config or AgentConfig()
        self.graph = self._build()

    # ── public API ────────────────────────────────────────────────────────────────

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        start: AgentState = {"question": question, "iteration": 0, "usage": Usage(), "trace": []}
        state = cast(AgentState, self.graph.invoke(start))
        text = state["draft"] if state.get("decision") == "accept" else REFUSAL
        timings: dict[str, float] = defaultdict(float)
        retrieve_rounds: dict[int, float] = defaultdict(float)
        for step in state["trace"]:
            if step["node"] == "retrieve":  # parallel within a round: wall time is the slowest
                it = step["iteration"]
                retrieve_rounds[it] = max(retrieve_rounds[it], step["seconds"])
            else:
                timings[f"{step['node']}_s"] += step.get("seconds", 0.0)
        timings["retrieve_s"] = sum(retrieve_rounds.values())
        timings["total_s"] = time.perf_counter() - t0
        return Answer(
            question=question,
            text=text,
            contexts=state.get("evidence", []),
            candidates=state.get("candidates", []),
            usage=state["usage"],
            timings=dict(timings),
            trace=state["trace"],
        )

    # ── graph ─────────────────────────────────────────────────────────────────────

    def _build(self) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("retrieve", self._retrieve)  # type: ignore[arg-type]  # input is a Send task
        g.add_node("gather", self._gather)
        g.add_node("draft", self._draft)
        g.add_node("critique", self._critique)
        g.add_node("replan", self._replan)

        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self._fan_out, ["retrieve"])
        g.add_edge("retrieve", "gather")
        g.add_edge("gather", "draft")
        if self.config.critic:
            g.add_edge("draft", "critique")
            g.add_conditional_edges(
                "critique", self._route, {"retry": "replan", "accept": END, "refuse": END}
            )
            g.add_conditional_edges("replan", self._search_or_widen, ["retrieve", "gather"])
        else:
            g.add_edge("draft", END)
        return g.compile()

    @staticmethod
    def _fan_out(state: AgentState) -> list[Send]:
        return [
            Send("retrieve", RetrieveTask(query=q, index=i, iteration=state["iteration"]))
            for i, q in enumerate(state["queries"])
        ]

    @classmethod
    def _search_or_widen(cls, state: AgentState) -> list[Send] | str:
        return cls._fan_out(state) if state["queries"] else "gather"

    @staticmethod
    def _route(state: AgentState) -> Decision:
        return state["decision"]

    # ── nodes ─────────────────────────────────────────────────────────────────────

    def _plan(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        question = state["question"]
        sub: list[str] = []
        subjects: list[str] = []
        usage = state["usage"]
        if self.config.decompose:
            try:
                plan, u = self.llm.generate_structured(
                    [
                        Message("system", PLANNER_SYSTEM),
                        Message("user", planner_user_prompt(question)),
                    ],
                    Plan,
                    max_tokens=200,
                )
                usage = usage + u
                subjects = plan.subjects
                if plan.needs_decomposition:
                    sub = plan.sub_questions[: self.config.max_sub_questions]
            except LLMError as exc:  # a broken plan must not break answering
                log.warning("plan_failed", error=str(exc))
        # Sub-questions first: they get first pick of evidence slots, the full question last.
        queries = ev.unique_queries([*sub, question])
        step = {
            "node": "plan",
            "subjects": subjects,
            "queries": queries,
            "seconds": time.perf_counter() - t0,
        }
        return {
            "subjects": subjects,
            "queries": queries,
            "past_queries": queries,
            "usage": usage,
            "trace": [step],
        }

    def _retrieve(self, task: RetrieveTask) -> AgentState:
        t0 = time.perf_counter()
        result = self.retriever.retrieve(task["query"])
        hits = QueryHits(
            task["iteration"], task["index"], task["query"], result.contexts, result.candidates
        )
        step = {
            "node": "retrieve",
            "iteration": task["iteration"],
            "query": task["query"],
            "hits": [h.chunk_id for h in result.contexts],
            "seconds": time.perf_counter() - t0,
        }
        return {"retrieved": [hits], "trace": [step]}

    @staticmethod
    def _current_round(state: AgentState) -> list[QueryHits]:
        it = state["iteration"]
        return sorted((r for r in state["retrieved"] if r.iteration == it), key=lambda r: r.index)

    def _gather(self, state: AgentState) -> AgentState:
        rounds = self._current_round(state)
        shown = set(state.get("shown", []))
        unseen = [[h for h in r.contexts if h.chunk_id not in shown] for r in rounds]
        fresh = ev.interleave(unseen, self.config.max_evidence)
        if state["iteration"] == 0:
            evidence = fresh
        else:
            prev = state.get("evidence", [])
            cited = cited_ids(state.get("draft", ""), len(prev))
            evidence = ev.carry_forward(prev, cited, fresh, self.config.max_evidence)
        all_candidates = [*state.get("candidates", []), *(h for r in rounds for h in r.candidates)]
        return {
            "evidence": evidence,
            "shown": [*shown, *(h.chunk_id for h in evidence if h.chunk_id not in shown)],
            "candidates": ev.interleave([all_candidates], len(all_candidates)),
        }

    def _draft(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        evidence = state["evidence"]
        usage = state["usage"]
        if evidence:
            res = self.llm.generate(
                [
                    Message("system", DRAFT_SYSTEM),
                    Message("user", answer_user_prompt(state["question"], evidence)),
                ],
                max_tokens=self.config.max_answer_tokens,
            )
            draft, usage = res.text.strip(), usage + res.usage
        else:
            draft = REFUSAL
        step = {
            "node": "draft",
            "iteration": state["iteration"],
            "refused": is_refusal(draft),
            "seconds": time.perf_counter() - t0,
        }
        # With the critic disabled the first draft is final; refusals stay refusals.
        decision: Decision = "refuse" if is_refusal(draft) else "accept"
        return {"draft": draft, "usage": usage, "decision": decision, "trace": [step]}

    def _critique(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        draft, evidence = state["draft"], state["evidence"]
        usage = state["usage"]
        try:
            critique, u = self.critic_llm.generate_structured(
                [
                    Message("system", CRITIC_SYSTEM),
                    Message(
                        "user",
                        critic_user_prompt(
                            state["question"], draft, evidence, state["past_queries"]
                        ),
                    ),
                ],
                Critique,
                max_tokens=400,
            )
            usage = usage + u
        except LLMError as exc:
            # No verdict: fall back to baseline behaviour (a cited draft is accepted).
            log.warning("critique_failed", error=str(exc))
            critique = Critique.passing()

        next_queries = ev.unique_queries(critique.follow_up_queries, exclude=state["past_queries"])[
            : self.config.max_sub_questions
        ]
        shown = set(state["shown"])
        can_widen = any(
            h.chunk_id not in shown for r in self._current_round(state) for h in r.contexts
        )
        cited = [evidence[i - 1] for i in cited_ids(draft, len(evidence))]
        subjects = state.get("subjects", [])
        uncovered = ev.missing_subjects(subjects, cited)
        decision = decide(
            critique,
            refused=is_refusal(draft),
            has_citations=bool(cited),
            subject_coverage=ev.subject_coverage(subjects, cited),
            retries_left=self.config.max_retries - state["iteration"],
            can_search_more=bool(next_queries) or can_widen,
        )
        step = {
            "node": "critique",
            "iteration": state["iteration"],
            "critique": critique.model_dump(),
            "uncovered_subjects": uncovered,
            "decision": decision,
            "seconds": time.perf_counter() - t0,
        }
        return {
            "critique": critique,
            "next_queries": next_queries,
            "decision": decision,
            "usage": usage,
            "trace": [step],
        }

    @classmethod
    def _replan(cls, state: AgentState) -> AgentState:
        queries = state["next_queries"]
        it = state["iteration"] + 1
        update: AgentState = {
            "iteration": it,
            "queries": queries,
            "past_queries": [*state["past_queries"], *queries],
            "trace": [
                {
                    "node": "replan",
                    "strategy": "search+widen" if queries else "widen",
                    "queries": queries,
                    "seconds": 0.0,
                }
            ],
        }
        # Always widen too: carry this round's rankings into the next one (after the new
        # queries), so gather can also take their next-best, not-yet-shown passages.
        carried = cls._current_round(state)
        update["retrieved"] = [
            replace(r, iteration=it, index=len(queries) + i) for i, r in enumerate(carried)
        ]
        return update
