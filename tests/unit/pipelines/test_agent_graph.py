"""End-to-end behaviour of the agent graph with a scripted LLM and a fake retriever."""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence

from fakes import FakeLLM, FakeRetriever, make_hit

from agentic_rag.pipelines.agent import AgentConfig, AgenticRAG
from agentic_rag.pipelines.prompts import REFUSAL

Q = "Which model does CRAG use as evaluator, and which model runs Ragas prompts?"
HITS = {
    Q: [make_hit("generic")],
    "CRAG retrieval evaluator model": [make_hit("crag", "CRAG fine-tunes T5-large.")],
    "Ragas evaluation LLM": [make_hit("ragas", "Ragas uses gpt-3.5-turbo-16k.")],
}
PLAN_SPLIT = {
    "subjects": ["CRAG", "Ragas"],
    "needs_decomposition": True,
    "sub_questions": ["CRAG retrieval evaluator model", "Ragas evaluation LLM"],
}
PLAN_SIMPLE = {"subjects": [], "needs_decomposition": False, "sub_questions": []}
OK = {
    "grounded": True,
    "answers_question": True,
    "unsupported_claims": [],
    "missing": [],
    "follow_up_queries": [],
}


def script(
    plan: Mapping[str, object] | str,
    drafts: Sequence[str],
    critiques: Sequence[Mapping[str, object]],
) -> Callable[[str], str]:
    """Route each LLM call by its prompt: planner, drafter or critic."""
    draft_iter: Iterator[str] = iter(drafts)
    critique_iter: Iterator[Mapping[str, object]] = iter(critiques)

    def reply(prompt: str) -> str:
        if "Draft answer:" in prompt:
            return json.dumps(next(critique_iter))
        if "<evidence" in prompt:
            return next(draft_iter)
        return plan if isinstance(plan, str) else json.dumps(plan)

    return reply


def agent(
    reply: Callable[[str], str], **config: object
) -> tuple[AgenticRAG, FakeRetriever, FakeLLM]:
    llm, retriever = FakeLLM(reply), FakeRetriever(HITS)
    return AgenticRAG(retriever, llm, config=AgentConfig(**config)), retriever, llm  # type: ignore[arg-type]


def nodes(trace: list[dict[str, object]]) -> list[object]:
    return [s["node"] for s in trace]


def test_simple_question_is_answered_in_one_pass() -> None:
    rag, retriever, _ = agent(script(PLAN_SIMPLE, ["Generic answer [1]."], [OK]))
    ans = rag.answer(Q)
    assert ans.text == "Generic answer [1]."
    assert retriever.queries == [Q]
    assert nodes(ans.trace) == ["plan", "retrieve", "draft", "critique"]
    assert ans.usage.prompt_tokens == 30  # plan + draft + critique
    assert {"plan_s", "retrieve_s", "draft_s", "critique_s", "total_s"} <= ans.timings.keys()


def test_decomposition_retrieves_every_sub_question() -> None:
    rag, retriever, llm = agent(script(PLAN_SPLIT, ["T5-large [1] and gpt-3.5 [2]."], [OK]))
    ans = rag.answer(Q)
    assert sorted(retriever.queries) == sorted([Q, *PLAN_SPLIT["sub_questions"]])  # type: ignore[misc]
    assert [h.chunk_id for h in ans.contexts] == ["crag", "ragas", "generic"]
    assert ans.cited_ids == [1, 2] and not ans.refused
    draft_prompt = next(c[-1].content for c in llm.calls if "<evidence" in c[-1].content)
    assert "T5-large" in draft_prompt and "gpt-3.5-turbo-16k" in draft_prompt


def test_critic_triggers_a_targeted_retry() -> None:
    off_target = {
        **OK,
        "answers_question": False,
        "missing": ["Ragas model"],
        "follow_up_queries": ["Ragas evaluation LLM"],
    }
    rag, retriever, _ = agent(
        script(PLAN_SIMPLE, ["Wrong paper [1].", "gpt-3.5-turbo-16k [2]."], [off_target, OK])
    )
    ans = rag.answer(Q)
    assert retriever.queries == [Q, "Ragas evaluation LLM"]
    assert ans.text == "gpt-3.5-turbo-16k [2]."
    assert [h.chunk_id for h in ans.contexts] == ["generic", "ragas"]  # cited first, then new
    assert ans.cited_contexts[0].chunk_id == "ragas"
    assert nodes(ans.trace) == [
        "plan",
        "retrieve",
        "draft",
        "critique",
        "replan",
        "retrieve",
        "draft",
        "critique",
    ]


def test_unfixable_draft_is_refused_when_budget_runs_out() -> None:
    bad = {**OK, "grounded": False, "follow_up_queries": ["q2"]}
    bad2 = {**OK, "grounded": False, "follow_up_queries": ["q3"]}
    rag, retriever, _ = agent(
        script(PLAN_SIMPLE, ["Made up [1].", "Still made up [1]."], [bad, bad2]), max_retries=1
    )
    ans = rag.answer(Q)
    assert ans.text == REFUSAL and ans.refused
    assert retriever.queries == [Q, "q2"]  # one retry only, then refuse


def test_critic_disabled_returns_first_draft() -> None:
    rag, _, llm = agent(script(PLAN_SIMPLE, ["Draft [1]."], []), critic=False)
    ans = rag.answer(Q)
    assert ans.text == "Draft [1]."
    assert not any("Draft answer:" in c[-1].content for c in llm.calls)


def test_broken_plan_falls_back_to_the_original_question() -> None:
    rag, retriever, _ = agent(script("not json", ["Answer [1]."], [OK]))
    ans = rag.answer(Q)
    assert retriever.queries == [Q]
    assert ans.text == "Answer [1]."


def test_decompose_disabled_skips_the_planner_llm_call() -> None:
    rag, _, llm = agent(script(PLAN_SPLIT, ["Answer [1]."], [OK]), decompose=False)
    rag.answer(Q)
    assert len(llm.calls) == 2  # draft + critique


def test_no_evidence_refuses_without_drafting() -> None:
    llm, retriever = FakeLLM(script(PLAN_SIMPLE, [], [OK])), FakeRetriever({})
    ans = AgenticRAG(retriever, llm).answer("unknown topic")
    assert ans.refused and ans.contexts == []


def test_refusal_without_new_queries_widens_into_deeper_ranks() -> None:
    deep = {Q: [make_hit("r1"), make_hit("r2"), make_hit("r3"), make_hit("gold")]}
    llm, retriever = (
        FakeLLM(script(PLAN_SIMPLE, [REFUSAL, "Found it [1]."], [{**OK}, OK])),
        FakeRetriever(deep),
    )
    rag = AgenticRAG(retriever, llm, config=AgentConfig(max_evidence=2, max_retries=2))
    ans = rag.answer(Q)
    assert retriever.queries == [Q]  # no new search: the existing ranking is reused
    assert [h.chunk_id for h in ans.contexts] == ["r3", "gold"]  # unseen hits only
    assert ans.text == "Found it [1]."
    replan = next(s for s in ans.trace if s["node"] == "replan")
    assert replan["strategy"] == "widen"


def test_answer_about_the_wrong_subject_is_refused() -> None:
    hits = {"What GPU did the Ragas authors use?": [make_hit("selfrag", "Self-RAG used A100s.")]}
    plan = {"subjects": ["Ragas"], "needs_decomposition": False, "sub_questions": []}
    llm = FakeLLM(script(plan, ["They used A100s [1]."], [OK]))  # the critic is fooled
    ans = AgenticRAG(FakeRetriever(hits), llm).answer("What GPU did the Ragas authors use?")
    assert ans.refused
    critique = next(s for s in ans.trace if s["node"] == "critique")
    assert critique["uncovered_subjects"] == ["Ragas"]


def test_evidence_budget_scales_with_number_of_queries() -> None:
    many = {Q: [make_hit(f"h{i}") for i in range(12)]}
    rag = AgenticRAG(FakeRetriever(many), FakeLLM(script(PLAN_SIMPLE, ["A [1]."], [OK])))
    assert len(rag.answer(Q).contexts) == 5  # one query → baseline-sized evidence

    split = {
        **{q: [make_hit(f"{q}-{i}") for i in range(12)] for q in PLAN_SPLIT["sub_questions"]},  # type: ignore[attr-defined]
        Q: [make_hit(f"q-{i}") for i in range(12)],
    }
    llm = FakeLLM(script(PLAN_SPLIT, ["A [1] [2]."], [OK]))
    assert len(AgenticRAG(FakeRetriever(split), llm).answer(Q).contexts) == 10  # capped
