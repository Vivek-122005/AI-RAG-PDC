import json
from pathlib import Path

import pytest
from fakes import FakeLLM

from agentic_rag.domain.search import SearchHit
from agentic_rag.evaluation import report, runner
from agentic_rag.evaluation.dataset import Evidence, GoldenItem, load_dataset, save_dataset
from agentic_rag.evaluation.matching import covers, evidence_recall, first_relevant_rank
from agentic_rag.evaluation.synth import StoredChunk, synthesize
from agentic_rag.llm.base import Usage
from agentic_rag.pipelines.answer import Answer


def hit(text: str, source: str = "a.pdf") -> SearchHit:
    return SearchHit("id", 1.0, text, "d", "T", source, [], 1, 1)


EV = Evidence(source="a.pdf", quote="The retriever uses DPR with a BERT encoder.")


def test_covers_exact_fuzzy_and_source() -> None:
    assert covers("Intro. The retriever uses DPR with a BERT encoder. More.", "a.pdf", EV)
    assert covers("the retriever uses DPR with BERT encoder", "a.pdf", EV)  # fuzzy
    assert not covers("The retriever uses DPR with a BERT encoder.", "b.pdf", EV)
    assert not covers("Something else entirely.", "a.pdf", EV)


def test_rank_and_recall() -> None:
    ev2 = Evidence(source="b.pdf", quote="Generation uses BART large model weights.")
    hits = [hit("noise"), hit(EV.quote), hit("noise")]
    assert first_relevant_rank(hits, [EV]) == 2
    assert first_relevant_rank([hit("x")], [EV]) is None
    assert evidence_recall(hits, [EV, ev2]) == 0.5


def test_dataset_roundtrip_and_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    items = [GoldenItem(id="a", question="q?", type="single_hop", evidence=[EV])]
    save_dataset(items, p)
    assert load_dataset(p) == items
    save_dataset(items * 2, p)
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(p)


def test_synthesize_keeps_only_verbatim_quotes() -> None:
    text = "CRAG uses a lightweight retrieval evaluator to score documents for each query."
    chunks = [StoredChunk("a.pdf", "CRAG", "4.2", text), StoredChunk("b.pdf", "RAG", "2", text)]
    good = {"question": "What does CRAG use?", "answer": "An evaluator.", "quote": text}
    bad = {**good, "quote": "a sentence that is not in the passage at all"}
    replies = iter([json.dumps(bad), json.dumps(good), json.dumps(good)])
    items = synthesize(FakeLLM(lambda _: next(replies)), chunks, n_single=2, n_multi=0)
    assert len(items) == 2
    assert all(i.evidence[0].quote == text and not i.verified for i in items)


def test_generate_judge_report_end_to_end(tmp_path: Path) -> None:
    items = [
        GoldenItem(id="q1", question="What retriever?", reference_answer="DPR", type="single_hop", evidence=[EV]),
        GoldenItem(id="q2", question="CEO salary?", type="unanswerable", answerable=False),
    ]  # fmt: skip

    def answer_fn(q: str) -> Answer:
        if "salary" in q:
            return Answer(
                q, "It is $1M [1].", [hit("noise")], usage=Usage(1, 1, 0), timings={"total_s": 1.0}
            )
        return Answer(
            q,
            "It uses DPR [1].",
            [hit(EV.quote)],
            [hit(EV.quote)],
            Usage(5, 2, 0),
            {"total_s": 3.0},
        )

    run_dir = tmp_path / "run"
    runner.run_generate(items, answer_fn, run_dir, {"pipeline": "test"})
    runner.run_generate(items, answer_fn, run_dir, {"pipeline": "test"})  # resumable: no dupes
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 2

    def judge_reply(prompt: str) -> str:
        if "atomic factual claims" in prompt:
            return json.dumps({"claims": [{"claim": "uses DPR", "supported": True}]})
        return json.dumps({"reasoning": "matches", "score": 5, "addresses_question": True})

    runner.run_judge(items, FakeLLM(judge_reply), run_dir)
    m = report.write_report(items, run_dir)["overall"]
    assert m["retrieval"]["hit_rate@k"] == 1.0
    assert m["generation"]["faithfulness_mean"] == 1.0
    assert m["generation"]["accuracy(score>=4)"] == 1.0
    assert m["generation"]["cites_gold_evidence_rate"] == 1.0  # measured on answerable items only
    assert m["generation"]["correct_refusal_rate"] == 0.0  # q2 should have been refused
    report_md = (run_dir / "report.md").read_text()
    assert "## Failures (1)" in report_md and "q2" in report_md
