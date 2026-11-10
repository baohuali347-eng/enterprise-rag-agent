"""评测口径回归：多轮使用真实前文、裁判耗时隔离、错误不能成为成功。"""
import json

import pytest
from pydantic import ValidationError

from kb_agent.eval.run_scenarios import (
    Verdict, citation_check, evaluate, metrics, percentile, summarize,
)
from kb_agent.eval.scenarios import (
    Scenario, ScenarioDataset, corpus_fingerprint, load_dataset, validate_corpus,
)


def scenario(case_id="case", category="direct", questions=("问题",), expected="answer"):
    return Scenario(id=case_id, category=category, turns=[{
        "question": q, "expected_behavior": expected, "reference_answer": "参考要点",
        "evidence": [{"doc_id": "doc", "quote": "原文"}] if expected == "answer" else [],
        "annotation_note": "人工标注依据",
    } for q in questions])


def good_judge(case, turn, history, result):
    return Verdict(response_kind=turn.expected_behavior, fulfills_expected=True,
                   grounded=True, reason="匹配参考要点及检索依据")


def good_ask(question, **kwargs):
    return {"answer": "依据资料回答。[1]", "chunks": [{"text": "原文"}], "trace": []}


def test_dataset_has_required_scenarios_and_answer_evidence():
    dataset = load_dataset()
    assert dataset.origin == "manually_designed"
    assert {c.category for c in dataset.cases} == {
        "direct", "colloquial", "multi_turn", "in_domain_unanswerable", "out_of_domain", "ambiguous"}
    assert all(t.evidence for c in dataset.cases for t in c.turns if t.expected_behavior == "answer")


def test_multi_turn_uses_actual_answer_and_cases_do_not_share_history():
    calls = []

    def ask(question, history, **kwargs):
        calls.append((question, [m.content for m in history], kwargs))
        return {"answer": f"实际回答{len(calls)} [1]", "chunks": [{"text": "原文"}]}

    cases = [scenario("multi", "multi_turn", ("先问", "追问")), scenario("new")]
    records = evaluate(cases, ask_fn=ask, judge_fn=good_judge)
    assert calls[0][1] == []
    assert calls[1][1] == ["先问", "实际回答1 [1]"]
    assert calls[2][1] == []
    assert calls[0][2]["with_trace"] is True
    assert records[1]["history"][1]["content"] == "实际回答1 [1]"
    assert all(r["success"] for r in records)
    assert summarize(records)["overall"]["case_success_rate"] == 1


def test_judge_latency_is_not_in_answer_latency():
    ticks = iter([0, 2, 100, 105, 200, 203, 300, 307])
    records = evaluate([scenario("a"), scenario("b")], ask_fn=good_ask,
                       judge_fn=good_judge, clock=lambda: next(ticks))
    assert [r["latency_seconds"] for r in records] == [2, 3]
    assert [r["judge_latency_seconds"] for r in records] == [5, 7]
    report = metrics(records)
    assert report["latency_seconds"]["p50"] == 2.5
    assert report["latency_excluding_first_request_seconds"]["count"] == 1
    assert report["latency_excluding_first_request_seconds"]["p50"] == 3


def test_refusal_is_failure_on_answerable_question_even_if_grounded():
    def judge(*args):
        return Verdict(response_kind="decline", fulfills_expected=False, grounded=True, reason="漏答")

    records = evaluate([scenario()], ask_fn=lambda *a, **kw: {"answer": "资料不足", "chunks": []},
                       judge_fn=judge)
    report = metrics(records)
    assert report["answer_success_rate"] == 0
    assert report["false_refusal_rate"] == 1
    assert report["false_refusal_count"] == 1


@pytest.mark.parametrize("answer", ["没有引用", "错误编号[0]", "不存在[2]", "混合[1, 3]"])
def test_answer_requires_existing_citation_numbers(answer):
    records = evaluate([scenario()], ask_fn=lambda *a, **kw: {"answer": answer, "chunks": [{}]},
                       judge_fn=good_judge)
    assert not records[0]["success"]


@pytest.mark.parametrize("category,expected", [("in_domain_unanswerable", "decline"),
                                               ("ambiguous", "clarify")])
def test_correct_refusal_and_clarification_do_not_require_citations(category, expected):
    records = evaluate([scenario(category=category, expected=expected)],
                       ask_fn=lambda *a, **kw: {"answer": "资料不足或请补充具体对象", "chunks": []},
                       judge_fn=good_judge)
    assert records[0]["success"]


def test_unsupported_answer_is_failure_despite_correct_citation_number():
    def judge(*args):
        return Verdict(response_kind="answer", fulfills_expected=True, grounded=False, reason="引用不支持")

    assert not evaluate([scenario()], ask_fn=good_ask, judge_fn=judge)[0]["success"]


def test_ask_failure_blocks_followup_but_does_not_abort_next_case_or_expose_error():
    def ask(question, **kwargs):
        if question == "先问":
            raise RuntimeError("私密请求头不应写入产物")
        return good_ask(question)

    records = evaluate([scenario("multi", "multi_turn", ("先问", "追问")), scenario("new")],
                       ask_fn=ask, judge_fn=good_judge)
    assert [r["status"] for r in records] == ["ask_error", "blocked", "ok"]
    assert records[1]["latency_seconds"] is None
    assert "私密" not in json.dumps(records, ensure_ascii=False)
    assert metrics(records)["task_success_rate"] == pytest.approx(1 / 3)
    assert metrics(records)["case_success_rate"] == 0.5
    assert metrics(records)["latency_seconds"]["count"] == 1


def test_judge_error_is_visible_and_does_not_invalidate_actual_history():
    seen = []

    def judge(case, turn, history, result):
        if not history:
            raise ValueError("模拟裁判故障")
        seen.extend(m.content for m in history)
        return good_judge(case, turn, history, result)

    records = evaluate([scenario("multi", "multi_turn", ("先问", "追问"))],
                       ask_fn=good_ask, judge_fn=judge)
    report = metrics(records)
    assert [r["status"] for r in records] == ["judge_error", "ok"]
    assert report["task_success_rate"] is None
    assert report["answer_success_rate"] is None
    assert report["answerable_judged"] == 1
    assert report["latency_seconds"]["count"] == 2
    assert seen == ["先问", "依据资料回答。[1]"]


def test_schema_rejects_duplicate_ids_and_answer_without_evidence():
    dataset = load_dataset().model_dump()
    dataset["cases"].append(dataset["cases"][0])
    with pytest.raises(ValidationError, match="ID"):
        ScenarioDataset.model_validate(dataset)
    item = scenario().model_dump()
    item["turns"][0]["evidence"] = []
    with pytest.raises(ValidationError, match="原文依据"):
        Scenario.model_validate(item)


def test_corpus_validation_detects_missing_quote_and_snapshot_changes():
    corpus = {"doc": "原文"}
    dataset = ScenarioDataset(version="test", origin="manually_designed", description="test",
                              corpus_sha256=corpus_fingerprint(corpus), cases=[scenario()])
    validate_corpus(dataset, corpus)
    with pytest.raises(ValueError, match="快照"):
        validate_corpus(dataset, {"doc": "改动后的正文"})
    dataset.cases[0].turns[0].evidence[0].quote = "不存在的句子"
    with pytest.raises(ValueError, match="原文依据不存在"):
        validate_corpus(dataset, corpus)


def test_citation_groups_and_percentiles():
    assert citation_check("见[1, 2]和[3]", 3)["valid_numbers"]
    assert citation_check("无引用", 0)["valid_numbers"]
    assert percentile([], 0.95) is None
    assert percentile([5], 0.95) == 5
    assert percentile([0, 10], 0.95) == 9.5


def test_multi_turn_scenario_can_clarify_then_answer():
    item = scenario("mixed", "multi_turn", ("模糊问题", "补充对象")).model_dump()
    item["turns"][0].update(expected_behavior="clarify", evidence=[])
    case = Scenario.model_validate(item)
    assert [t.expected_behavior for t in case.turns] == ["clarify", "answer"]
