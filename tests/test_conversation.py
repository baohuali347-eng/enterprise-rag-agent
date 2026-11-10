"""真实执行 LangGraph，替换外部 LLM 和检索器，验证多轮数据传递。"""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from kb_agent import config
from kb_agent.graph import nodes
from kb_agent.graph.builder import ask, build_graph


FOLLOWUP = "它怎么设置？"
STANDALONE = "飞书云文档的链接有效期怎么设置？"
HISTORY = [HumanMessage(content="飞书云文档能设置链接有效期吗？"),
           AIMessage(content="可以设置链接有效期。")]


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, value):
        self.calls.append(value)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def services(monkeypatch):
    rewrite = FakeLLM(AIMessage(content=STANDALONE))
    generate = FakeLLM(AIMessage(content="打开分享设置，设置链接有效期。[1]"))
    route = FakeLLM(nodes.RouteDecision(kind="kb", reason="功能操作"))
    reflect = FakeLLM(nodes.ReflectionDecision(ok=True, reason="", suggestion=""))
    queries = []

    def retrieve(query):
        queries.append(query)
        return [SimpleNamespace(chunk_id="doc#p0#c0", parent_id="doc#p0", doc_id="doc",
                title="文档权限", text="打开分享设置，设置链接有效期。", category="云文档",
                url="https://example.com", score=0.9)]

    # rewrite 使用格式化字符串，generate 使用消息列表。
    def invoke(value):
        return rewrite.invoke(value) if isinstance(value, str) else generate.invoke(value)

    monkeypatch.setattr(nodes, "get_llm", lambda *args: SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(nodes, "get_structured_llm",
        lambda schema: route if schema is nodes.RouteDecision else reflect)
    monkeypatch.setattr(nodes, "get_retriever", lambda: SimpleNamespace(retrieve=retrieve))
    return SimpleNamespace(rewrite=rewrite, generate=generate, route=route,
                           reflect=reflect, queries=queries)


@pytest.mark.parametrize("legacy_history", [False, True])
def test_followup_uses_same_resolved_question_through_graph(services, legacy_history):
    history = list(HISTORY)
    if legacy_history:
        history.append(HumanMessage(content=FOLLOWUP))
    result = build_graph().invoke({"query": FOLLOWUP, "messages": history, "retry_count": 0})
    assert services.queries == [STANDALONE]
    assert STANDALONE in services.route.calls[0]
    messages = services.generate.calls[0]
    assert messages[-1].content == STANDALONE
    assert [m.content for m in messages[1:-1]] == [m.content for m in HISTORY]
    assert result["final_answer"] == "打开分享设置，设置链接有效期。[1]"
    assert STANDALONE in services.reflect.calls[0]
    assert FOLLOWUP in services.reflect.calls[0]
    assert HISTORY[0].content in services.reflect.calls[0]
    assert len(history) == 2 + legacy_history  # 节点不修改调用方的历史


def test_first_turn_skips_rewrite_even_with_legacy_history(services):
    build_graph().invoke({"query": STANDALONE,
                         "messages": [HumanMessage(content=STANDALONE)], "retry_count": 0})
    assert services.rewrite.calls == []
    assert services.queries == [STANDALONE]
    assert len(services.generate.calls[0]) == 2


def test_rewrite_failure_keeps_history_for_generation_and_reflection(services):
    services.rewrite.response = RuntimeError("模拟改写服务失败")
    result = build_graph().invoke({"query": FOLLOWUP, "messages": HISTORY, "retry_count": 0})
    assert result["standalone_query"] == FOLLOWUP
    assert services.queries == [FOLLOWUP]
    assert services.generate.calls[0][1].content == HISTORY[0].content
    assert HISTORY[0].content in services.reflect.calls[0]


def test_history_keeps_four_complete_rounds():
    history = []
    for i in range(6):
        history.extend([HumanMessage(content=f"问题{i}"), AIMessage(content=f"回答{i}")])
    recent = nodes.recent_history({"query": FOLLOWUP, "messages": history})
    assert len(recent) == config.HISTORY_MAX_MESSAGES
    assert recent[0].content == "问题2"
    assert recent[-1].content == "回答5"


@pytest.mark.parametrize("supplement", ["relevant", "below_threshold", "empty"])
def test_reflection_retry_preserves_evidence_and_rebuilds_citations(services, monkeypatch, supplement):
    first = SimpleNamespace(
        chunk_id="doc#p0#c0", parent_id="doc#p0", doc_id="doc", title="权限说明",
        text="只有文档所有者可以设置链接有效期。", category="云文档",
        url="https://example.com/permissions", score=0.8,
    )
    steps = SimpleNamespace(
        chunk_id="doc#p1#c0", parent_id="doc#p1", doc_id="doc", title="设置步骤",
        text="打开分享设置，设置链接有效期。", category="云文档",
        url="https://example.com/steps", score=0.95,
    )
    duplicate = SimpleNamespace(**{**vars(first), "chunk_id": "doc#p0#c1", "score": 0.99})
    supplement_query = "飞书云文档链接有效期的设置入口和操作步骤是什么？"

    def retrieve(query):
        services.queries.append(query)
        if query == STANDALONE:
            return [first]
        if supplement == "empty":
            return []
        if supplement == "below_threshold":
            steps.score = config.RELEVANCE_THRESHOLD - 0.01
        return [duplicate, steps]

    monkeypatch.setattr(nodes, "get_retriever", lambda: SimpleNamespace(retrieve=retrieve))
    services.reflect.response = nodes.ReflectionDecision(
        ok=False, reason="缺少有效期操作步骤", suggestion=supplement_query)
    result = ask(FOLLOWUP, history=HISTORY, with_trace=True)

    assert services.queries == [STANDALONE, supplement_query]
    assert len(services.generate.calls) == 2
    assert all(call[-1].content == STANDALONE for call in services.generate.calls)
    assert result["retry_count"] == config.MAX_RETRY
    expected = [first, steps] if supplement == "relevant" else [first]
    assert [c["chunk_id"] for c in result["chunks"]] == [c.chunk_id for c in expected]
    assert [s["url"] for s in result["sources"]] == [c.url for c in expected]
    prompt = services.generate.calls[-1][0].content
    assert "[1] 标题: 权限说明" in prompt and first.text in prompt
    assert ("[2] 标题: 设置步骤" in prompt) == (supplement == "relevant")
    assert (steps.text in prompt) == (supplement == "relevant")
    first_retrieval = next(event for event in result["trace"] if event["node"] == "retrieve")
    assert [c["chunk_id"] for c in first_retrieval["updates"]["chunks"]] == [first.chunk_id]


def test_ask_returns_resolved_query_for_debugging(services):
    result = ask(FOLLOWUP, history=HISTORY)
    assert result["standalone_query"] == STANDALONE
    assert result["answer"]


@pytest.mark.parametrize("with_reflect", [True, False])
def test_trace_preserves_graph_result_and_retry_attempts(services, with_reflect):
    import json

    services.reflect.response = nodes.ReflectionDecision(
        ok=False, reason="缺少步骤", suggestion="补充有效期步骤")
    result = ask(FOLLOWUP, history=HISTORY, with_reflect=with_reflect, with_trace=True)
    assert result["answer"]
    assert result["standalone_query"] == STANDALONE
    assert result["retry_count"] == (config.MAX_RETRY if with_reflect else 0)
    attempts = [entry for entry in result["trace"] if entry["node"] == "retrieve"]
    assert len(attempts) == (config.MAX_RETRY + 1 if with_reflect else 1)
    assert attempts[0]["updates"]["chunks"][0]["doc_id"] == "doc"
    assert result["trace"][-1]["node"] == "finalize"
    assert all(entry["duration_seconds"] >= 0 for entry in result["trace"])
    json.dumps(result)  # 逐题结果无需丢弃节点内容即可写入 JSONL


@pytest.mark.parametrize("with_reflect", [True, False])
def test_clarification_ends_without_retrieval_or_generation(services, monkeypatch, with_reflect):
    question = "你想修改哪个功能的设置？"
    services.route.response = nodes.RouteDecision(
        kind="clarify", reason="缺少设置对象", clarification_question=question)

    def forbidden():
        raise AssertionError("澄清分支不应初始化检索器")

    monkeypatch.setattr(nodes, "get_retriever", forbidden)
    result = ask("这个怎么操作？", with_reflect=with_reflect, with_trace=True)
    assert result["route"] == "clarify"
    assert result["answer"] == question
    assert result["sources"] == result["chunks"] == []
    assert result["route_reason"] == "缺少设置对象"
    assert [n["node"] for n in result["trace"]] == ["rewrite", "route", "clarify_answer"]
    assert services.generate.calls == []
    assert services.reflect.calls == []


def test_clarification_has_fallback_question_if_model_omits_it(services):
    services.route.response = nodes.RouteDecision(kind="clarify", reason="对象不明")
    result = ask("怎么取消？")
    assert "哪个功能或对象" in result["answer"]
    assert services.queries == []


def test_user_can_answer_clarification_then_continue_to_knowledge_query(services):
    services.route.response = nodes.RouteDecision(
        kind="clarify", reason="缺少对象", clarification_question="你说的是哪项设置？")
    first = ask("这个在哪里改？")
    history = [HumanMessage(content="这个在哪里改？"), AIMessage(content=first["answer"])]
    services.route.response = nodes.RouteDecision(kind="kb", reason="用户已补充文档链接有效期")
    second = ask("文档链接有效期", history=history)
    assert second["route"] == "kb"
    assert services.queries == [STANDALONE]
    assert first["answer"] in services.route.calls[-1]
    assert STANDALONE in services.route.calls[-1]
    assert first["answer"] in services.rewrite.calls[0]


def test_route_error_falls_back_to_retrieval(services):
    services.route.response = RuntimeError("服务不可用")
    result = ask(STANDALONE)
    assert result["route"] == "kb"
    assert result["answer"]
    assert services.queries == [STANDALONE]


def test_route_schema_rejects_unknown_kind():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        nodes.RouteDecision(kind="guess", reason="未知类别")
