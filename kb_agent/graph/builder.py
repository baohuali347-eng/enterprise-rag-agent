"""装配 LangGraph 状态图。

流程:
    START → rewrite → route ──chitchat──→ chat_answer ─────────────→ END
                      ├────clarify────→ clarify_answer ──────────→ END
                      │
                      kb
                      ↓
                 retrieve → relevance ──decline──→ decline ────────→ END
                      ↑            │
                      │          generate
                      │            ↓
                      │          reflect ──done──→ finalize ───────→ END
                      │            │
                      └──retry─────┘ (bump_retry 计数后回到 retrieve)

反思-重试循环上限 MAX_RETRY 次；耗尽后仍输出答案并附保守提示，
保证用户体验上不会死循环或空手而归。
"""
from __future__ import annotations

from time import perf_counter

from langgraph.graph import END, START, StateGraph

from kb_agent.graph import nodes
from kb_agent.graph.state import AgentState


def build_graph(with_reflect: bool = True):
    """装配状态图。with_reflect=False 时跳过反思节点（消融实验对照组）。"""
    graph = StateGraph(AgentState)

    graph.add_node("rewrite", nodes.rewrite_node)
    graph.add_node("route", nodes.route_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("reflect", nodes.reflect_node)
    graph.add_node("finalize", nodes.finalize_node)
    graph.add_node("chat_answer", nodes.chitchat_node)
    graph.add_node("clarify_answer", nodes.clarify_node)
    graph.add_node("decline", nodes.decline_node)
    graph.add_node("bump_retry", lambda s: {"retry_count": s.get("retry_count", 0) + 1})
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "route")
    graph.add_conditional_edges(
        "route", nodes.route_edge,
        {"kb": "retrieve", "chitchat": "chat_answer", "clarify": "clarify_answer"},
    )
    graph.add_conditional_edges(
        "retrieve", nodes.relevance_edge,
        {"generate": "generate", "decline": "decline"},
    )
    if with_reflect:
        graph.add_edge("generate", "reflect")
        graph.add_conditional_edges(
            "reflect", nodes.reflect_edge,
            {"done": "finalize", "retry": "bump_retry"},
        )
        graph.add_edge("bump_retry", "retrieve")
    else:
        graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("chat_answer", END)
    graph.add_edge("clarify_answer", END)
    graph.add_edge("decline", END)

    return graph.compile()


# 进程级单例：图的编译有开销，且节点内检索器是重量级单例
_GRAPHS: dict = {}


def get_graph(with_reflect: bool = True):
    if with_reflect not in _GRAPHS:
        _GRAPHS[with_reflect] = build_graph(with_reflect)
    return _GRAPHS[with_reflect]


def ask(query: str, history: list | None = None, with_reflect: bool = True,
        *, with_trace: bool = False) -> dict:
    """单轮问答入口。

    参数:
        query: 用户当前输入
        history: 已完成轮次的 LangChain 消息列表，不含当前问题（可为空；兼容旧调用方式）
        with_reflect: 是否启用反思-重试循环（消融实验可关）
        with_trace: 保存节点输出与近似节点耗时，供逐题评测和失败定位
    返回:
        {"answer": str, "sources": list[dict], "route": str, "chunks": ..., "reflection": ...}
    """
    graph = get_graph(with_reflect)
    inputs = {
        "query": query,
        "messages": history or [],
        "retry_count": 0,
    }
    trace = []
    if with_trace:
        result = inputs
        previous = perf_counter()
        for mode, event in graph.stream(inputs, stream_mode=["updates", "values"]):
            if mode == "values":
                result = event
            else:
                now = perf_counter()
                for node, updates in event.items():
                    trace.append({"node": node, "duration_seconds": now - previous,
                                  "updates": updates})
                previous = now
    else:
        result = graph.invoke(inputs)
    output = {
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources", []),
        "route": result.get("route", "kb"),
        "route_reason": result.get("route_reason", ""),
        "clarification_question": result.get("clarification_question", ""),
        # 检索过程明细（调试/演示用）
        "chunks": result.get("chunks", []),
        "reflection": result.get("reflection", {}),
        "standalone_query": result.get("standalone_query", query),
        "retry_count": result.get("retry_count", 0),
    }
    if with_trace:
        output["trace"] = trace
    return output
