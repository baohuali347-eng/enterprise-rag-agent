"""LangGraph 状态定义。

AgentState 是贯穿整个图的状态对象：
- messages 用 add_messages reducer 自动累积对话历史
- 其余字段是各节点写入、下游节点读取的工作内存
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RetrievedChunkDict(TypedDict, total=False):
    """检索结果单元（retriever.RetrievedChunk 的可序列化形式）。"""
    chunk_id: str
    parent_id: str
    doc_id: str
    title: str
    text: str
    category: str
    url: str
    score: float


class ReflectionDict(TypedDict, total=False):
    ok: bool
    reason: str        # 不合格原因
    suggestion: str    # 聚焦缺失证据的补查查询


class AgentState(TypedDict, total=False):
    # 对话
    messages: Annotated[list[BaseMessage], add_messages]
    query: str                      # 用户当前输入
    # 路由
    route: str                      # kb / chitchat / clarify
    route_reason: str
    clarification_question: str
    # 查询处理
    standalone_query: str           # 多轮改写后的独立查询
    # 检索
    chunks: list[RetrievedChunkDict]
    context: str                    # 编号后的上下文，直接进 prompt
    # 生成与反思
    generation: str
    reflection: ReflectionDict
    retry_count: int
    # 输出
    final_answer: str
    sources: list[dict]             # 引用的来源（标题+URL），供 UI 展示
