"""LangGraph 各节点实现。

图结构（builder.py 装配）:

    START → rewrite → route ──chitchat──→ chat_answer → END
                      │
                      kb
                      ↓
                 retrieve → relevance_check ──低相关──→ decline → END
                      ↑            │
                      │          通过
                      │            ↓
                      └──(重试)── generate → reflect ──通过──→ END

节点职责:
- rewrite:        多轮指代消解，把当前问题改写成独立查询
- route:          意图路由（知识库问答 / 闲聊），结构化输出
- retrieve:       混合检索；重试时用反思建议做查询扩展
- relevance_check: 检索结果相关性兜底（最高重排分低于阈值 → 拒答）
- generate:       基于编号上下文生成带引用答案
- reflect:        自检（答非所问？引用是否真实支撑答案？），不合格触发一次重试
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from kb_agent import config
from kb_agent.graph.state import AgentState
from kb_agent.llm import get_llm, get_structured_llm
from kb_agent.retriever import HybridRetriever

# 全局单例：检索器含两个模型，初始化开销大，进程内复用
_RETRIEVER: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


# ---------------- 路由：结构化输出 ----------------
class RouteDecision(BaseModel):
    """路由决策。"""
    kind: str = Field(description="kb=知识库问答, chitchat=闲聊")
    reason: str = Field(description="一句话说明判断依据")


ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是企业知识库问答助手的意图路由器。判断用户输入属于哪类:

- kb: 询问产品的功能、操作方法、设置、故障解决等，需要检索知识库回答
- chitchat: 问候、感谢、自我介绍等与知识库无关的闲聊

输出 JSON: {{"kind": "kb 或 chitchat", "reason": "一句话判断依据"}}"""),
    ("human", "{query}"),
])


# ---------------- 查询改写 ----------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """把用户的当前问题改写成不依赖对话历史的独立查询。

规则:
1. 消解指代词：把"它/这个/上面说的"替换成历史中实际指代的对象
2. 补充省略的主语/宾语
3. 只改写表述，不添加新问题、不改变意图
4. 如果当前问题已经独立完整，原样返回

只输出改写后的查询，不要解释。"""),
    ("human", "对话历史:\n{history}\n\n当前问题: {query}"),
])


def rewrite_node(state: AgentState) -> dict:
    query = state["query"].strip()
    history = state.get("messages", [])
    if not history:
        return {"standalone_query": query}
    # 只取最近 4 轮，防止历史过长干扰
    recent = history[-4:]
    hist_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in recent if hasattr(m, "content")
    )
    try:
        out = get_llm(0.0).invoke(
            REWRITE_PROMPT.format(history=hist_text, query=query)
        )
        rewritten = out.content.strip().strip('"')
        return {"standalone_query": rewritten or query}
    except Exception:
        # 改写失败不阻塞主流程，退化为原问题
        return {"standalone_query": query}


def route_node(state: AgentState) -> dict:
    try:
        llm = get_structured_llm(RouteDecision)
        decision = llm.invoke(ROUTE_PROMPT.format(query=state["standalone_query"]))
        kind = "chitchat" if decision.kind == "chitchat" else "kb"
    except Exception:
        # 分类失败默认走知识库检索，宁可检索也不漏答
        kind = "kb"
    return {"route": kind}


def route_edge(state: AgentState) -> str:
    return state["route"]


# ---------------- 检索 ----------------
def retrieve_node(state: AgentState) -> dict:
    retriever = get_retriever()
    query = state["standalone_query"]
    # 反思重试：把建议拼进查询做扩展检索
    if state.get("retry_count", 0) > 0:
        suggestion = state.get("reflection", {}).get("suggestion", "")
        if suggestion:
            query = f"{query}\n补充要求: {suggestion}"
    chunks = retriever.retrieve(query)
    return {
        "chunks": [{
            "chunk_id": c.chunk_id, "parent_id": c.parent_id, "doc_id": c.doc_id,
            "title": c.title, "text": c.text, "category": c.category,
            "url": c.url, "score": c.score,
        } for c in chunks],
        "retry_count": state.get("retry_count", 0),
    }


def relevance_edge(state: AgentState) -> str:
    """相关性兜底：最高分低于阈值说明知识库里没有可靠答案，直接拒答。"""
    chunks = state.get("chunks", [])
    if not chunks:
        return "decline"
    if chunks[0].get("score", 0) < config.RELEVANCE_THRESHOLD:
        return "decline"
    return "generate"


# ---------------- 生成 ----------------
GENERATE_SYSTEM = """你是企业知识库问答助手。严格依据给定的知识库资料回答用户问题。

要求:
1. 只使用资料中的内容回答，不得编造资料中没有的信息
2. 回答中在引用处标注来源编号，如 [1]、[2]；编号与资料编号一一对应
3. 如果资料不足以回答，明确说"知识库中没有找到相关信息"，不要猜测
4. 使用中文，条理清晰，操作步骤用编号列表

知识库资料:
{context}"""


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    sources = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] 标题: {c['title']}\n分类: {c['category']}\nURL: {c['url']}\n内容:\n{c['text']}"
        )
        sources.append({"title": c["title"], "url": c["url"], "category": c["category"]})
    return "\n\n---\n\n".join(parts), sources


def generate_node(state: AgentState) -> dict:
    context, sources = build_context(state["chunks"])
    llm = get_llm(0.0)
    messages = [
        SystemMessage(content=GENERATE_SYSTEM.format(context=context)),
        HumanMessage(content=state["query"]),
    ]
    out = llm.invoke(messages)
    return {
        "context": context,
        "generation": out.content,
        "sources": sources,
    }


# ---------------- 反思（自检 + 单次重试） ----------------
class ReflectionDecision(BaseModel):
    """生成质量自检结果。"""
    ok: bool = Field(description="答案是否合格")
    reason: str = Field(description="不合格原因（合格则留空）")
    suggestion: str = Field(description="改进建议，如需要补充哪方面资料")


REFLECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是 RAG 系统的质量审核员。检查助手答案是否合格:

不合格标准（任一即不合格）:
1. 答非所问：没有正面回答用户问题
2. 引用失真：标注的 [n] 引用处，对应资料实际不包含该信息（幻觉引用）
3. 资料明明够用却回答"找不到"（漏答）

输出 JSON；合格时 reason/suggestion 留空字符串。"""),
    ("human", "用户问题: {query}\n\n知识库资料:\n{context}\n\n助手答案:\n{answer}"),
])


def reflect_node(state: AgentState) -> dict:
    try:
        llm = get_structured_llm(ReflectionDecision)
        decision = llm.invoke(REFLECT_PROMPT.format(
            query=state["query"],
            context=state["context"],
            answer=state["generation"],
        ))
        reflection = {
            "ok": decision.ok,
            "reason": decision.reason,
            "suggestion": decision.suggestion,
        }
    except Exception:
        # 反思本身失败不阻塞输出
        reflection = {"ok": True, "reason": "", "suggestion": ""}
    return {"reflection": reflection}


def reflect_edge(state: AgentState) -> str:
    reflection = state.get("reflection", {})
    if reflection.get("ok"):
        return "done"
    if state.get("retry_count", 0) < config.MAX_RETRY:
        return "retry"
    return "done"


def finalize_node(state: AgentState) -> dict:
    """收敛输出：反思通过直接出答案；重试耗尽则附上保守提示。"""
    reflection = state.get("reflection", {})
    answer = state["generation"]
    if not reflection.get("ok") and state.get("retry_count", 0) >= config.MAX_RETRY:
        answer += "\n\n（提示：已尝试补充检索，如以上内容未能解答，建议在帮助中心搜索或联系管理员。）"
    return {"final_answer": answer}


# ---------------- 闲聊 / 拒答 ----------------
def chitchat_node(state: AgentState) -> dict:
    llm = get_llm(0.7)
    out = llm.invoke([
        SystemMessage(content="你是企业知识库助手的接待员，友好简短地回应问候与闲聊，一两句话即可。"),
        HumanMessage(content=state["query"]),
    ])
    return {"final_answer": out.content, "sources": []}


DECLINE_TEXT = (
    "抱歉，这个问题超出了当前知识库的覆盖范围（知识库内容为飞书帮助中心的官方文档）。\n\n"
    "我可以帮你解答：\n"
    "- 各功能的使用方法（云文档、多维表格、审批、考勤、即时消息等）\n"
    "- 企业管理后台配置与账号权限\n"
    "- 常见问题排查\n\n"
    "你可以换个问法，或说明具体想了解哪个功能。"
)


def decline_node(state: AgentState) -> dict:
    return {"final_answer": DECLINE_TEXT, "sources": []}
