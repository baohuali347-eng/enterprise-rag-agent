"""LangGraph 各节点实现。

图结构（builder.py 装配）:

    START → rewrite → route ──chitchat──→ chat_answer → END
                      ├────clarify────→ clarify_answer → END
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
- route:          意图路由（知识库问答 / 闲聊 / 澄清），结构化输出
- retrieve:       混合检索；重试时定向补查并合并资料
- relevance_check: 检索结果相关性兜底（最高重排分低于阈值 → 拒答）
- generate:       基于编号上下文生成带引用答案
- reflect:        自检（答非所问？引用是否真实支撑答案？），不合格触发一次重试
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from kb_agent import config
from kb_agent.graph.state import AgentState, RetrievedChunkDict
from kb_agent.llm import get_llm, get_structured_llm
if TYPE_CHECKING:
    from kb_agent.retriever import HybridRetriever

# 全局单例：检索器含两个模型，初始化开销大，进程内复用
_RETRIEVER: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        from kb_agent.retriever import HybridRetriever

        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


# ---------------- 路由：结构化输出 ----------------
class RouteDecision(BaseModel):
    """路由决策。"""
    kind: Literal["kb", "chitchat", "clarify"] = Field(
        description="kb=知识库问答, chitchat=闲聊, clarify=缺少功能或操作对象，需先澄清")
    reason: str = Field(description="一句话说明判断依据")
    clarification_question: str = Field(
        default="", description="仅 clarify 时提供一句询问缺失信息的中文问题；其余留空，不给操作步骤")


ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是企业知识库问答助手的意图路由器。判断用户输入属于哪类:

- kb: 询问产品的功能、操作方法、设置、故障解决等，需要检索知识库回答
- chitchat: 问候、感谢、自我介绍等与知识库无关的闲聊
- clarify: 无法从独立查询和历史判断用户要操作的具体对象或功能，需先补充必要信息

规则:
1. 先结合独立查询和历史判断，不因问句短或有代词就要求澄清；历史已明确对象时走 kb
2. 只有缺少具体功能或操作对象才走 clarify。孤立的通用词（权限、设置、入口、取消）不能确定对象
3. 已明确产品功能、要问入口或一般操作方法时，默认走 kb。具体期望值、实例名称或其他可选细节不是先回答一般方法的前提
4. 不得凭模型自己的产品常识臆造额外分支，把已明确的功能判为不明确；有多个适用场景也可以先给常规方法与适用条件
5. clarify 只询问缺失对象，可用简短选项帮助说明，不提供自行猜测的操作步骤，不判定知识库越界
6. 问题对象明确但本公司配置、私人数据或资料答案未知，走 kb，由检索和生成处理信息边界
7. 非产品主题但问题本身明确时也走 kb，不把领域外问题当作表达不清

输出 kind、reason、clarification_question。"""),
    ("human", "对话历史:\n{history}\n\n独立查询: {query}"),
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



def recent_history(state: AgentState) -> list:
    """只取已完成的最近四轮，兼容旧调用方把本轮问题放进 history。"""
    history = [m for m in state.get("messages", [])
               if isinstance(m, (HumanMessage, AIMessage))]
    if (history and isinstance(history[-1], HumanMessage)
            and history[-1].content == state["query"]):
        history = history[:-1]
    recent = history[-config.HISTORY_MAX_MESSAGES:]
    # 不把窗口开头失去对应用户问题的助手消息交给后续节点。
    if recent and isinstance(recent[0], AIMessage):
        recent = recent[1:]
    return recent


def history_text(state: AgentState) -> str:
    return "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in recent_history(state)
    )


def rewrite_node(state: AgentState) -> dict:
    query = state["query"].strip()
    hist_text = history_text(state)
    if not hist_text:
        return {"standalone_query": query}
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
        decision = llm.invoke(ROUTE_PROMPT.format(
            query=state["standalone_query"], history=history_text(state)))
        return {"route": decision.kind, "route_reason": decision.reason,
                "clarification_question": decision.clarification_question.strip()}
    except Exception:
        # 分类失败默认走知识库检索，宁可检索也不漏答
        return {"route": "kb", "route_reason": "路由服务失败，回退知识库检索",
                "clarification_question": ""}


def route_edge(state: AgentState) -> str:
    return state["route"]


# ---------------- 检索 ----------------
def retrieve_node(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0)
    query = state["standalone_query"]
    chunks: list[RetrievedChunkDict] = []
    if retry_count:
        query = state.get("reflection", {}).get("suggestion", "").strip() or query
        chunks = list(state.get("chunks", []))

    # 保留首轮顺序。补查分数对应另一个查询，只用来筛选本次新增结果。
    seen_parents = {chunk["parent_id"] for chunk in chunks}
    for c in get_retriever().retrieve(query):
        if retry_count and (c.parent_id in seen_parents or c.score < config.RELEVANCE_THRESHOLD):
            continue
        chunks.append({
            "chunk_id": c.chunk_id, "parent_id": c.parent_id, "doc_id": c.doc_id,
            "title": c.title, "text": c.text, "category": c.category,
            "url": c.url, "score": c.score,
        })
        seen_parents.add(c.parent_id)
    return {"chunks": chunks, "retry_count": retry_count}


def relevance_edge(state: AgentState) -> str:
    """首轮无相关资料时拒答；补查后仍可使用保留的首轮资料。"""
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
3. 问题包含多项需求时，逐项回答有资料支持的部分，明确指出哪些部分没有查到依据；如果都没有依据，明确说"知识库中没有找到相关信息"，不要猜测
4. 使用中文，条理清晰，操作步骤用编号列表
5. 对话历史仅用于理解指代和用户意图，历史回答中的事实也必须由本轮资料支持

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
        *recent_history(state),
        HumanMessage(content=state.get("standalone_query") or state["query"]),
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
    suggestion: str = Field(description="仅用于补充检索的独立查询，包含具体对象与缺失的信息")


REFLECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是 RAG 系统的质量审核员。检查助手答案是否合格:

不合格标准（任一即不合格）:
1. 答非所问：没有正面回答用户问题
2. 引用失真：标注的 [n] 引用处，对应资料实际不包含该信息（幻觉引用）
3. 资料明明够用却回答"找不到"（漏答）
4. 若还可补查，用户明确询问的某项信息缺少依据，且能形成知识库范围内的具体补查问题，即使答案已说明缺项，也应标记不合格并提出补查。补查额度用尽后，如已答部分均有依据且明确说明缺项，可判为合格。

输出 JSON；合格时 reason/suggestion 留空字符串。
需要补查时，suggestion 写一条可以独立检索的问题，包含产品或功能名称，只问缺少依据的部分。
例如权限已有依据但缺少客户端操作入口，写「飞书群主在哪里点击导出群成员信息？」。"""),
    ("human", "对话历史（仅用于理解指代，不作为事实依据）:\n{history}\n\n"
     "原始提问: {original_query}\n独立用户问题: {query}\n\n"
     "剩余补查次数: {retries_left}\n\n"
     "知识库资料:\n{context}\n\n助手答案:\n{answer}"),
])


def reflect_node(state: AgentState) -> dict:
    try:
        llm = get_structured_llm(ReflectionDecision)
        decision = llm.invoke(REFLECT_PROMPT.format(
            query=state.get("standalone_query") or state["query"],
            original_query=state["query"],
            history=history_text(state),
            retries_left=max(0, config.MAX_RETRY - state.get("retry_count", 0)),
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


# ---------------- 澄清 / 闲聊 / 拒答 ----------------
def clarify_node(state: AgentState) -> dict:
    question = state.get("clarification_question", "").strip()
    if not question:
        question = "你具体想操作哪个功能或对象？可以补充所在页面和想完成的操作。"
    return {"final_answer": question, "sources": [], "chunks": []}


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
