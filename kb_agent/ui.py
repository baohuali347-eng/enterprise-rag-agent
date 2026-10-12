"""Streamlit 对话界面。

用法: uv run streamlit run kb_agent/ui.py

功能:
- 多轮对话（历史自动带上指代消解）
- 每条回答旁展示来源文档（标题可点击跳转原文）
- 侧边栏展开可见本轮检索明细：命中 chunk、重排得分、反思结果
  （面试/演示时直接展开讲检索链路）
"""
from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from kb_agent.graph.builder import ask

st.set_page_config(page_title="企业知识库问答 Agent", page_icon="🤖", layout="wide")

st.title("🤖 企业知识库问答 Agent")
st.caption("语料：飞书帮助中心官方文档（400 篇）｜ LangChain + LangGraph + DeepSeek + BGE-M3 混合检索")


def init_session() -> None:
    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages = []      # 送入图的 LangChain 消息历史
    if "messages" not in st.session_state:
        st.session_state.messages = []         # 展示用
    if "last_debug" not in st.session_state:
        st.session_state.last_debug = None     # 最近一轮检索明细


init_session()

# ---------- 历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 来源（{len(msg['sources'])} 篇）"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"[{i}] [{s['title']}]({s['url']}) — {s['category']}")

# ---------- 输入 ----------
if prompt := st.chat_input("问点什么，比如：如何在文档中设置查看权限？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.lc_messages.append(HumanMessage(content=prompt))
    with st.chat_message("assistant"):
        with st.spinner("思考中（改写 → 路由 → 检索 → 生成 → 反思）…"):
            result = ask(prompt, history=st.session_state.lc_messages)
        st.markdown(result["answer"])
        sources = result["sources"]
        if sources:
            with st.expander(f"📚 来源（{len(sources)} 篇）"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"[{i}] [{s['title']}]({s['url']}) — {s['category']}")
        st.session_state.last_debug = result

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"], "sources": sources,
    })
    st.session_state.lc_messages.append(AIMessage(content=result["answer"]))

# ---------- 侧边栏：检索链路明细 ----------
with st.sidebar:
    st.header("🔍 本轮检索链路")
    dbg = st.session_state.last_debug
    if dbg is None:
        st.info("开始提问后，这里会展示路由决策、检索结果与反思过程。")
    else:
        st.markdown(f"**路由**: `{dbg['route']}`")
        reflection = dbg.get("reflection") or {}
        if reflection.get("ok"):
            st.markdown("**反思**: ✅ 通过")
        elif reflection.get("reason"):
            st.markdown(f"**反思**: ⚠️ {reflection['reason']}")
        chunks = dbg.get("chunks") or []
        st.markdown(f"**检索命中**: {len(chunks)} 个父文档")
        for i, c in enumerate(chunks, 1):
            with st.expander(f"[{i}] {c['title'][:28]}（{c['score']:.3f}）"):
                st.markdown(f"*{c['category']}*")
                st.markdown(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))
                st.markdown(f"[查看原文]({c['url']})")
    st.divider()
    st.markdown("**系统配置**")
    st.code("混合检索: BGE-M3 + BM25 + RRF\n重排: BGE-reranker-v2-m3\n生成: DeepSeek-V3")
