"""评测集构造。

思路:
1. 从父文档中按分类分层抽样（保证评测覆盖各功能模块）
2. 让 LLM 基于每篇文档生成 2 个"答案确实在这篇文档里"的问题，
   文档 id 即标准答案 → 检索指标可自动计算（recall@k 无需人工标注）
3. 另外生成一批知识库边界外的问题 → 检验拒答能力
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from kb_agent import config
from kb_agent.llm import get_llm, get_structured_llm

random.seed(42)


class QAQuestions(BaseModel):
    """评测问题生成结果。"""
    questions: list[str] = Field(description="两个用户可能提出的问题")


QA_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是 RAG 评测集构造器。基于给定文档生成 2 个用户可能提出的问题。

要求:
1. 问题的答案必须能直接从文档中找到（这是检索真值）
2. 用真实的用户口吻（"怎么…""如何…""为什么…"），不要照抄文档标题
3. 问题不提及"文档/资料/根据"等字样
4. 两个问题覆盖文档的不同部分"""),
    ("human", "文档标题: {title}\n文档分类: {category}\n\n文档内容:\n{text}"),
])

# 边界外问题：知识库不覆盖的主题（期望系统拒答）
OOD_QUESTIONS = [
    "帮我写一首关于秋天的诗",
    "北京明天天气怎么样？",
    "Python 的 GIL 是什么？",
    "推荐几家上海好吃的川菜馆",
    "帮我制定一个健身计划",
    "2024 年世界杯冠军是谁？",
    "如何申请美国研究生？",
    "量子力学的基本原理是什么？",
    "给我讲个笑话",
    "怎么给汽车换机油？",
]


def sample_parents(n_per_category: int = 4, max_total: int = 60) -> list[dict]:
    parents = [json.loads(l) for l in (config.INDEX_DIR / "parents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in parents:
        # 用第二层分类做分层键（如 "功能概览/云文档"），顶层分类都相同没有区分度
        parts = p.get("category", "未分类").split("/")
        key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
        by_cat[key].append(p)
    sampled = []
    for cat, items in sorted(by_cat.items()):
        # 每个分类随机抽若干篇
        sampled.extend(random.sample(items, min(n_per_category, len(items))))
    random.shuffle(sampled)
    sampled = sampled[:max_total]
    logger.info(f"从 {len(parents)} 个父文档、{len(by_cat)} 个分类中抽样 {len(sampled)} 篇")
    return sampled


def generate_questions(parents: list[dict], out_path=None) -> list[dict]:
    """为每篇抽样文档生成评测问题。"""
    llm = get_structured_llm(QAQuestions, temperature=0.3)
    out_path = out_path or (config.EVAL_DIR / "questions.jsonl")
    records = []
    with out_path.open("w", encoding="utf-8") as f:
        for p in tqdm(parents, desc="生成评测问题"):
            try:
                resp = llm.invoke(QA_GEN_PROMPT.format(
                    title=p.get("title", ""), category=p.get("category", ""),
                    text=p["text"][:2000],
                ))
                for q in resp.questions[:2]:
                    rec = {
                        "question": q,
                        "gt_parent_id": p["chunk_id"],
                        "category": p.get("category", ""),
                        "title": p.get("title", ""),
                        "type": "kb",
                    }
                    records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"生成失败 {p['chunk_id']}: {e}")
    logger.info(f"生成 {len(records)} 个知识库评测问题 → {out_path.name}")
    return records


def generate_ood(out_path=None) -> list[dict]:
    out_path = out_path or (config.EVAL_DIR / "questions.jsonl")
    records = [{"question": q, "gt_parent_id": "", "type": "ood"} for q in OOD_QUESTIONS]
    with out_path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"追加 {len(records)} 个边界外问题")
    return records


def build_all() -> None:
    parents = sample_parents()
    generate_questions(parents)
    generate_ood()


if __name__ == "__main__":
    build_all()
