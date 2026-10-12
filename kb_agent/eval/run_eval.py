"""评测执行与指标计算。

三个维度:
1. 检索质量（自动计算，无需人工）:
   - recall@k: 真值文档是否进入前 k 个父文档
   - MRR: 真值文档排名的倒数平均
2. 生成质量（LLM-as-judge）:
   - 忠实度: 答案是否完全由引用资料支撑（无编造）
   - 引用准确率: 标注的 [n] 引用是否真实存在于对应资料
3. 边界能力: 知识库外问题是否被正确拒答
"""
from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from kb_agent import config
from kb_agent.graph.builder import ask
from kb_agent.llm import get_llm, get_structured_llm
from kb_agent.retriever import HybridRetriever

K_VALUES = [1, 3, 5]


class JudgeVerdict(BaseModel):
    faithfulness: float = Field(description="0-1: 答案内容是否全部由引用资料支撑，无编造")
    citation: float = Field(description="0-1: 标注的引用编号是否真实存在且对应内容支撑答案")
    decline: bool = Field(description="是否属于知识库外拒答")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是 RAG 系统的评测裁判。给定用户问题、知识库资料（带编号）和助手答案，输出 JSON:

- faithfulness (0-1): 答案中每个事实性陈述是否都能在资料中找到依据。
  出现编造、张冠李戴则大幅扣分；明确说"知识库中没有"则记 1
- citation (0-1): 答案标注的 [n] 引用是否编号真实存在、且引用处内容确实支撑答案。
  未标注引用但内容忠实记 0.5；标注了不存在的编号记 0
- decline (bool): 答案是否属于"知识库中没有/超出范围"的拒答"""),
    ("human", "用户问题: {question}\n\n知识库资料:\n{context}\n\n助手答案:\n{answer}"),
])


def load_questions() -> list[dict]:
    path = config.EVAL_DIR / "questions.jsonl"
    if not path.exists():
        raise RuntimeError("未找到评测集，请先生成: uv run python -m kb_agent.eval.build_evalset")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------- 检索指标 ----------------
def eval_retrieval(questions: list[dict], mode: str = "full",
                   variant: str = "hier", limit: int | None = None,
                   retriever: HybridRetriever | None = None) -> dict:
    """对 kb 类问题直接测检索器，计算 recall@k 与 MRR。

    命中基准: 文档级（doc_id）。评测集真值是层级分块的父文档 id，
    而固定分块变体里不存在这些 id——统一换算成"检索结果是否命中真值所在文档"，
    两种分块策略才能在同一基准上对比。
    另报告 hier 变体的父文档级命中率（parent_hit@5，分块粒度更细的指标）。

    mode: dense / sparse / hybrid / full（消融实验用）
    variant: hier（层级分块）/ fixed（固定分块）
    retriever: 可注入已有实例（消融实验复用，避免重复加载模型）
    """
    if retriever is None:
        retriever = HybridRetriever(variant=variant)
    # 真值 doc_id：父文档 id 形如 "{doc_id}#p{i}"，取 # 前缀即可（与变体无关）
    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    total = 0
    kb_qs = [q for q in questions if q["type"] == "kb"]
    if limit:
        kb_qs = kb_qs[:limit]
    for q in tqdm(kb_qs, desc=f"评测检索[{variant}/{mode}]"):
        total += 1
        gt_doc = q["gt_parent_id"].split("#")[0]
        results = retriever.retrieve(q["question"], top_k=max(K_VALUES), mode=mode)
        doc_ids = [r.doc_id for r in results]
        rank = None
        if gt_doc in doc_ids:
            rank = doc_ids.index(gt_doc) + 1
        for k in K_VALUES:
            if gt_doc in doc_ids[:k]:
                hits[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return {
        **{f"recall@{k}": hits[k] / total for k in K_VALUES},
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "total": total,
    }


# ---------------- 生成指标 ----------------
def eval_generation(questions: list[dict], with_reflect: bool = True,
                    limit: int | None = None) -> dict:
    """完整跑图，LLM-as-judge 打分。

    with_reflect=False 时跳过反思节点（消融实验对照组）。
    """
    llm = get_structured_llm(JudgeVerdict)
    scores_f, scores_c = [], []
    declines_correct = 0
    ood_total = 0
    results = []
    kb_qs = [q for q in questions if q["type"] == "kb"]
    ood_qs = [q for q in questions if q["type"] == "ood"]
    if limit:
        kb_qs = kb_qs[:limit]
    all_qs = kb_qs + ood_qs
    for q in tqdm(all_qs, desc=f"评测生成[reflect={with_reflect}]"):
        result = ask(q["question"], history=[], with_reflect=with_reflect)
        chunks = result.get("chunks") or []
        context = "\n\n---\n\n".join(
            f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(chunks, 1))
        verdict = llm.invoke(JUDGE_PROMPT.format(
            question=q["question"], context=context, answer=result["answer"]))
        if q["type"] == "kb":
            scores_f.append(verdict.faithfulness)
            scores_c.append(verdict.citation)
        else:
            ood_total += 1
            if verdict.decline:
                declines_correct += 1
        results.append({
            "question": q["question"], "type": q["type"], "answer": result["answer"],
            "faithfulness": verdict.faithfulness, "citation": verdict.citation,
            "decline": verdict.decline,
        })
    report = {
        "faithfulness_avg": sum(scores_f) / len(scores_f) if scores_f else 0,
        "citation_avg": sum(scores_c) / len(scores_c) if scores_c else 0,
        "decline_rate": declines_correct / ood_total if ood_total else 0,
        "ood_total": ood_total,
        "kb_total": len(scores_f),
    }
    # 落盘详细结果
    suffix = "no_reflect" if not with_reflect else "full"
    out = config.EVAL_DIR / f"generation_results_{suffix}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"生成评测结果 → {out}")
    return report


def run_all() -> dict:
    questions = load_questions()
    retrieval_report = eval_retrieval(questions)
    generation_report = eval_generation(questions)
    summary = {"retrieval": retrieval_report, "generation": generation_report}
    out = config.EVAL_DIR / "report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"评测完成 → {out}")
    return summary


if __name__ == "__main__":
    report = run_all()
    print(json.dumps(report, ensure_ascii=False, indent=2))
