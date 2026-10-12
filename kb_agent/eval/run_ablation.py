"""消融实验：量化每个组件的贡献。

两组对照:

1. 检索消融（无需 LLM，纯指标计算）:
   分块策略 × 检索模式 的 8 组组合，统一在 recall@k / MRR 上对比
   - 分块: fixed（固定长度窗口，朴素基线） vs hier（标题层级+父子文档）
   - 模式: dense（纯向量）/ sparse（纯 BM25）/ hybrid（双路+RRF）/ full（+重排）

2. 生成消融（LLM-as-judge）:
   完整图（含反思-重试） vs 去掉反思节点，对比忠实度与引用准确率

用法: uv run python -m kb_agent.eval.run_ablation [--limit N]
输出: data/eval/ablation_report.json + 终端 Markdown 表格
"""
from __future__ import annotations

import argparse
import json

from loguru import logger

from kb_agent import config
from kb_agent.eval.run_eval import (
    eval_generation,
    eval_retrieval,
    load_questions,
)
from kb_agent.retriever import HybridRetriever

RETRIEVAL_MODES = ["dense", "sparse", "hybrid", "full"]
RETRIEVAL_VARIANTS = ["fixed", "hier"]


def run_retrieval_ablation(questions: list[dict], limit: int | None) -> list[dict]:
    """分块 × 模式 全组合的检索指标。

    模型加载开销大（Embedding + Reranker 各数 GB），两个变体共用一个模型实例：
    先跑完 fixed 的四种模式（模型随之加载），再注入 hier 实例复用。
    """
    rows = []
    retrievers = {v: HybridRetriever(variant=v) for v in RETRIEVAL_VARIANTS}
    for variant in RETRIEVAL_VARIANTS:
        retriever = retrievers[variant]
        if variant != RETRIEVAL_VARIANTS[0]:
            first = retrievers[RETRIEVAL_VARIANTS[0]]
            retriever._dense_model = first._dense_model
            retriever._reranker = first._reranker
        for mode in RETRIEVAL_MODES:
            metrics = eval_retrieval(questions, mode=mode, variant=variant,
                                     limit=limit, retriever=retriever)
            rows.append({
                "chunking": variant, "retrieval": mode,
                **metrics,
            })
            logger.info(f"[检索消融] {variant}/{mode}: "
                        f"recall@5={metrics['recall@5']:.3f}, MRR={metrics['mrr']:.3f}")
    return rows


def run_generation_ablation(questions: list[dict], limit: int | None) -> list[dict]:
    rows = []
    for with_reflect in (True, False):
        report = eval_generation(questions, with_reflect=with_reflect, limit=limit)
        rows.append({
            "graph": "with_reflect" if with_reflect else "no_reflect",
            **report,
        })
        logger.info(f"[生成消融] reflect={with_reflect}: "
                    f"忠实度={report['faithfulness_avg']:.3f}, "
                    f"引用准确率={report['citation_avg']:.3f}, "
                    f"拒答率={report['decline_rate']:.3f}")
    return rows


def fmt_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """输出 Markdown 表格。columns: (key, 表头, 格式)。"""
    header = "| " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            val = row.get(key, "-")
            cells.append(format(val, fmt) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_all(limit: int | None = None) -> dict:
    questions = load_questions()
    logger.info(f"评测集规模: {len(questions)} 题（检索消融 limit={limit}）")
    retrieval_rows = run_retrieval_ablation(questions, limit)
    generation_rows = run_generation_ablation(questions, limit)
    report = {"retrieval": retrieval_rows, "generation": generation_rows}
    out = config.EVAL_DIR / "ablation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"消融报告 → {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--limit", type=int, default=50,
                        help="每类问题抽样上限（检索消融建议 50，全量不传 0 之外的数）")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只跑检索消融（无需 API key，可独立复现）")
    parser.add_argument("--generation-only", action="store_true",
                        help="只跑生成消融")
    args = parser.parse_args()
    questions = load_questions()
    logger.info(f"评测集规模: {len(questions)} 题（limit={args.limit}）")

    report: dict = {}
    if not args.generation_only:
        report["retrieval"] = run_retrieval_ablation(questions, args.limit)
    if not args.retrieval_only:
        report["generation"] = run_generation_ablation(questions, args.limit)
    out = config.EVAL_DIR / "ablation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"消融报告 → {out}")

    if "retrieval" in report:
        print("\n## 检索消融（recall@k / MRR）\n")
        print(fmt_table(report["retrieval"], [
            ("chunking", "分块策略", "s"),
            ("retrieval", "检索模式", "s"),
            ("recall@1", "recall@1", ".3f"),
            ("recall@3", "recall@3", ".3f"),
            ("recall@5", "recall@5", ".3f"),
            ("mrr", "MRR", ".3f"),
        ]))
    if "generation" in report:
        print("\n## 生成消融（LLM-as-judge）\n")
        print(fmt_table(report["generation"], [
            ("graph", "图配置", "s"),
            ("faithfulness_avg", "忠实度", ".3f"),
            ("citation_avg", "引用准确率", ".3f"),
            ("decline_rate", "拒答率", ".3f"),
            ("kb_total", "样本数", "d"),
        ]))


if __name__ == "__main__":
    main()
