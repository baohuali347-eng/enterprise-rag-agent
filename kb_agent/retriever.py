"""混合检索：向量 + BM25 → RRF 融合 → 重排。

设计决策:
- 为什么需要 RRF（Reciprocal Rank Fusion）？
  向量相似度和 BM25 分数的量纲完全不同，不能直接相加。
  RRF 只看"排名"：score = Σ 1/(k + rank)，对两路排名做倒数融合，
  简单、无需调参、对分数分布不敏感。
- 为什么融合后还要重排？
  BM25 召回的 chunk 可能只是"词面沾边"，语义上并不真正相关。
  用 BGE-reranker（交叉编码器，query 和 document 同时过模型）做精排，
  能显著提升送入 LLM 的上下文质量——这是 RAG 效果提升性价比最高的一步。
- 为什么要取父文档？
  子 chunk 用于匹配（粒度细、噪音低），父文档用于生成（上下文完整）。

支持检索模式（消融实验）:
- mode="dense":   纯向量
- mode="sparse":  纯 BM25
- mode="hybrid":  向量+BM25+RRF，不重排
- mode="full":    混合+重排（生产默认）
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from kb_agent import config
from kb_agent.indexer import VARIANTS, tokenize

RRF_K = 60


@dataclass
class RetrievedChunk:
    """最终返回给 LLM 的检索结果单元。"""
    chunk_id: str
    parent_id: str
    doc_id: str
    title: str
    text: str          # 父文档全文
    category: str
    url: str
    score: float


class HybridRetriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR, variant: str = "hier"):
        self.variant = variant
        self.spec = VARIANTS[variant]
        self.index_dir = index_dir
        self._check_index()
        # 向量库
        client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        self.collection = client.get_collection(self.spec["collection"])
        # BM25
        with (index_dir / self.spec["bm25"]).open("rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.children = data["children"]
        self._child_by_id = {c["chunk_id"]: c for c in self.children}
        # 父文档
        self.parents: dict[str, dict] = {
            p["chunk_id"]: p for p in
            (json.loads(l) for l in (index_dir / self.spec["parents"]).read_text(encoding="utf-8").splitlines() if l.strip())
        }
        # 模型（懒加载，检索时再初始化，避免启动过慢）
        self._dense_model: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

    def _check_index(self) -> None:
        for f in ("chroma", self.spec["bm25"], self.spec["parents"]):
            if not (self.index_dir / f).exists():
                raise RuntimeError(
                    f"索引不完整（缺 {f}），请先构建: uv run python -m kb_agent.indexer")

    @property
    def dense_model(self) -> SentenceTransformer:
        if self._dense_model is None:
            self._dense_model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        return self._dense_model

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")
        return self._reranker

    # ---------- 两路召回 ----------
    def _dense_search(self, query: str, top_k: int = config.DENSE_TOP_K) -> dict[str, int]:
        qvec = self.dense_model.encode([query], normalize_embeddings=True)[0]
        res = self.collection.query(query_embeddings=[qvec.tolist()], n_results=top_k)
        return {cid: i for i, cid in enumerate(res["ids"][0])}

    def _dense_scores(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """纯向量模式：按余弦相似度排序（Chroma 返回距离，1-距离 = 相似度）。"""
        qvec = self.dense_model.encode([query], normalize_embeddings=True)[0]
        res = self.collection.query(query_embeddings=[qvec.tolist()], n_results=top_k)
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def _sparse_search(self, query: str, top_k: int = config.BM25_TOP_K) -> dict[str, int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return {self.children[i]["chunk_id"]: r for r, i in enumerate(ranked)}

    # ---------- RRF 融合 ----------
    @staticmethod
    def rrf_fuse(*rankings: dict[str, int], k: int = RRF_K) -> list[str]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for cid, rank in ranking.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return [cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])]

    # ---------- 结果装配 ----------
    def _to_results(self, ordered: list[tuple[str, float]], top_k: int) -> list[RetrievedChunk]:
        """按顺序取父文档、去重（同一父的多个子 chunk 只算一次）。"""
        seen_parents: set[str] = set()
        results: list[RetrievedChunk] = []
        for cid, score in ordered:
            child = self._child_by_id.get(cid)
            if child is None:
                continue
            parent = self.parents.get(child["parent_id"])
            if parent is None or parent["chunk_id"] in seen_parents:
                continue
            seen_parents.add(parent["chunk_id"])
            results.append(RetrievedChunk(
                chunk_id=cid,
                parent_id=parent["chunk_id"],
                doc_id=parent["doc_id"],
                title=parent.get("title", ""),
                text=parent["text"],
                category=parent.get("category", ""),
                url=parent.get("url", ""),
                score=score,
            ))
            if len(results) >= top_k:
                break
        return results

    # ---------- 主入口 ----------
    def retrieve(self, query: str, top_k: int = config.RERANK_TOP_K,
                 mode: str = "full") -> list[RetrievedChunk]:
        """按模式检索，返回去重后的父文档列表。"""
        if mode == "dense":
            scored = self._dense_scores(query, top_k=max(top_k, config.DENSE_TOP_K))
            return self._to_results(scored, top_k)

        if mode == "sparse":
            # 按 BM25 分数降序取 top_k
            scores = self.bm25.get_scores(tokenize(query))
            n = min(max(top_k, config.BM25_TOP_K), len(scores))
            top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
            ordered = [(self.children[i]["chunk_id"], float(scores[i])) for i in top_idx]
            return self._to_results(ordered, top_k)

        # hybrid / full：先融合
        dense_rank = self._dense_search(query)
        sparse_rank = self._sparse_search(query)
        fused = self.rrf_fuse(dense_rank, sparse_rank)
        candidates = fused[: max(top_k * 4, 20)]
        if not candidates:
            return []
        if mode == "hybrid":
            # 无重排：按融合分排序输出
            fuse_score = {}
            for ranking in (dense_rank, sparse_rank):
                for cid, rank in ranking.items():
                    fuse_score[cid] = fuse_score.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ordered = [(cid, fuse_score.get(cid, 0.0)) for cid in candidates]
            return self._to_results(ordered, top_k)

        # full：交叉编码器精排
        pairs = [(query, self._child_by_id[cid]["text"]) for cid in candidates if cid in self._child_by_id]
        if not pairs:
            return []
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(pairs, scores), key=lambda x: -x[1])
        ordered = []
        for (_, ctext), score in ranked:
            cid = next((c for c in candidates
                        if c in self._child_by_id and self._child_by_id[c]["text"] == ctext), None)
            if cid is not None:
                ordered.append((cid, float(score)))
        return self._to_results(ordered, top_k)


if __name__ == "__main__":
    retriever = HybridRetriever()
    for r in retriever.retrieve("如何设置文档的分享权限？"):
        print(f"[{r.score:.3f}] {r.title} | {r.url}")
        print(r.text[:150].replace("\n", " "))
        print()
