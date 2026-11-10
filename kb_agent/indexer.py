"""构建检索索引：向量（BGE-M3 + Chroma）与 BM25（jieba 分词）双路。

同时构建两套索引（供消融实验对比分块策略）:
- hier:  标题层级分块 + 父子文档（生产方案）
- fixed: 固定长度滑动窗口（朴素基线）

设计决策:
- 为什么双路？向量检索擅长语义相似（"怎么撤回消息" ≈ "如何撤销发送"），
  BM25 擅长精确词面匹配（产品名、报错码、菜单名）。二者互补，
  混合后召回更稳。中文场景下 BM25 必须先做分词（jieba），否则等于没开。
- 为什么 Embedding 用本地模型？BGE-M3 是中文效果最好的开源 Embedding 之一，
  本地跑零成本、数据不出本机；语料几千条规模 CPU 完全够用。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
import jieba
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kb_agent import config

jieba.setLogLevel("WARN")

# 两种分块变体的索引命名
VARIANTS = {
    "hier": {"collection": "kb_chunks", "bm25": "bm25.pkl",
             "children": "children.jsonl", "parents": "parents.jsonl"},
    "fixed": {"collection": "kb_chunks_fixed", "bm25": "bm25_fixed.pkl",
              "children": "children_fixed.jsonl", "parents": "parents_fixed.jsonl"},
}


def tokenize(text: str) -> list[str]:
    """jieba 搜索模式分词：细粒度切分，利于召回短词与专名。"""
    return [t for t in jieba.lcut_for_search(text) if t.strip()]


"""
build_dense()：将文本转换为向量，进行语义检索。
build_sparse()：基于关键词建立 BM25 索引，进行关键词检索。
"""
class Indexer:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.children: dict[str, list[dict]] = {}
        self.parents: dict[str, dict[str, dict]] = {}

    # ---------- 数据加载 ----------
    def load_corpus(self, variant: str = "hier") -> None:
        spec = VARIANTS[variant]
        children_path = config.CORPUS_DIR / spec["children"]
        parents_path = config.CORPUS_DIR / spec["parents"]
        if not children_path.exists():
            raise RuntimeError(f"未找到 {spec['children']}，请先运行分块: uv run python -m kb_agent.chunker")
        self.children[variant] = [
            json.loads(l) for l in children_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        self.parents[variant] = {
            p["chunk_id"]: p
            for p in (json.loads(l) for l in parents_path.read_text(encoding="utf-8").splitlines() if l.strip())
        }
        #打印结果
        logger.info(f"[{variant}] 加载语料: {len(self.children[variant])} 个子 chunk, "
                    f"{len(self.parents[variant])} 个父文档")

    # ---------- 向量索引 ----------
    # 加载模型 → 读取子分块 → 文本向量化 → 存入 ChromaDB → 建立向量索引。
    def build_dense(self, variant: str = "hier") -> None:
        logger.info("加载 BGE-M3 Embedding 模型（首次运行需下载模型）…")
        model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        spec = VARIANTS[variant]
        ## 创建一个持久化的 ChromaDB 客户端。向量数据可以保存在本地磁盘中，而不是只存在于程序运行期间的内存里。path就是路径
        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))

        # 重建集合（语料变化时全量重建，规模小、可接受）
        existing = [c.name for c in client.list_collections()]
        if spec["collection"] in existing:
            client.delete_collection(spec["collection"])
        collection = client.create_collection(
            name=spec["collection"],
            metadata={"hnsw:space": "cosine"},
        )
        children = self.children[variant]
        batch_size = 64
        for i in tqdm(range(0, len(children), batch_size), desc=f"[{variant}] 向量化"):
            batch = children[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                                      show_progress_bar=False)
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=[e.tolist() for e in embeddings],
                documents=texts,
                metadatas=[{"parent_id": c["parent_id"], "char_count": c.get("char_count", 0)}
                           for c in batch],
            )
        logger.info(f"[{variant}] 向量索引完成: {len(children)} 条向量 → {collection.count()}")

    # ---------- BM25 索引 ----------
    #给子块建立 BM25 关键词检索索引，并保存到本地。
    def build_sparse(self, variant: str = "hier") -> None:
        logger.info(f"[{variant}] 构建 BM25 索引（jieba 分词）…")
        children = self.children[variant]
        corpus = [tokenize(c["text"]) for c in tqdm(children, desc=f"[{variant}] 分词")]
        bm25 = BM25Okapi(corpus)
        with (self.index_dir / VARIANTS[variant]["bm25"]).open("wb") as f:
            pickle.dump({"bm25": bm25, "children": children}, f)
        logger.info(f"[{variant}] BM25 索引完成: {len(corpus)} 篇")

    # ---------- 主流程 ----------
    def build_all(self, variants: list[str] | None = None) -> None:
        variants = variants or list(VARIANTS)
        for v in variants:
            self.load_corpus(v)
            self.build_sparse(v)
        # 两套索引共用一个模型实例，避免重复加载
        logger.info("加载 BGE-M3 Embedding 模型…")
        model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        for v in variants:
            self._build_dense_with(v, model, client)
            # 父文档快照（检索命中后取父内容用）
            with (self.index_dir / VARIANTS[v]["parents"]).open("w", encoding="utf-8") as f:
                for p in self.parents[v].values():
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info("索引构建完成")

    def _build_dense_with(self, variant: str, model: SentenceTransformer,
                          client: chromadb.PersistentClient) -> None:
        spec = VARIANTS[variant]
        existing = [c.name for c in client.list_collections()]
        if spec["collection"] in existing:
            client.delete_collection(spec["collection"])
        collection = client.create_collection(name=spec["collection"],
                                              metadata={"hnsw:space": "cosine"})
        children = self.children[variant]
        batch_size = 64
        for i in tqdm(range(0, len(children), batch_size), desc=f"[{variant}] 向量化"):
            batch = children[i:i + batch_size]
            embeddings = model.encode(
                [c["text"] for c in batch],
                normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=[e.tolist() for e in embeddings],
                documents=[c["text"] for c in batch],
                metadatas=[{"parent_id": c["parent_id"], "char_count": c.get("char_count", 0)}
                           for c in batch],
            )
        logger.info(f"[{variant}] 向量索引完成: {collection.count()}")


if __name__ == "__main__":
    Indexer().build_all()
