"""全局配置：路径、模型、检索参数集中管理。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env（API key 等敏感信息不进 git）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"        # 爬取的原始 HTML
CORPUS_DIR = DATA_DIR / "corpus"  # 清洗后的 Markdown 语料
INDEX_DIR = DATA_DIR / "index"    # Chroma 向量库持久化目录
EVAL_DIR = DATA_DIR / "eval"      # 评测集与评测结果

for _d in (RAW_DIR, CORPUS_DIR, INDEX_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 模型 ----------
# DeepSeek 兼容 OpenAI 协议，走 langchain-openai 的 ChatOpenAI
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


def _local_model(name: str) -> str:
    """优先用项目内 models/ 下的本地权重（ModelScope 下载），
    不存在则回退 HuggingFace 模型名（sentence-transformers 自动联网下载）。"""
    local = ROOT / "models" / name
    return str(local) if (local / "config.json").exists() else f"BAAI/{name}"


EMBED_MODEL = os.getenv("EMBED_MODEL", _local_model("bge-m3"))
RERANK_MODEL = os.getenv("RERANK_MODEL", _local_model("bge-reranker-v2-m3"))

# ---------- 分块参数 ----------
# 检索单元（子 chunk）的目标长度，单位：字符（中文约 1 字 ≈ 1 token）
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
# 父文档：完整章节，送入 LLM 生成答案时使用
PARENT_MAX_CHARS = 2500

# ---------- 检索参数 ----------
DENSE_TOP_K = 30     # 向量检索候选数
BM25_TOP_K = 30      # BM25 候选数
RERANK_TOP_K = 5     # 重排后送入 LLM 的最终数量
# 相关性阈值：检索最高分低于此值直接拒答，防止用无关内容硬编
RELEVANCE_THRESHOLD = 0.35

# ---------- Agent 参数 ----------
MAX_RETRY = 1  # 反思节点不满意时，最多重新检索一次

# ---------- 爬虫参数 ----------
CRAWL_DELAY = 0.8  # 每次请求间隔（秒），对目标站点保持礼貌
CRAWL_TIMEOUT = 20
CRAWL_MAX_ARTICLES = 400  # 语料规模上限；0 = 不限制。400 篇 ≈ 覆盖核心分类的热门文章
