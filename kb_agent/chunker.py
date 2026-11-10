"""Markdown 语料分块：标题层级分块 + 父子文档。

设计决策（面试常问，务必能讲清楚）:

1. 为什么按标题层级分块而不是固定长度？
   帮助中心文章是结构化的（h1 文章标题 / h2 功能模块 / h3 子功能），
   标题边界天然是语义边界。固定长度切分会把"问题"和"操作步骤"拦腰截断，
   导致检索回的 chunk 缺上下文。层级分块保证每个 chunk 是一个语义完整的单元。

2. 什么是父子文档（parent-child），解决什么问题？
   检索需要小块（精确匹配、噪音少），生成需要大块（上下文完整、答案不残缺）。
   两者目标冲突，所以拆两层：
   - 子 chunk（~400 字）：建索引、做检索
   - 父 chunk（章节或章节片段，含标题≤2500 字）：检索命中后送入 LLM
   类比：目录（子 chunk）找到章节号，翻到整章（父 chunk）阅读。

3. 为什么子 chunk 要带标题路径前缀？
   很多 chunk 的正文是"点击右上角…"，离开标题无法判断在讲什么。
   前缀（如 "云文档 > 知识库 > 权限设置"）让 chunk 自带上下文，
   对向量检索和 BM25 检索都有明显帮助。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from kb_agent import config

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")
FRONT_MATTER_END = "---"


@dataclass
class Section:
    """一篇文章切出的一个语义单元。"""
    heading_path: list[str]   # 标题路径，如 ["云文档", "知识库", "权限设置"]
    level: int                # 当前标题层级（1=h1 … 4=h4）
    body: str                 # 本 section 正文（不含标题行）


@dataclass
class ParentChunk:
    chunk_id: str
    doc_id: str
    title: str
    heading_path: list[str]
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def strip_front_matter(text: str) -> tuple[str, str]:
    """剥离解析阶段写入的元信息头（# 标题 / > 来源），返回 (正文, 标题)。"""
    lines = text.splitlines()
    if not lines:
        return "", ""
    title = ""
    start = 0
    if lines[0].startswith("# "):
        title = lines[0][2:].strip()
        start = 1
    # 跳过引用元信息与分隔线，直到正文
    for i in range(start, len(lines)):
        line = lines[i].strip()
        if line.startswith(">") or line in ("---", ""):
            continue
        start = i
        break
    else:
        start = len(lines)
    return "\n".join(lines[start:]), title


def split_sections(body: str) -> list[Section]:
    """把正文按标题层级切成 Section 列表；无标题的正文整体为一个 Section。"""
    lines = body.splitlines()
    sections: list[Section] = [] #把正文内容进行拆分个sections
    stack: list[tuple[int, str]] = []  # (level, text)，表示标题部分。用栈来表示。
    buf: list[str] = [] #缓冲区，正文的部分

    # 先把当前缓冲区的正文写入 sections
    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        path = [t for _, t in stack]
        sections.append(Section(heading_path=path, level=stack[-1][0] if stack else 1, body=text))

    # 遍历每一行，遇到标题就 flush 之前的正文，更新标题路径
    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            flush()
            buf = []
            level, text = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        else:
            buf.append(line)
    flush()

    # 合并碎片：只有"过短且是上一 section 标题路径的后代"才并入。
    # 例: "设置查看权限 > 指定成员"（短）并入 "设置查看权限"；
    # 但 "常见问题 > 为什么收不到通知" 是 FAQ 的独立一问，不能并入别的 h2。
    merged: list[Section] = []
    for sec in sections:
        prev_path = merged[-1].heading_path if merged else []
        #判断是否是上一节的子标题
        is_descendant = (len(sec.heading_path) > len(prev_path)
                         and sec.heading_path[:len(prev_path)] == prev_path)
        if merged and len(sec.body) < 100 and is_descendant:
            # 合并时保留子标题，否则“开启/关闭”等操作会失去语义边界。
            subheading = " > ".join(sec.heading_path[len(prev_path):])
            merged[-1].body += f"\n{subheading}\n{sec.body}"
        else:
            merged.append(sec)
    return merged


def split_child_windows(text: str, size: int = config.CHILD_CHUNK_SIZE,
                        overlap: int = config.CHILD_CHUNK_OVERLAP) -> list[str]:
    """优先在换行/句末切分；超长单段按字符切分，严格限制窗口长度。

    返回原文的连续切片，不丢字符；相邻窗口保留 overlap 个字符。
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("要求 size > 0 且 0 <= overlap < size")
    if not text.strip():
        return []
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 只在窗口后半段寻找边界，避免切出过短片段或被重叠阻止前进。
            lower = start + max(size // 2, overlap + 1)
            for separators in ("\n", "。！？!?；;"):
                boundary = max(text.rfind(s, lower, end) for s in separators)
                if boundary >= lower:
                    end = boundary + 1
                    break
        windows.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return windows


def _chunk_prefix(path: list[str]) -> str:
    """为正文预留空间；极长标题仅缩短展示前缀，完整路径保留在元数据中。"""
    prefix = " > ".join(path)
    limit = min(config.CHILD_CHUNK_SIZE, config.PARENT_MAX_CHARS) // 2
    if limit < 2:
        raise ValueError("父子块长度限制至少为 4")
    return prefix if len(prefix) <= limit else "…" + prefix[-(limit - 1):]


def chunk_document(doc: dict, md_text: str) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """把单篇语料切成父/子两层 chunk。"""
    body, title = strip_front_matter(md_text)
    if not body.strip():
        return [], []
    sections = split_sections(body) or [Section(heading_path=[title], level=1, body=body)]

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    for i, sec in enumerate(sections):
        path = sec.heading_path or [title or f"未命名-{i}"]
        prefix = _chunk_prefix(path)
        prefix_size = len(prefix) + 1
        parent_budget = config.PARENT_MAX_CHARS - prefix_size
        child_budget = config.CHILD_CHUNK_SIZE - prefix_size
        # 超长章节继续切成父片段，边界保留重叠，防止操作步骤被切断。
        parent_windows = split_child_windows(
            sec.body, size=parent_budget,
            overlap=min(config.CHILD_CHUNK_OVERLAP, parent_budget - 1),
        )
        for part, parent_body in enumerate(parent_windows):
            # 未拆分章节沿用旧 id；新片段的 id 仍能通过 # 前缀定位原文档。
            pid = f"{doc['doc_id']}#p{i}" + (f".s{part}" if part else "")
            parents.append(ParentChunk(
                chunk_id=pid, doc_id=doc["doc_id"], title=title or path[-1],
                heading_path=path, text=f"{prefix}\n{parent_body}",
                metadata={"category": doc.get("category", ""), "url": doc.get("url", "")},
            ))
            windows = split_child_windows(
                parent_body, size=child_budget,
                overlap=min(config.CHILD_CHUNK_OVERLAP, child_budget - 1),
            )
            for j, win in enumerate(windows):
                children.append(ChildChunk(
                    chunk_id=f"{pid}#c{j}", parent_id=pid,
                    text=f"{prefix}\n{win}",
                    metadata={"char_count": prefix_size + len(win)},
                ))
    return parents, children


def load_corpus_manifest() -> list[dict]:
    path = config.CORPUS_DIR / "manifest.jsonl"
    if not path.exists():
        raise RuntimeError("未找到 corpus/manifest.jsonl，请先运行解析: uv run python -m kb_agent.parser")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_fixed(doc: dict, md_text: str) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """对照组分块：固定长度滑动窗口，无标题结构（消融实验用）。

    窗口即父文档（父=子），模拟最常见的"朴素分块"做法。
    """
    body, title = strip_front_matter(md_text)
    if not body.strip():
        return [], []
    text = f"{title}\n{body}".strip()
    if not text:
        return [], []
    windows = split_child_windows(text)
    parents, children = [], []
    for j, win in enumerate(windows):
        cid = f"{doc['doc_id']}#f{j}"
        parents.append(ParentChunk(
            chunk_id=cid, doc_id=doc["doc_id"], title=title or f"文档-{doc['doc_id']}",
            heading_path=[title], text=win,
            metadata={"category": doc.get("category", ""), "url": doc.get("url", "")},
        ))
        children.append(ChildChunk(
            chunk_id=cid, parent_id=cid, text=win,
            metadata={"char_count": len(win)},
        ))
    return parents, children


def chunk_all() -> dict[str, int]:
    """切分全部语料，输出 parents.jsonl / children.jsonl（层级分块）及
    parents_fixed.jsonl / children_fixed.jsonl（固定分块对照）。"""
    docs = load_corpus_manifest()
    parents_out = config.CORPUS_DIR / "parents.jsonl"
    children_out = config.CORPUS_DIR / "children.jsonl"
    parents_fixed_out = config.CORPUS_DIR / "parents_fixed.jsonl"
    children_fixed_out = config.CORPUS_DIR / "children_fixed.jsonl"
    p_count = c_count = pf_count = cf_count = 0
    with parents_out.open("w", encoding="utf-8") as pf, \
            children_out.open("w", encoding="utf-8") as cf, \
            parents_fixed_out.open("w", encoding="utf-8") as pff, \
            children_fixed_out.open("w", encoding="utf-8") as cff:
        for doc in tqdm(docs, desc="分块"):
            md_text = (config.ROOT / doc["md_path"]).read_text(encoding="utf-8")
            parents, children = chunk_document(doc, md_text)
            p_count += _dump_parents(parents, pf)
            c_count += _dump_children(children, cf)
            f_parents, f_children = chunk_fixed(doc, md_text)
            pf_count += _dump_parents(f_parents, pff)
            cf_count += _dump_children(f_children, cff)
    logger.info(
        f"分块完成: {len(docs)} 篇语料 → 层级: {p_count} 父 / {c_count} 子 | "
        f"固定: {pf_count} 父 / {cf_count} 子"
    )
    return {"docs": len(docs), "parents": p_count, "children": c_count,
            "parents_fixed": pf_count, "children_fixed": cf_count}


def _dump_parents(parents: list[ParentChunk], fh) -> int:
    for p in parents:
        fh.write(json.dumps({
            "chunk_id": p.chunk_id, "doc_id": p.doc_id, "title": p.title,
            "heading_path": p.heading_path, "text": p.text, **p.metadata,
        }, ensure_ascii=False) + "\n")
    return len(parents)


def _dump_children(children: list[ChildChunk], fh) -> int:
    for c in children:
        fh.write(json.dumps({
            "chunk_id": c.chunk_id, "parent_id": c.parent_id, "text": c.text,
            **c.metadata,
        }, ensure_ascii=False) + "\n")
    return len(children)


if __name__ == "__main__":
    chunk_all()
