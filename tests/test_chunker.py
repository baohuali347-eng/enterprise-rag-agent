"""分块器单元测试。

运行: uv run pytest tests/ -v
"""
from kb_agent.chunker import (
    chunk_document,
    chunk_fixed,
    split_child_windows,
    split_sections,
)


SAMPLE_MD = """# 如何设置文档权限

> 来源: https://example.com
> 分类: 云文档/知识库

---

## 打开权限设置

点击右上角的分享按钮，选择权限设置。

## 设置查看权限

你可以将文档设置为组织内可见，或仅指定成员可见。

### 指定成员

在成员选择器中输入成员姓名，勾选即可添加。

### 设置有效期

你可以设置链接有效期，过期后链接自动失效。

## 常见问题

### 为什么收不到通知

请检查消息通知设置。
"""


def test_split_sections_preserves_hierarchy():
    doc = {"doc_id": "123", "category": "测试", "url": "https://example.com"}
    parents, children = chunk_document(doc, SAMPLE_MD)
    # 三个 h2 section；h3 碎片并入其父 h2 section
    assert len(parents) == 3
    paths = [" > ".join(p.heading_path) for p in parents]
    assert "打开权限设置" in paths[0]
    assert any("常见问题" in p for p in paths)
    # 每个父文档至少一个子 chunk
    assert len(children) >= len(parents)
    # 子 chunk 引用父文档
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_child_chunk_size_limit():
    long_text = "\n".join([f"第 {i} 段，" + "内容" * 80 for i in range(50)])
    windows = split_child_windows(long_text, size=400, overlap=50)
    # 所有窗口都不应超过上限太多
    for w in windows:
        assert len(w) <= 500
    # 原文内容不丢失（拼接后去除重叠也应有大部分段落）
    joined = "\n".join(windows)
    for i in range(50):
        assert f"第 {i} 段" in joined


def test_empty_document():
    doc = {"doc_id": "empty", "category": "测试", "url": "https://example.com"}
    parents, children = chunk_document(doc, "# 空文档\n")
    # 无正文时不产出空 chunk
    assert all(p.text.strip() for p in parents)
    assert all(c.text.strip() for c in children)


def test_chunk_fixed_flat_windows():
    """固定分块（消融基线）：窗口即父文档，无层级结构。"""
    doc = {"doc_id": "456", "category": "测试", "url": "https://example.com"}
    body = "\n".join([f"第 {i} 段，" + "内容" * 100 for i in range(40)])
    md_text = "# 标题\n\n> 来源: https://example.com\n\n---\n\n" + body
    parents, children = chunk_fixed(doc, md_text)
    assert len(parents) >= 2           # 长文档被切成多个窗口
    assert len(parents) == len(children)
    # 固定分块父=子，且不保留标题层级（heading_path 只有文章标题）
    assert parents[0].chunk_id == children[0].parent_id
    assert all(len(p.heading_path) == 1 for p in parents)
    # 内容不丢失
    joined = "\n".join(p.text for p in parents)
    assert "第 39 段" in joined
