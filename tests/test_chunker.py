"""分块器单元测试。

运行: uv run pytest tests/ -v
"""
import pytest

from kb_agent import config
from kb_agent.chunker import (
    chunk_document,
    chunk_fixed,
    split_child_windows,
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
    # 长度上限是严格限制
    for w in windows:
        assert len(w) <= 400
    # 原文内容不丢失（拼接后去除重叠也应有大部分段落）
    joined = "\n".join(windows)
    for i in range(50):
        assert f"第 {i} 段" in joined


def test_empty_document():
    doc = {"doc_id": "empty", "category": "测试", "url": "https://example.com"}
    parents, children = chunk_document(doc, "# 空文档\n")
    # 无正文时不产出空 chunk
    assert parents == []
    assert children == []


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


def stitch(windows, overlap):
    """按已知重叠还原原文，防止只检查首尾而遗漏中间内容。"""
    return windows[0] + "".join(w[overlap:] for w in windows[1:]) if windows else ""


@pytest.mark.parametrize("text", [
    "内容" * 1500,
    "前文。" * 200 + "\n" + "后文！" * 300,
    "短段\n" * 500,
    "\n" + "内容" * 500 + "\n",
])
@pytest.mark.parametrize("size,overlap", [(400, 50), (11, 0), (7, 6)])
def test_windows_preserve_every_character_and_overlap(text, size, overlap):
    windows = split_child_windows(text, size=size, overlap=overlap)
    assert all(0 < len(w) <= size for w in windows)
    assert stitch(windows, overlap) == text
    if overlap:
        assert all(a[-overlap:] == b[:overlap] for a, b in zip(windows, windows[1:]))


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10)])
def test_invalid_window_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_child_windows("正文", size=size, overlap=overlap)


@pytest.mark.parametrize("text", ["", "# 空文档\n", "# 空文档\n\n> 来源: x\n\n---\n"])
def test_empty_documents_have_no_chunks(text):
    for chunker in (chunk_document, chunk_fixed):
        assert chunker({"doc_id": "empty"}, text) == ([], [])


@pytest.mark.parametrize("heading", ["权限设置", "超长标题" * 300])
def test_parent_child_limits_include_heading_and_preserve_body(heading):
    body = "".join(f"操作步骤{i:04d}：点击权限菜单。" for i in range(700))
    parents, children = chunk_document(
        {"doc_id": "long", "category": "云文档", "url": "https://example.com"},
        f"# 文档标题\n\n## {heading}\n{body}",
    )
    assert len(parents) > 1
    assert len({p.chunk_id for p in parents}) == len(parents)
    assert len({c.chunk_id for c in children}) == len(children)
    parent_map = {p.chunk_id: p for p in parents}
    parent_bodies = [p.text.split("\n", 1)[1] for p in parents]
    assert stitch(parent_bodies, config.CHILD_CHUNK_OVERLAP) == body
    for parent in parents:
        assert len(parent.text) <= config.PARENT_MAX_CHARS
        assert parent.heading_path == [heading]
        assert parent.doc_id == "long"
        own_children = [c for c in children if c.parent_id == parent.chunk_id]
        assert own_children
        assert stitch([c.text.split("\n", 1)[1] for c in own_children],
                      config.CHILD_CHUNK_OVERLAP) == parent.text.split("\n", 1)[1]
    for child in children:
        assert len(child.text) <= config.CHILD_CHUNK_SIZE
        assert child.metadata["char_count"] == len(child.text)
        assert child.parent_id in parent_map


def test_merged_subsections_keep_operation_labels():
    parents, _ = chunk_document({"doc_id": "labels"},
        "# 文档\n## 权限\n操作入口。\n### 开启\n点击确认。\n### 关闭\n点击取消。")
    assert "开启\n点击确认。" in parents[0].text
    assert "关闭\n点击取消。" in parents[0].text


def test_fixed_chunks_bound_long_unbroken_paragraph():
    parents, children = chunk_fixed({"doc_id": "fixed"}, "# 标题\n" + "长段落" * 2000)
    assert all(len(c.text) <= config.CHILD_CHUNK_SIZE for c in children)
    assert stitch([p.text for p in parents], config.CHILD_CHUNK_OVERLAP) == "标题\n" + "长段落" * 2000
