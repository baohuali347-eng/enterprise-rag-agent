"""原始 HTML → Markdown 语料清洗。

飞书帮助中心文章页的服务端渲染数据在 window._templateValue 里：
- 文章按"版本页签"组织（多数文章 1 个页签，功能变化路径类文章按年份分页签）
- 每个页签的 html.html 是保留 h2/h3 层级的富文本 HTML

处理流程:
1. 正则提取 _templateValue JSON
2. 找到含 tabName + html 字段的列表（字段名是混淆的，按结构识别）
3. 每个页签 HTML 用 markdownify 转 Markdown（ATX 标题，图片只留 alt 文本）
4. 写入 corpus/{category}/{article_id}.md，同时输出 manifest.jsonl 元数据
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from loguru import logger
from markdownify import markdownify as md
from tqdm import tqdm

from kb_agent import config

TEMPLATE_RE = re.compile(r"window\._templateValue\s*=\s*(\{.*?\})\s*;\s*</script>", re.S)
TITLE_RE = re.compile(r"<title>([^<]*)</title>")

# 过滤掉与正文无关的噪音段落（页脚/推广/问卷等）
NOISE_PATTERNS = [
    r"^\s*前往下载飞书最新版本.*尽在掌握\.?\s*$",
    r"^\s*感觉有帮助吗\?.*$",
    r"^\s*本文是否对你有帮助\?.*$",
]


def extract_template(html: str) -> dict | None:
    m = TEMPLATE_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning(f"_templateValue 解析失败: {e}")
        return None


def find_tabs(data: dict) -> list[tuple[str, str]]:
    """按结构识别版本页签列表，返回 [(tab_name, body_html)]。

    多数文章有 tabName（如 "2026 年"）；少数文章无该字段（此时页签名置空）。
    """
    for value in data.values():
        if not isinstance(value, list) or not value:
            continue
        tabs = []
        for item in value:
            if not isinstance(item, dict):
                break
            html = item.get("html")
            body = html.get("html") if isinstance(html, dict) else None
            if not isinstance(body, str) or len(body) < 100:
                break
            tab = item.get("tabName") or ""
            tabs.append((tab, body))
        else:
            if tabs:
                return tabs
    return []


def extract_title(html: str, data: dict) -> str:
    """优先页面 <title>，兜底模板数据里的标题字段。"""
    m = TITLE_RE.search(html)
    if m:
        title = m.group(1).strip()
        # 去掉站点后缀，如 "xxx - 飞书帮助中心"
        title = re.sub(r"[-_|]\s*飞书.*$", "", title).strip()
        if title:
            return title
    # 模板数据里常见的标题字段名（混淆 key，按值内容启发式匹配）
    for v in data.values():
        if isinstance(v, str) and 4 < len(v) < 80 and "\n" not in v:
            if v in html:
                return v
    return ""


def clean_markdown(text: str) -> str:
    """逐行过滤噪音段落，规整空白。"""
    lines = []
    for line in text.splitlines():
        line = unicodedata.normalize("NFKC", line).strip()
        if not line:
            continue
        if any(re.match(p, line) for p in NOISE_PATTERNS):
            continue
        lines.append(line)
    # 压缩连续空行
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def html_to_markdown(body_html: str) -> str:
    markdown = md(
        body_html,
        heading_style="ATX",
        bullets="*",
        strip=["img", "script", "style", "video", "iframe"],
        # 保留链接 URL（知识库问答需要给出可点击的来源）
        default_title=False,
    )
    return clean_markdown(markdown)


def category_slug(name: str) -> str:
    return re.sub(r"[^\w一-鿿-]+", "_", name).strip("_") or "未分类"


def parse_raw_file(path: Path, manifest: dict) -> list[dict]:
    """解析单篇原始 HTML，返回该文章各页签的语料记录。"""
    article_id = path.stem
    meta = manifest.get(article_id)
    if meta is None:
        logger.warning(f"{article_id} 不在爬虫 manifest 中，跳过")
        return []

    html = path.read_text(encoding="utf-8", errors="ignore")
    data = extract_template(html)
    if data is None:
        logger.warning(f"{article_id} 未找到 _templateValue，跳过")
        return []

    tabs = find_tabs(data)
    if not tabs:
        logger.warning(f"{article_id} 未识别出版本页签，跳过")
        return []

    title = extract_title(html, data)
    multi = len(tabs) > 1

    records = []
    for i, (tab_name, body_html) in enumerate(tabs):
        doc_id = f"{article_id}" if not multi else f"{article_id}-{i}"
        slug = category_slug(meta["category"])
        out_dir = config.CORPUS_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / f"{doc_id}.md"

        md_text = html_to_markdown(body_html)
        header = (
            f"# {title}\n\n"
            f"> 来源: {meta['url']}\n"
            f"> 分类: {meta['category']}\n"
            + (f"> 版本: {tab_name}\n" if multi else "")
            + "\n---\n\n"
        )
        md_path.write_text(header + md_text, encoding="utf-8")

        records.append({
            "doc_id": doc_id,
            "article_id": article_id,
            "title": title,
            "tab": tab_name,
            "category": meta["category"],
            "url": meta["url"],
            "md_path": str(md_path.relative_to(config.ROOT)),
            "char_count": len(md_text),
        })
    return records


def parse_all() -> list[dict]:
    """解析 data/raw 下全部 HTML，重写 corpus/manifest.jsonl。"""
    raw_manifest = config.RAW_DIR / "manifest.jsonl"
    if not raw_manifest.exists():
        raise RuntimeError("未找到 data/raw/manifest.jsonl，请先运行爬虫: uv run python -m kb_agent.crawler")
    manifest = {}
    for line in raw_manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        manifest[item["article_id"]] = item

    files = sorted(config.RAW_DIR.glob("*.html"))
    logger.info(f"待解析 {len(files)} 个原始文件")
    all_records = []
    for f in tqdm(files, desc="解析"):
        all_records.extend(parse_raw_file(f, manifest))

    manifest_path = config.CORPUS_DIR / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_chars = sum(r["char_count"] for r in all_records)
    logger.info(f"解析完成: {len(all_records)} 篇语料, 共 {total_chars:,} 字符")
    return all_records


if __name__ == "__main__":
    parse_all()
