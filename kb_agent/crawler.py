"""飞书帮助中心爬虫。

目标站点: https://www.feishu.cn/hc/zh-CN/ （robots.txt 对该路径完全开放）

数据源: /hc/api/categoryTree 接口一次返回完整站点地图
（364 个分类、2564 篇文章及分类归属），据此精确挑选目标分类的文章。

策略:
1. 拉取 categoryTree，构建分类树（id → 名称/父节点）
2. 按优先级关键词（管理员后台、审批、考勤、权限等更贴近"企业知识库"的分类）
   选取顶层分类，收集其子树下的全部文章
3. 按文章 weight（站内热度）排序，限制总量，逐篇下载原始 HTML 到 data/raw/
4. 分类归属、标题、更新时间写入 raw/manifest.jsonl 供解析阶段使用

合规与礼貌: 遵守 robots.txt，请求间隔 CRAWL_DELAY 秒，带 UA，失败重试。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

from kb_agent import config

BASE_URL = "https://www.feishu.cn"
HC_HOME = f"{BASE_URL}/hc/zh-CN/"
CATEGORY_TREE_API = f"{BASE_URL}/hc/api/categoryTree"

# 更贴近"企业内部知识库"的顶层分类（按优先级排序）；留空则抓取全部
PRIORITY_CATEGORIES = [
    "企业管理后台", "账号与设置", "审批", "假勤", "工作台",
    "云文档", "多维表格", "知识库", "即时消息", "日历",
    "邮箱", "视频会议", "通讯录", "安全",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": HC_HOME,
}


@dataclass
class Article:
    id: str
    title: str
    category_path: str   # 如 "云文档/知识库"
    url: str
    weight: int
    update_time: int


class HelpCenterCrawler:
    def __init__(self, raw_dir: Path = config.RAW_DIR, delay: float = config.CRAWL_DELAY):
        self.raw_dir = raw_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.manifest_path = self.raw_dir / "manifest.jsonl"
        self._manifest = self._load_manifest()

    # ---------- 站点地图 ----------
    def fetch_site_map(self) -> dict:
        """拉取 categoryTree，返回规范化后的 {root_id, categories, articles}。"""
        logger.info("拉取帮助中心站点地图…")
        resp = self._get(CATEGORY_TREE_API, params={"language": "zh-CN"})
        if resp is None:
            raise RuntimeError("无法访问 categoryTree 接口")
        payload = resp.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"categoryTree 返回错误: {payload}")
        data = payload["data"]
        return {
            "root_id": data["root_id"],
            "categories": data["category_list"],
            "articles": data["article_list"],
        }

    def build_tree(self, categories: list[dict]) -> dict[str, dict]:
        """构建 分类id → {name, parent_id, children} 的树（children 只存 id）。"""
        nodes = {c["id"]: {**c, "children": []} for c in categories}
        for c in categories:
            parent = nodes.get(c["parent_id"])
            if parent is not None:
                parent["children"].append(c["id"])
        return nodes

    def category_path(self, nodes: dict[str, dict], cat_id: str) -> str:
        """从分类 id 反推完整路径，如 "云文档/知识库"。"""
        parts = []
        cur = nodes.get(cat_id)
        depth = 0
        while cur is not None and depth < 10:
            parts.append(cur["name"])
            cur = nodes.get(cur["parent_id"])
            depth += 1
        return "/".join(reversed(parts))

    # ---------- 文章选择 ----------
    def select_articles(self, site_map: dict, max_articles: int) -> list[Article]:
        """按优先级关键词匹配文章的分类路径，按站内热度排序并限流。

        顶层分类只有 3 个（功能概览/新手引导/更多资源），目标分类都嵌套在深层，
        所以直接对每篇文章的完整分类路径（如 "功能概览/云文档/知识库"）做关键词匹配。
        """
        nodes = self.build_tree(site_map["categories"])
        by_id = {c["id"]: c for c in site_map["categories"]}
        path_cache: dict[str, str] = {}

        def path_of(cat_id: str) -> str:
            if cat_id not in path_cache:
                path_cache[cat_id] = self.category_path(nodes, cat_id)
            return path_cache[cat_id]

        matched: list[Article] = []
        skipped = 0
        for a in site_map["articles"]:
            cat_id = a.get("category_id")
            path = a.get("path", "")
            # 过滤占位/搜索等非正文页；文章真实 URL 在 path 字段
            # （条目的 id 是站点内部 ID，直接拼 URL 会 404）
            if not cat_id or cat_id == "0" or path.endswith("/search"):
                skipped += 1
                continue
            m = re.search(r"/articles/(\d+)$", path)
            if not m:
                skipped += 1
                continue
            if cat_id not in by_id:
                continue
            cat_path = path_of(cat_id)
            if PRIORITY_CATEGORIES and not any(k in cat_path for k in PRIORITY_CATEGORIES):
                skipped += 1
                continue
            matched.append(Article(
                id=m.group(1),
                title=a["title"],
                category_path=cat_path,
                url=f"{BASE_URL}{path}",
                weight=a.get("weight", 0),
                update_time=a.get("update_time", 0),
            ))
        # 按热度排序，去重（按 URL 文章 id），限制总量
        matched.sort(key=lambda x: -x.weight)
        seen_ids: set[str] = set()
        deduped = []
        for art in matched:
            if art.id not in seen_ids:
                seen_ids.add(art.id)
                deduped.append(art)
        matched = deduped
        if max_articles:
            matched = matched[:max_articles]
        logger.info(f"共 {len(site_map['articles'])} 篇文章，命中 {len(matched)} 篇（过滤 {skipped} 篇）")
        return matched

    # ---------- HTTP ----------
    def _get(self, url: str, params: dict | None = None, retries: int = 2) -> requests.Response | None:
        for attempt in range(retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=config.CRAWL_TIMEOUT)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 404:
                    # 文章已下线，重试无意义，直接返回供调用方跳过
                    return resp
                logger.warning(f"HTTP {resp.status_code}: {url} (第 {attempt + 1} 次)")
            except requests.RequestException as e:
                logger.warning(f"请求失败 {url}: {e} (第 {attempt + 1} 次)")
            time.sleep(self.delay * (attempt + 2))
        return None

    # ---------- manifest ----------
    def _load_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        out = {}
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                out[item["article_id"]] = item
        return out

    def _record(self, article: Article) -> None:
        if article.id in self._manifest:
            return
        item = {
            "article_id": article.id,
            "title": article.title,
            "category": article.category_path,
            "url": article.url,
            "weight": article.weight,
            "update_time": article.update_time,
        }
        self._manifest[article.id] = item
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---------- 下载 ----------
    def download_article(self, article: Article) -> Path | None:
        dest = self.raw_dir / f"{article.id}.html"
        if dest.exists():
            self._record(article)
            return dest
        resp = self._get(article.url)
        if resp is None or resp.status_code != 200:
            return None
        dest.write_text(resp.text, encoding="utf-8")
        self._record(article)
        time.sleep(self.delay)
        return dest

    # ---------- 主流程 ----------
    def crawl(self, max_articles: int = config.CRAWL_MAX_ARTICLES) -> dict[str, int]:
        site_map = self.fetch_site_map()
        articles = self.select_articles(site_map, max_articles)
        stats: dict[str, int] = {}
        not_found = 0
        for a in tqdm(articles, desc="下载文章"):
            top_cat = a.category_path.split("/")[0]
            path = self.download_article(a)
            if path is not None:
                stats[top_cat] = stats.get(top_cat, 0) + 1
            else:
                not_found += 1
        logger.info(f"爬取完成: {sum(stats.values())} 篇成功, {not_found} 篇不存在, 分布: {stats}")
        return stats


if __name__ == "__main__":
    HelpCenterCrawler().crawl()
