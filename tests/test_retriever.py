"""不加载模型或磁盘索引，验证候选截断、重排 ID 对齐和父文档去重。"""
from types import SimpleNamespace

import pytest

from kb_agent.retriever import HybridRetriever


def fake_retriever(count=45):
    r = HybridRetriever.__new__(HybridRetriever)
    r.children = [{"chunk_id": f"c{i}", "parent_id": f"p{i}", "text": f"text{i}"}
                  for i in range(count)]
    r._child_by_id = {c["chunk_id"]: c for c in r.children}
    r.parents = {f"p{i}": {"chunk_id": f"p{i}", "doc_id": f"d{i}", "text": f"parent{i}"}
                 for i in range(count)}
    ranking = {f"c{i}": i for i in range(count)}
    r._dense_search = lambda query: ranking
    r._sparse_search = lambda query: ranking
    calls = []

    def predict(pairs):
        calls.append(pairs)
        # 原候选预算外的第 35 个块，确实比前面的块更能回答问题。
        return [100 if text == "text34" else 1 for _, text in pairs]

    r._reranker = SimpleNamespace(predict=predict)
    return r, calls


def test_larger_candidate_pool_can_recover_relevant_section_without_increasing_output():
    r, calls = fake_retriever()
    old = r.retrieve("操作问题", top_k=5, candidate_k=20)
    new = r.retrieve("操作问题", top_k=5)
    assert all(c.chunk_id != "c34" for c in old)
    assert new[0].chunk_id == "c34"
    assert len(calls[0]) == 20
    assert len(calls[1]) == 40
    assert len(new) == len(old) == 5


def test_duplicate_texts_preserve_distinct_ids_and_documents():
    r, _ = fake_retriever(count=2)
    for child in r.children:
        child["text"] = "相同正文"
    r._reranker = SimpleNamespace(predict=lambda pairs: [0.2, 0.9])
    results = r.retrieve("问题", top_k=2)
    assert [x.chunk_id for x in results] == ["c1", "c0"]
    assert [x.doc_id for x in results] == ["d1", "d0"]
    assert [x.score for x in results] == [0.9, 0.2]


def test_missing_child_is_filtered_without_shifting_scores():
    r, _ = fake_retriever(count=3)
    del r._child_by_id["c1"]
    r._reranker = SimpleNamespace(predict=lambda pairs: [0.1, 0.8])
    results = r.retrieve("问题", top_k=2)
    assert [x.chunk_id for x in results] == ["c2", "c0"]


def test_parent_deduplication_keeps_best_child():
    r, _ = fake_retriever(count=3)
    r._child_by_id["c1"]["parent_id"] = "p0"
    r._reranker = SimpleNamespace(predict=lambda pairs: [0.1, 0.9, 0.8])
    results = r.retrieve("问题", top_k=2)
    assert [x.parent_id for x in results] == ["p0", "p2"]
    assert results[0].chunk_id == "c1"


def test_hybrid_ablation_keeps_original_pool_and_does_not_rerank():
    r, calls = fake_retriever()
    results = r.retrieve("问题", mode="hybrid")
    assert len(results) == 5
    assert calls == []


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"mode": "unknown"},
    {"candidate_k": 3}, {"mode": "dense", "candidate_k": 40}])
def test_invalid_retrieval_options(kwargs):
    r, _ = fake_retriever()
    with pytest.raises(ValueError):
        r.retrieve("问题", **kwargs)
