"""人工设计的用户场景评测集；与旧的 LLM 自动生成题集分开管理。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kb_agent import config

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "user_scenarios.json"
Behavior = Literal["answer", "decline", "clarify"]
Category = Literal["direct", "colloquial", "multi_turn", "in_domain_unanswerable",
                   "out_of_domain", "ambiguous"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    doc_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class Turn(StrictModel):
    question: str = Field(min_length=1)
    expected_behavior: Behavior
    reference_answer: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    annotation_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def answer_requires_evidence(self):
        if self.expected_behavior == "answer" and not self.evidence:
            raise ValueError("可回答问题必须标注原文依据")
        return self


class Scenario(StrictModel):
    id: str = Field(min_length=1)
    category: Category
    turns: list[Turn] = Field(min_length=1)

    @model_validator(mode="after")
    def check_turns(self):
        if self.category == "multi_turn" and len(self.turns) < 2:
            raise ValueError("多轮场景至少两轮")
        if self.category != "multi_turn" and len(self.turns) != 1:
            raise ValueError("单轮场景只允许一轮")
        if self.category == "multi_turn":
            # 完整对话可先澄清、补充信息后再回答，或在条件变化后诚实拒答。
            return self
        expected = {"in_domain_unanswerable": "decline", "out_of_domain": "decline",
                    "ambiguous": "clarify"}.get(self.category, "answer")
        if any(t.expected_behavior != expected for t in self.turns):
            raise ValueError("题目分类与预期行为不一致")
        return self


class ScenarioDataset(StrictModel):
    version: str
    origin: Literal["manually_designed"]
    description: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("场景 ID 不可重复")
        return self


def load_dataset(path: Path = DEFAULT_DATASET) -> ScenarioDataset:
    return ScenarioDataset.model_validate_json(path.read_text(encoding="utf-8"))


def read_corpus(root: Path = config.ROOT) -> dict[str, str]:
    manifest = root / "data/corpus/manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return {r["doc_id"]: (root / r["md_path"]).read_text(encoding="utf-8") for r in records}


def corpus_fingerprint(corpus: dict[str, str]) -> str:
    payload = json.dumps(corpus, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_corpus(dataset: ScenarioDataset, corpus: dict[str, str]) -> None:
    if corpus_fingerprint(corpus) != dataset.corpus_sha256:
        raise ValueError("语料快照与题集不一致，需复核原文依据及无答案题的标注")
    for case in dataset.cases:
        for turn in case.turns:
            for source in turn.evidence:
                if source.doc_id not in corpus or source.quote not in corpus[source.doc_id]:
                    raise ValueError(f"原文依据不存在: {case.id}/{source.doc_id}")
