from __future__ import annotations

import statistics
import time
import uuid

import src.chat.engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.models import Concept, ConceptType, ImportanceVector
from src.integrations.tools import _search_memory_handler


class _NoLLM:
    provider = "stub"
    model = "stub"
    temperature = 0.0
    timeout = 1

    def extract_concepts(self, _text):
        return []

    def _chat(self, _prompt: str, num_predict: int = 256) -> str:
        return ""


def _stub_embedding(seed: int) -> list[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def _work_fact(company: str, idx: int, session_id: str) -> Concept:
    concept = Concept(
        type=ConceptType.FACT,
        description=f"I currently work at {company}.",
        embedding=_stub_embedding(idx),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.95, repetition=0.2),
        salience_score=0.9,
        grasp_score=0.88,
    )
    concept.context_tags.update(
        {
            "session_id": session_id,
            "source": "user",
            "person": "user",
            "task": "profile",
        }
    )
    return concept


def test_brutal_lineage_contradiction_storm(monkeypatch):
    monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

    engine = ChatEngine(
        llm=_NoLLM(),
        enable_auto_sleep=False,
        session_id=f"brutal_lineage_{uuid.uuid4().hex}",
        sandbox_mode=True,
        enable_persistence=False,
    )
    engine.long_term_memory._persist_concept = lambda _c: None
    engine.long_term_memory._persist_relation = lambda _r: None

    total_updates = 120
    for i in range(total_updates):
        company = f"Atlas Labs Unit {i}"
        concept = _work_fact(company, i, engine.session_id)
        engine.long_term_memory.add_concept(
            concept,
            context_tags=concept.context_tags,
            allow_versioning=True,
        )

    res = _search_memory_handler(
        {"query": "where do I currently work?", "user_id": "u1", "limit": 5},
        engine,
    )
    assert res["ok"] is True
    assert res["memories"], res

    top = res["memories"][0]
    assert top["lineage"]["is_current_version"] is True

    lineage = engine.long_term_memory.get_lineage(top["id"])
    assert lineage["version_count"] >= 100
    assert lineage["conflict_count"] >= 100
    assert lineage["current_id"] == top["id"]

    latencies_ms: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        _payload = engine.long_term_memory.get_lineage(top["id"])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=100)[94]
    assert p95 < 25.0

    print(
        {
            "lineage_version_count": lineage["version_count"],
            "lineage_conflict_count": lineage["conflict_count"],
            "lineage_query_p50_ms": round(p50, 3),
            "lineage_query_p95_ms": round(p95, 3),
            "retrieval_citations": len(res.get("retrieval", {}).get("citations", [])),
        }
    )
