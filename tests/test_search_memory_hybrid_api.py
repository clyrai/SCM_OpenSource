from __future__ import annotations

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

    def extract_concepts(self, text):
        return []

    def _chat(self, prompt: str, num_predict: int = 256) -> str:
        return ""


def _stub_embedding(seed: int) -> list[float]:
    base = (seed + 1) / 1000.0
    return [base + ((i % 7) * 0.0001) for i in range(384)]


def test_search_memory_handler_exposes_hybrid_retrieval_metadata(monkeypatch):
    monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

    engine = ChatEngine(
        llm=_NoLLM(),
        enable_auto_sleep=False,
        session_id=f"search_meta_{uuid.uuid4().hex}",
        sandbox_mode=True,
        enable_persistence=False,
    )
    engine.long_term_memory._persist_concept = lambda _c: None
    engine.long_term_memory._persist_relation = lambda _r: None

    fact = Concept(
        type=ConceptType.FACT,
        description="I work at Atlas Labs in Zurich.",
        embedding=_stub_embedding(3),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.9),
        salience_score=0.9,
        grasp_score=0.9,
        context_tags={"source": "profile"},
    )
    engine.long_term_memory.add_concept(
        fact,
        context_tags={"session_id": engine.session_id, "source": "profile"},
    )

    res = _search_memory_handler(
        {"query": "Where do I work in Zurich?", "user_id": "u1", "limit": 5},
        engine,
    )

    assert res["ok"] is True
    assert "retrieval_stats" in res
    assert "retrieval" in res
    assert res["retrieval"]["fusion_mode"] == "weighted_rrf"
    assert res["retrieval"]["channel_count"] == 4
    assert isinstance(res["retrieval"]["channels"], dict)
    assert isinstance(res["retrieval"]["citations"], list)
    assert "memory_context" in res and "Retrieved Memories" in res["memory_context"]
    assert any("atlas labs" in (m.get("description", "").lower()) for m in res["memories"])
    first = res["memories"][0]
    assert "lineage" in first
    assert first["lineage"]["version_root"]
    assert "provenance" in first
    assert "source" in first["provenance"]


def test_search_memory_handler_applies_query_semantic_rerank(monkeypatch):
    monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

    engine = ChatEngine(
        llm=_NoLLM(),
        enable_auto_sleep=False,
        session_id=f"search_rerank_{uuid.uuid4().hex}",
        sandbox_mode=True,
        enable_persistence=False,
    )
    engine.long_term_memory._persist_concept = lambda _c: None
    engine.long_term_memory._persist_relation = lambda _r: None

    first = Concept(
        type=ConceptType.FACT,
        description="I volunteer at Atlas Museum in Zurich.",
        embedding=_stub_embedding(5),
        importance=ImportanceVector(novelty=0.7, task_relevance=0.7),
        salience_score=0.7,
        grasp_score=0.7,
        context_tags={"source": "profile"},
    )
    second = Concept(
        type=ConceptType.FACT,
        description="I work at Atlas Labs in Zurich.",
        embedding=_stub_embedding(6),
        importance=ImportanceVector(novelty=0.8, task_relevance=0.9),
        salience_score=0.9,
        grasp_score=0.9,
        context_tags={"source": "profile"},
    )
    engine.long_term_memory.add_concept(first, context_tags={"session_id": engine.session_id})
    engine.long_term_memory.add_concept(second, context_tags={"session_id": engine.session_id})

    monkeypatch.setattr(
        engine,
        "_fuse_hybrid_channels",
        lambda *args, **kwargs: (
            [first, second],
            {first.id: 1.0, second.id: 0.9},
            {first.id: 1, second.id: 1},
        ),
    )
    engine._force_query_semantic_rerank = True
    engine._query_semantic_rerank_hook = lambda **_kwargs: {
        "applied": True,
        "reason": "test_rerank",
        "boosts": {second.id: 0.5},
    }

    res = _search_memory_handler(
        {"query": "Which company am I at?", "user_id": "u1", "limit": 5},
        engine,
    )

    assert res["ok"] is True
    assert res["retrieval"]["semantic_rerank"]["applied"] is True
    assert res["retrieval"]["semantic_rerank"]["reason"] == "test_rerank"
    assert res["memories"][0]["description"] == second.description
    assert res["retrieval"]["citations"][0]["memory_id"] == res["memories"][0]["id"]
