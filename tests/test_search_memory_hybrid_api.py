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
    assert "memory_context" in res and "Retrieved Memories" in res["memory_context"]
    assert any("atlas labs" in (m.get("description", "").lower()) for m in res["memories"])
