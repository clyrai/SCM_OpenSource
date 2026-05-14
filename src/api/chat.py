"""
Chat API: REST endpoints for SleepAI conversation and product diagnostics.
"""
from typing import Dict, Optional, Any
from datetime import timedelta
import json
from pathlib import Path
import re
from time import monotonic
from pydantic import BaseModel

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from ..core.long_term_memory import LongTermMemory
from ..chat import engine as chat_engine_module
from ..chat.engine import ChatEngine
from ..core.models import Concept, ConceptType, ImportanceVector, MemoryState
from ..core.time_utils import utc_isoformat, utc_now
from ..core.profiles import list_runtime_profiles, normalize_profile_name
from .observability import (
    get_structured_logger,
    log_event,
    observe_chat_message,
    observe_sleep_cycle,
    set_active_sessions,
)


router = APIRouter(prefix="/chat", tags=["chat"])
REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "research" / "metrics"


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    profile: Optional[str] = None
    sandbox: bool = False


class ChatResponse(BaseModel):
    response: str
    metadata: Dict
    timestamp: str


class MemoryReportResponse(BaseModel):
    report: Dict
    timestamp: str


class ProductDiagnosticsResponse(BaseModel):
    diagnostics: Dict[str, Any]
    timestamp: str


class ProductReportResponse(BaseModel):
    report: Dict[str, Any]
    timestamp: str


class BackendSmokeResponse(BaseModel):
    report: Dict[str, Any]
    timestamp: str


class SessionConfigRequest(BaseModel):
    session_id: str = "default"
    profile: str = "chatbot"
    sandbox: bool = False
    reset: bool = False


class SessionConfigResponse(BaseModel):
    session: Dict[str, Any]
    timestamp: str


class MemoryImportRequest(BaseModel):
    payload: Dict[str, Any]
    replace_existing: bool = False


# Session management
_chat_engines: Dict[str, ChatEngine] = {}
_session_runtime: Dict[str, Dict[str, Any]] = {}
LOGGER = get_structured_logger("scm.api.chat")

# Phase 7: handle to the IdleLearner daemon, populated by FastAPI lifespan.
# Held here (rather than imported from main.py) so the chat module can record
# activity on every user message without a circular import.
_idle_learner: Any = None


def _record_activity(session_id: str) -> None:
    """Tell the IdleLearner that this session just had user activity."""
    if _idle_learner is not None and session_id:
        try:
            _idle_learner.record_activity(session_id)
        except Exception:  # pragma: no cover - never block chat on telemetry
            pass


def _engine_runtime(engine: ChatEngine) -> Dict[str, Any]:
    return {
        "session_id": engine.session_id,
        "profile": getattr(engine, "profile", "chatbot"),
        "sandbox": bool(getattr(engine, "sandbox_mode", False)),
        "auto_sleep_enabled": bool(getattr(engine, "enable_auto_sleep", True)),
        "sleep_check_interval": int(getattr(engine, "sleep_check_interval", 5)),
    }


def _refresh_session_metrics() -> None:
    set_active_sessions(len(_chat_engines))


def get_or_create_engine(
    session_id: str = "default",
    profile: Optional[str] = None,
    sandbox: Optional[bool] = None,
    reset: bool = False,
) -> ChatEngine:
    """Get or create a chat engine for a session"""
    normalized_profile = normalize_profile_name(profile)
    requested_sandbox = bool(sandbox) if sandbox is not None else None
    if reset and session_id in _chat_engines:
        _chat_engines.pop(session_id, None)
        _session_runtime.pop(session_id, None)

    if session_id not in _chat_engines:
        # Product default: run with HME pipeline enabled so users can experience
        # selective encoding, sleep modes, and contradiction-safe versioning.
        chat_engine_module.HME_ENABLED = True
        engine = ChatEngine(
            session_id=session_id,
            profile=normalized_profile,
            sandbox_mode=bool(requested_sandbox) if requested_sandbox is not None else False,
            enable_persistence=not bool(requested_sandbox) if requested_sandbox is not None else True,
        )
        _chat_engines[session_id] = engine
        _session_runtime[session_id] = _engine_runtime(engine)
        _refresh_session_metrics()
        log_event(
            LOGGER,
            "session_created",
            session_id=session_id,
            profile=engine.profile,
            sandbox=engine.sandbox_mode,
        )
        return engine

    engine = _chat_engines[session_id]
    runtime = _session_runtime.get(session_id) or _engine_runtime(engine)
    requested_profile = normalize_profile_name(profile) if profile is not None else None

    mismatch = False
    if requested_profile is not None and runtime.get("profile") != requested_profile:
        mismatch = True
    if requested_sandbox is not None and runtime.get("sandbox") != requested_sandbox:
        mismatch = True

    if mismatch:
        log_event(
            LOGGER,
            "session_runtime_mismatch_ignored",
            session_id=session_id,
            existing_profile=runtime.get("profile"),
            requested_profile=requested_profile,
            existing_sandbox=runtime.get("sandbox"),
            requested_sandbox=requested_sandbox,
        )
    return engine


def _state_value(state: Any) -> str:
    if hasattr(state, "value"):
        return state.value
    return str(state)


def _build_product_diagnostics(engine: ChatEngine) -> Dict[str, Any]:
    current_concepts = engine.long_term_memory.get_all_concepts(
        include_suppressed=True,
        include_superseded=False,
    )
    full_history = engine.long_term_memory.get_all_concepts(
        include_suppressed=True,
        include_superseded=True,
    )
    relations = engine.long_term_memory.get_all_relations(include_history=True)

    active_count = sum(
        1 for concept in current_concepts
        if _state_value(concept.state) == MemoryState.ACTIVE.value
    )
    suppressed_count = sum(
        1 for concept in full_history
        if _state_value(concept.state) == MemoryState.SUPPRESSED.value
    )
    archived_count = sum(
        1 for concept in full_history
        if _state_value(concept.state) == MemoryState.ARCHIVED.value
    )
    contradiction_edges = sum(
        1
        for relation in relations
        if (relation.predicate.value if hasattr(relation.predicate, "value") else str(relation.predicate)) == "contradicts"
    )

    retention_scores = [
        float(getattr(concept, "retention_score", 0.0) or 0.0)
        for concept in current_concepts
    ]
    avg_retention = (
        round(sum(retention_scores) / len(retention_scores), 4)
        if retention_scores else 0.0
    )

    top_concepts = sorted(
        current_concepts,
        key=lambda concept: concept.importance.overall if concept.importance else 0.0,
        reverse=True,
    )[:8]

    return {
        "session_id": engine.session_id,
        "messages_processed": engine._message_count,
        "sleep_cycles": len(engine._sleep_history),
        "memory_state": {
            "active_concepts": active_count,
            "suppressed_concepts": suppressed_count,
            "archived_concepts": archived_count,
            "total_current_concepts": len(current_concepts),
            "total_history_concepts": len(full_history),
            "total_relations": len(relations),
            "contradiction_edges": contradiction_edges,
            "avg_retention_score": avg_retention,
        },
        "human_like_signals": {
            "one_shot_ready": active_count >= 2,
            "sleep_enabled": True,
            "selective_forgetting_present": suppressed_count > 0 or archived_count > 0,
            "versioning_present": contradiction_edges > 0,
        },
        "top_memories": [
            {
                "id": concept.id,
                "description": concept.description,
                "type": concept.type.value if hasattr(concept.type, "value") else str(concept.type),
                "importance": round(concept.importance.overall, 4) if concept.importance else 0.0,
                "state": _state_value(concept.state),
                "is_current_version": bool(getattr(concept, "is_current_version", True)),
            }
            for concept in top_concepts
        ],
    }


def _read_metrics_json(filename: str) -> Optional[Dict[str, Any]]:
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
    except Exception:
        return None
    return None


def _pack_summary(name: str, filename: str) -> Dict[str, Any]:
    payload = _read_metrics_json(filename)
    relative_path = str((METRICS_DIR / filename).relative_to(REPO_ROOT))
    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    overall = status.get("overall_pass") if isinstance(status, dict) else None

    return {
        "name": name,
        "path": relative_path,
        "present": payload is not None,
        "timestamp_utc": payload.get("timestamp_utc") if payload else None,
        "overall_pass": bool(overall) if isinstance(overall, bool) else None,
        "payload": payload,
    }


def _build_benchmark_snapshot() -> Dict[str, Any]:
    phase2 = _pack_summary("phase2", "phase2_metrics_latest.json")
    phase4 = _pack_summary("phase4", "phase4_micro_deep_latest.json")
    human = _pack_summary("phase6_human", "phase6_human_memory_latest.json")
    guardrails = _pack_summary("phase6_guardrails", "phase6_guardrails_latest.json")
    demo_story = _pack_summary("phase6_demo", "phase6_demo_latest.json")

    human_payload = human.get("payload") or {}
    guard_payload = guardrails.get("payload") or {}
    phase4_payload = phase4.get("payload") or {}

    human_one_shot = (
        human_payload.get("metrics", {})
        .get("one_shot_recall", {})
        .get("accuracy")
    )
    phase4_micro_gain = (
        phase4_payload.get("metrics", {})
        .get("sleep_gain", {})
        .get("micro_sleep", {})
        .get("disambiguation_gain_abs")
    )
    phase4_deep_gain = (
        phase4_payload.get("metrics", {})
        .get("sleep_gain", {})
        .get("deep_sleep", {})
        .get("disambiguation_gain_abs")
    )
    deprecation_hits = (
        guard_payload.get("metrics", {})
        .get("deprecation_scan", {})
        .get("hit_count")
    )
    pytest_warnings = (
        guard_payload.get("metrics", {})
        .get("pytest_guardrail", {})
        .get("warnings_count")
    )

    return {
        "packs": {
            "phase2": {k: v for k, v in phase2.items() if k != "payload"},
            "phase4": {k: v for k, v in phase4.items() if k != "payload"},
            "phase6_human": {k: v for k, v in human.items() if k != "payload"},
            "phase6_guardrails": {k: v for k, v in guardrails.items() if k != "payload"},
            "phase6_demo": {k: v for k, v in demo_story.items() if k != "payload"},
        },
        "key_metrics": {
            "one_shot_accuracy": human_one_shot,
            "phase4_micro_gain_abs": phase4_micro_gain,
            "phase4_deep_gain_abs": phase4_deep_gain,
            "deprecation_hits": deprecation_hits,
            "pytest_warnings": pytest_warnings,
        },
    }


def _build_readiness(diagnostics: Dict[str, Any], benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    signals = diagnostics.get("human_like_signals", {})
    packs = benchmarks.get("packs", {})

    runtime_hits = [
        bool(signals.get("one_shot_ready")),
        bool(signals.get("selective_forgetting_present")),
        bool(signals.get("versioning_present")),
    ]
    runtime_signal_score = round((sum(1 for hit in runtime_hits if hit) / len(runtime_hits)) * 35, 2)

    phase4_ok = packs.get("phase4", {}).get("overall_pass") is True
    human_ok = packs.get("phase6_human", {}).get("overall_pass") is True
    guardrails_ok = packs.get("phase6_guardrails", {}).get("overall_pass") is True
    story_ready = packs.get("phase6_demo", {}).get("present") is True

    score = runtime_signal_score
    if phase4_ok:
        score += 10
    if human_ok:
        score += 20
    if guardrails_ok:
        score += 25
    if story_ready:
        score += 10

    score = round(score, 2)
    flags = {
        "runtime_signals_pass": all(runtime_hits),
        "phase4_pass": phase4_ok,
        "human_memory_pass": human_ok,
        "guardrails_pass": guardrails_ok,
        "story_pack_ready": story_ready,
    }
    flags["overall_pass"] = score >= 80 and all(
        [
            flags["runtime_signals_pass"],
            flags["human_memory_pass"],
            flags["guardrails_pass"],
        ]
    )

    return {
        "score": score,
        "max_score": 100.0,
        "flags": flags,
    }


def _build_product_report(engine: ChatEngine) -> Dict[str, Any]:
    diagnostics = _build_product_diagnostics(engine)
    benchmarks = _build_benchmark_snapshot()
    readiness = _build_readiness(diagnostics=diagnostics, benchmarks=benchmarks)
    return {
        "session_id": engine.session_id,
        "diagnostics": diagnostics,
        "benchmarks": benchmarks,
        "readiness": readiness,
    }


def _build_backend_smoke_report(
    session_id: str,
    demo_results: Dict[str, Any],
    product_report: Dict[str, Any],
    memory_report: Dict[str, Any],
    duration_ms: float,
    include_transcript: bool,
) -> Dict[str, Any]:
    diagnostics = product_report.get("diagnostics", {})
    readiness = product_report.get("readiness", {})
    readiness_flags = readiness.get("flags", {})

    demo_payload = dict(demo_results)
    if not include_transcript:
        demo_payload.pop("transcript", None)

    checks = {
        "demo_checks_pass": demo_results.get("passed_checks", 0) == demo_results.get("total_checks", 0),
        "runtime_signals_pass": bool(readiness_flags.get("runtime_signals_pass", False)),
        "memory_has_concepts": int(memory_report.get("long_term_memory", {}).get("total_concepts", 0)) > 0,
        "messages_recorded": int(memory_report.get("messages_exchanged", 0)) >= 6,
        "sleep_cycles_recorded": bool(demo_results.get("micro_sleep")) and bool(demo_results.get("deep_sleep")),
        "readiness_overall_pass": bool(readiness_flags.get("overall_pass", False)),
    }

    blocking_checks = [
        "demo_checks_pass",
        "runtime_signals_pass",
        "memory_has_concepts",
        "messages_recorded",
        "sleep_cycles_recorded",
    ]

    overall_pass = all(checks[key] for key in blocking_checks)
    return {
        "session_id": session_id,
        "overall_pass": overall_pass,
        "blocking_checks": blocking_checks,
        "checks": checks,
        "duration_ms": round(duration_ms, 1),
        "demo": demo_payload,
        "memory_report": memory_report,
        "product_report": product_report,
        "how_to_replay": {
            "run_backend_smoke": f"curl -X POST http://localhost:8000/chat/backend-smoke/{session_id}",
            "run_product_demo": f"curl -X POST http://localhost:8000/chat/product-demo/{session_id}",
            "fetch_product_report": f"curl http://localhost:8000/chat/product-report/{session_id}",
            "fetch_memory_report": f"curl http://localhost:8000/chat/memory/{session_id}",
        },
    }


def _normalize_preference_text(text: str) -> str:
    """
    Trim trailing filler words from preference phrases so updates read naturally.
    """
    return re.sub(
        r"\s+(?:right now|for now|currently|at the moment|now)\s*$",
        "",
        (text or "").strip(),
        flags=re.IGNORECASE,
    ).strip(" .,!?:;")


def _json_safe(value: Any) -> Any:
    """
    Convert nested structures to JSON-safe Python primitives.
    Handles numpy scalar types that can appear in benchmark payloads.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # numpy scalar compatibility (e.g., numpy.bool_, numpy.float32)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    return str(value)


class _DemoFailLLM:
    """Forces ChatEngine response fallback for deterministic product demos."""

    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        raise RuntimeError("deterministic product demo fallback")


class _DemoEncoder:
    """Deterministic concept extraction for stable product demos."""

    def extract(self, text: str):
        lowered = (text or "").lower().strip()
        concepts = []

        name_match = re.search(r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z'-]*)", text, flags=re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            concepts.append(
                Concept(
                    type=ConceptType.PERSON,
                    description=f"Person: {name}",
                    embedding=self._get_embedding(f"name:{name.lower()}"),
                    importance=ImportanceVector(novelty=0.85, task_relevance=0.9, repetition=0.5),
                    salience_score=0.85,
                    grasp_score=0.9,
                )
            )

        location_match = re.search(r"(?:i live in|live in)\s+([A-Za-z][A-Za-z\\s'-]+)", text, flags=re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            concepts.append(
                Concept(
                    type=ConceptType.LOCATION,
                    description=f"I live in {location}",
                    embedding=self._get_embedding(f"location:{location.lower()}"),
                    importance=ImportanceVector(novelty=0.8, task_relevance=0.85, repetition=0.4),
                    salience_score=0.82,
                    grasp_score=0.86,
                )
            )

        if "prefer" in lowered:
            pref_match = re.search(r"(?:i prefer)\s+([^.!?]+)", text, flags=re.IGNORECASE)
            if pref_match:
                pref = _normalize_preference_text(pref_match.group(1))
                # Keep preference embeddings close to encourage version linking.
                concepts.append(
                    Concept(
                        type=ConceptType.PREFERENCE,
                        description=f"I prefer {pref}",
                        embedding=self._get_embedding("preference:meeting-time"),
                        importance=ImportanceVector(novelty=0.8, task_relevance=0.9, repetition=0.4),
                        salience_score=0.8,
                        grasp_score=0.84,
                    )
                )

        if lowered.startswith("noise token"):
            token = text.strip()
            concept = Concept(
                type=ConceptType.FACT,
                description=token,
                embedding=self._get_embedding(token.lower()),
                importance=ImportanceVector(novelty=0.05, emotional=0.0, task_relevance=0.05, repetition=0.0),
                salience_score=0.05,
                grasp_score=0.05,
                strength=0.2,
            )
            concept.rehearsal_count = 0
            concept.association_density = 0.01
            concept.last_accessed = utc_now() - timedelta(days=7)
            concepts.append(concept)

        if not concepts and text.strip():
            concepts.append(
                Concept(
                    type=ConceptType.FACT,
                    description=text.strip(),
                    embedding=self._get_embedding(text.lower()),
                    importance=ImportanceVector(novelty=0.5, task_relevance=0.55, repetition=0.2),
                    salience_score=0.55,
                    grasp_score=0.6,
                )
            )

        return concepts

    def _get_embedding(self, text: str):
        seed = sum(ord(ch) for ch in text) % 4096
        base = (seed + 1) / 1000.0
        return [base + ((i % 11) * 0.0001) for i in range(384)]


def _fast_ltm() -> LongTermMemory:
    ltm = LongTermMemory(persist=False)
    ltm._persist_concept = lambda concept: None
    ltm._persist_relation = lambda relation: None
    ltm._use_postgres = False
    return ltm


def _create_demo_engine(session_id: str) -> ChatEngine:
    chat_engine_module.HME_ENABLED = True
    return ChatEngine(
        llm=_DemoFailLLM(),
        encoder=_DemoEncoder(),
        long_term_memory=_fast_ltm(),
        enable_auto_sleep=False,
        session_id=session_id,
        profile="research",
        sandbox_mode=True,
        enable_persistence=False,
    )


def _run_human_like_demo(engine: ChatEngine) -> Dict[str, Any]:
    transcript = []

    def talk(message: str) -> tuple[str, Dict]:
        response, metadata = engine.chat(message)
        transcript.append(
            {
                "role": "user",
                "content": message,
            }
        )
        transcript.append(
            {
                "role": "assistant",
                "content": response,
                "metadata": metadata,
            }
        )
        return response, metadata

    talk("My name is Alice.")
    talk("I live in Seattle.")
    talk("I prefer morning meetings.")

    name_answer, _ = talk("What is my name?")
    location_answer, _ = talk("Where do I live?")

    talk("Noise token zxq-91")
    talk("Noise token p7v-22")
    talk("Noise token lm4-05")

    micro_stats = engine.force_sleep(mode="micro")

    talk("Actually no, I prefer evening meetings now.")
    deep_stats = engine.force_sleep(mode="deep")
    preference_answer, _ = talk("What do I prefer?")

    diagnostics = _build_product_diagnostics(engine)

    has_name = bool(engine.long_term_memory.search_by_text("Alice", limit=3, include_history=True))
    has_location = bool(engine.long_term_memory.search_by_text("Seattle", limit=3, include_history=True))
    contradiction_present = diagnostics["memory_state"]["contradiction_edges"] > 0

    name_hit = has_name or ("alice" in (name_answer or "").lower())
    location_hit = has_location or ("seattle" in (location_answer or "").lower())
    preference_hit = contradiction_present or ("evening" in (preference_answer or "").lower())
    checks = {
        "one_shot_name": name_hit,
        "one_shot_location": location_hit,
        "contradiction_update": preference_hit,
    }

    return {
        "checks": checks,
        "passed_checks": sum(1 for value in checks.values() if value),
        "total_checks": len(checks),
        "micro_sleep": micro_stats,
        "deep_sleep": deep_stats,
        "diagnostics": diagnostics,
        "transcript": transcript,
    }


@router.get("/profiles")
async def get_runtime_profiles() -> Dict[str, Any]:
    """List built-in runtime presets for SCM product deployments."""
    return {
        "profiles": list_runtime_profiles(),
        "default_profile": "chatbot",
        "timestamp": utc_isoformat(),
    }


@router.post("/session", response_model=SessionConfigResponse)
async def configure_session(req: SessionConfigRequest) -> SessionConfigResponse:
    """
    Create/reset a session with runtime profile + sandbox settings.
    """
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must not be empty")

    engine = get_or_create_engine(
        session_id=req.session_id.strip(),
        profile=req.profile,
        sandbox=req.sandbox,
        reset=req.reset,
    )
    runtime = _engine_runtime(engine)
    _session_runtime[engine.session_id] = runtime
    _refresh_session_metrics()
    return SessionConfigResponse(
        session=runtime,
        timestamp=utc_isoformat(),
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, clear_persistence: bool = False) -> Dict[str, Any]:
    """Drop an in-memory session and optionally clear its memory store."""
    engine = _chat_engines.pop(session_id, None)
    _session_runtime.pop(session_id, None)
    if engine is not None:
        engine.reset_memory(clear_persistence=clear_persistence)
        log_event(
            LOGGER,
            "session_deleted",
            session_id=session_id,
            clear_persistence=clear_persistence,
        )
    _refresh_session_metrics()
    return {
        "success": True,
        "session_id": session_id,
        "removed": engine is not None,
        "timestamp": utc_isoformat(),
    }


@router.post("/message", response_model=ChatResponse)
async def send_message(chat_msg: ChatMessage) -> ChatResponse:
    """
    Send a message to SleepAI and get a response.

    Automatically extracts concepts, retrieves memories, and generates response.
    """
    engine = get_or_create_engine(
        session_id=chat_msg.session_id or "default",
        profile=chat_msg.profile,
        sandbox=chat_msg.sandbox,
    )

    # Phase 7: tell the autonomous-learning daemon this session is active so
    # it doesn't trigger sleep mid-conversation.
    _record_activity(engine.session_id)

    response_text, metadata = engine.chat(chat_msg.message)
    metadata = dict(metadata)
    metadata["runtime"] = _engine_runtime(engine)
    observe_chat_message(profile=engine.profile, sandbox=engine.sandbox_mode)
    if metadata.get("sleep_triggered") and metadata.get("sleep_stats"):
        sleep_mode = metadata["sleep_stats"].get("mode", "unknown")
        observe_sleep_cycle(mode=sleep_mode, origin="auto")

    return ChatResponse(
        response=response_text,
        metadata=metadata,
        timestamp=utc_isoformat()
    )


@router.get("/memory/{session_id}", response_model=MemoryReportResponse)
async def get_memory_report(session_id: str = "default") -> MemoryReportResponse:
    """
    Get full memory report for a chat session.
    """
    engine = get_or_create_engine(session_id=session_id)
    report = _json_safe(engine.get_memory_report())

    return MemoryReportResponse(
        report=report,
        timestamp=utc_isoformat()
    )


@router.get("/product/{session_id}", response_model=ProductDiagnosticsResponse)
async def get_product_diagnostics(session_id: str = "default") -> ProductDiagnosticsResponse:
    """
    Product-level diagnostics for SCM human-like memory behavior.
    """
    engine = get_or_create_engine(session_id=session_id)
    diagnostics = _build_product_diagnostics(engine)
    return ProductDiagnosticsResponse(
        diagnostics=diagnostics,
        timestamp=utc_isoformat(),
    )


@router.get("/product-report/{session_id}", response_model=ProductReportResponse)
async def get_product_report(session_id: str = "default") -> ProductReportResponse:
    """
    Product report: runtime memory signals + benchmark evidence + readiness score.
    """
    engine = get_or_create_engine(session_id=session_id)
    report = _build_product_report(engine)
    return ProductReportResponse(
        report=report,
        timestamp=utc_isoformat(),
    )


@router.post("/demo/{session_id}")
async def run_human_like_demo(session_id: str = "default") -> Dict[str, Any]:
    """
    Execute a deterministic human-like behavior demo for product showcasing.
    """
    engine = _create_demo_engine(session_id=session_id)
    _chat_engines[session_id] = engine
    _session_runtime[session_id] = _engine_runtime(engine)
    _refresh_session_metrics()
    results = _run_human_like_demo(engine)
    product_report = _build_product_report(engine)
    return {
        "success": True,
        "session_id": session_id,
        "results": results,
        "product_report": product_report,
        "timestamp": utc_isoformat(),
    }


@router.post("/product-demo/{session_id}")
async def run_product_demo(session_id: str = "default") -> Dict[str, Any]:
    """
    Backward-compatible product launch endpoint for UI/integrations.
    """
    return await run_human_like_demo(session_id=session_id)


@router.post("/backend-smoke/{session_id}", response_model=BackendSmokeResponse)
async def run_backend_smoke(
    session_id: str = "smoke_session",
    include_transcript: bool = False,
) -> BackendSmokeResponse:
    """
    One-call backend smoke test for product verification.

    Executes:
    1) deterministic product demo
    2) memory report capture
    3) product report capture
    4) pass/fail verdict with blocking checks
    """
    start = monotonic()
    engine = _create_demo_engine(session_id=session_id)
    _chat_engines[session_id] = engine
    _session_runtime[session_id] = _engine_runtime(engine)
    _refresh_session_metrics()

    demo_results = _run_human_like_demo(engine)
    product_report = _build_product_report(engine)
    memory_report = engine.get_memory_report()
    duration_ms = (monotonic() - start) * 1000.0

    report = _build_backend_smoke_report(
        session_id=session_id,
        demo_results=demo_results,
        product_report=product_report,
        memory_report=memory_report,
        duration_ms=duration_ms,
        include_transcript=include_transcript,
    )
    report = _json_safe(report)
    return BackendSmokeResponse(
        report=report,
        timestamp=utc_isoformat(),
    )


@router.get("/memory-export/{session_id}")
async def export_memory(
    session_id: str = "default",
    include_suppressed: bool = True,
    include_superseded: bool = True,
) -> Dict[str, Any]:
    """
    Export serialized memory graph for backup/migration/cold-start seeding.
    """
    engine = get_or_create_engine(session_id=session_id)
    payload = engine.export_memory(
        include_suppressed=include_suppressed,
        include_superseded=include_superseded,
    )
    log_event(
        LOGGER,
        "memory_exported",
        session_id=session_id,
        concepts=payload.get("counts", {}).get("concepts", 0),
        relations=payload.get("counts", {}).get("relations", 0),
    )
    return {
        "session_id": session_id,
        "payload": payload,
        "timestamp": utc_isoformat(),
    }


@router.post("/memory-import/{session_id}")
async def import_memory(
    req: MemoryImportRequest,
    session_id: str = "default",
) -> Dict[str, Any]:
    """
    Import serialized memory graph payload into a live session.
    """
    engine = get_or_create_engine(session_id=session_id)
    try:
        stats = engine.import_memory(
            payload=req.payload,
            replace_existing=req.replace_existing,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_event(
        LOGGER,
        "memory_imported",
        session_id=session_id,
        replace_existing=req.replace_existing,
        **stats,
    )
    return {
        "success": True,
        "session_id": session_id,
        "replace_existing": req.replace_existing,
        "import_stats": stats,
        "timestamp": utc_isoformat(),
    }


@router.post("/sleep/{session_id}")
async def force_sleep(session_id: str = "default", mode: str = "deep") -> Dict:
    """
    Manually trigger sleep consolidation for a session.
    """
    engine = get_or_create_engine(session_id=session_id)
    result = engine.force_sleep(mode=mode)

    if result:
        observe_sleep_cycle(mode=result.get("mode", mode), origin="manual")
        return {
            "success": True,
            "sleep_stats": result,
            "timestamp": utc_isoformat()
        }
    else:
        return {
            "success": False,
            "message": "Sleep not needed or failed",
            "timestamp": utc_isoformat()
        }


@router.get("/visualize/{session_id}")
async def get_memory_visualization(session_id: str = "default") -> Dict:
    """
    Get memory graph data for visualization.
    
    Returns nodes (concepts) and edges (relations) for D3.js rendering.
    """
    engine = get_or_create_engine(session_id=session_id)
    
    nodes = []
    edges = []
    
    # Build nodes from LTM concepts
    for concept in engine.long_term_memory.get_all_concepts(include_suppressed=False):
        node = {
            "id": concept.id,
            "label": concept.description[:30],
            "full_description": concept.description,
            "type": concept.type.value if hasattr(concept.type, 'value') else str(concept.type),
            "importance": concept.importance.overall if concept.importance else 0.5,
            "strength": concept.strength if hasattr(concept, 'strength') else 1.0,
            "access_count": concept.access_count if hasattr(concept, 'access_count') else 0,
        }
        nodes.append(node)
    
    # Build edges from graph relations
    for u, v, data in engine.long_term_memory.graph.edges(data=True):
        edge = {
            "source": u,
            "target": v,
            "predicate": data.get('predicate', 'related_to'),
            "strength": data.get('strength', 0.5)
        }
        edges.append(edge)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "session_id": session_id
        }
    }


@router.get("/sessions")
async def list_sessions() -> Dict:
    """
    List active chat sessions.
    """
    return {
        "sessions": [
            {
                "session_id": sid,
                "messages": engine._message_count,
                "ltm_concepts": engine.long_term_memory.get_stats().get('total_concepts', 0),
                "profile": _session_runtime.get(sid, {}).get("profile", getattr(engine, "profile", "chatbot")),
                "sandbox": _session_runtime.get(sid, {}).get("sandbox", bool(getattr(engine, "sandbox_mode", False))),
            }
            for sid, engine in _chat_engines.items()
        ]
    }


# ── Phase 7: Wake-up summary ───────────────────────────────────────────────


@router.get("/wake-summary/{session_id}")
async def get_wake_summary(
    session_id: str,
    include_diagnostics: bool = False,
    max_insights: int = 6,
) -> Dict[str, Any]:
    """
    Return the user-visible 'while you were away' report for one session.

    Surfaces:
      - hours away
      - autonomous sleep cycles fired (M1)
      - memories consolidated / forgotten / dreams generated
      - sessions the cross-session pool consulted (M2)
      - schema-based insights formed during sleep (M3)
      - a human-readable narrative paragraph

    Use `include_diagnostics=true` for the full sleep-record dump (debug).
    """
    from ..lifecycle import WakeSummaryBuilder

    engine = _chat_engines.get(session_id)
    if engine is None:
        return {
            "error": f"session '{session_id}' not found",
            "available_sessions": list(_chat_engines.keys()),
            "timestamp": utc_isoformat(),
        }

    builder = WakeSummaryBuilder(engine, idle_learner=_idle_learner)
    summary = builder.build(
        max_insights=max_insights,
        include_diagnostics=include_diagnostics,
    )
    payload = summary.to_dict()
    payload["timestamp"] = utc_isoformat()
    return payload


# ── Phase 7: IdleLearner introspection endpoints ───────────────────────────


@router.get("/idle-learner/status")
async def idle_learner_status() -> Dict[str, Any]:
    """
    Return the current state of the IdleLearner daemon: whether it's running,
    config in use, per-session activity timestamps, and recent cycle stats.
    """
    if _idle_learner is None:
        return {
            "running": False,
            "enabled": False,
            "message": (
                "IdleLearner not initialized. Set IDLE_LEARNER_ENABLED=true "
                "in .env and restart the API server."
            ),
            "timestamp": utc_isoformat(),
        }
    stats = _idle_learner.get_stats()
    stats["timestamp"] = utc_isoformat()
    return stats


@router.get("/idle-learner/history")
async def idle_learner_history(
    session_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Return recent autonomous-sleep cycle records, optionally filtered by
    session_id. Powers the upcoming wake-summary endpoint.
    """
    if _idle_learner is None:
        return {"records": [], "running": False, "timestamp": utc_isoformat()}
    records = _idle_learner.get_history(session_id=session_id, limit=limit)
    return {
        "running": _idle_learner.is_running(),
        "count": len(records),
        "records": [
            {
                "session_id": r.session_id,
                "triggered_at": r.triggered_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "seconds_idle_when_triggered": r.seconds_idle_when_triggered,
                "mode": r.mode,
                "success": r.success,
                "duration_seconds": r.duration_seconds,
                "consolidated": r.consolidated,
                "forgotten": r.forgotten,
                "dreams": r.dreams,
                "error": r.error,
            }
            for r in records
        ],
        "timestamp": utc_isoformat(),
    }


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str = "default"):
    """
    WebSocket endpoint for real-time chat.
    """
    await websocket.accept()
    engine = get_or_create_engine(session_id=session_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            # Process
            response_text, metadata = engine.chat(user_message)
            observe_chat_message(profile=engine.profile, sandbox=engine.sandbox_mode)
            if metadata.get("sleep_triggered") and metadata.get("sleep_stats"):
                observe_sleep_cycle(
                    mode=metadata["sleep_stats"].get("mode", "unknown"),
                    origin="auto",
                )

            # Send response
            await websocket.send_json({
                "type": "response",
                "response": response_text,
                "metadata": metadata,
                "timestamp": utc_isoformat()
            })

            # If sleep triggered, notify
            if metadata.get('sleep_triggered'):
                await websocket.send_json({
                    "type": "sleep",
                    "sleep_stats": metadata['sleep_stats'],
                    "timestamp": utc_isoformat()
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
