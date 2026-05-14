"""Ephemeral task-context slots for short-horizon conversational state.

Used to keep transient intent/constraints (e.g. travel origin) separate
from durable profile memory.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_TRAVEL_SIGNAL_RE = re.compile(
    r"\b(?:travel|trip|route|reach|get to|go to|flight|plane|train|drive|rome|zurich|swiss|switzerland)\b",
    re.IGNORECASE,
)
_ASK_ORIGIN_RE = re.compile(
    r"\b(?:which city.*starting from|where.*starting from|starting city|depart(?:ing|ure).+from)\b",
    re.IGNORECASE,
)
_ASK_DEST_RE = re.compile(
    r"\b(?:where.*going|destination|headed to|travell?ing to)\b",
    re.IGNORECASE,
)
_ORIGIN_PATTERNS = [
    re.compile(
        r"\b(?:from|starting from|start(?:ing)? in|departing from)\s+([A-Za-z][A-Za-z .'-]{1,48}?)(?=$|[,.!?]|\s+(?:to|for|with|and)\b)",
        re.IGNORECASE,
    ),
]
_DESTINATION_PATTERNS = [
    re.compile(
        r"\b(?:to|towards|heading to|going to)\s+([A-Za-z][A-Za-z .'-]{1,48}?)(?=$|[,.!?]|\s+(?:from|for|with|and)\b)",
        re.IGNORECASE,
    ),
]


@dataclass
class TaskContextSlot:
    key: str
    value: str
    confidence: float
    source: str
    updated_at: float


class TaskContextState:
    SLOT_TTL_SEC = 4 * 60 * 60
    PENDING_TTL_SEC = 20 * 60
    SLOT_ORDER = (
        "task_topic",
        "origin",
        "destination",
        "constraints",
        "preference",
        "time_hint",
    )

    def __init__(self):
        self._slots: Dict[str, TaskContextSlot] = {}
        self._pending_slot: Optional[str] = None
        self._pending_set_at: float = 0.0

    def _clear_expired(self) -> None:
        now = time.time()
        stale = [
            k for k, v in self._slots.items()
            if now - v.updated_at > self.SLOT_TTL_SEC
        ]
        for key in stale:
            self._slots.pop(key, None)
        if (
            self._pending_slot
            and now - self._pending_set_at > self.PENDING_TTL_SEC
        ):
            self._pending_slot = None
            self._pending_set_at = 0.0

    def clear(self) -> None:
        self._slots.clear()
        self._pending_slot = None
        self._pending_set_at = 0.0

    def _set_slot(self, key: str, value: str, confidence: float, source: str) -> None:
        cleaned = re.sub(r"\s+", " ", (value or "").strip(" .,!?:;")).strip()
        if not cleaned:
            return
        if len(cleaned) > 80:
            cleaned = cleaned[:80].rstrip()
        self._slots[key] = TaskContextSlot(
            key=key,
            value=cleaned,
            confidence=max(0.0, min(1.0, float(confidence))),
            source=source,
            updated_at=time.time(),
        )

    def _prime_pending_from_question(self, assistant_text: str) -> None:
        text = (assistant_text or "").strip()
        if not text:
            return
        lowered = text.lower()
        if "?" not in lowered:
            return
        if _ASK_ORIGIN_RE.search(text):
            self._pending_slot = "origin"
            self._pending_set_at = time.time()
            return
        if _ASK_DEST_RE.search(text):
            self._pending_slot = "destination"
            self._pending_set_at = time.time()

    def _looks_like_short_answer(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped or "?" in stripped:
            return False
        words = [w for w in re.split(r"\s+", stripped) if w]
        if len(words) > 4:
            return False
        return bool(re.search(r"[A-Za-z]", stripped))

    def ingest_user_message(self, text: str, previous_assistant: str = "") -> List[Dict[str, Any]]:
        self._clear_expired()
        self._prime_pending_from_question(previous_assistant)
        updates: List[Dict[str, Any]] = []
        msg = (text or "").strip()
        if not msg:
            return updates

        if self._pending_slot and self._looks_like_short_answer(msg):
            slot_key = self._pending_slot
            self._set_slot(slot_key, msg, 0.82, "followup_answer")
            slot = self._slots.get(slot_key)
            if slot is not None:
                updates.append({
                    "key": slot.key,
                    "value": slot.value,
                    "confidence": round(slot.confidence, 2),
                    "source": slot.source,
                })
            self._pending_slot = None
            self._pending_set_at = 0.0

        for pattern in _ORIGIN_PATTERNS:
            match = pattern.search(msg)
            if match:
                self._set_slot("origin", match.group(1), 0.9, "inline_parse")
                break

        for pattern in _DESTINATION_PATTERNS:
            match = pattern.search(msg)
            if match:
                self._set_slot("destination", match.group(1), 0.88, "inline_parse")
                break

        lower = msg.lower()
        if _TRAVEL_SIGNAL_RE.search(msg):
            self._set_slot("task_topic", "travel", 0.72, "topic_infer")

        if re.search(r"\b(?:fast|quick|quickest|speed|convenien)\w*", lower):
            self._set_slot("preference", "speed", 0.7, "preference_infer")
        elif re.search(r"\b(?:scenic|views?|journey|experience|alps)\b", lower):
            self._set_slot("preference", "scenic", 0.68, "preference_infer")

        if re.search(r"\b(?:today|tomorrow|tonight|this weekend|next week)\b", lower):
            matched = re.search(
                r"\b(today|tomorrow|tonight|this weekend|next week)\b",
                lower,
            )
            if matched:
                self._set_slot("time_hint", matched.group(1), 0.66, "time_infer")

        if re.search(r"\b(?:budget|cheap|expensive|avoid|must|can't|cannot)\b", lower):
            phrase = re.sub(r"\s+", " ", msg).strip()
            self._set_slot("constraints", phrase, 0.62, "constraint_infer")

        for key in self.SLOT_ORDER:
            slot = self._slots.get(key)
            if slot is None:
                continue
            if any(u.get("key") == key for u in updates):
                continue
            updates.append({
                "key": slot.key,
                "value": slot.value,
                "confidence": round(slot.confidence, 2),
                "source": slot.source,
            })
        return updates

    def ingest_assistant_message(self, text: str) -> None:
        self._clear_expired()
        self._prime_pending_from_question(text)

    def prompt_block(self) -> str:
        self._clear_expired()
        lines: List[str] = []
        for key in self.SLOT_ORDER:
            slot = self._slots.get(key)
            if slot is None:
                continue
            lines.append(f"- {slot.key}: {slot.value}")
        if not lines:
            return ""
        return (
            "Active task context (ephemeral; use for current problem, do not treat as durable profile):\n"
            + "\n".join(lines)
        )

    def snapshot(self) -> List[Dict[str, Any]]:
        self._clear_expired()
        slots = sorted(
            self._slots.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )
        payload: List[Dict[str, Any]] = []
        for slot in slots:
            payload.append({
                "key": slot.key,
                "value": slot.value,
                "confidence": round(slot.confidence, 2),
                "source": slot.source,
                "updated_at": datetime.fromtimestamp(
                    slot.updated_at, timezone.utc,
                ).isoformat(),
            })
        return payload
