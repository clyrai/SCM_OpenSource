"""
Deterministic matching primitives shared across metric scorers.

These functions are intentionally simple and explicit. They do NOT use
embeddings or LLMs — that would make scoring stochastic and ungameable.
The cost is some recall: a system that reports a semantically-correct
schema in unusual phrasing may not match. This is an honest tradeoff
documented in SPEC.md §9 (threats to validity).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Tokens shorter than this are skipped. Tuned to drop "a", "in", "is", etc.
MIN_TOKEN_LEN = 3

# A small, fixed stoplist. Deliberately tiny — most disambiguation is done
# by the type-specific match rules, not by stoplist.
STOP = frozenset({
    "the", "and", "for", "you", "user", "agent", "with", "that", "this",
    "are", "have", "has", "had", "was", "were", "from", "into", "out",
    "your", "their", "his", "her", "its", "our", "but", "not", "can",
    "all", "any", "some", "one", "two", "three", "very", "much", "many",
})


def tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric word tokenize. Stable, no language model."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def content_tokens(text: str) -> set:
    """Tokens minus stoplist minus short words. Used for fact overlap."""
    return {t for t in tokenize(text) if len(t) >= MIN_TOKEN_LEN and t not in STOP}


def schema_text_blob(schema: Any) -> str:
    """Concatenate all readable text from a Schema object for matching."""
    sig = getattr(schema, "signature", None) or {}
    raw = getattr(schema, "raw_text", "") or ""
    schema_type = getattr(schema, "type", "") or ""
    try:
        sig_str = json.dumps(sig, sort_keys=True)
    except (TypeError, ValueError):
        sig_str = str(sig)
    return " ".join([schema_type, sig_str, raw])


def match_pattern(
    gt_pattern: Dict[str, Any],
    schemas: Sequence[Any],
) -> Tuple[Optional[str], float]:
    """
    Try to match a ground-truth pattern against the system's reported schemas.

    Returns (matched_schema_id_or_None, match_score). Score is 1.0 on hit,
    0.0 otherwise — no partial credit in v0.1 (binary by design).

    Match rules per type:
      REPETITION: at least half (rounded down, min 1) of the fact's content
                  tokens appear in the schema text blob.
      COOCCUR: every entity in signature.entities appears in the blob.
      TEMPORAL_CADENCE: signature.event AND signature.day_of_week both appear.
      TRAJECTORY: every step in signature.sequence appears, in order.
    """
    ptype = gt_pattern["type"]
    sig = gt_pattern["signature"]

    if ptype == "REPETITION":
        fact_tokens = content_tokens(sig.get("fact", ""))
        if not fact_tokens:
            return (None, 0.0)
        threshold = max(1, len(fact_tokens) // 2)
        for s in schemas:
            blob = schema_text_blob(s).lower()
            blob_tokens = content_tokens(blob)
            overlap = fact_tokens & blob_tokens
            if len(overlap) >= threshold:
                return (getattr(s, "schema_id", None), 1.0)
        return (None, 0.0)

    if ptype == "COOCCUR":
        entities = [e.lower() for e in sig.get("entities", [])]
        if not entities:
            return (None, 0.0)
        for s in schemas:
            blob = schema_text_blob(s).lower()
            if all(e in blob for e in entities):
                return (getattr(s, "schema_id", None), 1.0)
        return (None, 0.0)

    if ptype == "TEMPORAL_CADENCE":
        event = (sig.get("event") or "").lower().strip()
        dow = (sig.get("day_of_week") or "").lower().strip()
        if not event or not dow:
            return (None, 0.0)
        for s in schemas:
            blob = schema_text_blob(s).lower()
            if event in blob and dow in blob:
                return (getattr(s, "schema_id", None), 1.0)
        return (None, 0.0)

    if ptype == "TRAJECTORY":
        seq = [str(x).lower() for x in sig.get("sequence", [])]
        if not seq:
            return (None, 0.0)
        for s in schemas:
            blob = schema_text_blob(s).lower()
            # Order-preserving substring search.
            cursor = 0
            ok = True
            for step in seq:
                idx = blob.find(step, cursor)
                if idx < 0:
                    ok = False
                    break
                cursor = idx + len(step)
            if ok:
                return (getattr(s, "schema_id", None), 1.0)
        return (None, 0.0)

    # Unknown type — score 0, do not raise. Future versions may add types
    # without breaking older personas.
    return (None, 0.0)


def score_keyword_response(
    response_text: str,
    must_contain_any_of: Sequence[str] = (),
    must_not_contain: Sequence[str] = (),
    min_match_count: int = 1,
) -> float:
    """
    Score a free-text response by keyword presence rules.

    Returns 1.0 iff:
      - at least min_match_count of must_contain_any_of appear, AND
      - none of must_not_contain appear.
    """
    if not response_text:
        return 0.0
    blob = response_text.lower()
    pos_hits = sum(1 for kw in must_contain_any_of if kw.lower() in blob)
    neg_hit = any(kw.lower() in blob for kw in must_not_contain)
    if pos_hits >= min_match_count and not neg_hit:
        return 1.0
    return 0.0
