"""
Sleep-time memory paraphrase.

During NREM consolidation, concepts that have been rehearsed enough times to
warrant durable storage are rewritten into clean, retrieval-friendly fact
forms. The rewrite is applied at consolidation time rather than ingestion
time, so the cost is paid only on memories that have already proven worth
keeping — fresh ingest stays fast, only durable memories incur the polish.

Two backends:
  - HeuristicParaphraser  (free, regex/template based)
  - LLMParaphraser        (OpenAI-compat LLM call; e.g. DeepSeek)

The default in production is heuristic, with LLM as opt-in.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..core import linguistic_resources as lr
from ..core.models import Concept


class HeuristicParaphraser:
    """
    Cheap, deterministic paraphrasing using regex rewrites loaded from the
    linguistic_resources file. NO hardcoded patterns live in this class.

    The active rules are defined in `locales/en.json` under
    `paraphrase_rules.rules`. Override at runtime via the
    LINGUISTIC_CONFIG_PATH env var to switch locales / domains.
    """

    # Loaded once per process from linguistic_resources. The legacy class
    # attribute is kept as an empty list for backward compatibility.
    REWRITES: List[tuple] = []

    def __init__(self):
        # Compile per-instance from current resources so test-time reloads
        # take effect without restarting the process.
        self._rewrites: List[tuple] = lr.compile_paraphrase_rules()

    def paraphrase(self, description: str) -> Optional[str]:
        """Return paraphrased description, or None if no rewrite applies."""
        text = (description or "").strip()
        if not text:
            return None
        for pattern, template in self._rewrites:
            m = pattern.match(text)  # match (anchored to start)
            if m:
                groups = m.groupdict()
                speaker = groups.get("sp")
                speaker = speaker.strip() if speaker else "Speaker"
                fmt = {"sp": speaker}
                # Pass through every named capture group so templates can use
                # whichever fields they like (obj, rest, verb, etc.).
                for k, v in groups.items():
                    if k in {"sp"}:
                        continue
                    fmt[k] = (v or "").rstrip(".!?")
                try:
                    rewritten = template.format(**fmt).strip()
                except (KeyError, IndexError):
                    continue
                rewritten = re.sub(r"\s+", " ", rewritten)
                if not rewritten:
                    continue
                # Avoid degenerate rewrites that don't change anything meaningful.
                if rewritten.lower().rstrip(".") == text.lower().rstrip("."):
                    continue
                return rewritten
        return None


class LLMParaphraser:
    """
    LLM-backed paraphraser. Single concept → clean fact statement.

    Caller passes an LLMExtractor (uses ._chat() internally). Costs ~1 LLM
    call per consolidated concept; only fired when concept rehearsal_count
    exceeds a threshold so cost is bounded.
    """

    PROMPT = (
        "Rewrite the following fragment as ONE concise, self-contained statement "
        "of fact. Do not add new information. Do not invent details. If the "
        "fragment is too vague to clean up, return it verbatim.\n\n"
        "Fragment: \"{description}\"\n\n"
        "Output:"
    )

    def __init__(self, llm):
        self.llm = llm

    def paraphrase(self, description: str) -> Optional[str]:
        try:
            out = self.llm._chat(
                self.PROMPT.format(description=description),
                num_predict=64,
            )
            out = (out or "").strip().strip('"').strip("'")
            return out if out and len(out) >= 4 else None
        except Exception:
            return None


class SleepParaphraser:
    """
    Orchestrates paraphrase application during sleep cycles.

    Only paraphrases concepts whose rehearsal_count >= min_rehearsals. The
    original description is preserved in context_tags["original_description"]
    for audit.
    """

    def __init__(
        self,
        min_rehearsals: int = 0,  # 0 = paraphrase every concept on first sleep
        backend=None,
        annotate_only: bool = False,
    ):
        self.min_rehearsals = max(0, int(min_rehearsals))
        self.backend = backend or HeuristicParaphraser()
        self.annotate_only = bool(annotate_only)
        self.stats = {
            "considered": 0,
            "rewritten": 0,
            "skipped_no_pattern": 0,
            "skipped_low_rehearsal": 0,
        }

    def apply(self, concepts: List[Concept]) -> List[Concept]:
        for c in concepts:
            self.stats["considered"] += 1
            if (c.rehearsal_count or 0) < self.min_rehearsals:
                self.stats["skipped_low_rehearsal"] += 1
                continue
            new_desc = self.backend.paraphrase(c.description)
            if not new_desc or new_desc == c.description:
                self.stats["skipped_no_pattern"] += 1
                continue
            # Preserve original for audit trail.
            if c.context_tags is None:
                c.context_tags = {}
            c.context_tags.setdefault("original_description", c.description)
            c.context_tags["paraphrased"] = True
            if not self.annotate_only:
                c.description = new_desc
            self.stats["rewritten"] += 1
        return concepts

    def get_stats(self):
        d = dict(self.stats)
        n = max(1, d["considered"])
        d["rewrite_rate"] = round(d["rewritten"] / n, 4)
        return d
