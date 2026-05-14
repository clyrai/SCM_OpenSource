"""LangChain `@tool`-decorated wrappers around SCM, for tool-calling agents.

Two integration shapes for LangChain:

  • The "manual" pattern (search → invoke → store on every turn) is in
    src/integrations/langchain_adapter.py. The developer drives the
    memory calls. Verified by tests/brutal_langchain/.

  • This module is the "agent decides" pattern. SCM is exposed as tools
    the agent can call autonomously inside an AgentExecutor / LangGraph
    loop. The agent reads the user's message, reasons about whether to
    recall, calls `search_memory(query=...)`, gets results back as
    observations, optionally calls `add_memory(text=...)`, then
    answers. No per-turn glue code in the developer's hands.

Use ``make_scm_tools(scm_client)`` to get a list of tools you can pass
into `create_tool_calling_agent` or attach to a LangGraph node.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _import_langchain_tools():
    try:
        from langchain_core.tools import tool  # type: ignore
        return tool
    except ImportError as e:
        raise ImportError(
            "LangChain tools are not installed. Run: pip install langchain-core"
        ) from e


def make_scm_tools(scm_client: Any) -> List[Any]:
    """Return a list of LangChain ``@tool``-decorated callables backed by
    a single SCMClient. The agent (or LangGraph node) gets to call these
    autonomously inside its own reasoning loop.

    Args:
        scm_client: an SCMClient bound to the calling user_id. The tools
            close over this object — pass one client per logical user.

    Returns:
        List of tool objects: [search_memory, add_memory, consolidate,
        wake_summary]. Pass them straight into
        ``create_tool_calling_agent(llm, tools=..., prompt=...)`` or
        attach to ``ToolNode`` in LangGraph.
    """
    tool = _import_langchain_tools()

    @tool
    def search_memory(query: str) -> str:
        """Search the agent's persistent memory for facts relevant to a query.

        IMPORTANT for the LLM: if the user's question has multiple distinct
        sub-questions (e.g. "where do I work AND where do I run?"), call
        search_memory ONCE PER sub-question. A single compound query may
        only return the top-ranked branch.

        Use this when:
          - The user asks about something they previously told you
            ("where do I work?", "what's my favorite coffee?")
          - You're about to answer a question and want to ground it in
            anything the user has said before
          - You suspect there's relevant context across earlier sessions

        The query should be natural-language ("the user's job" /
        "favorite drinks") — paraphrases work (SCM uses spreading-
        activation, not exact match).

        Args:
            query: a short natural-language phrase describing what to recall.

        Returns:
            A clean bulleted list of relevant memories, or
            "(no memories found …)" if nothing is found. Includes any
            pending wake-summary ("[While you were away: …]") if SCM
            auto-fired a sleep cycle.
        """
        try:
            r = scm_client.search_memory(
                query=query, limit=8, wait_for_pending=True,
            )
        except Exception as e:
            return f"[search_memory error: {e}]"

        retrieval = r.get("retrieval") if isinstance(r, dict) else {}
        low_confidence_hint = ""
        if isinstance(retrieval, dict):
            if retrieval.get("decision") == "clarify_or_bound_uncertainty":
                low_confidence_hint = (
                    "[retrieval confidence is low; ask one concise clarifying "
                    "question before asserting uncertain facts]\n\n"
                )

        wake = r.get("wake_summary_pending")
        wake_block = ""
        if wake and wake.get("narrative"):
            wake_block = f"[While the user was away: {wake['narrative']}]\n\n"

        # Prefer the structured `memories` list — it's clean concept
        # descriptions without the rank/salience/activation diagnostics
        # that pollute the formatted memory_context block.
        mems = r.get("memories") or []
        if mems:
            lines = []
            for m in mems:
                d = (m.get("description") or "").strip()
                if d:
                    lines.append(f"- {d}")
            if lines:
                return low_confidence_hint + wake_block + "\n".join(lines)

        # Fallback: the formatted block, with the per-memory diagnostic
        # tail stripped (everything from the first '[Rank' / '[' to EOL).
        ctx = (r.get("memory_context") or "").strip()
        if ctx:
            cleaned_lines = []
            for ln in ctx.splitlines():
                if ln.startswith("##"):
                    continue  # skip the "Retrieved Memories (confidence:…)" header
                # Strip per-line tail "[Rank N] (medium) [salience=…]…"
                idx = ln.find("[Rank")
                if idx > 0:
                    ln = ln[:idx].rstrip()
                if ln.strip():
                    cleaned_lines.append(ln)
            if cleaned_lines:
                return low_confidence_hint + wake_block + "\n".join(cleaned_lines)

        return low_confidence_hint + (wake_block or "") + "(no memories found for that query)"

    @tool
    def add_memory(text: str, replaces_prior: bool = False) -> str:
        """Store a fact in the agent's persistent memory.

        Use this whenever the user shares a substantive fact about
        themselves or their preferences that should outlive the
        current session — name, location, profession, allergies,
        habits, decisions, opinions. Don't use this to store the
        agent's own replies or transient chat noise.

        CRITICAL — the `replaces_prior` flag:

        Set `replaces_prior=True` ONLY when the user is EXPLICITLY
        correcting a fact THEY THEMSELVES stated earlier in this
        conversation. Examples that warrant True:
          - "I'm Saish, not Alex"        (name correction)
          - "Actually I moved to Seattle" (location correction)
          - "I left Atlas, I'm at Northstar now" (job correction)

        Leave `replaces_prior=False` (the default) for everything
        else, including:
          - A brand-new fact the user just shared
          - An additive detail ("I also like tea")
          - A different person joining the chat ("Hi I'm Shakira,
            Saish's friend") — Shakira's introduction is NOT a
            correction of Saish's facts. Storing it with
            replaces_prior=True would supersede Saish's name,
            location, allergies — everything.

        When in doubt, leave it False. The conservative default
        keeps facts; an aggressive True wipes them.

        Args:
            text: the verbatim or rephrased fact to remember,
                  written from the user's perspective when possible
                  ("I work at Atlas Labs" not "user works at Atlas Labs").
            replaces_prior: ONLY True when the user explicitly
                  corrects a previously-stated fact of their own.

        Returns:
            A short confirmation. The actual concept extraction +
            embedding happens server-side; this returns immediately.
        """
        try:
            scm_client.add_memory(text=text, replaces_prior=bool(replaces_prior))
            tag = " (correction)" if replaces_prior else ""
            return f"stored{tag}: {text[:120]}"
        except Exception as e:
            return f"[add_memory error: {e}]"

    @tool
    def consolidate(mode: str = "deep") -> str:
        """Force a sleep-consolidation cycle right now.

        Most agents do NOT need this — SCM fires consolidation
        automatically at the user's configured bedtime. Use this only
        when the user explicitly asks ("summarize what you've learned",
        "review your notes") or when wrapping up a long session.

        Args:
            mode: 'deep' (full NREM + REM, ~5-15s) or 'micro'
                  (fast intra-session pass, <1s).

        Returns:
            Stats: schemas formed, concepts consolidated, memories
            forgotten, contradictions resolved.
        """
        try:
            r = scm_client.consolidate(mode=mode)
            return (
                f"sleep cycle complete | "
                f"consolidated={r.get('concepts_consolidated', 0)} "
                f"forgotten={r.get('concepts_forgotten', 0)} "
                f"schemas={r.get('schemas_formed', 0)} "
                f"contradictions_resolved={r.get('contradictions_resolved', 0)}"
            )
        except Exception as e:
            return f"[consolidate error: {e}]"

    @tool
    def wake_summary(since_hours: float = 24.0) -> str:
        """Read what the agent learned during recent idle / sleep.

        Use this when the user comes back after a gap and you want to
        know what patterns or facts the agent abstracted while they
        were away. Most useful for opening lines like 'Welcome back —
        I noticed X while you were gone.'

        Args:
            since_hours: how far back to look. Defaults to 24h.

        Returns:
            A short narrative block, or an empty string if there's
            nothing recent.
        """
        try:
            r = scm_client.wake_summary(since_hours=since_hours)
            return (r.get("narrative") or "").strip() or "(no wake summary yet)"
        except Exception as e:
            return f"[wake_summary error: {e}]"

    @tool
    def get_user_profile() -> str:
        """Get a complete profile of everything you know about the user.

        Call this ONE tool when the user asks meta questions like:
          - "What do you know about me?"
          - "What's in your memory?"
          - "Tell me what you've learned about me so far."
          - "Sorry, what was my name?"
          - "Refresh my profile."

        Use this INSTEAD of `search_memory` for any "what's stored about
        me" question. `search_memory` uses cue-based spreading activation
        which is reliable for content queries ("when do I run?") but
        unreliable for meta queries ("the user's name") — the cue tokens
        don't overlap with the stored content tokens.

        Returns:
            A bullet list of everything currently in memory about the
            user, deduplicated, with patterns surfaced after sleep
            cycles. Or "(no profile yet …)" if nothing's stored yet.
        """
        bullets: list = []
        seen: set = set()

        # Preferred path: SCMClient has a list_facts() method that walks
        # the concept graph directly. This is the in-process /chat
        # client's behavior — bypasses spreading activation, always
        # returns the actual stored facts. Reliable.
        if hasattr(scm_client, "list_facts"):
            try:
                r = scm_client.list_facts()
                facts = r.get("facts") or []
                schemas = r.get("schemas") or []
                for f in facts:
                    d = (f.get("description") or "").strip()
                    if d and d not in seen:
                        seen.add(d); bullets.append(f"- {d}")
                if schemas:
                    bullets.append("\nPatterns I've noticed:")
                    for s in schemas:
                        d = (s.get("description") or "").strip()
                        if d and d not in seen:
                            seen.add(d); bullets.append(f"  • {d}")
                if bullets:
                    return "What I know about the user:\n" + "\n".join(bullets)
            except Exception:
                pass

        # Fallback path: client doesn't expose list_facts (older SCMClient
        # over HTTP). Issue several focused spreading-activation searches
        # to cover the dimensions a typical profile spans, dedupe, return.
        # Less reliable for meta queries but works for clients that don't
        # speak the newer list_facts protocol.
        cues = [
            "the user's name",
            "where the user lives",
            "the user's profession or job",
            "the user's preferences and favorites",
            "the user's allergies and health",
            "the user's hobbies and routines",
        ]
        for cue in cues:
            try:
                r = scm_client.search_memory(
                    query=cue, limit=5, wait_for_pending=True,
                )
            except Exception:
                continue
            for m in r.get("memories") or []:
                desc = (m.get("description") or "").strip()
                if not desc or desc in seen:
                    continue
                seen.add(desc)
                bullets.append(f"- {desc}")
        if not bullets:
            return "(no profile yet — the user hasn't shared anything substantive)"
        return "What I know about the user:\n" + "\n".join(bullets)

    return [search_memory, add_memory, consolidate, wake_summary, get_user_profile]


__all__ = ["make_scm_tools"]
