"""Multi-agent LangGraph test — three specialist agents sharing SCM memory.

A common SaaS pattern: a Researcher agent finds facts, a Profiler agent
condenses them, a Writer agent uses them in a reply. All three run in a
LangGraph state machine, all three share access to the same SCM tools.

This test verifies:
  1. Multiple agents can read+write the same SCM namespace concurrently
  2. Cross-account isolation still holds: a separate account's agents
     can't see this account's memories even with the same user_id
  3. The LangGraph orchestration doesn't break the SCM tool wiring

We use a hand-rolled StateGraph (not the built-in `create_supervisor`)
so the test stays small and the SCM tool calls are explicit.

Requires:
  - SCM server on :8000 with SCM_CLOUD_AUTH=1
  - DEEPSEEK_API_KEY in .env
  - langgraph (already pinned in venv: 1.1.10)

Run:
    python tests/agent_with_tools/test_multiagent_langgraph.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Annotated, List

from dotenv import load_dotenv
load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _signup(base: str, email: str) -> tuple[str, str]:
    import requests
    aid = requests.post(f"{base}/v1/cloud/accounts", json={"email": email}).json()["id"]
    token = requests.post(f"{base}/v1/cloud/accounts/{aid}/keys/initial",
                          json={"label": "multiagent"}).json()["token"]
    return aid, token


def main() -> int:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict

    from src.integrations.langchain_adapter import SCMClient
    from src.integrations.langchain_tools import make_scm_tools

    BASE = "http://localhost:8000"
    print("\n=== Multi-agent LangGraph + SCM test ===\n")

    # ── Two separate accounts to verify cross-tenant isolation ──────
    aid_a, token_a = _signup(BASE, f"team-a-{uuid.uuid4().hex[:8]}@example.com")
    aid_b, token_b = _signup(BASE, f"team-b-{uuid.uuid4().hex[:8]}@example.com")
    print(f"Account A: {aid_a}")
    print(f"Account B: {aid_b}")

    # All Account A's agents share ONE user_id (one end-user). They can
    # see each other's writes — that's the desired multi-agent shared
    # memory pattern.
    end_user = f"shared_{uuid.uuid4().hex[:8]}"
    print(f"end-user (shared across all 3 Account-A agents): {end_user}\n")

    # Build one SCM client per agent — same user_id, same token, but
    # they're separate clients so each agent can be moved between
    # threads / processes if needed.
    def client_for(token: str) -> SCMClient:
        return SCMClient(user_id=end_user, base_url=f"{BASE}/v1", api_key=token)

    scm_a_researcher = client_for(token_a)
    scm_a_profiler = client_for(token_a)
    scm_a_writer = client_for(token_a)
    scm_b = client_for(token_b)  # Account B — different account, same user_id

    tools_researcher = make_scm_tools(scm_a_researcher)
    tools_profiler = make_scm_tools(scm_a_profiler)
    tools_writer = make_scm_tools(scm_a_writer)

    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.0, timeout=30,
    )

    # ── State definition ────────────────────────────────────────────
    class TeamState(TypedDict):
        user_msg: str
        notes: List[str]
        profile: str
        reply: str

    # ── Researcher: stores raw facts the user shares + searches  ────
    researcher_llm = llm.bind_tools(tools_researcher)

    def researcher_node(state: TeamState) -> dict:
        sys = SystemMessage(content=(
            "You are the Researcher agent. The user just sent: "
            f"\"{state['user_msg']}\"\n\n"
            "If the user shared substantive facts about themselves "
            "(name, location, profession, allergies, habits), call "
            "`add_memory` to store them. Do NOT call any other tools. "
            "Output a one-line summary of what you stored, or 'no facts' "
            "if there were none."
        ))
        msgs = [sys, HumanMessage(content=state["user_msg"])]
        response = researcher_llm.invoke(msgs)
        # Execute any tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            stored = []
            for tc in response.tool_calls:
                if tc["name"] == "add_memory":
                    args = tc.get("args", {})
                    text = args.get("text", "")
                    if text:
                        scm_a_researcher.add_memory(text=text)
                        stored.append(text)
            return {"notes": stored}
        return {"notes": []}

    # ── Profiler: searches memory for everything known about the user  ─
    profiler_llm = llm.bind_tools(tools_profiler)

    def profiler_node(state: TeamState) -> dict:
        sys = SystemMessage(content=(
            "You are the Profiler agent. Call `search_memory` to retrieve "
            "everything stored about the user. Use the queries:\n"
            "  - 'name and personal details'\n"
            "  - 'location'\n"
            "  - 'profession or work'\n"
            "Then output ONE compact paragraph summarizing what you found. "
            "Do not invent anything not in the search results."
        ))
        # Run the agent loop manually (1-2 iterations is enough)
        msgs = [sys, HumanMessage(content="Build a profile of the user.")]
        for _ in range(3):  # cap iterations
            response = profiler_llm.invoke(msgs)
            msgs.append(response)
            tcs = getattr(response, "tool_calls", None) or []
            if not tcs:
                break
            for tc in tcs:
                if tc["name"] == "search_memory":
                    result = next(t for t in tools_profiler if t.name == "search_memory").invoke(tc["args"])
                    msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                else:
                    msgs.append(ToolMessage(content="(skipped)", tool_call_id=tc["id"]))
        # Final assistant content is the profile
        final = next((m.content for m in reversed(msgs)
                      if getattr(m, "type", "") == "ai" and m.content), "")
        return {"profile": final}

    # ── Writer: composes the user-facing reply ─────────────────────
    writer_llm = llm.bind_tools(tools_writer)

    def writer_node(state: TeamState) -> dict:
        sys = SystemMessage(content=(
            f"You are the Writer agent. The user said: \"{state['user_msg']}\"\n\n"
            f"The Profiler agent built this user profile:\n"
            f"{state['profile']}\n\n"
            f"Compose a 1-2 sentence reply that addresses the user's "
            f"message and shows you know who they are. If the profile is "
            f"empty, just respond conversationally without inventing facts."
        ))
        msgs = [sys, HumanMessage(content="Write the reply.")]
        response = writer_llm.invoke(msgs)
        return {"reply": response.content}

    # ── Wire the graph ─────────────────────────────────────────────
    graph = StateGraph(TeamState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("profiler", profiler_node)
    graph.add_node("writer", writer_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "profiler")
    graph.add_edge("profiler", "writer")
    graph.add_edge("writer", END)
    team = graph.compile()

    # ── Drive the team through scenarios ───────────────────────────
    scenarios = [
        ("Hi, I'm Sam. I work at Filtrum as a data engineer in Berlin.",
         []),
        ("I bike to work every morning — about 6km along the canal.",
         []),
        ("Quick question — based on what you know, where do I work and how do I commute?",
         ["filtrum", "data engineer", "bike", "canal"]),
    ]

    print("─" * 60)
    print("PHASE 1 — Account A's 3-agent team handles the conversation")
    print("─" * 60)
    passed, failed = 0, 0
    for i, (msg, expected) in enumerate(scenarios, 1):
        print(f"\nTurn {i}: {msg}")
        t0 = time.perf_counter()
        try:
            result = team.invoke({"user_msg": msg, "notes": [], "profile": "", "reply": ""})
        except Exception as e:
            print(f"  ✗ team crashed: {type(e).__name__}: {e}")
            failed += 1
            continue
        elapsed = time.perf_counter() - t0
        print(f"  Researcher stored: {result['notes']}")
        print(f"  Profile: {result['profile'][:200]}")
        print(f"  Final reply: {result['reply'][:200]}")
        print(f"  ({elapsed:.1f}s)")

        if expected:
            blob = (result["reply"] + " " + result["profile"]).lower()
            hits = [k for k in expected if k in blob]
            if hits:
                print(f"  ✓ recalled: {hits}")
                passed += 1
            else:
                print(f"  ✗ MISSING any of {expected}")
                failed += 1
        else:
            # Storage turn — should have stored at least one fact
            if result["notes"]:
                print(f"  ✓ stored {len(result['notes'])} notes")
                passed += 1
            else:
                print(f"  ✗ stored nothing")
                failed += 1

    # ── PHASE 2 — Cross-account isolation ──────────────────────────
    print("\n" + "─" * 60)
    print("PHASE 2 — Account B (different account, SAME user_id) cannot see A's memory")
    print("─" * 60)
    r = scm_b.search_memory(query="where does the user work", limit=5, wait_for_pending=True)
    b_blob = (r.get("memory_context", "") + " " +
              " ".join((m.get("description") or "") for m in (r.get("memories") or []))).lower()
    print(f"\nAccount B's search returned {r.get('retrieved_count', 0)} concepts")
    print(f"  blob preview: {b_blob[:200]}")
    leaked = any(kw in b_blob for kw in ["filtrum", "berlin", "data engineer", "sam"])
    if not leaked:
        print(f"  ✓ Account B sees nothing from Account A — cross-tenant isolation holds")
        passed += 1
    else:
        print(f"  ✗ LEAK: Account B can see Account A's facts")
        failed += 1

    # ── Summary ────────────────────────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Multi-agent test: {passed}/{total} passed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
