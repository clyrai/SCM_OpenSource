"""Supervisor + 3 specialist agents, all with SCM memory.

Pattern: LangChain 1.x recommended supervisor-via-tools (per the
langgraph-supervisor README's deprecation note pointing to manual
handoff tools). Each worker is a `create_react_agent` with the SCM
tools relevant to its specialty. The supervisor is also a
`create_react_agent`, but its tools are handoff tools that emit
`Command(goto=worker_name, graph=Command.PARENT)` to route the
conversation.

Topology:
                  ┌──────────────┐
                  │  SUPERVISOR  │  routes via delegate_to_* tools
                  └──────┬───────┘
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
   profile_agent    task_agent       recall_agent
   (add_memory      (search_memory)  (search_memory
    + search)                          + wake_summary)
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Annotated

from dotenv import load_dotenv
load_dotenv()

# Add repo root to sys.path so `src.*` imports resolve.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# These need to be at module scope (NOT inside main()) because the @tool
# decorator's pydantic validator evaluates type hints via get_type_hints()
# in the function's enclosing globals — which is this module, not main()'s
# closure. Importing inside main() would raise NameError on the
# `InjectedState` annotation in the handoff tool.
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command


def _signup(base: str, email: str) -> tuple[str, str]:
    """Sign up, mint initial API key. Returns (account_id, token)."""
    import requests
    aid = requests.post(f"{base}/v1/cloud/accounts", json={"email": email}).json()["id"]
    token = requests.post(
        f"{base}/v1/cloud/accounts/{aid}/keys/initial",
        json={"label": "supervisor-team"},
    ).json()["token"]
    return aid, token


def main() -> int:
    from src.integrations.langchain_adapter import SCMClient
    from src.integrations.langchain_tools import make_scm_tools

    BASE = "http://localhost:8000"
    print("\n=== Supervisor + 3 specialist agents ===\n")

    email = f"team-{uuid.uuid4().hex[:8]}@example.com"
    end_user = f"alex_{uuid.uuid4().hex[:6]}"
    print(f"signing up as {email}…")
    aid, token = _signup(BASE, email)
    print(f"account: {aid}")
    print(f"end-user: {end_user}\n")

    # ── SCM tools split per worker ──────────────────────────────────
    scm = SCMClient(api_key=token, base_url=f"{BASE}/v1", user_id=end_user)
    all_tools = make_scm_tools(scm)
    by_name = {t.name: t for t in all_tools}

    profile_tools = [by_name["add_memory"], by_name["search_memory"]]
    task_tools    = [by_name["search_memory"]]
    # recall_agent: prefer the high-level get_user_profile tool for
    # vague meta questions ("what do you know about me?") — it does
    # multi-cue search internally so the LLM doesn't have to invent a
    # query. search_memory and wake_summary stay available as fallbacks.
    recall_tools  = [
        by_name["get_user_profile"],
        by_name["search_memory"],
        by_name["wake_summary"],
    ]

    # ── LLM ────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.0, timeout=30,
    )

    # ── Handoff tool factory ───────────────────────────────────────
    def make_handoff_tool(agent_name: str, description: str):
        tool_name = f"delegate_to_{agent_name}"

        @tool(tool_name, description=description)
        def handoff(
            task: Annotated[str, "What this agent should do, in one sentence."],
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            ack = ToolMessage(
                content=f"Routed to {agent_name}.",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return Command(
                goto=agent_name,
                update={
                    "messages": state["messages"] + [
                        ack,
                        {"role": "user", "content": task},
                    ],
                },
                graph=Command.PARENT,
            )

        return handoff

    # ── Workers ────────────────────────────────────────────────────
    profile_agent = create_react_agent(
        model=llm,
        tools=profile_tools,
        name="profile_agent",
        prompt=(
            "You are the Profile agent. The supervisor delegates to you "
            "when the user shares substantive personal information "
            "(name, location, profession, allergies, preferences, "
            "habits) OR asks specifically about their stored profile. "
            "Use `add_memory` to store the new fact, then briefly "
            "confirm what you stored (1 sentence). Don't invent facts."
        ),
    )

    task_agent = create_react_agent(
        model=llm,
        tools=task_tools,
        name="task_agent",
        prompt=(
            "You are the Task agent. You handle general requests "
            "('help me draft an email', 'recommend a Python pattern'). "
            "BEFORE answering, call `search_memory` to ground your "
            "response in what you know about the user. Then answer "
            "concisely (1-3 sentences). Don't invent facts."
        ),
    )

    recall_agent = create_react_agent(
        model=llm,
        tools=recall_tools,
        name="recall_agent",
        prompt=(
            "You are the Recall agent. The supervisor delegates to you "
            "when the user wants to know what you remember about them, "
            "or wants the wake-summary of what you figured out while "
            "they were away. Call `search_memory` for direct questions "
            "or `wake_summary` for return-from-idle moments. Then "
            "summarize what you found in 1-3 sentences."
        ),
    )

    # ── Supervisor ─────────────────────────────────────────────────
    handoff_tools = [
        make_handoff_tool(
            "profile_agent",
            "Delegate when the user shares OR asks about personal facts "
            "(name, location, profession, allergies, preferences, habits).",
        ),
        make_handoff_tool(
            "task_agent",
            "Delegate when the user wants help with a task, question, "
            "or general assistance — and the answer would benefit from "
            "knowing the user.",
        ),
        make_handoff_tool(
            "recall_agent",
            "Delegate when the user asks what you remember, what you "
            "noticed while they were away, or any meta-memory question.",
        ),
    ]
    supervisor = create_react_agent(
        model=llm,
        tools=handoff_tools,
        name="supervisor",
        prompt=(
            "You are the team supervisor for a personal assistant. "
            "Your job is to delegate every user message to exactly one "
            "specialist using the delegate_to_* tools. You never answer "
            "the user directly — your specialists do. After a "
            "specialist replies, you stop (the user sees the specialist's "
            "answer)."
        ),
    )

    # ── Wire StateGraph ────────────────────────────────────────────
    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("profile_agent", profile_agent)
    graph.add_node("task_agent", task_agent)
    graph.add_node("recall_agent", recall_agent)
    graph.add_edge(START, "supervisor")
    graph.add_edge("profile_agent", "supervisor")
    graph.add_edge("task_agent", "supervisor")
    graph.add_edge("recall_agent", "supervisor")
    team = graph.compile()

    # ── Drive the team ─────────────────────────────────────────────
    # Each scenario: (user_msg, allowed_workers_set, recall_keywords)
    # Multiple allowed workers when the message is genuinely ambiguous.
    scenarios = [
        ("Hi, I'm Alex. I'm a backend engineer in Lisbon.",
         {"profile_agent"}, []),
        ("I have a peanut allergy.",
         {"profile_agent"}, []),
        ("What do you know about me so far?",
         {"recall_agent"}, ["alex", "lisbon", "peanut"]),
        # "Recommend a safe lunch" can route to either task_agent (it's a
        # task) or profile_agent (the answer hinges on the user's profile).
        # As long as the team uses the user's stored allergy, both routes
        # are correct. The reply must mention the allergy.
        ("Recommend a quick lunch I could grab near my office that's safe.",
         {"task_agent", "profile_agent"}, ["peanut", "allerg"]),
    ]

    print("─" * 60)
    print("Driving the team through 4 scenarios")
    print("─" * 60)
    passed, failed = 0, 0

    for i, (user_msg, allowed_workers, expected_kw) in enumerate(scenarios, 1):
        print(f"\nTurn {i}: {user_msg}")
        t0 = time.perf_counter()
        try:
            result = team.invoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                {"recursion_limit": 12},
            )
        except Exception as e:
            print(f"  ✗ team crashed: {type(e).__name__}: {e}")
            failed += 1
            continue
        elapsed = time.perf_counter() - t0

        # Trace which worker(s) the supervisor delegated to
        delegated_to = []
        for m in result["messages"]:
            tcs = getattr(m, "tool_calls", None) or []
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name and name.startswith("delegate_to_"):
                    delegated_to.append(name.replace("delegate_to_", ""))

        # Score the WORKER'S reply, not the supervisor's terminator. Each
        # AIMessage from create_react_agent has .name = the agent's name;
        # filter to non-supervisor names so we don't score the supervisor's
        # "the worker has answered" boilerplate.
        worker_replies = [
            m for m in result["messages"]
            if getattr(m, "type", "") == "ai"
            and (m.content or "").strip()
            and getattr(m, "name", None) not in (None, "supervisor")
        ]
        if worker_replies:
            final_reply = worker_replies[-1].content
            replied_by = getattr(worker_replies[-1], "name", "?")
        else:
            # Fallback: any AI message with content
            ai = [m for m in result["messages"]
                  if getattr(m, "type", "") == "ai" and (m.content or "").strip()]
            final_reply = ai[-1].content if ai else "(no reply)"
            replied_by = getattr(ai[-1], "name", "?") if ai else "?"

        print(f"  delegated_to: {delegated_to}")
        print(f"  reply by: {replied_by}")
        print(f"  reply: {final_reply[:240]}")
        print(f"  ({elapsed:.1f}s)")

        # Routing check — at least one of the allowed workers was hit
        hit = set(delegated_to) & allowed_workers
        if hit:
            print(f"  ✓ routed to {sorted(hit)}")
            routing_ok = True
        else:
            print(f"  ✗ expected route to one of {sorted(allowed_workers)}, got {delegated_to}")
            routing_ok = False

        # Recall check (only when keywords are expected)
        recall_ok = True
        if expected_kw:
            blob = (final_reply or "").lower()
            hits = [k for k in expected_kw if k in blob]
            if hits:
                print(f"  ✓ recalled: {hits}")
            else:
                print(f"  ✗ MISSING any of {expected_kw}")
                recall_ok = False

        if routing_ok and recall_ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Supervisor team test: {passed}/{passed+failed} passed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
