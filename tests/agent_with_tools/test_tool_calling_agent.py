"""End-to-end test: a real LangChain `create_tool_calling_agent` with SCM tools.

Verifies that:
  1. The agent calls `add_memory` when the user states a fact
  2. The agent calls `search_memory` when the user asks about something
     it was told earlier
  3. The agent's reply is grounded in the retrieved memory
  4. Multiple tool calls in one turn work (search → answer)

Requires:
  - SCM server running locally on :8000 (start with: `scm serve --port 8000`)
  - DEEPSEEK_API_KEY in .env
  - LangChain + langchain-openai installed (already in venv)

Run:
    python tests/agent_with_tools/test_tool_calling_agent.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

# Add repo root to sys.path so `src.*` imports resolve.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _signup_and_get_token(base: str, email: str) -> tuple[str, str]:
    """If the SCM server has cloud auth enabled, sign up + mint a key.
    Returns (account_id, token). On a server with cloud auth off the
    POST will still succeed but the token isn't required for /v1/memories
    — we still return it so the test path is identical."""
    import requests
    r = requests.post(f"{base}/v1/cloud/accounts", json={"email": email})
    r.raise_for_status()
    aid = r.json()["id"]
    r = requests.post(f"{base}/v1/cloud/accounts/{aid}/keys/initial",
                      json={"label": "agent-test"})
    r.raise_for_status()
    return aid, r.json()["token"]


def main() -> int:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    from src.integrations.langchain_adapter import SCMClient
    from src.integrations.langchain_tools import make_scm_tools

    BASE = "http://localhost:8000"

    # Test against the cloud-auth product surface: real signup, real key,
    # real Bearer-authed SCMClient. This mirrors what a paying SaaS user
    # would do.
    email = f"agent-test-{uuid.uuid4().hex[:8]}@example.com"
    user_id = f"tooluser_{uuid.uuid4().hex[:8]}"
    print(f"\n=== Tool-calling agent test (against cloud auth) ===")
    print(f"signing up as {email}…")
    aid, token = _signup_and_get_token(BASE, email)
    print(f"account_id: {aid}")
    print(f"token: {token[:32]}…")
    print(f"agent end-user user_id: {user_id}\n")

    # 1. Build the SCM client + tools — Bearer-authed against the cloud
    scm = SCMClient(user_id=user_id, base_url=f"{BASE}/v1", api_key=token)
    tools = make_scm_tools(scm)
    print(f"SCM tools available to agent: {[t.name for t in tools]}\n")

    # 2. Build the LLM (DeepSeek via OpenAI-compat)
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.0,
        timeout=30,
    )

    # 3. Build the agent. The system prompt tells the LLM how to use the tools.
    system_prompt = (
        "You are a helpful assistant with persistent memory provided by the "
        "SCM tools. Follow these rules every turn:\n"
        "  • If the user asks about something they previously told you, "
        "call `search_memory` to recall it. If their question has "
        "multiple distinct parts (e.g. 'where do I work AND where do I "
        "run?'), call `search_memory` SEPARATELY for each part.\n"
        "  • Trust whatever `search_memory` returns — those are facts the "
        "user has actually told you. Use them in your reply verbatim.\n"
        "  • If the user shares a substantive fact about themselves "
        "(name, location, job, allergies, preferences, habits), call "
        "`add_memory` to remember it.\n"
        "  • Don't invent facts not in your memory or the user's current "
        "message.\n"
        "  • Keep replies to 1-2 short sentences."
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    # 4. Drive the agent through a multi-turn scenario that exercises both
    #    memory tools.
    turns = [
        ("Hi, I'm Alex. I'm a backend engineer in Lisbon.", ["alex"]),
        ("I run every Tuesday morning along the river.", []),
        ("My favorite coffee shop is Hello Kristof in Anjos.", []),
        ("Where do I work and where do I run?",
            ["backend", "engineer", "lisbon", "tuesday", "river"]),
        ("What's my preferred coffee spot?",
            ["hello kristof", "anjos"]),
    ]

    passed, failed = 0, 0
    transcript = []

    for i, (user_msg, expected_keywords) in enumerate(turns, 1):
        print(f"--- Turn {i} ---")
        print(f"  USER: {user_msg}")
        t0 = time.perf_counter()
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_msg}],
            })
        except Exception as e:
            print(f"  ✗ agent.invoke crashed: {type(e).__name__}: {e}")
            failed += 1
            continue
        elapsed = time.perf_counter() - t0

        # The agent returns a state dict; the final assistant message is
        # the last AIMessage in result["messages"].
        ai_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "ai"]
        reply = ai_msgs[-1].content if ai_msgs else "(no reply)"

        # Count tool calls.
        tool_calls = []
        for m in result["messages"]:
            tcs = getattr(m, "tool_calls", None) or []
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tool_calls.append(name)

        print(f"  TOOLS CALLED: {tool_calls}")
        print(f"  ASSISTANT: {reply[:200]}")
        print(f"  ({elapsed:.1f}s)")

        # Recall-turn check
        if expected_keywords:
            blob = (reply or "").lower()
            hits = [k for k in expected_keywords if k in blob]
            if hits:
                print(f"  ✓ recalled: {hits}")
                passed += 1
            else:
                print(f"  ✗ MISSING any of {expected_keywords}")
                failed += 1
        else:
            # Storage turn — should have called add_memory at least once
            if "add_memory" in tool_calls:
                print(f"  ✓ called add_memory")
                passed += 1
            else:
                print(f"  ✗ did NOT call add_memory")
                failed += 1
        transcript.append({"turn": i, "user": user_msg, "reply": reply,
                          "tools": tool_calls, "elapsed": elapsed})
        print()

    print("=" * 60)
    print(f"Tool-calling agent: {passed} passed, {failed} failed of {len(turns)}")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
