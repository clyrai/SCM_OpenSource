"""
SCM command-line interface.

Sub-commands:
    scm chat                Interactive REPL with the agent.
    scm sleep               Force a deep-sleep cycle.
    scm wake-summary        Show what the agent learned during the most recent idle.
    scm status              Show memory stats (concept counts, schemas, gaps).
    scm export <path>       Dump memory state to JSON.
    scm import <path>       Load memory state from JSON.
    scm serve               Run the FastAPI service (delegates to uvicorn).
    scm version             Print the SCM version.
    scm config              Print resolved configuration.

Profiles are env-driven (see docs/DEPLOYMENT.md). The CLI honours:
    LLM_PROVIDER          ollama | deepseek | openai (default: ollama)
    SCM_EMBEDDING_BACKEND sentence_transformers | ollama | openai_compat | hash
    SCM_DATA_DIR          where to persist sessions (default: ~/.scm)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ---------- shared helpers -----------------------------------------------


def _scm_data_dir() -> Path:
    p = Path(os.environ.get("SCM_DATA_DIR", str(Path.home() / ".scm")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_engine(session_id: Optional[str] = None, profile: str = "chatbot"):
    """Construct a ChatEngine using env-configured LLM + embedding backends."""
    # Lazy import — keeps `scm version` and `scm --help` fast.
    from src.chat.engine import ChatEngine
    from src.chat import engine as engine_mod
    from src.core.encoder import MeaningEncoder
    from src.lifecycle.curiosity import (
        CuriosityConfig,
        CuriosityEngine,
        StaticDictionarySource,
    )
    from src.lifecycle.wake_summary import WakeSummaryBuilder
    from src.sleep.deep_sleep import DeepSleep
    from src.sleep.schema_extractor import SchemaExtractor, SchemaExtractorConfig
    from src.sleep.sleep_cycle import SleepCycleOrchestrator

    engine_mod.HME_ENABLED = True

    # LLM extractor: heuristic if no provider configured.
    llm = None
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider in ("ollama", "deepseek", "openai"):
        try:
            from src.llm import LLMExtractor
            llm = LLMExtractor(provider=provider)
        except Exception as e:
            print(f"[scm] LLM extractor unavailable ({e}); using heuristic.")

    # Embedding backend: respects env SCM_EMBEDDING_BACKEND.
    encoder = MeaningEncoder(llm=llm)

    deep = DeepSleep(
        enable_synthesis=False,
        enable_schema_extraction=True,
        schema_extractor=SchemaExtractor(
            config=SchemaExtractorConfig(enabled=True),
        ),
        enable_paraphrase=True,
        enable_curiosity=True,
        curiosity_engine=CuriosityEngine(
            sources=[StaticDictionarySource({})],  # extend via SCM_CURIOSITY_DICT
            config=CuriosityConfig(enabled=True, min_occurrences=2, max_gaps_per_cycle=3),
        ),
    )
    orch = SleepCycleOrchestrator(deep_sleep=deep)

    sid = session_id or os.environ.get("SCM_SESSION_ID") or "cli"
    engine = ChatEngine(
        llm=llm,
        encoder=encoder,
        sleep_orchestrator=orch,
        session_id=sid,
        profile=profile,
        sandbox_mode=False,
        enable_persistence=True,
        enable_auto_sleep=False,
    )
    # Attach a WakeSummaryBuilder so the wake-summary command works.
    engine._wake_summary_builder = WakeSummaryBuilder(engine=engine)  # type: ignore[attr-defined]
    return engine


def _format_concept_counts(stats: dict) -> str:
    lt = stats.get("long_term_memory", stats)
    return (
        f"  concepts active     : {lt.get('active_count', '?')}\n"
        f"  concepts archived   : {lt.get('archived_count', '?')}\n"
        f"  concepts suppressed : {lt.get('suppressed_count', '?')}\n"
        f"  relations           : {lt.get('relation_count', '?')}\n"
    )


# ---------- subcommands ---------------------------------------------------


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL: type a message, get a response, watch memory grow."""
    print("SCM interactive session. Type /help for commands, /quit to exit.")
    print(f"Data dir: {_scm_data_dir()} | session: {args.session}")
    engine = _build_engine(session_id=args.session)

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, *rest = line[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd in {"quit", "q", "exit"}:
                break
            elif cmd == "help":
                print(textwrap.dedent("""\
                    /quit            exit
                    /sleep           force a deep-sleep consolidation cycle
                    /wake [HOURS]    show wake-summary for the last N hours (default 24)
                    /status          memory stats
                    /export PATH     write memory state to JSON
                    /clear           reset all memory (dangerous)
                """))
            elif cmd == "sleep":
                print("(running deep sleep…)")
                stats = engine.force_sleep("deep") or {}
                schemas = stats.get("schemas_formed", 0)
                print(f"done. schemas formed: {schemas}")
            elif cmd == "wake":
                hours = float(arg) if arg.strip() else 24.0
                since = datetime.utcnow() - timedelta(hours=hours)
                ws = engine._wake_summary_builder.build(since=since)  # type: ignore[attr-defined]
                _print_wake_summary(ws)
            elif cmd == "status":
                print(_format_concept_counts(engine.get_memory_report()))
            elif cmd == "export":
                if not arg.strip():
                    print("usage: /export PATH")
                    continue
                path = Path(arg.strip())
                with path.open("w") as f:
                    json.dump(engine.export_memory(), f, indent=2, default=str)
                print(f"wrote {path}")
            elif cmd == "clear":
                engine.reset_memory()
                print("(memory reset)")
            else:
                print(f"unknown command: /{cmd}")
            continue

        response, meta = engine.chat(line)
        if response:
            print(f"agent> {response}")
        if args.show_meta:
            print(f"  [retrieved={meta.get('memories_retrieved', 0)}, "
                  f"new_concepts={meta.get('concepts_added', 0)}]")
    return 0


def cmd_sleep(args: argparse.Namespace) -> int:
    """Force a sleep cycle and report what changed."""
    engine = _build_engine(session_id=args.session)
    print(f"forcing {args.mode}-sleep cycle…")
    stats = engine.force_sleep(args.mode) or {}
    print(json.dumps(stats, indent=2, default=str))
    return 0


def cmd_wake_summary(args: argparse.Namespace) -> int:
    """Print the wake-summary for the last N hours."""
    engine = _build_engine(session_id=args.session)
    since = datetime.utcnow() - timedelta(hours=args.hours)
    ws = engine._wake_summary_builder.build(since=since)  # type: ignore[attr-defined]
    _print_wake_summary(ws, as_json=args.json)
    return 0


def _print_wake_summary(ws, as_json: bool = False) -> None:
    """Pretty-print or JSON-dump a WakeSummary object."""
    if as_json:
        payload = {
            "narrative": getattr(ws, "narrative", ""),
            "insights": [str(i) for i in getattr(ws, "insights", []) or []],
            "schemas": [str(s) for s in getattr(ws, "schemas", []) or []],
            "gaps_filled": [str(g) for g in getattr(ws, "gaps_filled", []) or []],
        }
        print(json.dumps(payload, indent=2))
        return

    narrative = getattr(ws, "narrative", "") or "(no narrative produced)"
    print()
    print("─" * 60)
    print(narrative)
    print("─" * 60)
    insights = getattr(ws, "insights", []) or []
    if insights:
        print(f"\ninsights ({len(insights)}):")
        for i in insights[:8]:
            print(f"  • {i}")
    print()


def cmd_status(args: argparse.Namespace) -> int:
    """Show memory stats."""
    engine = _build_engine(session_id=args.session)
    report = engine.get_memory_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Session: {args.session}")
        print(_format_concept_counts(report))
        wm = report.get("working_memory", {})
        if wm:
            print(f"  WM episodes         : {wm.get('episode_count', '?')}")
        sleep = report.get("sleep_cycles", {})
        if sleep:
            print(f"  sleep cycles run    : {sleep.get('total_cycles', 0)}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    engine = _build_engine(session_id=args.session)
    data = engine.export_memory()
    path = Path(args.path)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    engine = _build_engine(session_id=args.session)
    path = Path(args.path)
    with path.open() as f:
        data = json.load(f)
    engine.import_memory(data)
    print(f"loaded {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI service (delegates to uvicorn)."""
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn not installed. run: pip install uvicorn", file=sys.stderr)
        return 1
    print(f"starting SCM server on {args.host}:{args.port}…")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the SCM MCP server (stdio by default; for Claude Desktop / Cursor / any MCP client)."""
    if args.transport == "http":
        os.environ["SCM_MCP_TRANSPORT"] = "http"
        if args.host:
            os.environ["SCM_MCP_HOST"] = args.host
        if args.port:
            os.environ["SCM_MCP_PORT"] = str(args.port)
    if args.idle_seconds is not None:
        os.environ["SCM_IDLE_THRESHOLD_SEC"] = str(args.idle_seconds)
    if args.no_auto_sleep:
        os.environ["SCM_AUTO_SLEEP_DISABLE"] = "1"
    from src.integrations.mcp_server import main as mcp_main
    return mcp_main()


def cmd_version(args: argparse.Namespace) -> int:
    try:
        import importlib.metadata as md
        print(md.version("scm-memory"))
    except Exception:
        # Fallback to reading pyproject.
        try:
            import tomllib
            here = Path(__file__).resolve().parents[2]
            data = tomllib.loads((here / "pyproject.toml").read_text())
            print(data["project"]["version"])
        except Exception:
            print("(unknown)")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Print resolved configuration."""
    keys = [
        "LLM_PROVIDER", "LLM_MODEL", "DEEPSEEK_MODEL", "OPENAI_MODEL",
        "SCM_EMBEDDING_BACKEND", "SCM_EMBEDDING_MODEL", "EMBEDDING_DIM",
        "OLLAMA_BASE_URL", "SCM_DATA_DIR", "SCM_SESSION_ID",
    ]
    print(f"data_dir: {_scm_data_dir()}")
    for k in keys:
        v = os.environ.get(k)
        print(f"  {k}: {v or '(unset)'}")
    return 0


# ---------- argparse plumbing --------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scm",
        description="SCM — memory that works like yours. Wake + sleep phases for AI agents.",
    )
    parser.add_argument("--session", default=os.environ.get("SCM_SESSION_ID", "default"),
                        help="session identifier (default: 'default')")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="interactive REPL")
    p_chat.add_argument("--show-meta", action="store_true", help="print retrieval metadata per turn")
    p_chat.set_defaults(func=cmd_chat)

    p_sleep = sub.add_parser("sleep", help="force a sleep cycle")
    p_sleep.add_argument("--mode", choices=["deep", "micro"], default="deep")
    p_sleep.set_defaults(func=cmd_sleep)

    p_wake = sub.add_parser("wake-summary", help="show what was learned recently")
    p_wake.add_argument("--hours", type=float, default=24.0)
    p_wake.add_argument("--json", action="store_true")
    p_wake.set_defaults(func=cmd_wake_summary)

    p_status = sub.add_parser("status", help="memory stats")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="dump memory state to JSON")
    p_export.add_argument("path")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import", help="load memory state from JSON")
    p_import.add_argument("path")
    p_import.set_defaults(func=cmd_import)

    p_serve = sub.add_parser("serve", help="run the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_mcp = sub.add_parser(
        "mcp",
        help="run the MCP server (Claude Desktop, Cursor, ChatGPT-with-MCP, any MCP client)",
        description=(
            "Start the SCM MCP server, exposing five tools (add_memory, "
            "search_memory, consolidate, wake_summary, forget) to any "
            "MCP-compatible client. Sleep is auto-fired when a user has "
            "been idle past the threshold (default 300s)."
        ),
    )
    p_mcp.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                       help="stdio for Claude Desktop / Cursor; http for everything else")
    p_mcp.add_argument("--host", default=None, help="HTTP host (only with --transport http)")
    p_mcp.add_argument("--port", type=int, default=None, help="HTTP port (only with --transport http)")
    p_mcp.add_argument("--idle-seconds", type=float, default=None,
                       help="auto-sleep idle threshold in seconds (default 300)")
    p_mcp.add_argument("--no-auto-sleep", action="store_true",
                       help="disable autonomous sleep cycles (manual consolidate only)")
    p_mcp.set_defaults(func=cmd_mcp)

    p_version = sub.add_parser("version", help="print version")
    p_version.set_defaults(func=cmd_version)

    p_config = sub.add_parser("config", help="print resolved configuration")
    p_config.set_defaults(func=cmd_config)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
