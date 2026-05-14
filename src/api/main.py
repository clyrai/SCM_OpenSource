"""
SleepAI FastAPI Main Application
"""
from fastapi import FastAPI
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from time import monotonic

from .memory import router as memory_router, init_memory_components
from ..sleep.api import router as sleep_router
from ..api.chat import router as chat_router
from ..api import chat as chat_module
from ..core.config import (
    API_HOST,
    API_PORT,
    IDLE_LEARNER_ENABLED,
    IDLE_LEARNER_IDLE_THRESHOLD_SECONDS,
    IDLE_LEARNER_MAX_SLEEP_DURATION_SECONDS,
    IDLE_LEARNER_MIN_SLEEP_INTERVAL_SECONDS,
    IDLE_LEARNER_PERSIST_ENABLED,
    IDLE_LEARNER_PERSIST_EVERY_N_TICKS,
    IDLE_LEARNER_SLEEP_MODE,
    IDLE_LEARNER_STATE_PATH,
    IDLE_LEARNER_TICK_INTERVAL_SECONDS,
)
from ..lifecycle import IdleLearner, IdleLearnerConfig
from ..lifecycle.lifecycle_policy import build_default_policy_from_config
from ..lifecycle.state_store import IdleLearnerStateStore
from .observability import (
    get_structured_logger,
    log_event,
    observe_http_request,
    render_metrics_payload,
)


# Process-global so other modules (e.g. the future wake-summary endpoint) can
# import it. Initialized in lifespan.
idle_learner: IdleLearner | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup, gracefully shut down on exit."""
    global idle_learner

    print("Initializing SleepAI memory components...")
    init_memory_components()

    # Phase 7: start the autonomous-learning daemon if enabled. The daemon
    # watches the chat engine pool for inactivity and runs sleep cycles
    # when sessions go idle. This is the foundation of "while the user
    # sleeps, the agent learns."
    if IDLE_LEARNER_ENABLED:
        # M6: build policy + state store from env config (no hardcoded knobs).
        policy = build_default_policy_from_config()
        state_store = (
            IdleLearnerStateStore(IDLE_LEARNER_STATE_PATH)
            if IDLE_LEARNER_PERSIST_ENABLED else None
        )
        idle_learner = IdleLearner(
            engine_provider=lambda: dict(chat_module._chat_engines),
            config=IdleLearnerConfig(
                idle_threshold_seconds=IDLE_LEARNER_IDLE_THRESHOLD_SECONDS,
                min_sleep_interval_seconds=IDLE_LEARNER_MIN_SLEEP_INTERVAL_SECONDS,
                tick_interval_seconds=IDLE_LEARNER_TICK_INTERVAL_SECONDS,
                max_sleep_duration_seconds=IDLE_LEARNER_MAX_SLEEP_DURATION_SECONDS,
                sleep_mode=IDLE_LEARNER_SLEEP_MODE,
                enabled=True,
            ),
            policy=policy,
            state_store=state_store,
            persist_every_n_ticks=IDLE_LEARNER_PERSIST_EVERY_N_TICKS,
        )
        idle_learner.start()
        chat_module._idle_learner = idle_learner
        print(
            f"IdleLearner started: idle_threshold="
            f"{IDLE_LEARNER_IDLE_THRESHOLD_SECONDS}s, mode={IDLE_LEARNER_SLEEP_MODE}, "
            f"policy={type(policy).__name__}, persist={state_store is not None}"
        )
    else:
        print("IdleLearner disabled (set IDLE_LEARNER_ENABLED=true to enable).")

    print("SleepAI ready!")

    yield

    print("SleepAI shutting down...")
    if idle_learner is not None:
        idle_learner.stop()
        chat_module._idle_learner = None
        print("IdleLearner stopped cleanly.")


app = FastAPI(
    title="SleepAI",
    description="Brain-inspired memory system with sleep consolidation",
    version="0.1.0",
    lifespan=lifespan
)
LOGGER = get_structured_logger("scm.api.main")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memory_router)
app.include_router(sleep_router)
app.include_router(chat_router)

# Cloud auth — gates /v1/* behind Bearer API keys when SCM_CLOUD_AUTH=1.
# Off by default to preserve the open-source self-hosted shape.
from ..cloud.auth_middleware import CloudAuthMiddleware
app.add_middleware(CloudAuthMiddleware)

# /v1/memories REST API + tool-definition exports for
# ChatGPT/Claude/Gemini integration. This is the public product API.
from ..integrations.memories_api import router as memories_router
app.include_router(memories_router)

# /v1/cloud/* — account + API-key management endpoints. Only useful in
# cloud mode; harmless when self-hosted (the routes still exist but you
# don't need to call them).
from ..cloud.cloud_api import router as cloud_router
app.include_router(cloud_router)

# Public hosted demo at /demo. Standalone — does NOT reuse the legacy /chat
# router or the older /static/index.html debug UI. Self-contained: serves
# its own HTML, runs its own per-slug ChatEngine pool, fires its own auto-
# sleep sweeper, and uses DeepSeek for the chat reply.
from .demo_router import router as demo_router
app.include_router(demo_router)

# Free community chat at /chat — BYOK LLM, deepagents + SCM under the
# hood. Anyone, no signup, free. Anti-product to the paid SCM Cloud, and
# arguably the better acquisition surface.
# NOTE: aliased as `community_chat_router` because the legacy debug
# /chat router from src/api/chat.py was already imported as `chat_router`
# at the top of this file. Both mount under /chat/* but FastAPI resolves
# the first-registered matching route — the legacy one shadows our new
# routes for any path that overlaps. The new router uses /chat (root),
# /chat/s/{slug}, /chat/api/message/{slug}, /chat/api/history/{slug} —
# none of which the legacy /chat/* router defined, so they coexist.
from .chat_router import router as community_chat_router
app.include_router(community_chat_router)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    start = monotonic()
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    method = request.method
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as exc:
        duration = monotonic() - start
        observe_http_request(method=method, route=path, status_code=500, duration_seconds=duration)
        log_event(
            LOGGER,
            "http_request_error",
            method=method,
            path=path,
            duration_ms=round(duration * 1000.0, 2),
            error=str(exc),
        )
        raise

    duration = monotonic() - start
    observe_http_request(method=method, route=path, status_code=status, duration_seconds=duration)
    log_event(
        LOGGER,
        "http_request",
        method=method,
        path=path,
        status=status,
        duration_ms=round(duration * 1000.0, 2),
    )
    return response

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Public docs + research — so the marketing landing page's links resolve.
# Repo root = three parents up from src/api/main.py.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_docs_dir = os.path.join(_repo_root, "docs")
_research_dir = os.path.join(_repo_root, "research")
if os.path.exists(_docs_dir):
    app.mount("/docs", StaticFiles(directory=_docs_dir), name="docs")
if os.path.exists(_research_dir):
    app.mount("/research", StaticFiles(directory=_research_dir), name="research")


@app.get("/")
async def root():
    """Serve the public marketing landing page.

    landing.html is the long-form site (hero, thesis, how-it-works,
    code, paper link, pricing, footer). The legacy /static/index.html
    debug UI is no longer routed; it stays in static/ for now but
    isn't referenced from anywhere user-facing.
    """
    landing_path = os.path.join(static_dir, "landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path, media_type="text/html")
    return {
        "name": "SCM",
        "version": "0.7.9",
        "status": "running",
        "description": "Memory that works like yours — wake + sleep phases for AI agents",
    }


@app.get("/app")
async def cloud_dashboard():
    """SCM Cloud dashboard — signup, login, key management, BYOK config.

    Single-page UI served from src/api/static/app.html. Public route (no
    auth required) since this IS the login page; all the API calls the
    page makes go to /v1/cloud/* which IS auth-gated when SCM_CLOUD_AUTH=1.
    """
    return FileResponse(
        os.path.join(static_dir, "app.html"),
        media_type="text/html",
    )


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    body, content_type = render_metrics_payload()
    return Response(content=body, media_type=content_type)


def start_server():
    """Start the SleepAI server"""
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False
    )


if __name__ == "__main__":
    start_server()
