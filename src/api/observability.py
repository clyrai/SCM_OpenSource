"""
Product observability helpers: structured logs + optional Prometheus metrics.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured service logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": _utc_ts(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "event"):
            payload["event"] = getattr(record, "event")
        if hasattr(record, "fields"):
            payload["fields"] = getattr(record, "fields")

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_structured_logger(name: str = "scm.api") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"event": event, "fields": fields})


PROMETHEUS_ENABLED = False
PROMETHEUS_IMPORT_ERROR: str | None = None

HTTP_REQUESTS_TOTAL = None
HTTP_REQUEST_LATENCY_SECONDS = None
CHAT_MESSAGES_TOTAL = None
SLEEP_CYCLES_TOTAL = None
ACTIVE_SESSIONS = None

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

    HTTP_REQUESTS_TOTAL = Counter(
        "scm_http_requests_total",
        "Total HTTP requests handled by SCM API",
        ["method", "route", "status"],
    )
    HTTP_REQUEST_LATENCY_SECONDS = Histogram(
        "scm_http_request_latency_seconds",
        "HTTP request latency for SCM API",
        ["method", "route"],
    )
    CHAT_MESSAGES_TOTAL = Counter(
        "scm_chat_messages_total",
        "Total chat messages processed",
        ["profile", "sandbox"],
    )
    SLEEP_CYCLES_TOTAL = Counter(
        "scm_sleep_cycles_total",
        "Total sleep consolidations triggered",
        ["mode", "origin"],
    )
    ACTIVE_SESSIONS = Gauge(
        "scm_active_sessions",
        "Number of active in-memory chat sessions",
    )

    PROMETHEUS_ENABLED = True
except Exception as exc:  # pragma: no cover - depends on runtime dependency
    PROMETHEUS_IMPORT_ERROR = str(exc)
    generate_latest = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"  # type: ignore[assignment]


def observe_http_request(method: str, route: str, status_code: int, duration_seconds: float) -> None:
    if not PROMETHEUS_ENABLED:
        return
    assert HTTP_REQUESTS_TOTAL is not None
    assert HTTP_REQUEST_LATENCY_SECONDS is not None
    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status=str(status_code)).inc()
    HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, route=route).observe(max(0.0, duration_seconds))


def observe_chat_message(profile: str, sandbox: bool) -> None:
    if not PROMETHEUS_ENABLED or CHAT_MESSAGES_TOTAL is None:
        return
    CHAT_MESSAGES_TOTAL.labels(profile=profile, sandbox="true" if sandbox else "false").inc()


def observe_sleep_cycle(mode: str, origin: str) -> None:
    if not PROMETHEUS_ENABLED or SLEEP_CYCLES_TOTAL is None:
        return
    SLEEP_CYCLES_TOTAL.labels(mode=(mode or "unknown"), origin=(origin or "unknown")).inc()


def set_active_sessions(count: int) -> None:
    if not PROMETHEUS_ENABLED or ACTIVE_SESSIONS is None:
        return
    ACTIVE_SESSIONS.set(max(0, int(count)))


def render_metrics_payload() -> tuple[str, str]:
    if not PROMETHEUS_ENABLED:
        reason = PROMETHEUS_IMPORT_ERROR or "prometheus_client not installed"
        return (
            json.dumps(
                {
                    "enabled": False,
                    "reason": reason,
                    "hint": "Install prometheus_client to enable /metrics scraping.",
                }
            ),
            "application/json",
        )

    assert generate_latest is not None
    assert CONTENT_TYPE_LATEST is not None
    payload = generate_latest()
    if isinstance(payload, bytes):
        return payload.decode("utf-8"), CONTENT_TYPE_LATEST
    return str(payload), CONTENT_TYPE_LATEST

