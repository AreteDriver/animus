"""Health check, metrics, and root endpoints."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Response

from animus_forge import api_state as state
from animus_forge.api_clients.resilience import get_all_provider_stats
from animus_forge.security import get_brute_force_protection
from animus_forge.state import (
    PostgresBackend,
    SQLiteBackend,
    get_database,
    get_migration_status,
)
from animus_forge.utils.circuit_breaker import get_all_circuit_stats

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
def root() -> dict:
    """Root endpoint."""
    return {"app": "AI Workflow Orchestrator", "version": "0.1.0", "status": "running"}


@router.get("/health")
def health_check() -> dict:
    """Basic health check (liveness probe)."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/health/live")
def liveness_check() -> dict:
    """Liveness probe - is the process alive?"""
    return {"status": "alive"}


@router.get("/health/ready")
def readiness_check() -> dict:
    """Readiness probe - is the application ready to serve traffic?"""
    if not state._app_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "reason": "Application not initialized"},
        )

    if state._app_state["shutting_down"]:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "reason": "Application shutting down"},
        )

    return {"status": "ready"}


@router.get("/health/db")
def database_health_check() -> dict:
    """Database health check endpoint."""
    try:
        backend = get_database()
        backend.fetchone("SELECT 1 AS ping")

        if isinstance(backend, PostgresBackend):
            backend_type = "postgresql"
        elif isinstance(backend, SQLiteBackend):
            backend_type = "sqlite"
        else:
            backend_type = "unknown"

        migration_status = get_migration_status(backend)

        return {
            "status": "healthy",
            "database": "connected",
            "backend": backend_type,
            "migrations": migration_status,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "disconnected",
                "timestamp": datetime.now().isoformat(),
            },
        )


@router.get("/health/full")
def full_health_check() -> dict:
    """Comprehensive health check with all subsystem statuses."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": None,
        "application": {
            "ready": state._app_state["ready"],
            "shutting_down": state._app_state["shutting_down"],
            "active_requests": state._app_state["active_requests"],
        },
        "database": None,
        "circuit_breakers": get_all_circuit_stats(),
        "api_clients": get_all_provider_stats(),
        "security": {
            "brute_force": get_brute_force_protection().get_stats(),
        },
    }

    if state._app_state["start_time"]:
        uptime = datetime.now() - state._app_state["start_time"]
        health["uptime_seconds"] = uptime.total_seconds()

    try:
        backend = get_database()
        backend.fetchone("SELECT 1 AS ping")

        if isinstance(backend, PostgresBackend):
            backend_type = "postgresql"
        elif isinstance(backend, SQLiteBackend):
            backend_type = "sqlite"
        else:
            backend_type = "unknown"

        health["database"] = {"status": "connected", "backend": backend_type}
    except Exception as e:
        health["database"] = {"status": "disconnected", "error": str(e)}
        health["status"] = "degraded"

    for name, stats in health["circuit_breakers"].items():
        if stats["state"] == "open":
            health["status"] = "degraded"
            break

    if state._app_state["shutting_down"]:
        health["status"] = "shutting_down"

    return health


@router.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint."""
    from starlette.responses import PlainTextResponse

    from animus_forge.metrics import PrometheusExporter, get_collector

    collector = get_collector()
    exporter = PrometheusExporter(prefix="gorgon")
    metrics_output = exporter.export(collector)

    lines = [metrics_output.rstrip()]

    lines.append("# TYPE gorgon_app_ready gauge")
    lines.append(f"gorgon_app_ready {1 if state._app_state['ready'] else 0}")

    lines.append("# TYPE gorgon_app_shutting_down gauge")
    lines.append(f"gorgon_app_shutting_down {1 if state._app_state['shutting_down'] else 0}")

    lines.append("# TYPE gorgon_active_requests gauge")
    lines.append(f"gorgon_active_requests {state._app_state['active_requests']}")

    if state._app_state["start_time"]:
        uptime = (datetime.now() - state._app_state["start_time"]).total_seconds()
        lines.append("# TYPE gorgon_uptime_seconds counter")
        lines.append(f"gorgon_uptime_seconds {uptime:.2f}")

    for name, stats in get_all_circuit_stats().items():
        safe_name = name.replace("-", "_").replace(".", "_")
        lines.append(f"# TYPE gorgon_circuit_breaker_{safe_name}_state gauge")
        state_value = {"closed": 0, "open": 1, "half_open": 2}.get(stats["state"], -1)
        lines.append(f"gorgon_circuit_breaker_{safe_name}_state {state_value}")

        lines.append(f"# TYPE gorgon_circuit_breaker_{safe_name}_failures gauge")
        lines.append(f"gorgon_circuit_breaker_{safe_name}_failures {stats['failure_count']}")

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


@router.get("/ws/stats", include_in_schema=False)
def websocket_stats() -> dict:
    """Get WebSocket connection statistics (internal use)."""
    if state.ws_manager is None:
        return {"error": "WebSocket not initialized"}
    return state.ws_manager.get_stats()
