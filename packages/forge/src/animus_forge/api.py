"""FastAPI backend for AI Workflow Orchestrator."""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from animus_forge import api_state as state
from animus_forge.api_errors import (
    APIException,
    RateLimitErrorResponse,
    api_exception_handler,
    gorgon_exception_handler,
)
from animus_forge.config import configure_logging, get_settings
from animus_forge.errors import GorgonError
from animus_forge.security import (
    AuditLogMiddleware,
    BruteForceConfig,
    BruteForceMiddleware,
    RequestLimitConfig,
    RequestSizeLimitMiddleware,
    get_brute_force_protection,
)
from animus_forge.state import get_database, run_migrations
from animus_forge.tracing.middleware import TracingMiddleware
from animus_forge.utils.circuit_breaker import reset_all_circuits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def _handle_shutdown_signal(signum, frame) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    state._app_state["shutting_down"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage component lifecycles with graceful shutdown."""
    # Reset application state at startup
    state._app_state["ready"] = False
    state._app_state["shutting_down"] = False
    state._app_state["active_requests"] = 0
    state._app_state["start_time"] = datetime.now()

    # Configure logging
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        format=settings.log_format,
        sanitize_logs=settings.sanitize_logs,
    )

    # Register signal handlers for graceful shutdown (only in main thread)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_shutdown_signal)
        signal.signal(signal.SIGINT, _handle_shutdown_signal)

    # Get shared database backend
    backend = get_database()
    state._app_state["backend"] = backend  # Store for supervisor factory

    # Run migrations
    logger.info("Running database migrations...")
    try:
        applied = run_migrations(backend)
        if applied:
            logger.info(f"Applied migrations: {', '.join(applied)}")
        else:
            logger.info("Database schema is up to date")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

    # Initialize managers with shared backend
    from animus_forge.budget import PersistentBudgetManager
    from animus_forge.executions import ExecutionManager
    from animus_forge.jobs import JobManager
    from animus_forge.mcp import MCPConnectorManager
    from animus_forge.scheduler import ScheduleManager
    from animus_forge.settings import SettingsManager
    from animus_forge.webhooks import WebhookManager
    from animus_forge.webhooks.webhook_delivery import WebhookDeliveryManager
    from animus_forge.websocket import Broadcaster, ConnectionManager
    from animus_forge.workflow import WorkflowVersionManager

    state.execution_manager = ExecutionManager(backend=backend)
    state.schedule_manager = ScheduleManager(backend=backend)
    state.webhook_manager = WebhookManager(backend=backend)
    state.delivery_manager = WebhookDeliveryManager(backend=backend)
    state.job_manager = JobManager(backend=backend, execution_manager=state.execution_manager)
    state.version_manager = WorkflowVersionManager(backend=backend)
    state.mcp_manager = MCPConnectorManager(backend=backend)
    state.settings_manager = SettingsManager(backend=backend)
    state.budget_manager = PersistentBudgetManager(backend=backend)

    from animus_forge.db import TaskStore

    state.task_store = TaskStore(backend=backend)

    # Initialize WebSocket components
    state.ws_manager = ConnectionManager()
    state.ws_broadcaster = Broadcaster(state.ws_manager)

    # Start broadcaster with current event loop
    loop = asyncio.get_running_loop()
    state.ws_broadcaster.start(loop)

    # Register broadcaster callback with execution manager
    state.execution_manager.register_callback(state.ws_broadcaster.create_execution_callback())

    logger.info("WebSocket components initialized")

    # Create coordination bridge once (persistent across supervisor instances)
    try:
        from animus_forge.agents.convergence import create_bridge

        _coordination_bridge = create_bridge()
    except Exception:
        _coordination_bridge = None
    state.coordination_bridge = _coordination_bridge

    # Create coordination event log
    try:
        from animus_forge.agents.convergence import create_event_log

        state.coordination_event_log = create_event_log()
    except Exception:
        state.coordination_event_log = None

    from animus_forge.agents import SupervisorAgent, create_agent_provider

    def create_supervisor(backend=None):
        """Factory function to create Supervisor agent."""
        try:
            provider = create_agent_provider("anthropic")
            try:
                from animus_forge.agents.convergence import create_checker

                checker = create_checker()
            except Exception:
                checker = None
            return SupervisorAgent(
                provider,
                backend=backend or state._app_state.get("backend"),
                convergence_checker=checker,
                coordination_bridge=_coordination_bridge,
                budget_manager=state.budget_manager,
                event_log=state.coordination_event_log,
            )
        except Exception as e:
            logger.warning(f"Could not create supervisor: {e}")
            return None

    state.supervisor_factory = create_supervisor
    logger.info("Supervisor factory initialized")

    # Migrate existing workflows (one-time)
    try:
        workflows_dir = settings.workflows_dir
        migrated = state.version_manager.migrate_existing_workflows(workflows_dir)
        if migrated:
            logger.info(f"Migrated {len(migrated)} existing workflows to version control")
    except Exception as e:
        logger.warning(f"Workflow migration skipped: {e}")

    state.schedule_manager.start()

    # Mark application as ready
    state._app_state["ready"] = True
    logger.info("Application startup complete - ready to serve requests")

    yield

    # Begin graceful shutdown
    logger.info("Beginning graceful shutdown...")
    state._app_state["shutting_down"] = True
    state._app_state["ready"] = False

    # Wait for active requests to complete (with timeout)
    shutdown_timeout = 30  # seconds
    start = time.monotonic()
    while state._app_state["active_requests"] > 0:
        if time.monotonic() - start > shutdown_timeout:
            logger.warning(
                f"Shutdown timeout reached with {state._app_state['active_requests']} "
                "active requests still running"
            )
            break
        await asyncio.sleep(0.1)

    # Shutdown managers
    state.schedule_manager.shutdown()
    state.job_manager.shutdown()

    # Shutdown WebSocket broadcaster
    if state.ws_broadcaster:
        await state.ws_broadcaster.stop()

    # Close coordination bridge
    if _coordination_bridge is not None:
        try:
            _coordination_bridge.close()
        except Exception as e:
            logger.warning("Failed to close coordination bridge: %s", e)

    # Close coordination event log
    if state.coordination_event_log is not None:
        try:
            state.coordination_event_log.close()
        except Exception as e:
            logger.warning("Failed to close coordination event log: %s", e)

    # Reset circuit breakers
    reset_all_circuits()

    logger.info("Graceful shutdown complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Workflow Orchestrator", version="0.1.0", lifespan=lifespan)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests with timing and request IDs.

    Also tracks active requests for graceful shutdown and rejects new
    requests during shutdown.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Reject requests during shutdown (except health checks)
        path = request.url.path
        if state._app_state["shutting_down"] and not path.startswith("/health"):
            return JSONResponse(
                status_code=503,
                content={"detail": "Service shutting down"},
                headers={"Retry-After": "30"},
            )

        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        method = request.method

        # Track active requests for graceful shutdown
        await state.increment_active_requests()

        # Log request start with structured fields
        start_time = time.perf_counter()
        logger.info(
            f"[{request_id}] {method} {path} - client={client_ip}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
            },
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error and re-raise
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] {method} {path} - 500 ERROR in {duration_ms:.1f}ms - {e}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "client_ip": client_ip,
                    "status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        finally:
            await state.decrement_active_requests()

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log response with structured fields
        status_code = response.status_code
        log_level = logging.WARNING if status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            f"[{request_id}] {method} {path} - {status_code} in {duration_ms:.1f}ms",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        # Add request ID to response headers for tracing
        response.headers["X-Request-ID"] = request_id

        return response


# Register middleware (order matters: last added runs first on request)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AuditLogMiddleware)

_tracing_settings = get_settings()
if _tracing_settings.tracing_enabled:
    app.add_middleware(
        TracingMiddleware,
        service_name=_tracing_settings.tracing_service_name,
        exclude_paths=["/health", "/health/live", "/health/ready", "/metrics"],
    )

_settings = get_settings()
brute_force_config = BruteForceConfig(
    max_attempts_per_minute=_settings.brute_force_max_attempts_per_minute,
    max_attempts_per_hour=_settings.brute_force_max_attempts_per_hour,
    max_auth_attempts_per_minute=_settings.brute_force_max_auth_attempts_per_minute,
    max_auth_attempts_per_hour=_settings.brute_force_max_auth_attempts_per_hour,
    initial_block_seconds=_settings.brute_force_initial_block_seconds,
    max_block_seconds=_settings.brute_force_max_block_seconds,
    auth_paths=("/v1/auth/", "/auth/", "/login"),
)
app.add_middleware(BruteForceMiddleware, protection=get_brute_force_protection(brute_force_config))

request_limit_config = RequestLimitConfig(
    max_body_size=_settings.request_max_body_size,
    max_json_size=_settings.request_max_json_size,
    max_form_size=_settings.request_max_form_size,
    large_upload_paths=(),
)
app.add_middleware(RequestSizeLimitMiddleware, config=request_limit_config)

# Register rate limiter
app.state.limiter = state.limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors with structured response."""
    request_id = request.headers.get("X-Request-ID")
    response = RateLimitErrorResponse(
        error={
            "error_code": "RATE_LIMITED",
            "message": "Rate limit exceeded",
            "details": {"limit": str(exc.detail)},
            "request_id": request_id,
        },
        retry_after=int(exc.detail) if str(exc.detail).isdigit() else 60,
    )
    return JSONResponse(
        status_code=429,
        content=response.model_dump(),
        headers={"Retry-After": str(exc.detail)},
    )


app.add_exception_handler(GorgonError, gorgon_exception_handler)
app.add_exception_handler(APIException, api_exception_handler)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from animus_forge.api_routes import (  # noqa: E402
    auth,
    budgets,
    coordination,
    dashboard,
    executions,
    graph,
    health,
    history,
    jobs,
    mcp,
    prompts,
    schedules,
    settings,
    webhooks,
    websocket,
    workflows,
)

v1_router = APIRouter(prefix="/v1", tags=["v1"])
v1_router.include_router(auth.router)
v1_router.include_router(workflows.router)
v1_router.include_router(executions.router)
v1_router.include_router(schedules.router)
v1_router.include_router(webhooks.router)
v1_router.include_router(mcp.router)
v1_router.include_router(jobs.router)
v1_router.include_router(prompts.router)
v1_router.include_router(settings.router)
v1_router.include_router(budgets.router)
v1_router.include_router(dashboard.router)
v1_router.include_router(history.router)
v1_router.include_router(graph.router)
v1_router.include_router(coordination.router)

app.include_router(v1_router)
app.include_router(health.router)
app.include_router(webhooks.trigger_router)
app.include_router(websocket.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
