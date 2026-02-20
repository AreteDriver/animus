"""Audit logging middleware for compliance and security monitoring.

Records structured audit events for:
- Authentication attempts (success/failure)
- Resource mutations (POST/PUT/PATCH/DELETE)
- Admin operations
- Authorization failures
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("gorgon.audit")

# Paths that always generate audit events regardless of method
_ALWAYS_AUDIT_PATHS = frozenset(
    (
        "/v1/auth/",
        "/auth/",
        "/login",
    )
)

# HTTP methods that indicate mutations
_MUTATION_METHODS = frozenset(("POST", "PUT", "PATCH", "DELETE"))

# Paths to skip (health checks, metrics)
_SKIP_PATHS = frozenset(
    (
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    )
)


def _classify_event(method: str, path: str, status_code: int) -> str:
    """Classify the audit event type."""
    path_lower = path.lower()

    if any(path_lower.startswith(p) for p in _ALWAYS_AUDIT_PATHS):
        if status_code == 401 or status_code == 403:
            return "auth.failure"
        return "auth.attempt"

    if status_code == 401:
        return "authz.unauthorized"
    if status_code == 403:
        return "authz.forbidden"
    if status_code == 429:
        return "ratelimit.exceeded"

    if method in _MUTATION_METHODS:
        if status_code < 400:
            return "resource.mutated"
        return "resource.mutation_failed"

    return "request"


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Middleware that emits structured audit log entries.

    Uses a dedicated 'gorgon.audit' logger so audit events can be
    routed to a separate sink (file, SIEM, database) via logging config.
    """

    def __init__(
        self,
        app,
        log_reads: bool = False,
        exclude_paths: set[str] | None = None,
    ) -> None:
        """Initialize audit middleware.

        Args:
            app: ASGI application.
            log_reads: If True, also log GET/HEAD requests (verbose).
            exclude_paths: Additional paths to skip.
        """
        super().__init__(app)
        self.log_reads = log_reads
        self.exclude_paths = (_SKIP_PATHS | exclude_paths) if exclude_paths else _SKIP_PATHS

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        method = request.method

        # Skip non-interesting paths
        if path in self.exclude_paths:
            return await call_next(request)

        # Skip reads unless explicitly enabled
        if not self.log_reads and method in ("GET", "HEAD", "OPTIONS"):
            # Still audit auth paths even for GET
            if not any(path.lower().startswith(p) for p in _ALWAYS_AUDIT_PATHS):
                return await call_next(request)

        start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        client_ip = request.client.host if request.client else "unknown"

        # Extract user identity from auth header if present
        user_id = _extract_user(request)

        try:
            response = await call_next(request)
        except Exception:
            _emit(
                event_type="request.error",
                request_id=request_id,
                method=method,
                path=path,
                client_ip=client_ip,
                user_id=user_id,
                status_code=500,
                duration_ms=_elapsed(start),
            )
            raise

        event_type = _classify_event(method, path, response.status_code)

        _emit(
            event_type=event_type,
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            user_id=user_id,
            status_code=response.status_code,
            duration_ms=_elapsed(start),
        )

        return response


def _extract_user(request: Request) -> str | None:
    """Best-effort extraction of user identity from the request."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Don't log the full token — just indicate it was present
        return "bearer_token"
    if auth_header:
        return "other_auth"
    return None


def _elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _emit(
    *,
    event_type: str,
    request_id: str,
    method: str,
    path: str,
    client_ip: str,
    user_id: str | None,
    status_code: int,
    duration_ms: float,
) -> None:
    """Emit a structured audit log entry."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event_type,
        "request_id": request_id,
        "method": method,
        "path": path,
        "client_ip": client_ip,
        "user_id": user_id,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }

    # Use WARNING for auth failures and errors so they stand out
    if "failure" in event_type or "forbidden" in event_type or "error" in event_type:
        logger.warning(json.dumps(entry))
    else:
        logger.info(json.dumps(entry))
