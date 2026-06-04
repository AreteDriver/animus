"""Path-based bearer-auth middleware for the dashboard.

Protects only the PWA surface (the JSON/WS API) while leaving the local
HTMX dashboard pages and static assets untouched. This keeps the local
browser dashboard frictionless and secures everything the remote phone
touches over the Tailscale tunnel.

The active configuration is read from ``request.app.state.config`` at
dispatch time so that a token generated during ``serve()`` is picked up
without rebuilding the middleware stack.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from animus_bootstrap.config.schema import AnimusConfig
from animus_bootstrap.dashboard.auth import (
    auth_required_for,
    is_local_client,
    verify_bearer,
)

# Path prefixes that form the remote PWA surface and must be protected.
# Everything else (HTMX page routes, /static) is treated as local-only UI.
_PROTECTED_PREFIXES: tuple[str, ...] = ("/api",)


def _is_protected(path: str) -> bool:
    """Return ``True`` if *path* is part of the protected PWA API surface."""
    return any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES)


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a bearer token for the PWA API surface from non-local clients."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[object]],
    ) -> object:
        config: AnimusConfig | None = getattr(request.app.state, "config", None)
        if (
            config is not None
            and _is_protected(request.url.path)
            and auth_required_for(config)
            and not is_local_client(request)
        ):
            header = request.headers.get("Authorization")
            if not verify_bearer(header, config.services.auth_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid bearer token."},
                )
        return await call_next(request)  # type: ignore[return-value]
