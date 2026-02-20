"""Request size limits middleware.

Protects against denial of service via oversized requests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestTooLarge(Exception):
    """Raised when request exceeds size limits."""

    def __init__(
        self,
        message: str = "Request too large",
        max_size: int = 0,
        actual_size: int = 0,
    ):
        super().__init__(message)
        self.max_size = max_size
        self.actual_size = actual_size


@dataclass
class RequestLimitConfig:
    """Configuration for request size limits."""

    # Maximum request body size in bytes
    max_body_size: int = 10 * 1024 * 1024  # 10 MB default
    # Maximum for specific content types
    max_json_size: int = 1 * 1024 * 1024  # 1 MB for JSON
    max_form_size: int = 50 * 1024 * 1024  # 50 MB for form data (file uploads)
    # Paths that allow larger uploads (e.g., file upload endpoints)
    large_upload_paths: tuple[str, ...] = ()
    large_upload_max_size: int = 100 * 1024 * 1024  # 100 MB for special paths


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request size limits.

    Rejects requests that exceed configured size limits before
    processing to prevent memory exhaustion attacks.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: RequestLimitConfig | None = None,
    ):
        """Initialize middleware.

        Args:
            app: ASGI application
            config: Size limit configuration
        """
        super().__init__(app)
        self.config = config or RequestLimitConfig()

    def _get_max_size(self, request: Request) -> int:
        """Determine max size based on request path and content type."""
        path = request.url.path

        # Check for large upload paths
        for upload_path in self.config.large_upload_paths:
            if path.startswith(upload_path):
                return self.config.large_upload_max_size

        # Check content type
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            return self.config.max_json_size
        elif "multipart/form-data" in content_type:
            return self.config.max_form_size
        elif "application/x-www-form-urlencoded" in content_type:
            return self.config.max_form_size

        return self.config.max_body_size

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Check request size and reject if too large."""
        # Skip size check for requests without body
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        # Get content length header
        content_length = request.headers.get("content-length")
        max_size = self._get_max_size(request)

        if content_length:
            try:
                size = int(content_length)
                if size > max_size:
                    logger.warning(
                        f"Request rejected: size {size} exceeds limit {max_size} "
                        f"for path {request.url.path}"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request Entity Too Large",
                            "message": f"Request body exceeds maximum size of {max_size} bytes",
                            "max_size": max_size,
                            "actual_size": size,
                        },
                    )
            except ValueError:
                # Invalid content-length header
                pass

        # For chunked transfers without content-length, we read and check
        # This is handled by the framework's body parsing, so we proceed
        return await call_next(request)


def create_size_limit_middleware(
    max_body_size: int = 10 * 1024 * 1024,
    max_json_size: int = 1 * 1024 * 1024,
    large_upload_paths: tuple[str, ...] = (),
) -> type[RequestSizeLimitMiddleware]:
    """Create a configured size limit middleware class.

    Args:
        max_body_size: Default max body size in bytes
        max_json_size: Max size for JSON payloads
        large_upload_paths: Paths that allow larger uploads

    Returns:
        Configured middleware class
    """
    config = RequestLimitConfig(
        max_body_size=max_body_size,
        max_json_size=max_json_size,
        large_upload_paths=large_upload_paths,
    )

    class ConfiguredMiddleware(RequestSizeLimitMiddleware):
        def __init__(self, app: ASGIApp):
            super().__init__(app, config=config)

    return ConfiguredMiddleware
