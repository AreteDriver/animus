"""HTTP client with connection pooling.

Provides centralized HTTP clients with connection reuse for better performance.
Both sync (requests) and async (httpx) clients are supported.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Global client instances
_sync_client: requests.Session | None = None
_async_client: httpx.AsyncClient | None = None


@dataclass
class HTTPClientConfig:
    """Configuration for HTTP clients.

    Attributes:
        pool_connections: Number of connection pools to cache
        pool_maxsize: Maximum connections per pool
        max_retries: Maximum retries for failed requests
        timeout: Default timeout in seconds
        retry_backoff_factor: Backoff factor for retries
        retry_statuses: HTTP status codes to retry on
    """

    pool_connections: int = 10
    pool_maxsize: int = 20
    max_retries: int = 3
    timeout: float = 30.0
    retry_backoff_factor: float = 0.5
    retry_statuses: tuple = field(default_factory=lambda: (429, 500, 502, 503, 504))
    headers: dict[str, str] = field(default_factory=dict)


# Default config
_default_config = HTTPClientConfig()


def configure_http_client(config: HTTPClientConfig) -> None:
    """Configure the default HTTP client settings.

    Must be called before first client access.
    """
    global _default_config
    _default_config = config


def get_sync_client(config: HTTPClientConfig | None = None) -> requests.Session:
    """Get or create a shared sync HTTP client with connection pooling.

    The client is a singleton that reuses connections across requests.
    Call close_sync_client() to clean up when done.

    Args:
        config: Optional custom configuration

    Returns:
        requests.Session with connection pooling configured
    """
    global _sync_client

    if _sync_client is None:
        cfg = config or _default_config
        _sync_client = _create_sync_client(cfg)
        logger.info(
            f"Created sync HTTP client (pool_connections={cfg.pool_connections}, "
            f"pool_maxsize={cfg.pool_maxsize})"
        )

    return _sync_client


def _create_sync_client(config: HTTPClientConfig) -> requests.Session:
    """Create a new requests Session with connection pooling."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=config.max_retries,
        backoff_factor=config.retry_backoff_factor,
        status_forcelist=list(config.retry_statuses),
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
    )

    # Create adapters with connection pooling
    adapter = HTTPAdapter(
        pool_connections=config.pool_connections,
        pool_maxsize=config.pool_maxsize,
        max_retries=retry_strategy,
    )

    # Mount for both HTTP and HTTPS
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set default headers
    if config.headers:
        session.headers.update(config.headers)

    return session


def close_sync_client() -> None:
    """Close the shared sync HTTP client and release resources."""
    global _sync_client

    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None
        logger.info("Closed sync HTTP client")


@asynccontextmanager
async def get_async_client(
    config: HTTPClientConfig | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    """Get an async HTTP client with connection pooling.

    This is a context manager that provides an httpx.AsyncClient.
    For single requests, prefer using the shared client pattern.

    Args:
        config: Optional custom configuration

    Yields:
        httpx.AsyncClient configured with connection pooling

    Example:
        async with get_async_client() as client:
            response = await client.get("https://api.example.com")
    """
    cfg = config or _default_config

    # Configure connection limits
    limits = httpx.Limits(
        max_keepalive_connections=cfg.pool_connections,
        max_connections=cfg.pool_maxsize,
        keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
    )

    # Configure timeout
    timeout = httpx.Timeout(cfg.timeout, connect=10.0)

    # Create transport with retry capability
    transport = httpx.AsyncHTTPTransport(
        retries=cfg.max_retries,
    )

    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        transport=transport,
        headers=cfg.headers,
    ) as client:
        yield client


async def get_shared_async_client(
    config: HTTPClientConfig | None = None,
) -> httpx.AsyncClient:
    """Get or create a shared async HTTP client.

    Unlike get_async_client(), this returns a long-lived client that
    should be manually closed with close_async_client().

    Args:
        config: Optional custom configuration

    Returns:
        Shared httpx.AsyncClient instance
    """
    global _async_client

    if _async_client is None:
        cfg = config or _default_config
        _async_client = _create_async_client(cfg)
        logger.info(f"Created async HTTP client (max_connections={cfg.pool_maxsize})")

    return _async_client


def _create_async_client(config: HTTPClientConfig) -> httpx.AsyncClient:
    """Create a new httpx AsyncClient with connection pooling."""
    limits = httpx.Limits(
        max_keepalive_connections=config.pool_connections,
        max_connections=config.pool_maxsize,
        keepalive_expiry=30.0,
    )

    timeout = httpx.Timeout(config.timeout, connect=10.0)

    transport = httpx.AsyncHTTPTransport(
        retries=config.max_retries,
    )

    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        transport=transport,
        headers=config.headers,
    )


async def close_async_client() -> None:
    """Close the shared async HTTP client and release resources."""
    global _async_client

    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None
        logger.info("Closed async HTTP client")


class PooledHTTPClient:
    """Context manager for sync HTTP client with automatic cleanup.

    Usage:
        with PooledHTTPClient() as client:
            response = client.get("https://api.example.com")
    """

    def __init__(self, config: HTTPClientConfig | None = None):
        self.config = config or _default_config
        self._session: requests.Session | None = None

    def __enter__(self) -> requests.Session:
        self._session = _create_sync_client(self.config)
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            self._session.close()
            self._session = None


def get_pool_stats() -> dict[str, Any]:
    """Get statistics about connection pool usage.

    Returns:
        Dict with pool statistics for both sync and async clients
    """
    stats = {
        "sync_client_active": _sync_client is not None,
        "async_client_active": _async_client is not None,
    }

    if _sync_client is not None:
        # Get adapter stats
        for prefix in ("http://", "https://"):
            adapter = _sync_client.get_adapter(prefix)
            if hasattr(adapter, "poolmanager"):
                pm = adapter.poolmanager
                stats[f"{prefix}_pools"] = len(pm.pools) if hasattr(pm, "pools") else 0

    return stats
