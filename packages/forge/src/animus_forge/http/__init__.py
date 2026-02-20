"""HTTP client module with connection pooling.

Provides shared HTTP clients with connection reuse:
- sync_client: requests.Session with connection pooling
- async_client: httpx.AsyncClient with connection pooling

Usage:
    # Sync
    from animus_forge.http import get_sync_client
    client = get_sync_client()
    response = client.get("https://api.example.com/data")

    # Async
    from animus_forge.http import get_async_client
    async with get_async_client() as client:
        response = await client.get("https://api.example.com/data")
"""

from animus_forge.http.client import (
    HTTPClientConfig,
    close_async_client,
    close_sync_client,
    get_async_client,
    get_sync_client,
)

__all__ = [
    "get_sync_client",
    "get_async_client",
    "close_sync_client",
    "close_async_client",
    "HTTPClientConfig",
]
