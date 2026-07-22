"""BrowserCache — DurableObjectStore integration for rendered pages.

Keys by URL + content hash of extraction config.  Avoids re-fetching
static pages across Research Guild harvester runs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from animus.logging import get_logger

logger = get_logger("browser.cache")


def _cache_key(url: str, config_dict: dict[str, Any]) -> str:
    """Deterministic cache key."""
    payload = json.dumps({"url": url, "cfg": config_dict}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class BrowserCache:
    """Wraps DurableObjectStore (or local dict) for page caching."""

    def __init__(self, store: Any | None = None, ttl_sec: int = 3600) -> None:
        self.store = store
        self.ttl_sec = ttl_sec
        self._local: dict[str, tuple[float, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, url: str, config: Any) -> dict[str, Any] | None:
        """Return cached result dict or None."""
        key = _cache_key(url, config.__dict__ if hasattr(config, "__dict__") else dict(config))

        # Try local first (fast path)
        if key in self._local:
            ts, payload = self._local[key]
            if time.monotonic() - ts < self.ttl_sec:
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    del self._local[key]

        # Try DurableObjectStore if available
        if self.store is not None:
            try:
                record = self.store.get(key)
                if record and record.payload:
                    # TODO: wire TTL check through store metadata when supported
                    return record.payload
            except Exception as exc:
                logger.debug("DurableObjectStore cache miss: %s", exc)

        return None

    async def put(self, url: str, config: Any, result: dict[str, Any]) -> None:
        """Store result dict."""
        key = _cache_key(url, config.__dict__ if hasattr(config, "__dict__") else dict(config))
        payload = json.dumps(result, default=str)

        # Always cache locally
        self._local[key] = (time.monotonic(), payload)

        # Persist to DurableObjectStore if available
        if self.store is not None:
            try:
                # ObjectType.SOURCE is closest semantic
                from animus.durability.postgres_store import ObjectRecord, ObjectType

                record = ObjectRecord(
                    object_id=f"browser_cache:{key}",
                    schema_id="browser_result",
                    payload=result,
                )
                self.store.store(record)
            except Exception as exc:
                logger.debug("DurableObjectStore cache write failed: %s", exc)

    async def invalidate(self, url: str, config: Any) -> None:
        """Remove a specific cached entry."""
        key = _cache_key(url, config.__dict__ if hasattr(config, "__dict__") else dict(config))
        self._local.pop(key, None)
