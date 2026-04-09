"""Proxy rotation for residential IP cycling."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from animus_faucet.config import ProxyConfig

logger = logging.getLogger("animus_faucet.proxy")


@dataclass
class ProxyState:
    """Tracks proxy usage and rotation."""

    current_proxy: str | None = None
    last_rotated_at: float = 0.0
    request_count: int = 0
    ban_count: int = 0
    banned_proxies: set[str] = field(default_factory=set)


class ProxyRotator:
    """Manages proxy rotation for faucet requests.

    Supports two modes:
    1. Bright Data / Oxylabs style: single endpoint with session rotation
       (proxy provider handles IP rotation server-side)
    2. Custom proxy list: round-robin through a list of proxy URLs
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self._state = ProxyState()
        self._proxy_list: list[str] = []
        self._proxy_index = 0

    def set_proxy_list(self, proxies: list[str]) -> None:
        """Set a custom list of proxy URLs for round-robin rotation."""
        self._proxy_list = [p for p in proxies if p not in self._state.banned_proxies]

    def get_proxy(self) -> str | None:
        """Get the current proxy URL, rotating if needed."""
        if not self.config.enabled:
            return None

        now = time.time()
        should_rotate = now - self._state.last_rotated_at > self.config.rotation_interval_seconds

        if self._proxy_list:
            return self._get_from_list(rotate=should_rotate)
        else:
            return self._get_from_provider(rotate=should_rotate)

    def report_ban(self, proxy_url: str) -> None:
        """Report that a proxy was banned by a faucet."""
        self._state.ban_count += 1
        self._state.banned_proxies.add(proxy_url)
        if proxy_url in self._proxy_list:
            self._proxy_list.remove(proxy_url)
        logger.warning(f"Proxy banned: {_redact(proxy_url)} (total bans: {self._state.ban_count})")

    def report_success(self) -> None:
        """Report a successful request through the proxy."""
        self._state.request_count += 1

    @property
    def stats(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "current_proxy": _redact(self._state.current_proxy)
            if self._state.current_proxy
            else None,
            "request_count": self._state.request_count,
            "ban_count": self._state.ban_count,
            "available_proxies": len(self._proxy_list),
        }

    def _get_from_list(self, rotate: bool) -> str | None:
        """Get proxy from custom list using round-robin."""
        if not self._proxy_list:
            return None

        if rotate or self._state.current_proxy is None:
            self._proxy_index = (self._proxy_index + 1) % len(self._proxy_list)
            self._state.current_proxy = self._proxy_list[self._proxy_index]
            self._state.last_rotated_at = time.time()

        return self._state.current_proxy

    def _get_from_provider(self, rotate: bool) -> str | None:
        """Get proxy from provider endpoint with session rotation."""
        base_url = self.config.proxy_url
        if not base_url:
            return None

        if rotate or self._state.current_proxy is None:
            # For Bright Data style: append session ID for rotation
            session_id = random.randint(100000, 999999)
            if "brightdata" in self.config.provider:
                # Bright Data uses -session-{id} in username
                rotated = base_url.replace(
                    self.config.username,
                    f"{self.config.username}-session-{session_id}",
                )
            else:
                rotated = base_url

            self._state.current_proxy = rotated
            self._state.last_rotated_at = time.time()

        return self._state.current_proxy


def _redact(url: str | None) -> str:
    """Redact credentials from proxy URL for logging."""
    if not url:
        return "<none>"
    if "@" in url:
        # http://user:pass@host:port -> http://***@host:port
        scheme_rest = url.split("://", 1)
        if len(scheme_rest) == 2:
            at_split = scheme_rest[1].split("@", 1)
            if len(at_split) == 2:
                return f"{scheme_rest[0]}://***@{at_split[1]}"
    return url
