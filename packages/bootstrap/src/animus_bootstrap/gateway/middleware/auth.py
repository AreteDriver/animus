"""Authentication middleware — allowlist-based sender authorization."""

from __future__ import annotations

import logging

from animus_bootstrap.gateway.models import GatewayMessage

logger = logging.getLogger(__name__)


class GatewayAuthMiddleware:
    """Authorize inbound messages against a channel+sender allowlist.

    When the allowlist is empty the gateway operates in **open mode** and
    permits all messages.  Once at least one entry is added the gateway
    switches to **restricted mode** where only explicitly allowed
    (channel, sender_id) pairs are accepted.
    """

    def __init__(self) -> None:
        self._allowed: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Allowlist management
    # ------------------------------------------------------------------

    def add_allowed(self, channel: str, sender_id: str) -> None:
        """Permit *sender_id* on *channel* through the gateway."""
        self._allowed.add((channel, sender_id))
        logger.info("auth: added (%s, %s) to allowlist", channel, sender_id)

    def remove_allowed(self, channel: str, sender_id: str) -> None:
        """Revoke permission for *sender_id* on *channel*."""
        self._allowed.discard((channel, sender_id))
        logger.info("auth: removed (%s, %s) from allowlist", channel, sender_id)

    # ------------------------------------------------------------------
    # Authorization check
    # ------------------------------------------------------------------

    def is_allowed(self, message: GatewayMessage) -> bool:
        """Return ``True`` if *message* is authorized to pass through.

        In open mode (empty allowlist) every message is accepted.
        In restricted mode only messages whose ``(channel, sender_id)``
        pair is in the allowlist are accepted.
        """
        if not self._allowed:
            return True
        return (message.channel, message.sender_id) in self._allowed

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Return ``"open"`` when the allowlist is empty, else ``"restricted"``."""
        return "open" if not self._allowed else "restricted"

    @property
    def allowlist(self) -> set[tuple[str, str]]:
        """Return a copy of the current allowlist."""
        return set(self._allowed)
