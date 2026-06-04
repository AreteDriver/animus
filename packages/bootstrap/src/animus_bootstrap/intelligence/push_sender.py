"""Web Push (VAPID) key generation and notification delivery.

VAPID keys are generated with ``cryptography`` (always available). Actual
delivery uses ``pywebpush``, imported lazily so this module — and the rest of
the dashboard — load even when that optional dependency is absent.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_bootstrap.config.manager import ConfigManager
    from animus_bootstrap.config.schema import AnimusConfig
    from animus_bootstrap.intelligence.push_store import PushSubscriptionStore

logger = logging.getLogger(__name__)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_vapid_keys() -> tuple[str, str]:
    """Generate a VAPID keypair.

    Returns:
        ``(private_pem, public_b64url)`` — the PKCS8 PEM private key used to
        sign push requests and the base64url uncompressed public key passed to
        the browser as the ``applicationServerKey``.
    """
    # Imported lazily so the dashboard loads even where cryptography is absent
    # or its native bindings are unavailable; push is an optional feature.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    raw_public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return private_pem, _b64url(raw_public)


def ensure_vapid_keys(config: AnimusConfig, manager: ConfigManager) -> tuple[str, str]:
    """Return the configured VAPID keypair, generating + persisting if absent."""
    svc = config.services
    if svc.vapid_public_key and svc.vapid_private_key:
        return svc.vapid_private_key, svc.vapid_public_key

    private_pem, public_b64 = generate_vapid_keys()
    svc.vapid_private_key = private_pem
    svc.vapid_public_key = public_b64
    manager.save(config)
    logger.info("Generated VAPID keypair for Web Push")
    return private_pem, public_b64


class PushSender:
    """Send Web Push notifications to all stored subscriptions."""

    def __init__(
        self,
        store: PushSubscriptionStore,
        private_key: str,
        subject: str,
    ) -> None:
        self._store = store
        self._private_key = private_key
        self._subject = subject

    def send(self, title: str, body: str, url: str = "/") -> int:
        """Push a notification to every subscription.

        Stale subscriptions (HTTP 404/410) are pruned. Returns the number of
        successful deliveries. No-ops with a warning if ``pywebpush`` is not
        installed.
        """
        try:
            from pywebpush import WebPushException, webpush
        except ImportError:
            logger.warning("pywebpush not installed — skipping push notification")
            return 0

        payload = json.dumps({"title": title, "body": body, "url": url})
        claims = {"sub": self._subject}
        sent = 0
        for subscription in self._store.all():
            try:
                webpush(
                    subscription_info=subscription,
                    data=payload,
                    vapid_private_key=self._private_key,
                    vapid_claims=dict(claims),
                )
                sent += 1
            except WebPushException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (404, 410):
                    endpoint = subscription.get("endpoint", "")
                    self._store.remove(endpoint)
                    logger.info("Pruned expired push subscription")
                else:
                    logger.warning("Push delivery failed: %s", exc)
        return sent
