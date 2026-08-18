"""Webhooks module for event-driven workflow execution."""

from animus_forge.webhooks.webhook_manager import (
    PayloadMapping,
    Webhook,
    WebhookManager,
    WebhookStatus,
    WebhookTriggerLog,
)

_DELIVERY_EXPORTS = {
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "DeliveryStatus",
    "RetryStrategy",
    "WebhookDelivery",
    "WebhookDeliveryManager",
}

__all__ = [
    "WebhookManager",
    "Webhook",
    "WebhookStatus",
    "PayloadMapping",
    "WebhookTriggerLog",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "WebhookDeliveryManager",
    "WebhookDelivery",
    "DeliveryStatus",
    "RetryStrategy",
]


def __getattr__(name: str):
    if name in _DELIVERY_EXPORTS:
        from animus_forge.webhooks import webhook_delivery

        value = getattr(webhook_delivery, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
