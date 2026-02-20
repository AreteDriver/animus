"""Webhook delivery with retries, circuit breaker, and dead-letter queue.

Provides reliable outbound webhook delivery with:
- Configurable retry strategy with exponential backoff
- Per-URL circuit breaker (closed -> open -> half_open -> closed)
- Dead-letter queue for failed deliveries with batch reprocessing
- Async and sync delivery options
- Delivery status tracking and history
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import httpx
import requests
from pydantic import BaseModel, Field

from animus_forge.http import get_async_client, get_sync_client
from animus_forge.state import DatabaseBackend, get_database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreakerConfig:
    """Configuration for per-URL circuit breaker."""

    failure_threshold: int = 5  # Consecutive failures to trip
    recovery_timeout: float = 300.0  # Seconds before half-open attempt


@dataclass
class CircuitBreakerState:
    """Mutable state for a single circuit breaker instance."""

    failures: int = 0
    state: str = "closed"  # closed, open, half_open
    last_failure_at: float | None = None


class CircuitBreaker:
    """Per-URL circuit breaker that protects downstream endpoints.

    State machine:
        closed  --[threshold failures]--> open
        open    --[recovery timeout]----> half_open
        half_open --[success]-----------> closed
        half_open --[failure]-----------> open
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._states: dict[str, CircuitBreakerState] = {}

    def _get_state(self, url: str) -> CircuitBreakerState:
        """Get or create state for a URL."""
        if url not in self._states:
            self._states[url] = CircuitBreakerState()
        return self._states[url]

    def allow_request(self, url: str) -> bool:
        """Return True if the circuit allows a request to *url*."""
        cb = self._get_state(url)

        if cb.state == "closed":
            return True

        if cb.state == "open":
            # Check if recovery timeout has elapsed
            if (
                cb.last_failure_at is not None
                and (time.monotonic() - cb.last_failure_at) >= self.config.recovery_timeout
            ):
                cb.state = "half_open"
                logger.info(f"Circuit breaker half-open for {url}")
                return True
            return False

        # half_open — allow exactly one probe request
        return True

    def record_success(self, url: str) -> None:
        """Record a successful delivery, resetting the breaker."""
        cb = self._get_state(url)
        cb.failures = 0
        cb.state = "closed"
        cb.last_failure_at = None

    def record_failure(self, url: str) -> None:
        """Record a failed delivery, potentially tripping the breaker."""
        cb = self._get_state(url)
        cb.failures += 1
        cb.last_failure_at = time.monotonic()

        if cb.state == "half_open":
            # Half-open probe failed — reopen
            cb.state = "open"
            logger.warning(f"Circuit breaker re-opened for {url}")
        elif cb.failures >= self.config.failure_threshold:
            cb.state = "open"
            logger.warning(f"Circuit breaker opened for {url} after {cb.failures} failures")

    def get_state(self, url: str) -> str:
        """Return the current breaker state for *url*."""
        return self._get_state(url).state

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Return snapshot of all tracked URLs and their breaker state."""
        return {
            url: {
                "state": cb.state,
                "failures": cb.failures,
                "last_failure_at": cb.last_failure_at,
            }
            for url, cb in self._states.items()
        }

    def reset(self, url: str) -> None:
        """Manually reset a circuit breaker to closed."""
        if url in self._states:
            self._states[url] = CircuitBreakerState()


# ---------------------------------------------------------------------------
# Delivery models
# ---------------------------------------------------------------------------


class DeliveryStatus(str, Enum):
    """Status of a webhook delivery attempt."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"
    CIRCUIT_BROKEN = "circuit_broken"


class WebhookDelivery(BaseModel):
    """A webhook delivery record."""

    id: int | None = None
    webhook_url: str
    payload: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempt_count: int = 0
    max_retries: int = 3
    last_error: str | None = None
    last_attempt_at: datetime | None = None
    next_retry_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    response_status: int | None = None
    response_body: str | None = None


class RetryStrategy:
    """Configurable retry strategy with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry strategy.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Add randomization to prevent thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random

            delay *= 0.5 + random.random()  # 50-150% of calculated delay

        return delay


class WebhookDeliveryManager:
    """Manages webhook delivery with retries and dead-letter queue."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS webhook_deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            webhook_url TEXT NOT NULL,
            payload TEXT NOT NULL,
            headers TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            last_error TEXT,
            last_attempt_at TIMESTAMP,
            next_retry_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            response_status INTEGER,
            response_body TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_deliveries_status
        ON webhook_deliveries(status);

        CREATE INDEX IF NOT EXISTS idx_deliveries_next_retry
        ON webhook_deliveries(next_retry_at)
        WHERE status = 'retrying';

        CREATE TABLE IF NOT EXISTS webhook_dead_letter (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            delivery_id INTEGER NOT NULL,
            webhook_url TEXT NOT NULL,
            payload TEXT NOT NULL,
            headers TEXT,
            error TEXT,
            attempt_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reprocessed_at TIMESTAMP,
            FOREIGN KEY (delivery_id) REFERENCES webhook_deliveries(id)
        );

        CREATE INDEX IF NOT EXISTS idx_dlq_reprocessed
        ON webhook_dead_letter(reprocessed_at)
        WHERE reprocessed_at IS NULL;
    """

    def __init__(
        self,
        backend: DatabaseBackend | None = None,
        retry_strategy: RetryStrategy | None = None,
        timeout: float = 10.0,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        """Initialize delivery manager.

        Args:
            backend: Database backend (defaults to global)
            retry_strategy: Custom retry strategy
            timeout: Request timeout in seconds
            circuit_breaker: Per-URL circuit breaker (created with defaults if None)
        """
        self.backend = backend or get_database()
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.backend.executescript(self.SCHEMA)

    def _generate_signature(self, payload: bytes, secret: str, algorithm: str = "sha256") -> str:
        """Generate HMAC signature for payload."""
        return (
            f"{algorithm}="
            + hmac.new(secret.encode(), payload, getattr(hashlib, algorithm)).hexdigest()
        )

    def deliver(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        secret: str | None = None,
        max_retries: int | None = None,
    ) -> WebhookDelivery:
        """Deliver a webhook synchronously with retries.

        Args:
            url: Webhook URL
            payload: JSON payload
            headers: Custom headers
            secret: Secret for HMAC signature
            max_retries: Override default max retries

        Returns:
            WebhookDelivery with final status
        """
        headers = headers or {}
        max_retries = max_retries if max_retries is not None else self.retry_strategy.max_retries

        delivery = WebhookDelivery(
            webhook_url=url,
            payload=payload,
            headers=headers,
            max_retries=max_retries,
        )
        delivery = self._save_delivery(delivery)

        # Circuit breaker check — skip delivery if circuit is open
        if not self.circuit_breaker.allow_request(url):
            delivery.status = DeliveryStatus.CIRCUIT_BROKEN
            delivery.last_error = "Circuit breaker open"
            delivery.completed_at = datetime.now()
            self._save_delivery(delivery)
            self._add_to_dlq(delivery)
            logger.warning(f"Circuit breaker open, skipping delivery to DLQ: {url}")
            return delivery

        # Add signature if secret provided
        payload_bytes = json.dumps(payload).encode("utf-8")
        if secret:
            headers["X-Webhook-Signature"] = self._generate_signature(payload_bytes, secret)

        headers["Content-Type"] = "application/json"

        # Use pooled HTTP client for connection reuse
        client = get_sync_client()

        while delivery.attempt_count <= max_retries:
            delivery.attempt_count += 1
            delivery.last_attempt_at = datetime.now()
            delivery.status = (
                DeliveryStatus.RETRYING if delivery.attempt_count > 1 else DeliveryStatus.PENDING
            )

            try:
                response = client.post(
                    url,
                    data=payload_bytes,
                    headers=headers,
                    timeout=self.timeout,
                )

                delivery.response_status = response.status_code
                delivery.response_body = response.text[:1000]

                if response.ok:
                    delivery.status = DeliveryStatus.SUCCESS
                    delivery.completed_at = datetime.now()
                    self._save_delivery(delivery)
                    self.circuit_breaker.record_success(url)
                    logger.info(f"Webhook delivered successfully: {url}")
                    return delivery

                # Non-2xx response
                delivery.last_error = f"HTTP {response.status_code}"

            except requests.exceptions.Timeout:
                delivery.last_error = "Request timeout"
                logger.warning(f"Webhook delivery timeout (attempt {delivery.attempt_count})")

            except requests.exceptions.ConnectionError as e:
                delivery.last_error = f"Connection error: {e}"
                logger.warning(
                    f"Webhook delivery failed (attempt {delivery.attempt_count}): {delivery.last_error}"
                )

            except requests.exceptions.RequestException as e:
                delivery.last_error = str(e)
                logger.warning(
                    f"Webhook delivery failed (attempt {delivery.attempt_count}): {delivery.last_error}"
                )

            # Check if we should retry
            if delivery.attempt_count < max_retries:
                delay = self.retry_strategy.get_delay(delivery.attempt_count)
                delivery.next_retry_at = datetime.now() + timedelta(seconds=delay)
                self._save_delivery(delivery)
                time.sleep(delay)
            else:
                break

        # Max retries exceeded - record failure in circuit breaker, move to DLQ
        self.circuit_breaker.record_failure(url)
        delivery.status = DeliveryStatus.DEAD_LETTER
        delivery.completed_at = datetime.now()
        self._save_delivery(delivery)
        self._add_to_dlq(delivery)
        logger.error(f"Webhook delivery failed after {max_retries} retries, moved to DLQ: {url}")

        return delivery

    async def deliver_async(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        secret: str | None = None,
        max_retries: int | None = None,
    ) -> WebhookDelivery:
        """Deliver a webhook asynchronously with retries.

        Args:
            url: Webhook URL
            payload: JSON payload
            headers: Custom headers
            secret: Secret for HMAC signature
            max_retries: Override default max retries

        Returns:
            WebhookDelivery with final status
        """
        headers = headers or {}
        max_retries = max_retries if max_retries is not None else self.retry_strategy.max_retries

        delivery = WebhookDelivery(
            webhook_url=url,
            payload=payload,
            headers=headers,
            max_retries=max_retries,
        )
        delivery = self._save_delivery(delivery)

        # Circuit breaker check
        if not self.circuit_breaker.allow_request(url):
            delivery.status = DeliveryStatus.CIRCUIT_BROKEN
            delivery.last_error = "Circuit breaker open"
            delivery.completed_at = datetime.now()
            self._save_delivery(delivery)
            self._add_to_dlq(delivery)
            logger.warning(f"Circuit breaker open, skipping async delivery to DLQ: {url}")
            return delivery

        # Add signature if secret provided
        payload_bytes = json.dumps(payload).encode("utf-8")
        if secret:
            headers["X-Webhook-Signature"] = self._generate_signature(payload_bytes, secret)

        headers["Content-Type"] = "application/json"

        # Use pooled async HTTP client for connection reuse
        async with get_async_client() as client:
            while delivery.attempt_count <= max_retries:
                delivery.attempt_count += 1
                delivery.last_attempt_at = datetime.now()
                delivery.status = (
                    DeliveryStatus.RETRYING
                    if delivery.attempt_count > 1
                    else DeliveryStatus.PENDING
                )

                try:
                    response = await client.post(url, content=payload_bytes, headers=headers)
                    delivery.response_status = response.status_code
                    delivery.response_body = response.text[:1000]

                    if response.is_success:
                        delivery.status = DeliveryStatus.SUCCESS
                        delivery.completed_at = datetime.now()
                        self._save_delivery(delivery)
                        self.circuit_breaker.record_success(url)
                        logger.info(f"Webhook delivered successfully (async): {url}")
                        return delivery

                    delivery.last_error = f"HTTP {response.status_code}"

                except httpx.TimeoutException:
                    delivery.last_error = "Request timeout"
                    logger.warning(f"Webhook delivery timeout (attempt {delivery.attempt_count})")

                except httpx.RequestError as e:
                    delivery.last_error = str(e)
                    logger.warning(
                        f"Webhook delivery failed (attempt {delivery.attempt_count}): {delivery.last_error}"
                    )

                except Exception as e:
                    delivery.last_error = str(e)
                    logger.warning(
                        f"Webhook delivery failed (attempt {delivery.attempt_count}): {delivery.last_error}"
                    )

                # Check if we should retry
                if delivery.attempt_count < max_retries:
                    delay = self.retry_strategy.get_delay(delivery.attempt_count)
                    delivery.next_retry_at = datetime.now() + timedelta(seconds=delay)
                    self._save_delivery(delivery)
                    await asyncio.sleep(delay)
                else:
                    break

        # Max retries exceeded
        self.circuit_breaker.record_failure(url)
        delivery.status = DeliveryStatus.DEAD_LETTER
        delivery.completed_at = datetime.now()
        self._save_delivery(delivery)
        self._add_to_dlq(delivery)
        logger.error(f"Webhook delivery failed after {max_retries} retries, moved to DLQ: {url}")

        return delivery

    def _save_delivery(self, delivery: WebhookDelivery) -> WebhookDelivery:
        """Save delivery to database."""
        try:
            if delivery.id is None:
                # Insert new
                with self.backend.transaction():
                    self.backend.execute(
                        """
                        INSERT INTO webhook_deliveries
                        (webhook_url, payload, headers, status, attempt_count,
                         max_retries, last_error, last_attempt_at, next_retry_at,
                         created_at, completed_at, response_status, response_body)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            delivery.webhook_url,
                            json.dumps(delivery.payload),
                            json.dumps(delivery.headers),
                            delivery.status.value,
                            delivery.attempt_count,
                            delivery.max_retries,
                            delivery.last_error,
                            delivery.last_attempt_at.isoformat()
                            if delivery.last_attempt_at
                            else None,
                            delivery.next_retry_at.isoformat() if delivery.next_retry_at else None,
                            delivery.created_at.isoformat(),
                            delivery.completed_at.isoformat() if delivery.completed_at else None,
                            delivery.response_status,
                            delivery.response_body,
                        ),
                    )
                    row = self.backend.fetchone("SELECT last_insert_rowid() as id")
                    delivery.id = row["id"]
            else:
                # Update existing
                with self.backend.transaction():
                    self.backend.execute(
                        """
                        UPDATE webhook_deliveries
                        SET status = ?, attempt_count = ?, last_error = ?,
                            last_attempt_at = ?, next_retry_at = ?, completed_at = ?,
                            response_status = ?, response_body = ?
                        WHERE id = ?
                        """,
                        (
                            delivery.status.value,
                            delivery.attempt_count,
                            delivery.last_error,
                            delivery.last_attempt_at.isoformat()
                            if delivery.last_attempt_at
                            else None,
                            delivery.next_retry_at.isoformat() if delivery.next_retry_at else None,
                            delivery.completed_at.isoformat() if delivery.completed_at else None,
                            delivery.response_status,
                            delivery.response_body,
                            delivery.id,
                        ),
                    )
        except Exception as e:
            logger.error(f"Failed to save delivery: {e}")

        return delivery

    def _add_to_dlq(self, delivery: WebhookDelivery):
        """Add failed delivery to dead-letter queue."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    INSERT INTO webhook_dead_letter
                    (delivery_id, webhook_url, payload, headers, error, attempt_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        delivery.id,
                        delivery.webhook_url,
                        json.dumps(delivery.payload),
                        json.dumps(delivery.headers),
                        delivery.last_error,
                        delivery.attempt_count,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to add to DLQ: {e}")

    def get_dlq_items(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get items from dead-letter queue that haven't been reprocessed."""
        rows = self.backend.fetchall(
            """
            SELECT * FROM webhook_dead_letter
            WHERE reprocessed_at IS NULL
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]

    def reprocess_dlq_item(self, dlq_id: int) -> WebhookDelivery:
        """Reprocess a single item from the dead-letter queue.

        Args:
            dlq_id: Dead-letter queue item ID

        Returns:
            New WebhookDelivery result
        """
        row = self.backend.fetchone("SELECT * FROM webhook_dead_letter WHERE id = ?", (dlq_id,))
        if not row:
            raise ValueError(f"DLQ item {dlq_id} not found")

        # Mark as reprocessed
        with self.backend.transaction():
            self.backend.execute(
                "UPDATE webhook_dead_letter SET reprocessed_at = ? WHERE id = ?",
                (datetime.now().isoformat(), dlq_id),
            )

        # Redeliver
        return self.deliver(
            url=row["webhook_url"],
            payload=json.loads(row["payload"]),
            headers=json.loads(row["headers"]) if row["headers"] else None,
        )

    def get_delivery_stats(self) -> dict[str, Any]:
        """Get delivery statistics."""
        stats = {}

        # Count by status
        for status in list(DeliveryStatus):
            row = self.backend.fetchone(
                "SELECT COUNT(*) as count FROM webhook_deliveries WHERE status = ?",
                (status.value,),
            )
            stats[f"{status.value}_count"] = row["count"]

        # DLQ count
        row = self.backend.fetchone(
            "SELECT COUNT(*) as count FROM webhook_dead_letter WHERE reprocessed_at IS NULL"
        )
        stats["dlq_pending_count"] = row["count"]

        # Average attempts for successful deliveries
        row = self.backend.fetchone(
            """
            SELECT AVG(attempt_count) as avg_attempts
            FROM webhook_deliveries
            WHERE status = 'success'
            """
        )
        stats["avg_attempts_success"] = row["avg_attempts"] or 0

        return stats

    def cleanup_old_deliveries(self, days: int = 30) -> int:
        """Remove delivery records older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now() - timedelta(days=days)

        with self.backend.transaction():
            # Delete DLQ entries first (FK constraint)
            self.backend.execute(
                """
                DELETE FROM webhook_dead_letter
                WHERE delivery_id IN (
                    SELECT id FROM webhook_deliveries
                    WHERE created_at < ? AND status IN ('success', 'dead_letter')
                )
                """,
                (cutoff.isoformat(),),
            )

            self.backend.execute(
                """
                DELETE FROM webhook_deliveries
                WHERE created_at < ? AND status IN ('success', 'dead_letter')
                """,
                (cutoff.isoformat(),),
            )

        row = self.backend.fetchone("SELECT changes() as deleted")
        return row["deleted"]

    # ------------------------------------------------------------------
    # DLQ batch management
    # ------------------------------------------------------------------

    def reprocess_all_dlq(self, max_items: int = 50) -> list[dict[str, Any]]:
        """Batch reprocess pending DLQ items.

        Args:
            max_items: Maximum number of items to reprocess in one batch.

        Returns:
            List of dicts with ``dlq_id``, ``url``, and ``status`` for each item.
        """
        items = self.get_dlq_items(limit=max_items)
        results: list[dict[str, Any]] = []
        for item in items:
            try:
                delivery = self.reprocess_dlq_item(item["id"])
                results.append(
                    {
                        "dlq_id": item["id"],
                        "url": item["webhook_url"],
                        "status": delivery.status.value,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to reprocess DLQ item {item['id']}: {e}")
                results.append(
                    {
                        "dlq_id": item["id"],
                        "url": item["webhook_url"],
                        "status": "error",
                        "error": str(e),
                    }
                )
        return results

    def purge_dlq(self, older_than_days: int = 30) -> int:
        """Remove old DLQ entries (both reprocessed and unprocessed).

        Args:
            older_than_days: Age threshold in days.

        Returns:
            Number of DLQ records deleted.
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        with self.backend.transaction():
            self.backend.execute(
                "DELETE FROM webhook_dead_letter WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
        row = self.backend.fetchone("SELECT changes() as deleted")
        return row["deleted"]

    def get_dlq_stats(self) -> dict[str, Any]:
        """Get DLQ statistics: count by URL, oldest item age.

        Returns:
            Dict with ``total_pending``, ``by_url`` counts, and ``oldest_age_seconds``.
        """
        # Total pending
        row = self.backend.fetchone(
            "SELECT COUNT(*) as cnt FROM webhook_dead_letter WHERE reprocessed_at IS NULL"
        )
        total = row["cnt"]

        # Count by URL
        rows = self.backend.fetchall(
            """
            SELECT webhook_url, COUNT(*) as cnt
            FROM webhook_dead_letter
            WHERE reprocessed_at IS NULL
            GROUP BY webhook_url
            ORDER BY cnt DESC
            """
        )
        by_url = {r["webhook_url"]: r["cnt"] for r in rows}

        # Oldest item age
        row = self.backend.fetchone(
            """
            SELECT MIN(created_at) as oldest
            FROM webhook_dead_letter
            WHERE reprocessed_at IS NULL
            """
        )
        oldest_age: float | None = None
        if row and row["oldest"]:
            try:
                oldest_dt = datetime.fromisoformat(row["oldest"])
                # SQLite CURRENT_TIMESTAMP stores UTC; compare in UTC
                now_utc = datetime.now(tz=UTC).replace(tzinfo=None)
                oldest_age = (now_utc - oldest_dt).total_seconds()
            except (ValueError, TypeError):
                oldest_age = None

        return {
            "total_pending": total,
            "by_url": by_url,
            "oldest_age_seconds": oldest_age,
        }

    def delete_dlq_item(self, dlq_id: int) -> bool:
        """Delete a single DLQ item by ID.

        Args:
            dlq_id: Dead-letter queue item ID.

        Returns:
            True if the item was deleted, False if not found.
        """
        row = self.backend.fetchone("SELECT id FROM webhook_dead_letter WHERE id = ?", (dlq_id,))
        if not row:
            return False
        with self.backend.transaction():
            self.backend.execute("DELETE FROM webhook_dead_letter WHERE id = ?", (dlq_id,))
        return True
