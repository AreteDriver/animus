"""Webhook trigger manager for event-driven workflow execution."""

import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from animus_forge.config import get_settings
from animus_forge.orchestrator import WorkflowEngineAdapter
from animus_forge.state import DatabaseBackend, get_database

logger = logging.getLogger(__name__)


def _parse_datetime(value) -> datetime | None:
    """Parse datetime from database (handles both strings and datetime objects)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class WebhookStatus(str, Enum):
    """Webhook status."""

    ACTIVE = "active"
    DISABLED = "disabled"


class PayloadMapping(BaseModel):
    """Maps incoming webhook payload fields to workflow variables."""

    source_path: str = Field(..., description="JSON path in payload (e.g., 'data.user.id')")
    target_variable: str = Field(..., description="Workflow variable name")
    default: Any | None = Field(None, description="Default value if path not found")


class Webhook(BaseModel):
    """A webhook definition."""

    id: str = Field(..., description="Webhook identifier (used in URL)")
    name: str = Field(..., description="Webhook name")
    description: str = Field("", description="Webhook description")
    workflow_id: str = Field(..., description="Workflow to trigger")
    secret: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret for signature verification",
    )
    payload_mappings: list[PayloadMapping] = Field(
        default_factory=list, description="Map payload fields to workflow variables"
    )
    static_variables: dict[str, Any] = Field(
        default_factory=dict, description="Static variables to pass to workflow"
    )
    status: WebhookStatus = Field(WebhookStatus.ACTIVE, description="Webhook status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_triggered: datetime | None = Field(None, description="Last trigger timestamp")
    trigger_count: int = Field(0, ge=0, description="Total trigger count")


class WebhookTriggerLog(BaseModel):
    """Log entry for a webhook trigger."""

    webhook_id: str
    workflow_id: str
    triggered_at: datetime
    source_ip: str | None = None
    payload_size: int
    status: str
    duration_seconds: float
    error: str | None = None


class WebhookManager:
    """Manages webhook definitions and triggers."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS webhooks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            workflow_id TEXT NOT NULL,
            secret TEXT NOT NULL,
            payload_mappings TEXT,
            static_variables TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_triggered TIMESTAMP,
            trigger_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_webhooks_status ON webhooks(status);
        CREATE INDEX IF NOT EXISTS idx_webhooks_workflow ON webhooks(workflow_id);

        CREATE TABLE IF NOT EXISTS webhook_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            webhook_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_ip TEXT,
            payload_size INTEGER,
            status TEXT NOT NULL,
            duration_seconds REAL,
            error TEXT,
            FOREIGN KEY (webhook_id) REFERENCES webhooks(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_webhook_logs_webhook
        ON webhook_logs(webhook_id, triggered_at DESC);
    """

    def __init__(self, backend: DatabaseBackend | None = None):
        self.settings = get_settings()
        self.backend = backend or get_database()
        self.workflow_engine = WorkflowEngineAdapter()
        self._webhooks: dict[str, Webhook] = {}
        self._init_schema()
        self._load_all_webhooks()

    def _init_schema(self):
        """Initialize database schema."""
        self.backend.executescript(self.SCHEMA)

    def _load_all_webhooks(self):
        """Load all webhooks from database."""
        rows = self.backend.fetchall("SELECT * FROM webhooks")
        for row in rows:
            try:
                webhook = self._row_to_webhook(row)
                if webhook:
                    self._webhooks[webhook.id] = webhook
            except Exception as e:
                logger.error(f"Failed to load webhook from row: {e}")

    def _row_to_webhook(self, row: dict) -> Webhook | None:
        """Convert database row to Webhook."""
        try:
            payload_mappings = []
            if row.get("payload_mappings"):
                mappings_data = json.loads(row["payload_mappings"])
                payload_mappings = [PayloadMapping(**m) for m in mappings_data]

            return Webhook(
                id=row["id"],
                name=row["name"],
                description=row.get("description", ""),
                workflow_id=row["workflow_id"],
                secret=row["secret"],
                payload_mappings=payload_mappings,
                static_variables=json.loads(row["static_variables"])
                if row.get("static_variables")
                else {},
                status=WebhookStatus(row["status"]),
                created_at=_parse_datetime(row.get("created_at")) or datetime.now(),
                last_triggered=_parse_datetime(row.get("last_triggered")),
                trigger_count=row.get("trigger_count", 0),
            )
        except Exception as e:
            logger.error(f"Failed to parse webhook row: {e}")
            return None

    def _save_webhook(self, webhook: Webhook) -> bool:
        """Save a webhook to database (insert or update)."""
        try:
            existing = self.backend.fetchone("SELECT id FROM webhooks WHERE id = ?", (webhook.id,))
            if existing:
                return self._update_webhook_in_db(webhook)
            else:
                return self._insert_webhook_in_db(webhook)
        except Exception as e:
            logger.error(f"Failed to save webhook {webhook.id}: {e}")
            return False

    def _insert_webhook_in_db(self, webhook: Webhook) -> bool:
        """Insert a new webhook into the database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    INSERT INTO webhooks
                    (id, name, description, workflow_id, secret, payload_mappings,
                     static_variables, status, created_at, last_triggered, trigger_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        webhook.id,
                        webhook.name,
                        webhook.description,
                        webhook.workflow_id,
                        webhook.secret,
                        json.dumps([m.model_dump() for m in webhook.payload_mappings])
                        if webhook.payload_mappings
                        else None,
                        json.dumps(webhook.static_variables) if webhook.static_variables else None,
                        webhook.status.value,
                        webhook.created_at.isoformat() if webhook.created_at else None,
                        webhook.last_triggered.isoformat() if webhook.last_triggered else None,
                        webhook.trigger_count,
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to insert webhook {webhook.id}: {e}")
            return False

    def _update_webhook_in_db(self, webhook: Webhook) -> bool:
        """Update an existing webhook in the database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    UPDATE webhooks
                    SET name = ?, description = ?, workflow_id = ?, secret = ?,
                        payload_mappings = ?, static_variables = ?, status = ?,
                        last_triggered = ?, trigger_count = ?
                    WHERE id = ?
                    """,
                    (
                        webhook.name,
                        webhook.description,
                        webhook.workflow_id,
                        webhook.secret,
                        json.dumps([m.model_dump() for m in webhook.payload_mappings])
                        if webhook.payload_mappings
                        else None,
                        json.dumps(webhook.static_variables) if webhook.static_variables else None,
                        webhook.status.value,
                        webhook.last_triggered.isoformat() if webhook.last_triggered else None,
                        webhook.trigger_count,
                        webhook.id,
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to update webhook {webhook.id}: {e}")
            return False

    def create_webhook(self, webhook: Webhook) -> bool:
        """Create a new webhook."""
        # Validate workflow exists
        workflow = self.workflow_engine.load_workflow(webhook.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {webhook.workflow_id} not found")

        # Check for duplicate ID
        if webhook.id in self._webhooks:
            raise ValueError(f"Webhook {webhook.id} already exists")

        webhook.created_at = datetime.now()
        if self._save_webhook(webhook):
            self._webhooks[webhook.id] = webhook
            return True
        return False

    def update_webhook(self, webhook: Webhook) -> bool:
        """Update an existing webhook."""
        if webhook.id not in self._webhooks:
            raise ValueError(f"Webhook {webhook.id} not found")

        # Preserve creation time, trigger count, and secret if not provided
        existing = self._webhooks[webhook.id]
        webhook.created_at = existing.created_at
        webhook.trigger_count = existing.trigger_count
        webhook.last_triggered = existing.last_triggered
        if webhook.secret == "":
            webhook.secret = existing.secret

        if self._save_webhook(webhook):
            self._webhooks[webhook.id] = webhook
            return True
        return False

    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook."""
        if webhook_id not in self._webhooks:
            return False

        # Remove from database (logs cascade due to FK)
        try:
            with self.backend.transaction():
                self.backend.execute("DELETE FROM webhook_logs WHERE webhook_id = ?", (webhook_id,))
                self.backend.execute("DELETE FROM webhooks WHERE id = ?", (webhook_id,))
        except Exception as e:
            logger.error(f"Failed to delete webhook {webhook_id} from database: {e}")

        del self._webhooks[webhook_id]
        return True

    def get_webhook(self, webhook_id: str) -> Webhook | None:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)

    def list_webhooks(self) -> list[dict]:
        """List all webhooks (without secrets)."""
        webhooks = []
        for webhook in self._webhooks.values():
            webhooks.append(
                {
                    "id": webhook.id,
                    "name": webhook.name,
                    "workflow_id": webhook.workflow_id,
                    "status": webhook.status.value,
                    "last_triggered": webhook.last_triggered.isoformat()
                    if webhook.last_triggered
                    else None,
                    "trigger_count": webhook.trigger_count,
                }
            )
        return webhooks

    def verify_signature(self, webhook_id: str, payload: bytes, signature: str) -> bool:
        """Verify webhook signature using HMAC-SHA256."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return False

        expected = hmac.new(webhook.secret.encode(), payload, hashlib.sha256).hexdigest()

        # Support both raw hex and prefixed formats
        if signature.startswith("sha256="):
            signature = signature[7:]

        return hmac.compare_digest(expected, signature)

    def generate_signature(self, webhook_id: str, payload: bytes) -> str:
        """Generate signature for a payload (useful for testing)."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            raise ValueError(f"Webhook {webhook_id} not found")

        return "sha256=" + hmac.new(webhook.secret.encode(), payload, hashlib.sha256).hexdigest()

    def _extract_payload_value(self, payload: dict, path: str) -> Any:
        """Extract value from payload using dot notation path."""
        keys = path.split(".")
        value = payload
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _map_payload_to_variables(self, webhook: Webhook, payload: dict) -> dict[str, Any]:
        """Map webhook payload to workflow variables."""
        variables = webhook.static_variables.copy()

        for mapping in webhook.payload_mappings:
            value = self._extract_payload_value(payload, mapping.source_path)
            if value is not None:
                variables[mapping.target_variable] = value
            elif mapping.default is not None:
                variables[mapping.target_variable] = mapping.default

        return variables

    def trigger(
        self,
        webhook_id: str,
        payload: dict,
        source_ip: str | None = None,
    ) -> dict[str, Any]:
        """Trigger a webhook and execute the associated workflow."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            raise ValueError(f"Webhook {webhook_id} not found")

        if webhook.status != WebhookStatus.ACTIVE:
            raise ValueError(f"Webhook {webhook_id} is disabled")

        logger.info(f"Triggering webhook: {webhook_id} -> {webhook.workflow_id}")
        start_time = datetime.now()
        error_msg = None
        status = "success"

        try:
            workflow = self.workflow_engine.load_workflow(webhook.workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {webhook.workflow_id} not found")

            # Map payload to variables
            variables = self._map_payload_to_variables(webhook, payload)
            workflow.variables.update(variables)

            result = self.workflow_engine.execute_workflow(workflow)
            status = result.status
            workflow_result = result.model_dump(mode="json")

        except Exception as e:
            logger.error(f"Webhook trigger failed: {e}")
            status = "failed"
            error_msg = str(e)
            workflow_result = None

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Update webhook stats
        webhook.last_triggered = start_time
        webhook.trigger_count += 1
        self._save_webhook(webhook)
        self._webhooks[webhook_id] = webhook

        # Log trigger
        self._save_trigger_log(
            WebhookTriggerLog(
                webhook_id=webhook_id,
                workflow_id=webhook.workflow_id,
                triggered_at=start_time,
                source_ip=source_ip,
                payload_size=len(json.dumps(payload)),
                status=status,
                duration_seconds=duration,
                error=error_msg,
            )
        )

        return {
            "status": status,
            "webhook_id": webhook_id,
            "workflow_id": webhook.workflow_id,
            "duration_seconds": duration,
            "result": workflow_result,
            "error": error_msg,
        }

    def _save_trigger_log(self, log: WebhookTriggerLog):
        """Save trigger log entry to database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    INSERT INTO webhook_logs
                    (webhook_id, workflow_id, triggered_at, source_ip, payload_size,
                     status, duration_seconds, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log.webhook_id,
                        log.workflow_id,
                        log.triggered_at.isoformat(),
                        log.source_ip,
                        log.payload_size,
                        log.status,
                        log.duration_seconds,
                        log.error,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to save trigger log: {e}")

    def get_trigger_history(self, webhook_id: str, limit: int = 10) -> list[WebhookTriggerLog]:
        """Get trigger history for a webhook."""
        rows = self.backend.fetchall(
            """
            SELECT * FROM webhook_logs
            WHERE webhook_id = ?
            ORDER BY triggered_at DESC
            LIMIT ?
            """,
            (webhook_id, limit),
        )

        logs = []
        for row in rows:
            try:
                logs.append(
                    WebhookTriggerLog(
                        webhook_id=row["webhook_id"],
                        workflow_id=row["workflow_id"],
                        triggered_at=_parse_datetime(row.get("triggered_at")) or datetime.now(),
                        source_ip=row.get("source_ip"),
                        payload_size=row.get("payload_size", 0),
                        status=row["status"],
                        duration_seconds=row.get("duration_seconds", 0),
                        error=row.get("error"),
                    )
                )
            except Exception as e:
                logger.error(f"Failed to parse trigger log: {e}")
                continue

        return logs

    def regenerate_secret(self, webhook_id: str) -> str:
        """Regenerate the secret for a webhook."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            raise ValueError(f"Webhook {webhook_id} not found")

        webhook.secret = secrets.token_urlsafe(32)
        self._save_webhook(webhook)
        self._webhooks[webhook_id] = webhook
        return webhook.secret
