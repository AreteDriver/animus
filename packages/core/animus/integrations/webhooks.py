"""
Webhook Integration

Receive and process webhooks from external services.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

from animus.integrations.base import AuthType, BaseIntegration
from animus.logging import get_logger
from animus.tools import Tool, ToolResult

logger = get_logger("integrations.webhooks")


@dataclass
class WebhookEvent:
    """Received webhook event."""

    id: str
    source: str
    event_type: str
    payload: dict[str, Any]
    received_at: datetime
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "event_type": self.event_type,
            "payload": self.payload,
            "received_at": self.received_at.isoformat(),
            "headers": self.headers,
        }


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler for incoming webhooks."""

    integration: WebhookIntegration | None = None

    def do_POST(self):
        """Handle POST webhook requests."""
        if not WebhookHandler.integration:
            self.send_error(500, "Integration not initialized")
            return

        # Parse path: /webhooks/{source}/{event_type}
        path_parts = self.path.strip("/").split("/")
        if len(path_parts) < 2:
            self.send_error(400, "Invalid webhook path. Expected /webhooks/{source}/{event_type}")
            return

        source = path_parts[0]
        event_type = path_parts[1] if len(path_parts) > 1 else "default"

        # Get headers
        headers = {k: v for k, v in self.headers.items()}

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Verify signature if secret is configured
        if WebhookHandler.integration._secret:
            if not self._verify_signature(body, headers):
                self.send_error(401, "Invalid signature")
                return

        # Parse payload
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", errors="ignore")}

        # Create event
        event = WebhookEvent(
            id=f"{source}-{event_type}-{datetime.now().timestamp()}",
            source=source,
            event_type=event_type,
            payload=payload,
            received_at=datetime.now(),
            headers=headers,
        )

        # Store event
        WebhookHandler.integration._receive_event(event)

        # Send response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "received", "id": event.id}).encode())

    def _verify_signature(self, body: bytes, headers: dict[str, str]) -> bool:
        """Verify webhook signature."""
        if not WebhookHandler.integration or not WebhookHandler.integration._secret:
            return True

        # Support common signature headers
        signature = None
        for header in ["X-Hub-Signature-256", "X-Signature-256", "X-Webhook-Signature"]:
            if header in headers:
                signature = headers[header]
                break

        if not signature:
            return False

        # Compute expected signature
        expected = hmac.new(
            WebhookHandler.integration._secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        # Compare (handle "sha256=" prefix)
        if signature.startswith("sha256="):
            signature = signature[7:]

        return hmac.compare_digest(signature, expected)

    def log_message(self, format: str, *args: object) -> None:
        """Custom logging."""
        logger.debug(f"Webhook request: {format % args}")


class WebhookIntegration(BaseIntegration):
    """
    Webhook integration for receiving external events.

    Runs an HTTP server to receive webhooks from external services.
    Events are stored in a ring buffer for later retrieval.
    """

    name = "webhooks"
    display_name = "Webhooks"
    auth_type = AuthType.NONE

    def __init__(self, max_events: int = 100):
        super().__init__()
        self._port: int = 8421
        self._secret: str | None = None
        self._server: HTTPServer | None = None
        self._server_thread: Thread | None = None
        self._events: deque[WebhookEvent] = deque(maxlen=max_events)
        self._callbacks: dict[str, list[Callable[[WebhookEvent], None]]] = {}

    async def connect(self, credentials: dict[str, Any]) -> bool:
        """
        Start webhook server.

        Credentials:
            port: Port to listen on (default: 8421)
            secret: Optional secret for signature verification
        """
        self._port = credentials.get("port", 8421)
        self._secret = credentials.get("secret")

        try:
            # Set up handler
            WebhookHandler.integration = self

            # Create and start server
            self._server = HTTPServer(("0.0.0.0", self._port), WebhookHandler)
            self._server_thread = Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()

            self._set_connected()
            logger.info(f"Webhook server started on port {self._port}")
            return True
        except Exception as e:
            self._set_error(f"Failed to start webhook server: {e}")
            return False

    async def disconnect(self) -> bool:
        """Stop webhook server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._server_thread:
            self._server_thread.join(timeout=5)
            self._server_thread = None

        WebhookHandler.integration = None
        self._set_disconnected()
        logger.info("Webhook server stopped")
        return True

    async def verify(self) -> bool:
        """Verify webhook server is running."""
        return self._server is not None and self.is_connected

    def get_tools(self) -> list[Tool]:
        """Get webhook tools."""
        return [
            Tool(
                name="webhook_list_events",
                description="List recent webhook events",
                parameters={
                    "source": {
                        "type": "string",
                        "description": "Filter by source",
                        "required": False,
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type",
                        "required": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum events to return (default: 20)",
                        "required": False,
                    },
                },
                handler=self._tool_list_events,
            ),
            Tool(
                name="webhook_get_event",
                description="Get a specific webhook event by ID",
                parameters={
                    "event_id": {
                        "type": "string",
                        "description": "Event ID",
                        "required": True,
                    },
                },
                handler=self._tool_get_event,
            ),
            Tool(
                name="webhook_info",
                description="Get webhook server information",
                parameters={},
                handler=self._tool_info,
            ),
        ]

    def _receive_event(self, event: WebhookEvent) -> None:
        """Process received webhook event."""
        self._events.append(event)
        logger.info(f"Received webhook: {event.source}/{event.event_type}")

        # Call registered callbacks
        key = f"{event.source}:{event.event_type}"
        for callback in self._callbacks.get(key, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error for {key}: {e}")

        # Also call wildcard callbacks
        for callback in self._callbacks.get("*", []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Wildcard callback error: {e}")

    def register_callback(
        self, source: str, event_type: str, callback: Callable[[WebhookEvent], None]
    ) -> None:
        """
        Register a callback for specific webhook events.

        Args:
            source: Source to match (or "*" for all)
            event_type: Event type to match (or "*" for all)
            callback: Function to call with event
        """
        key = f"{source}:{event_type}" if source != "*" else "*"
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    async def _tool_list_events(
        self,
        source: str | None = None,
        event_type: str | None = None,
        limit: int = 20,
    ) -> ToolResult:
        """List recent webhook events."""
        events = list(self._events)

        # Filter
        if source:
            events = [e for e in events if e.source == source]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Sort by newest first and limit
        events = sorted(events, key=lambda e: e.received_at, reverse=True)[:limit]

        return ToolResult(
            tool_name="webhook_tool",
            success=True,
            output={
                "count": len(events),
                "events": [e.to_dict() for e in events],
            },
        )

    async def _tool_get_event(self, event_id: str) -> ToolResult:
        """Get a specific event by ID."""
        for event in self._events:
            if event.id == event_id:
                return ToolResult(tool_name="webhook_tool", success=True, output=event.to_dict())

        return ToolResult(
            tool_name="webhook_tool",
            success=False,
            output=None,
            error=f"Event not found: {event_id}",
        )

    async def _tool_info(self) -> ToolResult:
        """Get webhook server information."""
        return ToolResult(
            tool_name="webhook_tool",
            success=True,
            output={
                "port": self._port,
                "running": self.is_connected,
                "events_stored": len(self._events),
                "signature_required": self._secret is not None,
                "endpoint_pattern": f"http://localhost:{self._port}/{{source}}/{{event_type}}",
            },
        )
