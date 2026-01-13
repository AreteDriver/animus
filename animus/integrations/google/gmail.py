"""
Gmail Integration

Email access via Gmail API.
"""

from __future__ import annotations

import base64
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from animus.integrations.base import AuthType, BaseIntegration
from animus.integrations.oauth import (
    GOOGLE_AUTH_AVAILABLE,
    OAuth2Flow,
    OAuth2Token,
    load_token,
    save_token,
)
from animus.logging import get_logger
from animus.tools import Tool, ToolResult

logger = get_logger("integrations.google.gmail")

# Check if Google API client is available
GOOGLE_API_AVAILABLE = False
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    GOOGLE_API_AVAILABLE = True
except ImportError:
    pass

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]


class GmailIntegration(BaseIntegration):
    """
    Gmail integration.

    Provides tools for:
    - Listing inbox messages
    - Reading emails
    - Searching emails
    - Drafting emails
    - Sending emails
    """

    name = "gmail"
    display_name = "Gmail"
    auth_type = AuthType.OAUTH2

    def __init__(self, data_dir: Path | None = None):
        super().__init__()
        self._data_dir = data_dir or Path.home() / ".animus" / "integrations"
        self._service: Any = None
        self._token: OAuth2Token | None = None

    @property
    def _token_path(self) -> Path:
        """Path to stored OAuth token."""
        return self._data_dir / "gmail_token.json"

    async def connect(self, credentials: dict[str, Any]) -> bool:
        """
        Connect to Gmail.

        Credentials:
            client_id: Google OAuth2 client ID
            client_secret: Google OAuth2 client secret
            token: Optional OAuth2Token dict if already authorized
        """
        if not GOOGLE_AUTH_AVAILABLE or not GOOGLE_API_AVAILABLE:
            self._set_error(
                "Google API libraries not installed. Install with: "
                "pip install google-api-python-client google-auth google-auth-oauthlib"
            )
            return False

        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")

        if not client_id or not client_secret:
            self._set_error("client_id and client_secret required")
            return False

        # Check for existing token
        self._token = load_token(self._token_path)

        # Check for token passed in credentials
        if token_data := credentials.get("token"):
            self._token = OAuth2Token.from_dict(token_data)

        # If no token or expired, run OAuth flow
        if not self._token or self._token.is_expired():
            if self._token and self._token.refresh_token:
                # Try to refresh
                flow = OAuth2Flow(client_id, client_secret, GMAIL_SCOPES)
                self._token = flow.refresh_token(self._token)

            if not self._token:
                # Run full OAuth flow
                flow = OAuth2Flow(client_id, client_secret, GMAIL_SCOPES)
                self._token = flow.run_local_server()

            if not self._token:
                self._set_error("OAuth2 authorization failed")
                return False

            # Save token for future use
            self._data_dir.mkdir(parents=True, exist_ok=True)
            save_token(self._token, self._token_path)

        # Build the Gmail service
        try:
            creds = Credentials(
                token=self._token.access_token,
                refresh_token=self._token.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
            )
            self._service = build("gmail", "v1", credentials=creds)
            self._credentials = credentials
            self._set_connected(expires_at=self._token.expires_at)
            logger.info("Connected to Gmail")
            return True
        except Exception as e:
            self._set_error(f"Failed to build Gmail service: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Gmail."""
        self._service = None
        self._token = None
        if self._token_path.exists():
            self._token_path.unlink()
        self._set_disconnected()
        logger.info("Disconnected from Gmail")
        return True

    async def verify(self) -> bool:
        """Verify Gmail connection."""
        if not self._service:
            return False
        try:
            self._service.users().getProfile(userId="me").execute()
            return True
        except Exception:
            self._set_expired()
            return False

    def get_tools(self) -> list[Tool]:
        """Get Gmail tools."""
        return [
            Tool(
                name="gmail_list_inbox",
                description="List recent inbox messages",
                parameters={
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum messages to return (default: 20)",
                        "required": False,
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only show unread messages (default: false)",
                        "required": False,
                    },
                },
                handler=self._tool_list_inbox,
            ),
            Tool(
                name="gmail_read_email",
                description="Read a specific email by ID",
                parameters={
                    "message_id": {
                        "type": "string",
                        "description": "Email message ID",
                        "required": True,
                    },
                },
                handler=self._tool_read_email,
            ),
            Tool(
                name="gmail_search",
                description="Search emails using Gmail search syntax",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'from:alice subject:meeting')",
                        "required": True,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "required": False,
                    },
                },
                handler=self._tool_search,
            ),
            Tool(
                name="gmail_draft_email",
                description="Create an email draft",
                parameters={
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                        "required": True,
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                        "required": True,
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text",
                        "required": True,
                    },
                },
                handler=self._tool_draft_email,
            ),
            Tool(
                name="gmail_send_email",
                description="Send an email directly",
                parameters={
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                        "required": True,
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                        "required": True,
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text",
                        "required": True,
                    },
                },
                handler=self._tool_send_email,
                requires_approval=True,  # Sending emails should require approval
            ),
        ]

    def _parse_message(self, message: dict) -> dict[str, Any]:
        """Parse Gmail message into readable format."""
        headers = message.get("payload", {}).get("headers", [])
        header_dict = {h["name"].lower(): h["value"] for h in headers}

        # Get body
        body = ""
        payload = message.get("payload", {})

        if "body" in payload and payload["body"].get("data"):
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode(
                "utf-8", errors="ignore"
            )
        elif "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode(
                        "utf-8", errors="ignore"
                    )
                    break

        return {
            "id": message["id"],
            "thread_id": message.get("threadId"),
            "from": header_dict.get("from", ""),
            "to": header_dict.get("to", ""),
            "subject": header_dict.get("subject", "(No subject)"),
            "date": header_dict.get("date", ""),
            "snippet": message.get("snippet", ""),
            "body": body[:2000] if body else "",
            "labels": message.get("labelIds", []),
        }

    async def _tool_list_inbox(
        self, max_results: int = 20, unread_only: bool = False
    ) -> ToolResult:
        """List inbox messages."""
        if not self._service:
            return ToolResult(
                tool_name="gmail_tool", success=False, output=None, error="Not connected to Gmail"
            )

        try:
            query = "in:inbox"
            if unread_only:
                query += " is:unread"

            results = (
                self._service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

            messages = results.get("messages", [])
            message_list = []

            for msg in messages:
                full_msg = (
                    self._service.users()
                    .messages()
                    .get(userId="me", id=msg["id"], format="metadata")
                    .execute()
                )
                headers = full_msg.get("payload", {}).get("headers", [])
                header_dict = {h["name"].lower(): h["value"] for h in headers}

                message_list.append(
                    {
                        "id": msg["id"],
                        "from": header_dict.get("from", ""),
                        "subject": header_dict.get("subject", "(No subject)"),
                        "date": header_dict.get("date", ""),
                        "snippet": full_msg.get("snippet", "")[:100],
                        "unread": "UNREAD" in full_msg.get("labelIds", []),
                    }
                )

            return ToolResult(
                tool_name="gmail_tool",
                success=True,
                output={
                    "count": len(message_list),
                    "messages": message_list,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="gmail_tool",
                success=False,
                output=None,
                error=f"Failed to list inbox: {e}",
            )

    async def _tool_read_email(self, message_id: str) -> ToolResult:
        """Read a specific email."""
        if not self._service:
            return ToolResult(
                tool_name="gmail_tool", success=False, output=None, error="Not connected to Gmail"
            )

        try:
            message = (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

            parsed = self._parse_message(message)
            return ToolResult(tool_name="gmail_tool", success=True, output=parsed)
        except Exception as e:
            return ToolResult(
                tool_name="gmail_tool",
                success=False,
                output=None,
                error=f"Failed to read email: {e}",
            )

    async def _tool_search(self, query: str, max_results: int = 20) -> ToolResult:
        """Search emails."""
        if not self._service:
            return ToolResult(
                tool_name="gmail_tool", success=False, output=None, error="Not connected to Gmail"
            )

        try:
            results = (
                self._service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

            messages = results.get("messages", [])
            message_list = []

            for msg in messages:
                full_msg = (
                    self._service.users()
                    .messages()
                    .get(userId="me", id=msg["id"], format="metadata")
                    .execute()
                )
                headers = full_msg.get("payload", {}).get("headers", [])
                header_dict = {h["name"].lower(): h["value"] for h in headers}

                message_list.append(
                    {
                        "id": msg["id"],
                        "from": header_dict.get("from", ""),
                        "subject": header_dict.get("subject", "(No subject)"),
                        "date": header_dict.get("date", ""),
                        "snippet": full_msg.get("snippet", "")[:100],
                    }
                )

            return ToolResult(
                tool_name="gmail_tool",
                success=True,
                output={
                    "query": query,
                    "count": len(message_list),
                    "messages": message_list,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="gmail_tool", success=False, output=None, error=f"Search failed: {e}"
            )

    async def _tool_draft_email(self, to: str, subject: str, body: str) -> ToolResult:
        """Create an email draft."""
        if not self._service:
            return ToolResult(
                tool_name="gmail_tool", success=False, output=None, error="Not connected to Gmail"
            )

        try:
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            draft = (
                self._service.users()
                .drafts()
                .create(userId="me", body={"message": {"raw": raw}})
                .execute()
            )

            return ToolResult(
                tool_name="gmail_tool",
                success=True,
                output={
                    "draft_id": draft["id"],
                    "to": to,
                    "subject": subject,
                    "message": "Draft created successfully",
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="gmail_tool",
                success=False,
                output=None,
                error=f"Failed to create draft: {e}",
            )

    async def _tool_send_email(self, to: str, subject: str, body: str) -> ToolResult:
        """Send an email."""
        if not self._service:
            return ToolResult(
                tool_name="gmail_tool", success=False, output=None, error="Not connected to Gmail"
            )

        try:
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            sent = self._service.users().messages().send(userId="me", body={"raw": raw}).execute()

            return ToolResult(
                tool_name="gmail_tool",
                success=True,
                output={
                    "message_id": sent["id"],
                    "to": to,
                    "subject": subject,
                    "status": "sent",
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="gmail_tool",
                success=False,
                output=None,
                error=f"Failed to send email: {e}",
            )
