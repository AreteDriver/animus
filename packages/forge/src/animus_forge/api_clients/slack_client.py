"""Slack integration client for notifications and approvals."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    WebClient = None
    AsyncWebClient = None
    SlackApiError = Exception


class MessageType(Enum):
    """Types of Slack messages."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    APPROVAL_REQUEST = "approval_request"


@dataclass
class SlackMessage:
    """Represents a Slack message."""

    channel: str
    text: str
    message_type: MessageType = MessageType.INFO
    blocks: list[dict] | None = None
    thread_ts: str | None = None
    attachments: list[dict] | None = None


class SlackClient:
    """Client for Slack notifications and interactive messages."""

    # Color mapping for message types
    COLORS = {
        MessageType.INFO: "#3498db",
        MessageType.SUCCESS: "#2ecc71",
        MessageType.WARNING: "#f39c12",
        MessageType.ERROR: "#e74c3c",
        MessageType.APPROVAL_REQUEST: "#9b59b6",
    }

    # Emoji mapping for message types
    EMOJIS = {
        MessageType.INFO: ":information_source:",
        MessageType.SUCCESS: ":white_check_mark:",
        MessageType.WARNING: ":warning:",
        MessageType.ERROR: ":x:",
        MessageType.APPROVAL_REQUEST: ":raised_hand:",
    }

    def __init__(self, token: str | None = None):
        """Initialize Slack client.

        Args:
            token: Slack Bot token. If not provided, attempts to load from settings.
        """
        self.token = token
        self.client = None
        self.async_client = None

        if token and WebClient:
            self.client = WebClient(token=token)
        if token and AsyncWebClient:
            self.async_client = AsyncWebClient(token=token)

    def is_configured(self) -> bool:
        """Check if Slack client is configured."""
        return self.client is not None

    def send_message(
        self,
        channel: str,
        text: str,
        message_type: MessageType = MessageType.INFO,
        blocks: list[dict] | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to a Slack channel.

        Args:
            channel: Channel ID or name
            text: Message text
            message_type: Type of message for styling
            blocks: Optional Block Kit blocks
            thread_ts: Optional thread timestamp for replies

        Returns:
            Dict with 'success', 'ts' (message timestamp), and optionally 'error'
        """
        if not self.is_configured():
            return {"success": False, "error": "Slack client not configured"}

        try:
            # Build attachments with color coding
            attachments = [
                {
                    "color": self.COLORS[message_type],
                    "text": text,
                    "fallback": text,
                }
            ]

            response = self.client.chat_postMessage(
                channel=channel,
                text=f"{self.EMOJIS[message_type]} {text}",
                attachments=attachments if not blocks else None,
                blocks=blocks,
                thread_ts=thread_ts,
            )

            return {
                "success": True,
                "ts": response["ts"],
                "channel": response["channel"],
            }
        except SlackApiError as e:
            return {"success": False, "error": str(e)}

    def send_workflow_notification(
        self,
        channel: str,
        workflow_name: str,
        status: str,
        details: dict | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Send a workflow status notification.

        Args:
            channel: Channel ID or name
            workflow_name: Name of the workflow
            status: Workflow status (started, completed, failed, etc.)
            details: Optional additional details
            thread_ts: Optional thread timestamp

        Returns:
            Dict with 'success' and message details
        """
        status_map = {
            "started": MessageType.INFO,
            "completed": MessageType.SUCCESS,
            "failed": MessageType.ERROR,
            "warning": MessageType.WARNING,
            "pending_approval": MessageType.APPROVAL_REQUEST,
        }

        message_type = status_map.get(status, MessageType.INFO)

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Workflow: {workflow_name}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{status.replace('_', ' ').title()}",
                    },
                    {"type": "mrkdwn", "text": f"*Type:*\n{message_type.value}"},
                ],
            },
        ]

        if details:
            detail_text = "\n".join(f"• *{k}:* {v}" for k, v in details.items())
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": detail_text},
                }
            )

        return self.send_message(
            channel=channel,
            text=f"Workflow '{workflow_name}' {status}",
            message_type=message_type,
            blocks=blocks,
            thread_ts=thread_ts,
        )

    def send_approval_request(
        self,
        channel: str,
        title: str,
        description: str,
        requester: str | None = None,
        callback_id: str | None = None,
        details: dict | None = None,
    ) -> dict[str, Any]:
        """Send an approval request with interactive buttons.

        Args:
            channel: Channel ID or name
            title: Approval request title
            description: Description of what needs approval
            requester: Optional requester name/ID
            callback_id: Optional callback ID for tracking
            details: Optional additional details

        Returns:
            Dict with 'success' and message details
        """
        if not self.is_configured():
            return {"success": False, "error": "Slack client not configured"}

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":raised_hand: Approval Required: {title}",
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": description},
            },
        ]

        if requester:
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"Requested by: {requester}"}],
                }
            )

        if details:
            detail_text = "\n".join(f"• *{k}:* {v}" for k, v in details.items())
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Details:*\n{detail_text}"},
                }
            )

        # Add approval buttons
        blocks.append(
            {
                "type": "actions",
                "block_id": callback_id or "approval_actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "approve",
                        "value": callback_id or "approve",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "action_id": "reject",
                        "value": callback_id or "reject",
                    },
                ],
            }
        )

        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=f"Approval required: {title}",
                blocks=blocks,
            )
            return {
                "success": True,
                "ts": response["ts"],
                "channel": response["channel"],
            }
        except SlackApiError as e:
            return {"success": False, "error": str(e)}

    def send_code_review_notification(
        self,
        channel: str,
        review_result: dict,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Send a code review result notification.

        Args:
            channel: Channel ID or name
            review_result: Review results from reviewer agent
            thread_ts: Optional thread timestamp

        Returns:
            Dict with 'success' and message details
        """
        approved = review_result.get("approved", False)
        score = review_result.get("score", 0)
        findings = review_result.get("findings", [])

        message_type = MessageType.SUCCESS if approved else MessageType.WARNING

        # Build findings summary
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        major = sum(1 for f in findings if f.get("severity") == "major")
        minor = sum(1 for f in findings if f.get("severity") == "minor")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{'✅' if approved else '⚠️'} Code Review {'Approved' if approved else 'Needs Work'}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Score:* {score}/10"},
                    {"type": "mrkdwn", "text": f"*Findings:* {len(findings)} total"},
                ],
            },
        ]

        if findings:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Breakdown:*\n• Critical: {critical}\n• Major: {major}\n• Minor: {minor}",
                    },
                }
            )

        if review_result.get("summary"):
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Summary:*\n{review_result['summary']}",
                    },
                }
            )

        return self.send_message(
            channel=channel,
            text=f"Code review {'approved' if approved else 'needs work'} (score: {score}/10)",
            message_type=message_type,
            blocks=blocks,
            thread_ts=thread_ts,
        )

    async def send_message_async(
        self,
        channel: str,
        text: str,
        message_type: MessageType = MessageType.INFO,
        blocks: list[dict] | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Async version of send_message."""
        if not self.async_client:
            return {"success": False, "error": "Async Slack client not configured"}

        try:
            attachments = [
                {
                    "color": self.COLORS[message_type],
                    "text": text,
                    "fallback": text,
                }
            ]

            response = await self.async_client.chat_postMessage(
                channel=channel,
                text=f"{self.EMOJIS[message_type]} {text}",
                attachments=attachments if not blocks else None,
                blocks=blocks,
                thread_ts=thread_ts,
            )

            return {
                "success": True,
                "ts": response["ts"],
                "channel": response["channel"],
            }
        except SlackApiError as e:
            return {"success": False, "error": str(e)}

    def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Update an existing message.

        Args:
            channel: Channel ID
            ts: Message timestamp
            text: New message text
            blocks: Optional new blocks

        Returns:
            Dict with 'success' and optionally 'error'
        """
        if not self.is_configured():
            return {"success": False, "error": "Slack client not configured"}

        try:
            response = self.client.chat_update(
                channel=channel,
                ts=ts,
                text=text,
                blocks=blocks,
            )
            return {"success": True, "ts": response["ts"]}
        except SlackApiError as e:
            return {"success": False, "error": str(e)}

    def add_reaction(self, channel: str, ts: str, emoji: str) -> dict[str, Any]:
        """Add a reaction to a message.

        Args:
            channel: Channel ID
            ts: Message timestamp
            emoji: Emoji name (without colons)

        Returns:
            Dict with 'success' and optionally 'error'
        """
        if not self.is_configured():
            return {"success": False, "error": "Slack client not configured"}

        try:
            self.client.reactions_add(channel=channel, timestamp=ts, name=emoji)
            return {"success": True}
        except SlackApiError as e:
            return {"success": False, "error": str(e)}
