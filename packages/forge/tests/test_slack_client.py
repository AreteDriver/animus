"""Tests for Slack integration client."""

from unittest.mock import MagicMock, patch

import pytest

from animus_forge.api_clients.slack_client import (
    MessageType,
    SlackClient,
    SlackMessage,
)


class TestSlackMessage:
    """Tests for SlackMessage dataclass."""

    def test_defaults(self):
        msg = SlackMessage(channel="#general", text="hello")
        assert msg.message_type == MessageType.INFO
        assert msg.blocks is None
        assert msg.thread_ts is None
        assert msg.attachments is None

    def test_all_fields(self):
        msg = SlackMessage(
            channel="#ops",
            text="alert",
            message_type=MessageType.ERROR,
            blocks=[{"type": "section"}],
            thread_ts="1234.5678",
            attachments=[{"text": "detail"}],
        )
        assert msg.channel == "#ops"
        assert msg.message_type == MessageType.ERROR


class TestSlackClientInit:
    """Tests for SlackClient initialization."""

    @patch("animus_forge.api_clients.slack_client.WebClient")
    @patch("animus_forge.api_clients.slack_client.AsyncWebClient")
    def test_init_with_token(self, mock_async, mock_sync):
        client = SlackClient(token="xoxb-test")
        assert client.is_configured()
        mock_sync.assert_called_once_with(token="xoxb-test")

    def test_init_without_token(self):
        client = SlackClient()
        assert not client.is_configured()
        assert client.client is None

    @patch("animus_forge.api_clients.slack_client.WebClient", None)
    def test_init_sdk_not_installed(self):
        """Client degrades gracefully when slack_sdk is missing."""
        client = SlackClient(token="xoxb-test")
        assert not client.is_configured()


class TestSendMessage:
    """Tests for send_message."""

    def _make_client(self):
        client = SlackClient(token=None)
        client.client = MagicMock()
        return client

    def test_not_configured_returns_error(self):
        client = SlackClient()
        result = client.send_message("#test", "hello")
        assert result["success"] is False
        assert "not configured" in result["error"]

    def test_successful_send(self):
        client = self._make_client()
        client.client.chat_postMessage.return_value = {
            "ts": "111.222",
            "channel": "C123",
        }
        result = client.send_message("#test", "hello", MessageType.SUCCESS)
        assert result["success"] is True
        assert result["ts"] == "111.222"

        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        assert ":white_check_mark:" in call_kwargs["text"]
        assert call_kwargs["attachments"][0]["color"] == "#2ecc71"

    def test_blocks_suppress_attachments(self):
        client = self._make_client()
        client.client.chat_postMessage.return_value = {"ts": "1", "channel": "C1"}
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "hi"}}]
        client.send_message("#test", "hello", blocks=blocks)

        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        assert call_kwargs["attachments"] is None
        assert call_kwargs["blocks"] == blocks

    def test_thread_reply(self):
        client = self._make_client()
        client.client.chat_postMessage.return_value = {"ts": "2", "channel": "C1"}
        client.send_message("#test", "reply", thread_ts="1.0")

        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        assert call_kwargs["thread_ts"] == "1.0"

    def test_api_error_returns_failure(self):
        client = self._make_client()
        client.client.chat_postMessage.side_effect = Exception("channel_not_found")
        result = client.send_message("#bad", "hello")
        assert result["success"] is False
        assert "channel_not_found" in result["error"]


class TestWorkflowNotification:
    """Tests for send_workflow_notification."""

    def _make_client(self):
        client = SlackClient(token=None)
        client.client = MagicMock()
        client.client.chat_postMessage.return_value = {"ts": "1", "channel": "C1"}
        return client

    def test_started_notification(self):
        client = self._make_client()
        result = client.send_workflow_notification("#ops", "deploy", "started")
        assert result["success"] is True

    def test_failed_notification_uses_error_type(self):
        client = self._make_client()
        client.send_workflow_notification("#ops", "deploy", "failed")
        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        # Error message type should use the error emoji in fallback text
        assert "failed" in call_kwargs["text"]

    def test_details_included_in_blocks(self):
        client = self._make_client()
        client.send_workflow_notification(
            "#ops", "deploy", "completed", details={"duration": "5m", "steps": "3"}
        )
        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        blocks = call_kwargs["blocks"]
        # Should have header, status section, and details section
        assert len(blocks) == 3
        detail_block = blocks[2]
        assert "duration" in detail_block["text"]["text"]

    def test_unknown_status_defaults_to_info(self):
        client = self._make_client()
        client.send_workflow_notification("#ops", "deploy", "paused")
        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        # Should still succeed with INFO type
        assert ":information_source:" in call_kwargs["text"]


class TestApprovalRequest:
    """Tests for send_approval_request."""

    def _make_client(self):
        client = SlackClient(token=None)
        client.client = MagicMock()
        client.client.chat_postMessage.return_value = {"ts": "1", "channel": "C1"}
        return client

    def test_not_configured(self):
        client = SlackClient()
        result = client.send_approval_request("#test", "Deploy", "Deploy to prod?")
        assert result["success"] is False

    def test_basic_approval(self):
        client = self._make_client()
        result = client.send_approval_request("#ops", "Deploy", "Deploy to prod?")
        assert result["success"] is True

        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        blocks = call_kwargs["blocks"]
        # Last block should be actions with approve/reject buttons
        actions = blocks[-1]
        assert actions["type"] == "actions"
        button_ids = [el["action_id"] for el in actions["elements"]]
        assert "approve" in button_ids
        assert "reject" in button_ids

    def test_callback_id_used(self):
        client = self._make_client()
        client.send_approval_request("#ops", "Deploy", "desc", callback_id="deploy-123")
        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        actions = call_kwargs["blocks"][-1]
        assert actions["block_id"] == "deploy-123"

    def test_requester_context(self):
        client = self._make_client()
        client.send_approval_request("#ops", "Deploy", "desc", requester="alice")
        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        blocks = call_kwargs["blocks"]
        context_blocks = [b for b in blocks if b["type"] == "context"]
        assert len(context_blocks) == 1
        assert "alice" in context_blocks[0]["elements"][0]["text"]


class TestCodeReviewNotification:
    """Tests for send_code_review_notification."""

    def _make_client(self):
        client = SlackClient(token=None)
        client.client = MagicMock()
        client.client.chat_postMessage.return_value = {"ts": "1", "channel": "C1"}
        return client

    def test_approved_review(self):
        client = self._make_client()
        result = client.send_code_review_notification(
            "#reviews",
            {"approved": True, "score": 9, "findings": []},
        )
        assert result["success"] is True

    def test_rejected_review_with_findings(self):
        client = self._make_client()
        findings = [
            {"severity": "critical", "msg": "SQL injection"},
            {"severity": "major", "msg": "Missing auth"},
            {"severity": "minor", "msg": "Typo"},
        ]
        client.send_code_review_notification(
            "#reviews",
            {
                "approved": False,
                "score": 4,
                "findings": findings,
                "summary": "Needs work",
            },
        )
        call_kwargs = client.client.chat_postMessage.call_args.kwargs
        blocks = call_kwargs["blocks"]
        # Should have header, score section, breakdown, and summary
        assert len(blocks) == 4


class TestUpdateMessage:
    """Tests for update_message."""

    def test_not_configured(self):
        client = SlackClient()
        result = client.update_message("C1", "1.0", "new text")
        assert result["success"] is False

    def test_successful_update(self):
        client = SlackClient(token=None)
        client.client = MagicMock()
        client.client.chat_update.return_value = {"ts": "1.0"}
        result = client.update_message("C1", "1.0", "updated")
        assert result["success"] is True


class TestAddReaction:
    """Tests for add_reaction."""

    def test_not_configured(self):
        client = SlackClient()
        result = client.add_reaction("C1", "1.0", "thumbsup")
        assert result["success"] is False

    def test_successful_reaction(self):
        client = SlackClient(token=None)
        client.client = MagicMock()
        result = client.add_reaction("C1", "1.0", "thumbsup")
        assert result["success"] is True
        client.client.reactions_add.assert_called_once_with(
            channel="C1", timestamp="1.0", name="thumbsup"
        )


class TestAsyncSendMessage:
    """Tests for async send_message."""

    @pytest.mark.asyncio
    async def test_not_configured(self):
        client = SlackClient()
        result = await client.send_message_async("#test", "hello")
        assert result["success"] is False
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_async_send(self):
        client = SlackClient(token=None)
        mock_async = MagicMock()

        async def mock_post(**kwargs):
            return {"ts": "1.0", "channel": "C1"}

        mock_async.chat_postMessage = mock_post
        client.async_client = mock_async

        result = await client.send_message_async("#test", "hello")
        assert result["success"] is True
        assert result["ts"] == "1.0"
