"""Tests for notification channel send methods.

Covers the uncovered send/post paths for Slack, Discord, Webhook, Email, Teams, PagerDuty.
"""

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from animus_forge.notifications.notifier import (
    DiscordChannel,
    EmailChannel,
    EventType,
    NotificationEvent,
    Notifier,
    PagerDutyChannel,
    SlackChannel,
    TeamsChannel,
    WebhookChannel,
)


def _make_event(event_type=EventType.WORKFLOW_COMPLETED, severity="success", details=None):
    return NotificationEvent(
        event_type=event_type,
        workflow_name="test-wf",
        message="Test message",
        severity=severity,
        details=details or {},
    )


class TestSlackChannelSend:
    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        event = _make_event(details={"tokens": 100, "nested": {"a": 1}})
        result = channel.send(event)
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_with_channel_override(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = SlackChannel(
            webhook_url="https://hooks.slack.com/test",
            channel="#alerts",
        )
        result = channel.send(_make_event())
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_failure(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        result = channel.send(_make_event())
        assert result is False

    def test_build_fields_with_details(self):
        channel = SlackChannel(webhook_url="https://example.com")
        event = _make_event(details={"tokens_used": 100, "duration_ms": 5000, "nested": {"a": 1}})
        fields = channel._build_fields(event)
        field_titles = [f["title"] for f in fields]
        assert "Event" in field_titles
        assert "Severity" in field_titles
        assert "Tokens Used" in field_titles
        # nested dict should not appear
        assert "Nested" not in field_titles

    def test_unknown_severity_color(self):
        channel = SlackChannel(webhook_url="https://example.com")
        assert channel._severity_to_color("unknown") == "#95a5a6"

    def test_unknown_event_emoji(self):
        channel = SlackChannel(webhook_url="https://example.com")
        assert channel._event_emoji(EventType.SCHEDULE_TRIGGERED) == ":alarm_clock:"


class TestDiscordChannelSend:
    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = DiscordChannel(webhook_url="https://discord.com/api/webhooks/test")
        result = channel.send(_make_event())
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_with_avatar(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = DiscordChannel(
            webhook_url="https://discord.com/api/webhooks/test",
            avatar_url="https://example.com/avatar.png",
        )
        result = channel.send(_make_event())
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_failure(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("timeout")

        channel = DiscordChannel(webhook_url="https://discord.com/api/webhooks/test")
        result = channel.send(_make_event())
        assert result is False

    def test_build_fields(self):
        channel = DiscordChannel(webhook_url="https://example.com")
        event = _make_event(details={"duration_ms": 5000})
        fields = channel._build_fields(event)
        assert any(f["name"] == "Duration Ms" for f in fields)

    def test_event_emojis(self):
        channel = DiscordChannel(webhook_url="https://example.com")
        assert channel._event_emoji(EventType.BUDGET_WARNING) == "\U0001f4b0"

    def test_unknown_severity(self):
        channel = DiscordChannel(webhook_url="https://example.com")
        assert channel._severity_to_color("unknown") == 9807270


class TestWebhookChannelSend:
    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = WebhookChannel(url="https://example.com/hook")
        result = channel.send(_make_event())
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_with_custom_headers(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = WebhookChannel(
            url="https://example.com/hook",
            headers={"X-Custom": "value"},
        )
        result = channel.send(_make_event())
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_failure(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("refused")

        channel = WebhookChannel(url="https://example.com/hook")
        result = channel.send(_make_event())
        assert result is False

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_non_2xx(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = WebhookChannel(url="https://example.com/hook")
        result = channel.send(_make_event())
        assert result is False


class TestEmailChannelSend:
    @patch("smtplib.SMTP")
    def test_send_success_with_tls_and_auth(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        channel = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_addr="bot@example.com",
            to_addrs=["admin@example.com"],
            use_tls=True,
        )
        result = channel.send(_make_event())
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")

    @patch("smtplib.SMTP")
    def test_send_without_tls(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        channel = EmailChannel(
            smtp_host="smtp.example.com",
            to_addrs=["admin@example.com"],
            use_tls=False,
        )
        result = channel.send(_make_event())
        assert result is True
        mock_server.starttls.assert_not_called()

    @patch("smtplib.SMTP")
    def test_send_without_auth(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        channel = EmailChannel(
            smtp_host="smtp.example.com",
            to_addrs=["admin@example.com"],
            use_tls=False,
        )
        result = channel.send(_make_event())
        assert result is True
        mock_server.login.assert_not_called()

    @patch("smtplib.SMTP")
    def test_send_smtp_error(self, mock_smtp_cls):
        mock_smtp_cls.side_effect = Exception("Connection refused")

        channel = EmailChannel(
            smtp_host="smtp.example.com",
            to_addrs=["admin@example.com"],
        )
        result = channel.send(_make_event())
        assert result is False

    def test_event_emojis(self):
        channel = EmailChannel(smtp_host="smtp.example.com")
        assert "✅" in channel._event_emoji(EventType.WORKFLOW_COMPLETED)
        assert "🔔" in channel._event_emoji(
            EventType.SCHEDULE_TRIGGERED
        ) or "⏰" in channel._event_emoji(EventType.SCHEDULE_TRIGGERED)


class TestTeamsChannelSend:
    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = TeamsChannel(webhook_url="https://outlook.office.com/webhook/test")
        result = channel.send(_make_event(event_type=EventType.WORKFLOW_FAILED, severity="error"))
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_failure(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("timeout")

        channel = TeamsChannel(webhook_url="https://outlook.office.com/webhook/test")
        result = channel.send(_make_event())
        assert result is False

    def test_unknown_severity(self):
        channel = TeamsChannel(webhook_url="https://example.com")
        assert channel._severity_to_color("unknown") == "808080"


class TestPagerDutyChannelSend:
    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_error_event(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 202
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = PagerDutyChannel(routing_key="test-key")
        event = _make_event(event_type=EventType.WORKFLOW_FAILED, severity="error")
        result = channel.send(event)
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_warning_event(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 202
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = PagerDutyChannel(routing_key="test-key")
        event = _make_event(event_type=EventType.BUDGET_EXCEEDED, severity="warning")
        result = channel.send(event)
        assert result is True

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_send_failure(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("timeout")

        channel = PagerDutyChannel(routing_key="test-key")
        event = _make_event(event_type=EventType.WORKFLOW_FAILED, severity="error")
        result = channel.send(event)
        assert result is False

    @patch("animus_forge.notifications.notifier.urlopen")
    def test_resolve(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 202
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = PagerDutyChannel(routing_key="test-key")
        result = channel.resolve("test-wf", EventType.WORKFLOW_FAILED)
        assert result is True

    def test_unknown_severity_mapping(self):
        channel = PagerDutyChannel(routing_key="test-key")
        assert channel._map_severity("unknown") == "info"


class TestNotifierChannelException:
    def test_channel_exception_caught(self):
        """If a channel raises an exception (not just returns False), notify catches it."""
        notifier = Notifier()

        class ExplodingChannel:
            def name(self):
                return "exploding"

            def send(self, event):
                raise RuntimeError("Boom")

        notifier.add_channel(ExplodingChannel())
        result = notifier.notify(_make_event())
        assert result["exploding"] is False

    def test_remove_nonexistent_channel(self):
        notifier = Notifier()
        assert notifier.remove_channel("nonexistent") is False
