"""Coverage push tests for low-coverage modules.

Targets: gmail.py, oauth.py, calendar.py, sync/client.py, learning/preferences.py
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus.integrations.oauth import OAuth2Token

# ===================================================================
# Gmail Coverage (51% → 85%+)
# ===================================================================


class TestGmailCoveragePush:
    """Cover uncovered lines in gmail.py."""

    def test_connect_missing_credentials(self, tmp_path: Path):
        """Lines 90-92: missing client_id/client_secret."""
        import animus.integrations.google.gmail as gmail_mod
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        with (
            patch.object(gmail_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(gmail_mod, "GOOGLE_AUTH_AVAILABLE", True),
        ):
            result = asyncio.run(gmail.connect({}))
        assert result is False
        assert "client_id" in gmail._error_message

    def test_connect_token_in_credentials(self, tmp_path: Path):
        """Line 98-99: token passed in credentials dict."""
        import animus.integrations.google.gmail as gmail_mod
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        token_dict = {
            "access_token": "tok",
            "refresh_token": "ref",
            "token_type": "Bearer",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            "scopes": [],
        }
        with (
            patch.object(gmail_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(gmail_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(gmail_mod, "load_token", return_value=None),
            patch.object(gmail_mod, "Credentials", create=True, return_value=MagicMock()),
            patch.object(gmail_mod, "build", create=True, return_value=MagicMock()),
        ):
            result = asyncio.run(
                gmail.connect({"client_id": "id", "client_secret": "s", "token": token_dict})
            )
        assert result is True
        assert gmail._token is not None
        assert gmail._token.access_token == "tok"

    def test_connect_expired_with_refresh(self, tmp_path: Path):
        """Lines 102-106: expired token triggers refresh."""
        import animus.integrations.google.gmail as gmail_mod
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        expired_token = OAuth2Token(
            access_token="old",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() - timedelta(hours=1),
            scopes=[],
        )
        refreshed_token = OAuth2Token(
            access_token="new",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            scopes=[],
        )
        mock_flow = MagicMock()
        mock_flow.refresh_token.return_value = refreshed_token

        with (
            patch.object(gmail_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(gmail_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(gmail_mod, "load_token", return_value=expired_token),
            patch.object(gmail_mod, "OAuth2Flow", return_value=mock_flow),
            patch.object(gmail_mod, "save_token"),
            patch.object(gmail_mod, "Credentials", create=True, return_value=MagicMock()),
            patch.object(gmail_mod, "build", create=True, return_value=MagicMock()),
        ):
            result = asyncio.run(gmail.connect({"client_id": "id", "client_secret": "s"}))
        assert result is True

    def test_connect_no_token_full_flow(self, tmp_path: Path):
        """Lines 108-119: no token, run full OAuth flow."""
        import animus.integrations.google.gmail as gmail_mod
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        new_token = OAuth2Token(
            access_token="new",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            scopes=[],
        )
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_token

        with (
            patch.object(gmail_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(gmail_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(gmail_mod, "load_token", return_value=None),
            patch.object(gmail_mod, "OAuth2Flow", return_value=mock_flow),
            patch.object(gmail_mod, "save_token"),
            patch.object(gmail_mod, "Credentials", create=True, return_value=MagicMock()),
            patch.object(gmail_mod, "build", create=True, return_value=MagicMock()),
        ):
            result = asyncio.run(gmail.connect({"client_id": "id", "client_secret": "s"}))
        assert result is True

    def test_connect_flow_fails(self, tmp_path: Path):
        """Lines 113-115: OAuth flow returns None."""
        import animus.integrations.google.gmail as gmail_mod
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = None

        with (
            patch.object(gmail_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(gmail_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(gmail_mod, "load_token", return_value=None),
            patch.object(gmail_mod, "OAuth2Flow", return_value=mock_flow),
        ):
            result = asyncio.run(gmail.connect({"client_id": "id", "client_secret": "s"}))
        assert result is False

    def test_disconnect_removes_token_file(self, tmp_path: Path):
        """Line 143-144: disconnect unlinks token file."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        token_file = tmp_path / "gmail_token.json"
        token_file.write_text("{}")
        assert token_file.exists()

        asyncio.run(gmail.disconnect())
        assert not token_file.exists()

    def test_verify_success(self):
        """Lines 153-155: verify succeeds."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.getProfile.return_value.execute.return_value = {}
        gmail._service = mock_service

        assert asyncio.run(gmail.verify()) is True

    def test_verify_expired(self):
        """Lines 156-158: verify catches exception, sets expired."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.getProfile.return_value.execute.side_effect = Exception(
            "expired"
        )
        gmail._service = mock_service

        assert asyncio.run(gmail.verify()) is False

    def test_parse_message_direct_body(self):
        """Lines 265-268: body from payload.body.data."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        body_text = "Hello, world!"
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        message = {
            "id": "m1",
            "threadId": "t1",
            "snippet": "Hello",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [{"name": "Subject", "value": "Test"}],
                "body": {"data": encoded},
            },
        }
        parsed = gmail._parse_message(message)
        assert "Hello, world!" in parsed["body"]
        assert parsed["id"] == "m1"

    def test_parse_message_multipart(self):
        """Lines 269-275: body from multipart parts."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        body_text = "Multipart body"
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        message = {
            "id": "m2",
            "payload": {
                "headers": [],
                "parts": [
                    {"mimeType": "text/html", "body": {"data": "aHRtbA=="}},
                    {"mimeType": "text/plain", "body": {"data": encoded}},
                ],
            },
        }
        parsed = gmail._parse_message(message)
        assert "Multipart body" in parsed["body"]

    def test_tool_list_inbox_unread_only(self):
        """Line 300-301: unread_only=True appends 'is:unread'."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": []
        }
        gmail._service = mock_service

        asyncio.run(gmail._tool_list_inbox(unread_only=True))
        call_args = mock_service.users().messages().list.call_args
        assert "is:unread" in call_args[1]["q"]

    def test_tool_list_inbox_error(self):
        """Lines 342-348: list inbox exception."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.list.return_value.execute.side_effect = RuntimeError(
            "API error"
        )
        gmail._service = mock_service

        result = asyncio.run(gmail._tool_list_inbox())
        assert result.success is False
        assert "Failed to list inbox" in result.error

    def test_tool_read_email_success(self):
        """Lines 357-366: read email happy path."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        body_text = "Email body"
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "m1",
            "threadId": "t1",
            "snippet": "snippet",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test"},
                    {"name": "From", "value": "alice@test.com"},
                ],
                "body": {"data": encoded},
            },
        }
        gmail._service = mock_service

        result = asyncio.run(gmail._tool_read_email("m1"))
        assert result.success is True
        assert result.output["subject"] == "Test"
        assert "Email body" in result.output["body"]

    def test_tool_search_success(self):
        """Lines 382-421: search happy path."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "m1"}]
        }
        mock_service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "m1",
            "snippet": "Found it",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Result"},
                    {"name": "From", "value": "bob@test.com"},
                    {"name": "Date", "value": "2025-01-01"},
                ]
            },
        }
        gmail._service = mock_service

        result = asyncio.run(gmail._tool_search("from:bob"))
        assert result.success is True
        assert result.output["count"] == 1
        assert result.output["query"] == "from:bob"

    def test_tool_search_error(self):
        """Lines 422-425: search exception."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.list.return_value.execute.side_effect = RuntimeError(
            "API"
        )
        gmail._service = mock_service

        result = asyncio.run(gmail._tool_search("query"))
        assert result.success is False
        assert "Search failed" in result.error

    def test_tool_draft_email_success(self):
        """Lines 434-457: draft email happy path."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
            "id": "draft-1"
        }
        gmail._service = mock_service

        result = asyncio.run(gmail._tool_draft_email("to@test.com", "Subject", "Body"))
        assert result.success is True
        assert result.output["draft_id"] == "draft-1"

    def test_tool_draft_email_error(self):
        """Lines 458-464: draft email exception."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.drafts.return_value.create.return_value.execute.side_effect = RuntimeError(
            "quota"
        )
        gmail._service = mock_service

        result = asyncio.run(gmail._tool_draft_email("to@test.com", "Subject", "Body"))
        assert result.success is False
        assert "Failed to create draft" in result.error

    def test_tool_send_email_error(self):
        """Lines 492-498: send email exception."""
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.send.return_value.execute.side_effect = RuntimeError(
            "send failed"
        )
        gmail._service = mock_service

        result = asyncio.run(gmail._tool_send_email("to@test.com", "Sub", "Body"))
        assert result.success is False
        assert "Failed to send email" in result.error


# ===================================================================
# OAuth2 Coverage (59% → 85%+)
# ===================================================================


class TestOAuth2FlowCoverage:
    """Cover uncovered lines in oauth.py."""

    def test_flow_init_with_google_auth(self):
        """Lines 149-153: constructor when GOOGLE_AUTH_AVAILABLE=True."""
        import animus.integrations.oauth as oauth_mod

        with patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True):
            flow = oauth_mod.OAuth2Flow("my-id", "my-secret", ["scope1"])
        assert flow.client_id == "my-id"
        assert flow.client_secret == "my-secret"
        assert flow.scopes == ["scope1"]
        assert flow.redirect_port == 8422
        assert flow.redirect_uri == "http://localhost:8422"

    def test_run_local_server_success(self):
        """Lines 166-231: full OAuth browser flow success."""
        import animus.integrations.oauth as oauth_mod

        mock_credentials = MagicMock()
        mock_credentials.token = "access-tok"
        mock_credentials.refresh_token = "refresh-tok"
        mock_credentials.expiry = datetime(2099, 1, 1)
        mock_credentials.scopes = ["scope1"]

        mock_flow_instance = MagicMock()
        mock_flow_instance.authorization_url.return_value = ("https://auth.url", "state")
        mock_flow_instance.credentials = mock_credentials

        with (
            patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(oauth_mod, "Flow", create=True) as mock_flow_cls,
            patch.object(oauth_mod, "HTTPServer"),
            patch.object(oauth_mod, "Thread") as mock_thread_cls,
            patch("webbrowser.open"),
        ):
            mock_flow_cls.from_client_config.return_value = mock_flow_instance

            # Simulate the callback handler receiving a code
            def join_side_effect(timeout=None):
                oauth_mod.OAuth2CallbackHandler.authorization_code = "auth-code-123"

            mock_thread = MagicMock()
            mock_thread.join.side_effect = join_side_effect
            mock_thread_cls.return_value = mock_thread

            flow = oauth_mod.OAuth2Flow("id", "secret", ["scope1"])
            token = flow.run_local_server()

        assert token is not None
        assert token.access_token == "access-tok"
        assert token.refresh_token == "refresh-tok"

    def test_run_local_server_error_callback(self):
        """Lines 208-210: OAuth callback returns error."""
        import animus.integrations.oauth as oauth_mod

        with (
            patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(oauth_mod, "Flow", create=True) as mock_flow_cls,
            patch.object(oauth_mod, "HTTPServer"),
            patch.object(oauth_mod, "Thread") as mock_thread_cls,
            patch("webbrowser.open"),
        ):
            mock_flow_cls.from_client_config.return_value = MagicMock(
                authorization_url=MagicMock(return_value=("https://auth", "state"))
            )

            def join_side_effect(timeout=None):
                oauth_mod.OAuth2CallbackHandler.error = "access_denied"
                oauth_mod.OAuth2CallbackHandler.authorization_code = None

            mock_thread = MagicMock()
            mock_thread.join.side_effect = join_side_effect
            mock_thread_cls.return_value = mock_thread

            flow = oauth_mod.OAuth2Flow("id", "secret", ["scope1"])
            token = flow.run_local_server()

        assert token is None

    def test_run_local_server_no_code(self):
        """Lines 212-214: no authorization code received (timeout)."""
        import animus.integrations.oauth as oauth_mod

        with (
            patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(oauth_mod, "Flow", create=True) as mock_flow_cls,
            patch.object(oauth_mod, "HTTPServer"),
            patch.object(oauth_mod, "Thread") as mock_thread_cls,
            patch("webbrowser.open"),
        ):
            mock_flow_cls.from_client_config.return_value = MagicMock(
                authorization_url=MagicMock(return_value=("https://auth", "state"))
            )

            def join_side_effect(timeout=None):
                oauth_mod.OAuth2CallbackHandler.authorization_code = None
                oauth_mod.OAuth2CallbackHandler.error = None

            mock_thread = MagicMock()
            mock_thread.join.side_effect = join_side_effect
            mock_thread_cls.return_value = mock_thread

            flow = oauth_mod.OAuth2Flow("id", "secret", ["scope1"])
            token = flow.run_local_server()

        assert token is None

    def test_run_local_server_fetch_token_exception(self):
        """Lines 232-234: fetch_token raises exception."""
        import animus.integrations.oauth as oauth_mod

        mock_flow_instance = MagicMock()
        mock_flow_instance.authorization_url.return_value = ("https://auth", "state")
        mock_flow_instance.fetch_token.side_effect = RuntimeError("exchange failed")

        with (
            patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(oauth_mod, "Flow", create=True) as mock_flow_cls,
            patch.object(oauth_mod, "HTTPServer"),
            patch.object(oauth_mod, "Thread") as mock_thread_cls,
            patch("webbrowser.open"),
        ):
            mock_flow_cls.from_client_config.return_value = mock_flow_instance

            def join_side_effect(timeout=None):
                oauth_mod.OAuth2CallbackHandler.authorization_code = "code"

            mock_thread = MagicMock()
            mock_thread.join.side_effect = join_side_effect
            mock_thread_cls.return_value = mock_thread

            flow = oauth_mod.OAuth2Flow("id", "secret", ["scope1"])
            token = flow.run_local_server()

        assert token is None

    def test_refresh_token_no_refresh_token(self):
        """Lines 246-248: no refresh token available."""
        import animus.integrations.oauth as oauth_mod

        token = OAuth2Token(
            access_token="tok",
            refresh_token=None,
            token_type="Bearer",
            expires_at=datetime.now() - timedelta(hours=1),
            scopes=[],
        )
        with patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True):
            flow = oauth_mod.OAuth2Flow("id", "secret", ["scope"])
        result = flow.refresh_token(token)
        assert result is None

    def test_refresh_token_success(self):
        """Lines 250-272: refresh succeeds."""
        import animus.integrations.oauth as oauth_mod

        token = OAuth2Token(
            access_token="old",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() - timedelta(hours=1),
            scopes=["scope1"],
        )

        mock_creds = MagicMock()
        mock_creds.token = "new-access"
        mock_creds.refresh_token = "new-refresh"
        mock_creds.expiry = datetime(2099, 1, 1)

        with (
            patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(oauth_mod, "Credentials", create=True, return_value=mock_creds),
            patch.object(oauth_mod, "Request", create=True, return_value=MagicMock()),
        ):
            flow = oauth_mod.OAuth2Flow("id", "secret", ["scope1"])
            result = flow.refresh_token(token)

        assert result is not None
        assert result.access_token == "new-access"
        assert result.refresh_token == "new-refresh"

    def test_refresh_token_exception(self):
        """Lines 273-275: refresh raises exception."""
        import animus.integrations.oauth as oauth_mod

        token = OAuth2Token(
            access_token="old",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=None,
            scopes=[],
        )

        mock_creds = MagicMock()
        mock_creds.refresh.side_effect = RuntimeError("network error")

        with (
            patch.object(oauth_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(oauth_mod, "Credentials", create=True, return_value=mock_creds),
            patch.object(oauth_mod, "Request", create=True, return_value=MagicMock()),
        ):
            flow = oauth_mod.OAuth2Flow("id", "secret", ["scope"])
            result = flow.refresh_token(token)

        assert result is None


# ===================================================================
# Calendar Coverage (61% → 85%+)
# ===================================================================


class TestCalendarCoveragePush:
    """Cover uncovered lines in calendar.py."""

    def test_connect_missing_credentials(self, tmp_path: Path):
        """Lines 88-90."""
        import animus.integrations.google.calendar as cal_mod
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration(data_dir=tmp_path)
        with (
            patch.object(cal_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(cal_mod, "GOOGLE_AUTH_AVAILABLE", True),
        ):
            result = asyncio.run(cal.connect({}))
        assert result is False

    def test_connect_token_in_credentials(self, tmp_path: Path):
        """Lines 96-97."""
        import animus.integrations.google.calendar as cal_mod
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration(data_dir=tmp_path)
        token_dict = {
            "access_token": "tok",
            "refresh_token": "ref",
            "token_type": "Bearer",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            "scopes": [],
        }
        with (
            patch.object(cal_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(cal_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(cal_mod, "load_token", return_value=None),
            patch.object(cal_mod, "Credentials", create=True, return_value=MagicMock()),
            patch.object(cal_mod, "build", create=True, return_value=MagicMock()),
        ):
            result = asyncio.run(
                cal.connect({"client_id": "id", "client_secret": "s", "token": token_dict})
            )
        assert result is True

    def test_connect_expired_with_refresh(self, tmp_path: Path):
        """Lines 100-104."""
        import animus.integrations.google.calendar as cal_mod
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration(data_dir=tmp_path)
        expired_token = OAuth2Token(
            access_token="old",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() - timedelta(hours=1),
            scopes=[],
        )
        refreshed_token = OAuth2Token(
            access_token="new",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            scopes=[],
        )
        mock_flow = MagicMock()
        mock_flow.refresh_token.return_value = refreshed_token

        with (
            patch.object(cal_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(cal_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(cal_mod, "load_token", return_value=expired_token),
            patch.object(cal_mod, "OAuth2Flow", return_value=mock_flow),
            patch.object(cal_mod, "save_token"),
            patch.object(cal_mod, "Credentials", create=True, return_value=MagicMock()),
            patch.object(cal_mod, "build", create=True, return_value=MagicMock()),
        ):
            result = asyncio.run(cal.connect({"client_id": "id", "client_secret": "s"}))
        assert result is True

    def test_connect_full_flow(self, tmp_path: Path):
        """Lines 106-117: no token, full OAuth flow."""
        import animus.integrations.google.calendar as cal_mod
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration(data_dir=tmp_path)
        new_token = OAuth2Token(
            access_token="new",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            scopes=[],
        )
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_token

        with (
            patch.object(cal_mod, "GOOGLE_API_AVAILABLE", True),
            patch.object(cal_mod, "GOOGLE_AUTH_AVAILABLE", True),
            patch.object(cal_mod, "load_token", return_value=None),
            patch.object(cal_mod, "OAuth2Flow", return_value=mock_flow),
            patch.object(cal_mod, "save_token"),
            patch.object(cal_mod, "Credentials", create=True, return_value=MagicMock()),
            patch.object(cal_mod, "build", create=True, return_value=MagicMock()),
        ):
            result = asyncio.run(cal.connect({"client_id": "id", "client_secret": "s"}))
        assert result is True

    def test_disconnect_removes_token_file(self, tmp_path: Path):
        """Line 142-143."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration(data_dir=tmp_path)
        token_file = tmp_path / "google_calendar_token.json"
        token_file.write_text("{}")

        asyncio.run(cal.disconnect())
        assert not token_file.exists()

    def test_verify_success(self):
        """Lines 152-154."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        cal._service = mock_service
        assert asyncio.run(cal.verify()) is True

    def test_verify_expired(self):
        """Lines 155-157."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.calendarList.return_value.list.return_value.execute.side_effect = Exception(
            "expired"
        )
        cal._service = mock_service
        assert asyncio.run(cal.verify()) is False

    def test_tool_list_events_error(self):
        """Lines 313-314."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.events.return_value.list.return_value.execute.side_effect = RuntimeError("API")
        cal._service = mock_service
        result = asyncio.run(cal._tool_list_events())
        assert result.success is False
        assert "Failed to list events" in result.error

    def test_tool_create_event_with_desc_and_location(self):
        """Lines 350-353."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.events.return_value.insert.return_value.execute.return_value = {
            "id": "e1",
            "summary": "Meeting",
            "htmlLink": "https://cal/e1",
            "start": {"dateTime": "2025-01-01T10:00:00"},
            "end": {"dateTime": "2025-01-01T11:00:00"},
        }
        cal._service = mock_service

        result = asyncio.run(
            cal._tool_create_event(
                "Meeting",
                "2025-01-01T10:00:00",
                "2025-01-01T11:00:00",
                description="Discuss Q1",
                location="Room 42",
            )
        )
        assert result.success is True
        # Verify description and location were included in the API call
        call_args = mock_service.events().insert.call_args
        body = call_args[1]["body"]
        assert body["description"] == "Discuss Q1"
        assert body["location"] == "Room 42"

    def test_tool_create_event_error(self):
        """Lines 368-369."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.events.return_value.insert.return_value.execute.side_effect = RuntimeError(
            "quota"
        )
        cal._service = mock_service

        result = asyncio.run(
            cal._tool_create_event("Event", "2025-01-01T10:00:00", "2025-01-01T11:00:00")
        )
        assert result.success is False

    def test_tool_check_availability_success(self):
        """Lines 391-414."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.freebusy.return_value.query.return_value.execute.return_value = {
            "calendars": {
                "primary": {
                    "busy": [{"start": "2025-01-01T10:00:00Z", "end": "2025-01-01T11:00:00Z"}]
                }
            }
        }
        cal._service = mock_service

        result = asyncio.run(
            cal._tool_check_availability("2025-01-01T09:00:00", "2025-01-01T12:00:00")
        )
        assert result.success is True
        assert result.output["busy_count"] == 1

    def test_tool_check_availability_error(self):
        """Lines 415-421."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.freebusy.return_value.query.return_value.execute.side_effect = RuntimeError(
            "API"
        )
        cal._service = mock_service

        result = asyncio.run(
            cal._tool_check_availability("2025-01-01T09:00:00", "2025-01-01T12:00:00")
        )
        assert result.success is False

    def test_tool_list_calendars_success(self):
        """Lines 433-454."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.calendarList.return_value.list.return_value.execute.return_value = {
            "items": [
                {"id": "primary", "summary": "My Cal", "primary": True, "accessRole": "owner"},
                {"id": "work", "summary": "Work", "primary": False, "accessRole": "writer"},
            ]
        }
        cal._service = mock_service

        result = asyncio.run(cal._tool_list_calendars())
        assert result.success is True
        assert result.output["count"] == 2

    def test_tool_list_calendars_no_service(self):
        """Lines 425-431."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        result = asyncio.run(cal._tool_list_calendars())
        assert result.success is False

    def test_tool_list_calendars_error(self):
        """Lines 455-461."""
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.calendarList.return_value.list.return_value.execute.side_effect = RuntimeError(
            "API"
        )
        cal._service = mock_service

        result = asyncio.run(cal._tool_list_calendars())
        assert result.success is False


# ===================================================================
# Sync Client Coverage (62% → 85%+)
# ===================================================================


class TestSyncClientCoverage:
    """Cover uncovered lines in sync/client.py."""

    def _make_client(self):
        from animus.sync.client import SyncClient

        state = MagicMock()
        state.device_id = "dev-1"
        state.version = 1
        state.get_peer_version.return_value = 0
        state.collect_state.return_value = {"key": "val"}
        state.apply_delta.return_value = True
        client = SyncClient(state=state, shared_secret="test-secret")
        return client

    def test_disconnect_websocket_close_raises(self):
        """Lines 132-136: websocket close raises but doesn't propagate."""
        client = self._make_client()
        mock_ws = AsyncMock()
        mock_ws.close.side_effect = RuntimeError("already closed")
        client._websocket = mock_ws
        client._connected = True

        asyncio.run(client.disconnect())
        assert client._connected is False
        assert client._websocket is None

    def test_sync_snapshot_response(self):
        """Lines 154-246: full sync happy path."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        client._connected = True
        client._peer_device_id = "peer-1"

        mock_ws = AsyncMock()
        # Return a SNAPSHOT_RESPONSE then DELTA_ACK
        snapshot_msg = SyncMessage(
            type=MessageType.SNAPSHOT_RESPONSE,
            device_id="peer-1",
            payload={
                "snapshot": {
                    "id": "snap-1",
                    "device_id": "peer-1",
                    "version": 2,
                    "data": {"key": "peer-val"},
                    "checksum": "abc",
                    "timestamp": datetime.now().isoformat(),
                },
                "version": 2,
            },
        )
        ack_msg = SyncMessage(
            type=MessageType.DELTA_ACK,
            device_id="peer-1",
            payload={"success": True},
        )
        mock_ws.recv = AsyncMock(side_effect=[snapshot_msg.to_json(), ack_msg.to_json()])
        client._websocket = mock_ws

        # Make StateDelta.compute return non-empty deltas
        delta_with_changes = MagicMock()
        delta_with_changes.is_empty.return_value = False
        delta_with_changes.to_dict.return_value = {"changes": {"added": {"k": "v"}}}
        delta_with_changes.changes = {"added": {"k": "v"}, "modified": {}, "deleted": []}

        with patch("animus.sync.client.StateDelta") as mock_delta_cls:
            mock_delta_cls.compute.return_value = delta_with_changes

            result = asyncio.run(client.sync())

        assert result.success is True

    def test_sync_unexpected_response(self):
        """Lines 169-173: non-SNAPSHOT_RESPONSE message."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        client._connected = True
        client._peer_device_id = "peer-1"

        wrong_msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer-1",
            payload={"status": "ok"},
        )
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=wrong_msg.to_json())
        client._websocket = mock_ws

        result = asyncio.run(client.sync())
        assert result.success is False
        assert "Unexpected" in result.error

    def test_sync_timeout(self):
        """Lines 248-249: sync timeout."""
        client = self._make_client()
        client._connected = True
        client._peer_device_id = "peer-1"

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)
        client._websocket = mock_ws

        result = asyncio.run(client.sync())
        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_push_changes_success(self):
        """Lines 268-278: push delta and get ACK."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        client._connected = True

        ack_msg = SyncMessage(
            type=MessageType.DELTA_ACK,
            device_id="peer-1",
            payload={"success": True},
        )
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=ack_msg.to_json())
        client._websocket = mock_ws

        delta = MagicMock()
        delta.to_dict.return_value = {"changes": {}}

        result = asyncio.run(client.push_changes(delta))
        assert result is True

    def test_push_changes_wrong_response(self):
        """Line 280: wrong message type returns False."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        client._connected = True

        wrong_msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer-1",
            payload={},
        )
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=wrong_msg.to_json())
        client._websocket = mock_ws

        delta = MagicMock()
        delta.to_dict.return_value = {}

        result = asyncio.run(client.push_changes(delta))
        assert result is False

    def test_push_changes_exception(self):
        """Lines 282-284: push raises exception."""
        client = self._make_client()
        client._connected = True

        mock_ws = AsyncMock()
        mock_ws.send.side_effect = RuntimeError("disconnected")
        client._websocket = mock_ws

        delta = MagicMock()
        delta.to_dict.return_value = {}

        result = asyncio.run(client.push_changes(delta))
        assert result is False

    def test_ping_success(self):
        """Lines 303-304: ping returns RTT."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        client._connected = True

        pong_msg = SyncMessage(
            type=MessageType.PONG,
            device_id="peer-1",
            payload={},
        )
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=pong_msg.to_json())
        client._websocket = mock_ws

        result = asyncio.run(client.ping())
        assert result is not None
        assert isinstance(result, int)

    def test_ping_wrong_response(self):
        """Line 306: non-PONG returns None."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        client._connected = True

        wrong_msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer-1",
            payload={},
        )
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=wrong_msg.to_json())
        client._websocket = mock_ws

        result = asyncio.run(client.ping())
        assert result is None

    def test_handle_message_delta_push(self):
        """Lines 333-346: handle DELTA_PUSH with callback."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        callback = MagicMock()
        client.add_delta_callback(callback)

        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="peer-1",
            payload={
                "delta": {
                    "id": "delta-1",
                    "source_device": "peer-1",
                    "target_device": "dev-1",
                    "changes": {"added": {}, "modified": {}, "deleted": []},
                    "base_version": 1,
                    "new_version": 2,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

        asyncio.run(client._handle_message(msg))
        callback.assert_called_once()

    def test_handle_message_status(self):
        """Lines 348-350: handle STATUS message."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()

        msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer-1",
            payload={"status": "healthy"},
        )

        asyncio.run(client._handle_message(msg))  # should not raise

    def test_listen_processes_messages(self):
        """Lines 320-329: listen loop processes messages."""
        from animus.sync.protocol import MessageType, SyncMessage

        client = self._make_client()
        client._connected = True

        msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer-1",
            payload={"status": "ok"},
        )

        # Simulate async iterator that yields one message then stops
        mock_ws = MagicMock()

        async def _aiter():
            yield msg.to_json()

        mock_ws.__aiter__ = _aiter
        client._websocket = mock_ws

        asyncio.run(client.listen())

    def test_listen_connection_drops(self):
        """Lines 327-329: listen catches outer exception."""
        client = self._make_client()
        client._connected = True

        mock_ws = MagicMock()

        async def _aiter():
            raise ConnectionError("lost connection")
            yield  # noqa: F401 — make it a generator

        mock_ws.__aiter__ = _aiter
        client._websocket = mock_ws

        asyncio.run(client.listen())
        assert client._connected is False


# ===================================================================
# Preferences Coverage (64% → 85%+)
# ===================================================================


class TestPreferencesCoveragePush:
    """Cover uncovered lines in learning/preferences.py."""

    def test_preference_apply(self):
        """Lines 61-62."""
        from animus.learning.preferences import Preference

        pref = Preference.create(
            domain="communication",
            key="tone",
            value="formal",
            confidence=0.8,
            source_patterns=["p1"],
        )
        assert pref.last_applied is None
        assert pref.application_count == 0

        pref.apply()
        assert pref.last_applied is not None
        assert pref.application_count == 1

    def test_preference_from_dict_with_last_applied(self):
        """Lines 90-91: from_dict with non-None last_applied."""
        from animus.learning.preferences import Preference

        now = datetime.now()
        data = {
            "id": "p1",
            "domain": "tools",
            "key": "editor",
            "value": "vim",
            "confidence": 0.9,
            "source_patterns": ["pat-1"],
            "created_at": now.isoformat(),
            "last_applied": now.isoformat(),
            "application_count": 5,
            "metadata": {},
        }
        pref = Preference.from_dict(data)
        assert pref.last_applied is not None
        assert pref.application_count == 5

    def test_load_preferences_from_file(self, tmp_path: Path):
        """Lines 117-124: load preferences from disk."""
        from animus.learning.preferences import PreferenceEngine

        now = datetime.now()
        prefs = [
            {
                "id": "p1",
                "domain": "communication",
                "key": "tone",
                "value": "formal",
                "confidence": 0.8,
                "source_patterns": ["pat-1"],
                "created_at": now.isoformat(),
                "last_applied": None,
                "application_count": 0,
                "metadata": {},
            }
        ]
        (tmp_path / "preferences.json").write_text(json.dumps(prefs))

        engine = PreferenceEngine(tmp_path)
        assert len(engine._preferences) == 1
        assert "p1" in engine._preferences

    def test_load_preferences_corrupt_file(self, tmp_path: Path):
        """Lines 125-126: corrupt file logged, no crash."""
        from animus.learning.preferences import PreferenceEngine

        (tmp_path / "preferences.json").write_text("not json!")
        engine = PreferenceEngine(tmp_path)
        assert len(engine._preferences) == 0

    def test_infer_from_pattern_new(self, tmp_path: Path):
        """Lines 190-201: infer creates new preference."""
        from animus.learning.preferences import PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        pattern = MagicMock()
        pattern.id = "pat-1"
        pattern.description = "tone: formal writing style"
        pattern.confidence = 0.7

        from animus.learning.patterns import PatternType

        pattern.pattern_type = PatternType.PREFERENCE

        result = engine.infer_from_pattern(pattern)
        assert result is not None
        assert result.key == "tone"
        assert result.value == "formal writing style"
        assert len(engine._preferences) == 1

    def test_infer_from_pattern_reinforces_existing(self, tmp_path: Path):
        """Lines 182-187: second call reinforces confidence."""
        from animus.learning.preferences import PreferenceEngine

        engine = PreferenceEngine(tmp_path)

        from animus.learning.patterns import PatternType

        pattern1 = MagicMock()
        pattern1.id = "pat-1"
        pattern1.description = "tone: formal"
        pattern1.confidence = 0.5
        pattern1.pattern_type = PatternType.PREFERENCE

        pref1 = engine.infer_from_pattern(pattern1)
        original_confidence = pref1.confidence

        pattern2 = MagicMock()
        pattern2.id = "pat-2"
        pattern2.description = "tone: formal"
        pattern2.confidence = 0.6
        pattern2.pattern_type = PatternType.PREFERENCE

        pref2 = engine.infer_from_pattern(pattern2)
        assert pref2.id == pref1.id  # Same preference
        assert pref2.confidence == min(1.0, original_confidence + 0.1)
        assert "pat-2" in pref2.source_patterns

    def test_infer_domain_refinement_scheduling(self, tmp_path: Path):
        """Lines 162-163: keyword-based domain refinement for scheduling."""
        from animus.learning.preferences import PreferenceEngine

        engine = PreferenceEngine(tmp_path)

        from animus.learning.patterns import PatternType

        pattern = MagicMock()
        pattern.id = "pat-1"
        pattern.description = "prefers morning meetings"
        pattern.confidence = 0.6
        pattern.pattern_type = PatternType.FREQUENCY

        result = engine.infer_from_pattern(pattern)
        assert result.domain == "scheduling"

    def test_infer_domain_refinement_tools(self, tmp_path: Path):
        """Lines 166-167: keyword-based domain refinement for tools."""
        from animus.learning.preferences import PreferenceEngine

        engine = PreferenceEngine(tmp_path)

        from animus.learning.patterns import PatternType

        pattern = MagicMock()
        pattern.id = "pat-1"
        pattern.description = "prefers tool grep for searching"
        pattern.confidence = 0.6
        pattern.pattern_type = PatternType.FREQUENCY

        result = engine.infer_from_pattern(pattern)
        assert result.domain == "tools"

    def test_find_by_key(self, tmp_path: Path):
        """Lines 206-207: find existing preference by domain+key."""
        from animus.learning.preferences import Preference, PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        pref = Preference.create(
            domain="tools", key="editor", value="vim", confidence=0.8, source_patterns=["p1"]
        )
        engine._preferences[pref.id] = pref

        found = engine._find_by_key("tools", "editor")
        assert found is pref

        not_found = engine._find_by_key("tools", "nonexistent")
        assert not_found is None

    def test_apply_to_context_confident(self, tmp_path: Path):
        """Lines 226, 243-248: apply confident preferences to context."""
        from animus.learning.preferences import Preference, PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        pref = Preference.create(
            domain="communication",
            key="tone",
            value="formal",
            confidence=0.8,
            source_patterns=["p1"],
        )
        engine._preferences[pref.id] = pref

        context = {"user": "Alice"}
        modified = engine.apply_to_context(context, "communication")
        assert "preferences" in modified
        assert len(modified["preferences"]) == 1
        assert modified["preferences"][0]["key"] == "tone"
        assert pref.application_count == 1

    def test_apply_to_context_low_confidence_skipped(self, tmp_path: Path):
        """Low confidence (<0.6) not applied."""
        from animus.learning.preferences import Preference, PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        pref = Preference.create(
            domain="communication",
            key="tone",
            value="casual",
            confidence=0.3,
            source_patterns=["p1"],
        )
        engine._preferences[pref.id] = pref

        modified = engine.apply_to_context({}, "communication")
        assert "preferences" not in modified

    def test_update_confidence_found(self, tmp_path: Path):
        """Lines 266-270: update existing preference confidence."""
        from animus.learning.preferences import Preference, PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        pref = Preference.create(
            domain="tools", key="editor", value="vim", confidence=0.5, source_patterns=["p1"]
        )
        engine._preferences[pref.id] = pref

        result = engine.update_confidence(pref.id, 0.3)
        assert result is True
        assert pref.confidence == pytest.approx(0.8)

    def test_update_confidence_not_found(self, tmp_path: Path):
        """Line 271: ID not found returns False."""
        from animus.learning.preferences import PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        assert engine.update_confidence("nonexistent", 0.1) is False

    def test_update_confidence_clamped(self, tmp_path: Path):
        """Confidence clamped to [0.0, 1.0]."""
        from animus.learning.preferences import Preference, PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        pref = Preference.create(
            domain="tools", key="editor", value="vim", confidence=0.9, source_patterns=["p1"]
        )
        engine._preferences[pref.id] = pref

        engine.update_confidence(pref.id, 0.5)
        assert pref.confidence == 1.0

    def test_remove_preference_found(self, tmp_path: Path):
        """Lines 283-287."""
        from animus.learning.preferences import Preference, PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        pref = Preference.create(
            domain="tools", key="editor", value="vim", confidence=0.5, source_patterns=["p1"]
        )
        engine._preferences[pref.id] = pref

        result = engine.remove_preference(pref.id)
        assert result is True
        assert len(engine._preferences) == 0

    def test_remove_preference_not_found(self, tmp_path: Path):
        """Line 288."""
        from animus.learning.preferences import PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        assert engine.remove_preference("nonexistent") is False

    def test_get_statistics_with_data(self, tmp_path: Path):
        """Lines 296-310."""
        from animus.learning.preferences import Preference, PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        for i, (domain, conf) in enumerate([("communication", 0.8), ("tools", 0.6)]):
            pref = Preference.create(
                domain=domain,
                key=f"key{i}",
                value=f"val{i}",
                confidence=conf,
                source_patterns=[f"p{i}"],
            )
            pref.application_count = i + 1
            engine._preferences[pref.id] = pref

        stats = engine.get_statistics()
        assert stats["total"] == 2
        assert stats["by_domain"]["communication"] == 1
        assert stats["by_domain"]["tools"] == 1
        assert stats["total_applications"] == 3  # 1 + 2
        assert stats["avg_confidence"] == pytest.approx(0.7)  # (0.8 + 0.6) / 2

    def test_get_statistics_empty(self, tmp_path: Path):
        """Empty engine returns zeros."""
        from animus.learning.preferences import PreferenceEngine

        engine = PreferenceEngine(tmp_path)
        stats = engine.get_statistics()
        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0.0
