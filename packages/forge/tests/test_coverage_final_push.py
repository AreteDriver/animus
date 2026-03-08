"""Final coverage push tests targeting remaining gaps to reach 97%."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# GmailClient tests (16 missed lines, 78% → ~100%)
# ---------------------------------------------------------------------------


class TestGmailClient:
    """Cover gmail_client.py authenticate, list_messages, get_message, extract_email_body."""

    def _make_client(self):
        with patch("animus_forge.api_clients.gmail_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(gmail_credentials_path="/fake/creds.json")
            from animus_forge.api_clients.gmail_client import GmailClient

            return GmailClient()

    def test_authenticate_not_configured(self):
        with patch("animus_forge.api_clients.gmail_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(gmail_credentials_path=None)
            from animus_forge.api_clients.gmail_client import GmailClient

            client = GmailClient()
            assert not client.authenticate()

    def test_authenticate_success_new_creds(self):
        client = self._make_client()
        mock_flow = MagicMock()
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.to_json.return_value = '{"token": "abc"}'
        mock_flow.run_local_server.return_value = mock_creds
        mock_service = MagicMock()

        with (
            patch("os.path.exists", return_value=False),
            patch(
                "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
            patch("googleapiclient.discovery.build", return_value=mock_service),
        ):
            result = client.authenticate()
            assert result is True
            assert client.service is mock_service

    def test_authenticate_exception_returns_false(self):
        client = self._make_client()
        with patch("os.path.exists", side_effect=RuntimeError("boom")):
            assert client.authenticate() is False

    def test_authenticate_with_existing_token(self):
        client = self._make_client()
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_service = MagicMock()

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file",
                return_value=mock_creds,
            ),
            patch("googleapiclient.discovery.build", return_value=mock_service),
        ):
            result = client.authenticate()
            assert result is True

    def test_authenticate_refresh_expired_token(self):
        client = self._make_client()
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_tok"
        mock_creds.to_json.return_value = '{"token": "new"}'
        mock_service = MagicMock()

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file",
                return_value=mock_creds,
            ),
            patch("googleapiclient.discovery.build", return_value=mock_service),
        ):
            result = client.authenticate()
            assert result is True
            mock_creds.refresh.assert_called_once()

    def test_authenticate_no_creds_runs_flow(self):
        client = self._make_client()
        mock_flow = MagicMock()
        mock_new_creds = MagicMock()
        mock_new_creds.valid = True
        mock_new_creds.to_json.return_value = '{"token": "new"}'
        mock_flow.run_local_server.return_value = mock_new_creds
        mock_service = MagicMock()

        mock_old_creds = MagicMock()
        mock_old_creds.valid = False
        mock_old_creds.expired = False  # Not expired, just invalid — triggers flow

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file",
                return_value=mock_old_creds,
            ),
            patch(
                "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
            patch("googleapiclient.discovery.build", return_value=mock_service),
            patch("os.open", return_value=99),
            patch("os.fdopen", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
        ):
            result = client.authenticate()
            assert result is True

    def test_list_messages_no_service(self):
        client = self._make_client()
        assert client.list_messages() == []

    def test_list_messages_with_service(self):
        client = self._make_client()
        client.service = MagicMock()
        msgs = [{"id": "1"}, {"id": "2"}]

        with patch.object(client, "_list_messages_with_retry", return_value=msgs):
            result = client.list_messages(max_results=5, query="test")
        assert result == msgs

    def test_list_messages_exception(self):
        client = self._make_client()
        client.service = MagicMock()
        with patch.object(client, "_list_messages_with_retry", side_effect=RuntimeError("fail")):
            assert client.list_messages() == []

    def test_get_message_no_service(self):
        client = self._make_client()
        assert client.get_message("123") is None

    def test_get_message_success(self):
        client = self._make_client()
        client.service = MagicMock()
        msg = {"id": "123", "payload": {}}
        with patch.object(client, "_get_message_with_retry", return_value=msg):
            assert client.get_message("123") == msg

    def test_get_message_exception(self):
        client = self._make_client()
        client.service = MagicMock()
        with patch.object(client, "_get_message_with_retry", side_effect=RuntimeError("fail")):
            assert client.get_message("123") is None

    def test_extract_email_body_with_parts(self):
        client = self._make_client()
        body_text = "Hello World"
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        message = {
            "payload": {
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": encoded}},
                ]
            }
        }
        assert client.extract_email_body(message) == body_text

    def test_extract_email_body_no_parts(self):
        client = self._make_client()
        body_text = "No parts body"
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        message = {"payload": {"body": {"data": encoded}}}
        assert client.extract_email_body(message) == body_text

    def test_extract_email_body_exception(self):
        client = self._make_client()
        assert client.extract_email_body({}) == ""


# ---------------------------------------------------------------------------
# Workflow CLI _output_yaml_results (lines 129-149)
# ---------------------------------------------------------------------------


class TestWorkflowOutputYaml:
    """Cover _output_yaml_results function."""

    def test_output_yaml_success_with_outputs(self):
        from animus_forge.cli.commands.workflow import _output_yaml_results

        result = MagicMock()
        result.status = "success"
        step = MagicMock()
        step.status.value = "success"
        step.step_id = "step1"
        step.tokens_used = 500
        result.steps = [step]
        result.total_tokens = 500
        result.error = None
        result.outputs = {"key": "value"}

        with patch("animus_forge.cli.commands.workflow.console"):
            _output_yaml_results(result)

    def test_output_yaml_error(self):
        from animus_forge.cli.commands.workflow import _output_yaml_results

        result = MagicMock()
        result.status = "failed"
        result.steps = []
        result.total_tokens = 0
        result.error = "Something went wrong"
        result.outputs = {}

        with patch("animus_forge.cli.commands.workflow.console"):
            _output_yaml_results(result)

    def test_output_yaml_no_tokens_no_outputs(self):
        from animus_forge.cli.commands.workflow import _output_yaml_results

        result = MagicMock()
        result.status = "success"
        step = MagicMock()
        step.status.value = "success"
        step.step_id = "step1"
        step.tokens_used = 0
        result.steps = [step]
        result.total_tokens = 0
        result.error = None
        result.outputs = None

        with patch("animus_forge.cli.commands.workflow.console"):
            _output_yaml_results(result)


# ---------------------------------------------------------------------------
# Admin CLI follow-logs (lines 232-244)
# ---------------------------------------------------------------------------


class TestAdminLogFollow:
    """Cover admin.py log_follow mode."""

    def test_logs_follow_keyboard_interrupt(self):
        from animus_forge.cli.commands.admin import logs

        mock_tracker = MagicMock()
        mock_tracker.get_logs.return_value = [
            {
                "timestamp": "2026-03-07T10:00:00",
                "level": "INFO",
                "message": "test",
                "workflow_id": "",
                "execution_id": "",
            }
        ]

        with (
            patch("animus_forge.cli.commands.admin.console"),
            patch("animus_forge.cli.commands.admin.get_tracker", return_value=mock_tracker),
            patch("time.sleep", side_effect=KeyboardInterrupt),
        ):
            # follow=True enters the loop, time.sleep raises KeyboardInterrupt
            logs(follow=True, json_output=False)


# ---------------------------------------------------------------------------
# Filesystem edge cases (14 missed lines)
# ---------------------------------------------------------------------------


class TestFilesystemEdgeCases:
    """Cover filesystem.py edge cases."""

    def _make_fs(self, tmp_path):
        from animus_forge.tools.filesystem import FilesystemTools
        from animus_forge.tools.safety import PathValidator

        validator = PathValidator(project_path=tmp_path)
        return FilesystemTools(validator=validator)

    def test_list_files_excluded_path(self, tmp_path):
        fs = self._make_fs(tmp_path)
        excluded = tmp_path / ".git"
        excluded.mkdir()
        (excluded / "config").write_text("test")
        (tmp_path / "real.py").write_text("real")

        result = fs.list_files(".")
        names = [e.name for e in result.entries]
        assert "real.py" in names

    def test_list_files_value_error(self, tmp_path):
        fs = self._make_fs(tmp_path)
        (tmp_path / "test.py").write_text("test")

        original_relative_to = Path.relative_to

        def flaky_relative_to(self_path, *args, **kwargs):
            if self_path.name == "test.py":
                raise ValueError("not relative")
            return original_relative_to(self_path, *args, **kwargs)

        with patch.object(Path, "relative_to", flaky_relative_to):
            result = fs.list_files(".")
            assert result is not None

    def test_search_code_excluded_file(self, tmp_path):
        fs = self._make_fs(tmp_path)
        (tmp_path / "test.py").write_text("hello world")

        with patch.object(fs.validator, "is_excluded", return_value=True):
            result = fs.search_code("hello")
            assert result.total_matches == 0

    def test_search_code_unreadable_file(self, tmp_path):
        fs = self._make_fs(tmp_path)
        f = tmp_path / "test.py"
        f.write_text("hello")

        original_read = Path.read_text

        def flaky_read(self_path, *args, **kwargs):
            if self_path.name == "test.py":
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            return original_read(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", flaky_read):
            result = fs.search_code("hello")
            assert result.total_matches == 0

    def test_search_code_security_error(self, tmp_path):
        fs = self._make_fs(tmp_path)
        (tmp_path / "test.py").write_text("hello")

        from animus_forge.tools.safety import SecurityError

        def deny_all(path):
            raise SecurityError("denied")

        with patch.object(fs.validator, "validate_file_for_read", deny_all):
            result = fs.search_code("hello")
            assert result.total_matches == 0

    def test_glob_files_excluded(self, tmp_path):
        fs = self._make_fs(tmp_path)
        (tmp_path / "test.py").write_text("test")

        with patch.object(fs.validator, "is_excluded", return_value=True):
            result = fs.glob_files("*.py")
            assert result == []

    def test_glob_files_value_error(self, tmp_path):
        fs = self._make_fs(tmp_path)
        (tmp_path / "test.py").write_text("test")

        original_relative_to = Path.relative_to

        def flaky_relative_to(self_path, *args, **kwargs):
            if self_path.name == "test.py":
                raise ValueError("not relative")
            return original_relative_to(self_path, *args, **kwargs)

        with patch.object(Path, "relative_to", flaky_relative_to):
            result = fs.glob_files("*.py")
            assert result == []


# ---------------------------------------------------------------------------
# Workflow CLI _validate_cli_required_fields
# ---------------------------------------------------------------------------


class TestWorkflowValidation:
    """Cover _validate_cli_required_fields in workflow.py."""

    def test_validate_missing_id_and_steps(self):
        from animus_forge.cli.commands.workflow import _validate_cli_required_fields

        errors, warnings = _validate_cli_required_fields({})
        assert any("id" in e for e in errors)
        assert any("steps" in e for e in errors)

    def test_validate_valid_data(self):
        from animus_forge.cli.commands.workflow import _validate_cli_required_fields

        errors, warnings = _validate_cli_required_fields({"id": "test", "steps": []})
        assert len(errors) == 0
