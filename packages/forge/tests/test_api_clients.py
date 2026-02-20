"""Tests for API client wrappers (GitHub, Gmail, Claude Code)."""

from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestGitHubClient:
    """Tests for GitHubClient."""

    @patch("animus_forge.api_clients.github_client.get_settings")
    @patch("animus_forge.api_clients.github_client.Github")
    def test_init_with_token(self, mock_github, mock_settings):
        """Test initialization with GitHub token."""
        mock_settings.return_value.github_token = "test_token"

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()

        assert client.is_configured()
        mock_github.assert_called_once_with("test_token")

    @patch("animus_forge.api_clients.github_client.get_settings")
    def test_init_without_token(self, mock_settings):
        """Test initialization without GitHub token."""
        mock_settings.return_value.github_token = None

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()

        assert not client.is_configured()
        assert client.client is None

    @patch("animus_forge.api_clients.github_client.get_settings")
    @patch("animus_forge.api_clients.github_client.Github")
    def test_create_issue_not_configured(self, mock_github, mock_settings):
        """Test create_issue when not configured."""
        mock_settings.return_value.github_token = None

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()
        result = client.create_issue("repo", "title", "body")

        assert result is None

    @patch("animus_forge.api_clients.github_client.get_settings")
    @patch("animus_forge.api_clients.github_client.Github")
    def test_create_issue_success(self, mock_github, mock_settings):
        """Test successful issue creation."""
        mock_settings.return_value.github_token = "token"

        mock_issue = MagicMock()
        mock_issue.number = 123
        mock_issue.html_url = "https://github.com/repo/issues/123"
        mock_issue.title = "Test Issue"

        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = mock_issue

        mock_github.return_value.get_repo.return_value = mock_repo

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()
        result = client.create_issue("owner/repo", "Test Issue", "Body", ["bug"])

        assert result["number"] == 123
        assert result["url"] == "https://github.com/repo/issues/123"
        mock_repo.create_issue.assert_called_once_with(
            title="Test Issue", body="Body", labels=["bug"]
        )

    @patch("animus_forge.api_clients.github_client.get_settings")
    @patch("animus_forge.api_clients.github_client.Github")
    def test_create_issue_exception(self, mock_github, mock_settings):
        """Test issue creation with exception."""
        from github import GithubException

        mock_settings.return_value.github_token = "token"
        mock_github.return_value.get_repo.side_effect = GithubException(404, "Not found", {})

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()
        result = client.create_issue("owner/repo", "Title", "Body")

        assert "error" in result

    @patch("animus_forge.api_clients.github_client.get_settings")
    @patch("animus_forge.api_clients.github_client.Github")
    def test_commit_file_creates_new(self, mock_github, mock_settings):
        """Test committing a new file."""
        from github import GithubException

        mock_settings.return_value.github_token = "token"

        mock_repo = MagicMock()
        # File doesn't exist
        mock_repo.get_contents.side_effect = GithubException(404, "Not found", {})

        mock_result = {
            "commit": MagicMock(sha="abc123"),
            "content": MagicMock(html_url="https://github.com/repo/blob/main/file.txt"),
        }
        mock_repo.create_file.return_value = mock_result

        mock_github.return_value.get_repo.return_value = mock_repo

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()
        result = client.commit_file("owner/repo", "file.txt", "content", "Add file")

        assert result["commit_sha"] == "abc123"
        mock_repo.create_file.assert_called_once()

    @patch("animus_forge.api_clients.github_client.get_settings")
    @patch("animus_forge.api_clients.github_client.Github")
    def test_commit_file_updates_existing(self, mock_github, mock_settings):
        """Test updating an existing file."""
        mock_settings.return_value.github_token = "token"

        mock_file = MagicMock(sha="old_sha")
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_file

        mock_result = {
            "commit": MagicMock(sha="new_sha"),
            "content": MagicMock(html_url="https://github.com/repo/blob/main/file.txt"),
        }
        mock_repo.update_file.return_value = mock_result

        mock_github.return_value.get_repo.return_value = mock_repo

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()
        result = client.commit_file("owner/repo", "file.txt", "new content", "Update file")

        assert result["commit_sha"] == "new_sha"
        mock_repo.update_file.assert_called_once()

    @patch("animus_forge.api_clients.github_client.get_settings")
    @patch("animus_forge.api_clients.github_client.Github")
    def test_list_repositories(self, mock_github, mock_settings):
        """Test listing repositories."""
        mock_settings.return_value.github_token = "token"

        mock_repos = [
            MagicMock(full_name="user/repo1", description="Repo 1", html_url="url1"),
            MagicMock(full_name="user/repo2", description="Repo 2", html_url="url2"),
        ]
        mock_github.return_value.get_user.return_value.get_repos.return_value = mock_repos

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()
        result = client.list_repositories()

        assert len(result) == 2
        assert result[0]["name"] == "user/repo1"

    @patch("animus_forge.api_clients.github_client.get_settings")
    def test_list_repositories_not_configured(self, mock_settings):
        """Test listing repos when not configured."""
        mock_settings.return_value.github_token = None

        from animus_forge.api_clients.github_client import GitHubClient

        client = GitHubClient()
        result = client.list_repositories()

        assert result == []


class TestGmailClient:
    """Tests for GmailClient."""

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_init_with_credentials(self, mock_settings):
        """Test initialization with credentials path."""
        mock_settings.return_value.gmail_credentials_path = "/path/to/creds.json"

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()

        assert client.is_configured()
        assert client.credentials_path == "/path/to/creds.json"

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_init_without_credentials(self, mock_settings):
        """Test initialization without credentials."""
        mock_settings.return_value.gmail_credentials_path = None

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()

        assert not client.is_configured()

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_authenticate_not_configured(self, mock_settings):
        """Test authenticate when not configured."""
        mock_settings.return_value.gmail_credentials_path = None

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()
        result = client.authenticate()

        assert result is False

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_list_messages_no_service(self, mock_settings):
        """Test list_messages when not authenticated."""
        mock_settings.return_value.gmail_credentials_path = "/path"

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()
        result = client.list_messages()

        assert result == []

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_get_message_no_service(self, mock_settings):
        """Test get_message when not authenticated."""
        mock_settings.return_value.gmail_credentials_path = "/path"

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()
        result = client.get_message("msg_id")

        assert result is None

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_extract_email_body_with_parts(self, mock_settings):
        """Test extracting email body from multipart message."""
        mock_settings.return_value.gmail_credentials_path = "/path"

        import base64

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()

        body_text = "Hello, World!"
        encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()

        message = {
            "payload": {
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": encoded_body}},
                    {"mimeType": "text/html", "body": {"data": "html"}},
                ]
            }
        }
        result = client.extract_email_body(message)

        assert result == "Hello, World!"

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_extract_email_body_simple(self, mock_settings):
        """Test extracting email body from simple message."""
        mock_settings.return_value.gmail_credentials_path = "/path"

        import base64

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()

        body_text = "Simple message"
        encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()

        message = {"payload": {"body": {"data": encoded_body}}}
        result = client.extract_email_body(message)

        assert result == "Simple message"

    @patch("animus_forge.api_clients.gmail_client.get_settings")
    def test_extract_email_body_error(self, mock_settings):
        """Test extract_email_body with invalid message."""
        mock_settings.return_value.gmail_credentials_path = "/path"

        from animus_forge.api_clients.gmail_client import GmailClient

        client = GmailClient()
        result = client.extract_email_body({})

        assert result == ""


class TestClaudeCodeClient:
    """Tests for ClaudeCodeClient."""

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_init_api_mode(self, mock_settings):
        """Test initialization in API mode."""
        mock_settings.return_value.claude_mode = "api"
        mock_settings.return_value.anthropic_api_key = "test_key"
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        with patch("animus_forge.api_clients.claude_code_client.anthropic") as mock_anthropic:
            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            client = ClaudeCodeClient()

            assert client.mode == "api"
            mock_anthropic.Anthropic.assert_called_once_with(api_key="test_key")

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_init_cli_mode(self, mock_settings):
        """Test initialization in CLI mode."""
        mock_settings.return_value.claude_mode = "cli"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "/usr/bin/claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()

        assert client.mode == "cli"
        assert client.client is None

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    @patch("animus_forge.api_clients.claude_code_client.subprocess.run")
    def test_is_configured_cli_mode(self, mock_run, mock_settings):
        """Test is_configured check in CLI mode."""
        mock_settings.return_value.claude_mode = "cli"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        mock_run.return_value = MagicMock(returncode=0)

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()
        assert client.is_configured()

        mock_run.assert_called_once()

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_is_configured_api_mode_no_client(self, mock_settings):
        """Test is_configured when API client is None."""
        mock_settings.return_value.claude_mode = "api"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()
        assert not client.is_configured()

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_execute_agent_not_configured(self, mock_settings):
        """Test execute_agent when not configured."""
        mock_settings.return_value.claude_mode = "api"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()
        result = client.execute_agent("planner", "test task")

        assert result["success"] is False
        assert "not configured" in result["error"]

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_execute_agent_unknown_role(self, mock_settings):
        """Test execute_agent with unknown role."""
        mock_settings.return_value.claude_mode = "api"
        mock_settings.return_value.anthropic_api_key = "key"
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        with patch("animus_forge.api_clients.claude_code_client.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()

            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            client = ClaudeCodeClient()
            # Clear role prompts to trigger unknown role
            client.role_prompts = {}
            result = client.execute_agent("unknown_role", "test task")

            assert result["success"] is False
            assert "Unknown role" in result["error"]

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_execute_agent_api_success(self, mock_settings):
        """Test successful agent execution via API."""
        mock_settings.return_value.claude_mode = "api"
        mock_settings.return_value.anthropic_api_key = "key"
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        with patch("animus_forge.api_clients.claude_code_client.anthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Generated plan")]
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            client = ClaudeCodeClient()
            result = client.execute_agent("planner", "Build a feature", context="Some context")

            assert result["success"] is True
            assert result["output"] == "Generated plan"
            assert result["role"] == "planner"

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    @patch("animus_forge.api_clients.claude_code_client.subprocess.run")
    def test_execute_agent_cli_success(self, mock_run, mock_settings):
        """Test successful agent execution via CLI."""
        mock_settings.return_value.claude_mode = "cli"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        # First call is for is_configured check, second is for actual execution
        mock_run.side_effect = [
            MagicMock(returncode=0),  # is_configured
            MagicMock(returncode=0, stdout="CLI output", stderr=""),  # execute
        ]

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()
        result = client.execute_agent("builder", "Write code")

        assert result["success"] is True
        assert result["output"] == "CLI output"

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_generate_completion_api(self, mock_settings):
        """Test generate_completion via API."""
        mock_settings.return_value.claude_mode = "api"
        mock_settings.return_value.anthropic_api_key = "key"
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        with patch("animus_forge.api_clients.claude_code_client.anthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Completion text")]
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            client = ClaudeCodeClient()
            result = client.generate_completion("Explain this", system_prompt="Be helpful")

            assert result["success"] is True
            assert result["output"] == "Completion text"

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    @patch("animus_forge.api_clients.claude_code_client.subprocess.run")
    def test_execute_cli_command(self, mock_run, mock_settings):
        """Test execute_cli_command."""
        mock_settings.return_value.claude_mode = "cli"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        mock_run.return_value = MagicMock(returncode=0, stdout="Command output", stderr="")

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()
        result = client.execute_cli_command("Do something", working_dir="/tmp", timeout=60)

        assert result["success"] is True
        assert result["output"] == "Command output"

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    @patch("animus_forge.api_clients.claude_code_client.subprocess.run")
    def test_execute_cli_command_error(self, mock_run, mock_settings):
        """Test execute_cli_command with error."""
        mock_settings.return_value.claude_mode = "cli"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error message")

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()
        result = client.execute_cli_command("Do something")

        assert result["success"] is False
        assert "error" in result

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_load_role_prompts_from_file(self, mock_settings):
        """Test loading role prompts from config file."""
        mock_settings.return_value.claude_mode = "cli"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_settings.return_value.base_dir.__truediv__.return_value = mock_path

        import json

        config_data = {
            "planner": {"system_prompt": "Custom planner prompt"},
            "builder": {"system_prompt": "Custom builder prompt"},
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            client = ClaudeCodeClient()

            assert client.role_prompts["planner"] == "Custom planner prompt"
            assert client.role_prompts["builder"] == "Custom builder prompt"

    @patch("animus_forge.api_clients.claude_code_client.get_settings")
    def test_execute_via_api_no_client(self, mock_settings):
        """Test _execute_via_api raises when client is None."""
        mock_settings.return_value.claude_mode = "api"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.claude_cli_path = "claude"
        mock_settings.return_value.base_dir = MagicMock()
        mock_settings.return_value.base_dir.__truediv__.return_value.exists.return_value = False

        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient()

        with pytest.raises(RuntimeError, match="not initialized"):
            client._execute_via_api("system", "user", "model", 1000)
