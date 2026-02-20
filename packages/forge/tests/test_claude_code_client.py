"""Tests for ClaudeCodeClient skill context injection."""

import json
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.api_clients.claude_code_client import (
    DEFAULT_ROLE_PROMPTS,
    DEFAULT_ROLE_SKILL_AGENTS,
    ClaudeCodeClient,
)


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock settings pointing to a temp directory."""
    settings = MagicMock()
    settings.claude_mode = "api"
    settings.anthropic_api_key = None  # Skip real client init
    settings.claude_cli_path = "claude"
    settings.base_dir = tmp_path
    (tmp_path / "config").mkdir()
    return settings


@pytest.fixture
def client_no_anthropic(mock_settings):
    """Create client with mocked settings and no anthropic SDK."""
    with (
        patch(
            "animus_forge.api_clients.claude_code_client.get_settings",
            return_value=mock_settings,
        ),
        patch("animus_forge.api_clients.claude_code_client.anthropic", None),
    ):
        return ClaudeCodeClient()


class TestSkillContextInjection:
    """Test that skill context is injected into role prompts."""

    def test_mapped_roles_get_skill_context(self, mock_settings):
        """Roles in DEFAULT_ROLE_SKILL_AGENTS get skill context appended."""
        fake_context = "# Skills for system agent\n## file_operations (v1.0.0)"
        mock_library = MagicMock()
        mock_library.build_skill_context.return_value = fake_context

        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
            patch("animus_forge.skills.SkillLibrary", return_value=mock_library),
        ):
            client = ClaudeCodeClient()

        # builder is in the default mapping
        assert "file_operations" in client.role_prompts["builder"]
        assert "---" in client.role_prompts["builder"]

    def test_unmapped_roles_no_skill_context(self, mock_settings):
        """Roles not in the mapping (e.g., tester) should not have skill context."""
        fake_context = "# Skills for system agent\n## file_operations (v1.0.0)"
        mock_library = MagicMock()
        mock_library.build_skill_context.return_value = fake_context

        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
            patch("animus_forge.skills.SkillLibrary", return_value=mock_library),
        ):
            client = ClaudeCodeClient()

        # tester is NOT in the default mapping
        original_tester = DEFAULT_ROLE_PROMPTS["tester"]
        assert client.role_prompts["tester"] == original_tester

    def test_graceful_fallback_skills_dir_missing(self, mock_settings):
        """Client initializes normally when SkillLibrary fails to load."""
        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
            patch(
                "animus_forge.skills.SkillLibrary",
                side_effect=FileNotFoundError("no skills dir"),
            ),
        ):
            client = ClaudeCodeClient()

        # Should still have default prompts, unmodified
        assert client.role_prompts["builder"] == DEFAULT_ROLE_PROMPTS["builder"]

    def test_graceful_fallback_import_error(self, mock_settings):
        """Client initializes normally when skills module can't be imported."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "animus_forge.skills":
                raise ImportError("no skills module")
            return real_import(name, *args, **kwargs)

        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
            patch("builtins.__import__", side_effect=mock_import),
        ):
            client = ClaudeCodeClient()

        assert client.role_prompts["builder"] == DEFAULT_ROLE_PROMPTS["builder"]

    def test_json_config_override(self, mock_settings, tmp_path):
        """role_skill_mapping.json overrides default mapping."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        mapping = {"tester": ["system"]}
        (config_dir / "role_skill_mapping.json").write_text(json.dumps(mapping))

        fake_context = "# Skills for system agent\n## file_operations (v1.0.0)"
        mock_library = MagicMock()
        mock_library.build_skill_context.return_value = fake_context

        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
            patch("animus_forge.skills.SkillLibrary", return_value=mock_library),
        ):
            client = ClaudeCodeClient()

        # tester should NOW have skill context (from JSON override)
        assert "file_operations" in client.role_prompts["tester"]
        # builder should NOT have skill context (not in JSON override)
        assert client.role_prompts["builder"] == DEFAULT_ROLE_PROMPTS["builder"]

    def test_empty_skill_context_not_appended(self, mock_settings):
        """Roles with empty skill context don't get separators appended."""
        mock_library = MagicMock()
        mock_library.build_skill_context.return_value = ""

        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
            patch("animus_forge.skills.SkillLibrary", return_value=mock_library),
        ):
            client = ClaudeCodeClient()

        # No separator should be appended when context is empty
        assert client.role_prompts["builder"] == DEFAULT_ROLE_PROMPTS["builder"]


class TestEnforcementIntegration:
    """Test enforcement is wired into execute_agent results."""

    def test_execute_agent_includes_enforcement_key(self, mock_settings):
        """Successful execution includes enforcement dict."""
        mock_library = MagicMock()
        mock_library.build_skill_context.return_value = ""
        mock_library.get_skills_for_agent.return_value = []

        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
            patch("animus_forge.skills.SkillLibrary", return_value=mock_library),
        ):
            client = ClaudeCodeClient()

        # Mock CLI execution to return output
        with (
            patch.object(client, "is_configured", return_value=True),
            patch.object(client, "_execute_via_cli", return_value="safe output"),
        ):
            client.mode = "cli"
            result = client.execute_agent("builder", "write tests")

        assert result["success"] is True
        assert "enforcement" in result
        assert result["enforcement"]["action"] == "allow"
        assert result["enforcement"]["passed"] is True

    def test_enforcement_failure_is_fail_open(self, mock_settings):
        """If enforcer raises, execution still succeeds."""
        with (
            patch(
                "animus_forge.api_clients.claude_code_client.get_settings",
                return_value=mock_settings,
            ),
            patch("animus_forge.api_clients.claude_code_client.anthropic", None),
        ):
            client = ClaudeCodeClient()

        # Force enforcer to raise
        with (
            patch.object(client, "is_configured", return_value=True),
            patch.object(client, "_execute_via_cli", return_value="output"),
            patch.object(
                type(client),
                "enforcer",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
                ),
            ),
        ):
            client.mode = "cli"
            result = client.execute_agent("builder", "task")

        assert result["success"] is True
        assert result["enforcement"]["action"] == "allow"


class TestDefaultRoleSkillAgents:
    """Verify the default mapping structure."""

    def test_expected_roles_present(self):
        assert "builder" in DEFAULT_ROLE_SKILL_AGENTS
        assert "planner" in DEFAULT_ROLE_SKILL_AGENTS
        assert "reviewer" in DEFAULT_ROLE_SKILL_AGENTS

    def test_builder_has_system_and_browser(self):
        assert "system" in DEFAULT_ROLE_SKILL_AGENTS["builder"]
        assert "browser" in DEFAULT_ROLE_SKILL_AGENTS["builder"]
