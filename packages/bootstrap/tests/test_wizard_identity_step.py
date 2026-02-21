"""Tests for the identity files wizard step and wizard integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from animus_bootstrap.setup.steps.identity_files import run_identity_files


class TestRunIdentityFiles:
    """Test the identity files step function."""

    @patch("animus_bootstrap.setup.steps.identity_files.Prompt.ask", return_value="I'm a developer")
    def test_returns_about_text(self, mock_ask):
        console = MagicMock()
        result = run_identity_files(console)
        assert result["about"] == "I'm a developer"
        assert result["generate_identity_files"] is True

    @patch("animus_bootstrap.setup.steps.identity_files.Prompt.ask", return_value="")
    def test_empty_about_is_valid(self, mock_ask):
        console = MagicMock()
        result = run_identity_files(console)
        assert result["about"] == ""
        assert result["generate_identity_files"] is True

    @patch("animus_bootstrap.setup.steps.identity_files.Prompt.ask", return_value="")
    def test_prints_file_descriptions(self, mock_ask):
        console = MagicMock()
        run_identity_files(console)
        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "CORE_VALUES.md" in printed
        assert "IDENTITY.md" in printed
        assert "LEARNED.md" in printed


class TestWizardGenerateIdentityFiles:
    """Test the wizard's _generate_identity_files integration."""

    def test_generates_all_files(self, tmp_path):
        from animus_bootstrap.config.schema import AnimusConfig, IdentitySection
        from animus_bootstrap.setup.wizard import AnimusWizard

        config = AnimusConfig()
        config.identity = IdentitySection(
            name="TestUser",
            timezone="UTC",
            locale="en_US",
            identity_dir=str(tmp_path / "identity"),
        )

        wizard = AnimusWizard(MagicMock())
        wizard._generate_identity_files(
            config,
            {"about": "I like Python", "generate_identity_files": True},
        )

        identity_dir = tmp_path / "identity"
        assert (identity_dir / "CORE_VALUES.md").exists()
        assert (identity_dir / "IDENTITY.md").exists()
        assert (identity_dir / "CONTEXT.md").exists()
        assert (identity_dir / "GOALS.md").exists()
        assert (identity_dir / "PREFERENCES.md").exists()
        assert (identity_dir / "LEARNED.md").exists()

        # Verify name is injected
        core = (identity_dir / "CORE_VALUES.md").read_text()
        assert "TestUser" in core

    def test_skips_existing_files(self, tmp_path):
        from animus_bootstrap.config.schema import AnimusConfig, IdentitySection
        from animus_bootstrap.setup.wizard import AnimusWizard

        identity_dir = tmp_path / "identity"
        identity_dir.mkdir()
        (identity_dir / "IDENTITY.md").write_text("custom")

        config = AnimusConfig()
        config.identity = IdentitySection(
            name="User",
            timezone="UTC",
            identity_dir=str(identity_dir),
        )

        wizard = AnimusWizard(MagicMock())
        wizard._generate_identity_files(config, {"about": "", "generate_identity_files": True})

        assert (identity_dir / "IDENTITY.md").read_text() == "custom"


class TestWizardStepCount:
    """Verify wizard step count is updated."""

    def test_total_steps_is_nine(self):
        from animus_bootstrap.setup.wizard import _TOTAL_STEPS

        assert _TOTAL_STEPS == 9
