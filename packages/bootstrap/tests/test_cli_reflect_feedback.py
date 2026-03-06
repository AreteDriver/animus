"""Tests for reflect and feedback CLI commands."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from animus_bootstrap.cli import app

runner = CliRunner()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_feedback_db(tmp_path: Path) -> Path:
    """Create a feedback.db with some entries."""
    db_path = tmp_path / "feedback.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE feedback (
            id TEXT PRIMARY KEY,
            message_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT DEFAULT '',
            channel TEXT DEFAULT '',
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("id1", "How do I X?", "You do Y.", 1, "helpful", "cli", "2026-03-06T12:00:00"),
    )
    conn.execute(
        "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("id2", "What about Z?", "Try W.", -1, "wrong answer", "cli", "2026-03-06T12:01:00"),
    )
    conn.commit()
    conn.close()
    return db_path


def _mock_config_manager(tmp_path):
    """Create a mock ConfigManager that points to tmp_path."""
    mock_cm = MagicMock()
    mock_cm.return_value.load.return_value = MagicMock(
        api=MagicMock(
            anthropic_key="", ollama_host="http://localhost:11434", ollama_model="llama3.2",
        ),
    )
    mock_cm.return_value.get_config_path.return_value = tmp_path / "config.toml"
    return mock_cm


# ------------------------------------------------------------------
# Reflect command
# ------------------------------------------------------------------


class TestReflectCommand:
    """Tests for 'animus-bootstrap reflect'."""

    def test_reflect_no_feedback(self, tmp_path):
        mock_cm = _mock_config_manager(tmp_path)
        mock_run = AsyncMock(return_value=None)

        # Import submodules so patch can resolve the dotted path
        import animus_bootstrap.identity.manager  # noqa: F401
        import animus_bootstrap.intelligence.proactive.checks.reflection  # noqa: F401

        with (
            patch("animus_bootstrap.config.ConfigManager", mock_cm),
            patch("animus_bootstrap.identity.manager.IdentityFileManager"),
            patch(
                "animus_bootstrap.intelligence.proactive.checks.reflection.set_reflection_deps",
            ),
            patch(
                "animus_bootstrap.intelligence.proactive.checks.reflection._run_reflection",
                mock_run,
            ),
        ):
            result = runner.invoke(app, ["reflect"])
            assert result.exit_code == 0
            assert "No reflection needed" in result.output

    def test_reflect_with_results(self, tmp_path):
        mock_cm = _mock_config_manager(tmp_path)
        mock_run = AsyncMock(return_value="Reflection complete — 3 new insights recorded.")

        import animus_bootstrap.identity.manager  # noqa: F401
        import animus_bootstrap.intelligence.proactive.checks.reflection  # noqa: F401

        with (
            patch("animus_bootstrap.config.ConfigManager", mock_cm),
            patch("animus_bootstrap.identity.manager.IdentityFileManager"),
            patch(
                "animus_bootstrap.intelligence.proactive.checks.reflection.set_reflection_deps",
            ),
            patch(
                "animus_bootstrap.intelligence.proactive.checks.reflection._run_reflection",
                mock_run,
            ),
        ):
            result = runner.invoke(app, ["reflect"])
            assert result.exit_code == 0
            assert "3 new insights" in result.output


# ------------------------------------------------------------------
# Feedback add
# ------------------------------------------------------------------


class TestFeedbackAdd:
    """Tests for 'animus-bootstrap feedback add'."""

    def test_add_thumbs_up(self, tmp_path):
        mock_cm = _mock_config_manager(tmp_path)

        with patch("animus_bootstrap.config.ConfigManager", mock_cm):
            result = runner.invoke(
                app,
                ["feedback", "add", "up", "-m", "How do I X?", "-c", "great answer"],
            )
            assert result.exit_code == 0
            assert "thumbs up" in result.output

            # Verify DB was created
            db_path = tmp_path / "feedback.db"
            assert db_path.exists()

    def test_add_thumbs_down(self, tmp_path):
        mock_cm = _mock_config_manager(tmp_path)

        with patch("animus_bootstrap.config.ConfigManager", mock_cm):
            result = runner.invoke(
                app,
                ["feedback", "add", "down", "-c", "wrong answer"],
            )
            assert result.exit_code == 0
            assert "thumbs down" in result.output

    def test_add_invalid_rating(self, tmp_path):
        mock_cm = _mock_config_manager(tmp_path)

        with patch("animus_bootstrap.config.ConfigManager", mock_cm):
            result = runner.invoke(app, ["feedback", "add", "maybe"])
            assert result.exit_code == 1


# ------------------------------------------------------------------
# Feedback list
# ------------------------------------------------------------------


class TestFeedbackList:
    """Tests for 'animus-bootstrap feedback list'."""

    def test_list_no_db(self, tmp_path):
        mock_cm = _mock_config_manager(tmp_path)

        with patch("animus_bootstrap.config.ConfigManager", mock_cm):
            result = runner.invoke(app, ["feedback", "list"])
            assert result.exit_code == 0
            assert "No feedback recorded" in result.output

    def test_list_with_entries(self, tmp_path):
        _make_feedback_db(tmp_path)
        mock_cm = _mock_config_manager(tmp_path)

        with patch("animus_bootstrap.config.ConfigManager", mock_cm):
            result = runner.invoke(app, ["feedback", "list"])
            assert result.exit_code == 0
            assert "Recent Feedback" in result.output


# ------------------------------------------------------------------
# Feedback stats
# ------------------------------------------------------------------


class TestFeedbackStats:
    """Tests for 'animus-bootstrap feedback stats'."""

    def test_stats_no_db(self, tmp_path):
        mock_cm = _mock_config_manager(tmp_path)

        with patch("animus_bootstrap.config.ConfigManager", mock_cm):
            result = runner.invoke(app, ["feedback", "stats"])
            assert result.exit_code == 0
            assert "No feedback recorded" in result.output

    def test_stats_with_data(self, tmp_path):
        _make_feedback_db(tmp_path)
        mock_cm = _mock_config_manager(tmp_path)

        with patch("animus_bootstrap.config.ConfigManager", mock_cm):
            result = runner.invoke(app, ["feedback", "stats"])
            assert result.exit_code == 0
            assert "Feedback Stats" in result.output
