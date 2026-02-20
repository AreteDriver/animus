"""Tests for the CLI inspector (``python -m convergent``)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from convergent.__main__ import main
from convergent.intent import (
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(name: str, tags: list[str] | None = None) -> InterfaceSpec:
    return InterfaceSpec(
        name=name,
        kind=InterfaceKind.FUNCTION,
        signature="(x: int) -> int",
        module_path="mod",
        tags=tags or [],
    )


def _make_intent(
    agent_id: str,
    intent: str,
    provides: list[InterfaceSpec] | None = None,
    evidence: list[Evidence] | None = None,
) -> Intent:
    return Intent(
        agent_id=agent_id,
        intent=intent,
        provides=provides or [],
        evidence=evidence or [],
    )


@pytest.fixture
def db_with_data(tmp_path):
    """Create a temporary SQLite DB with two agents' intents."""
    db_path = str(tmp_path / "test.db")
    b = SQLiteBackend(db_path)
    b.publish(
        _make_intent(
            "alpha",
            "build api",
            provides=[_make_spec("handler", tags=["web", "api"])],
            evidence=[Evidence.test_pass("unit tests")],
        )
    )
    b.publish(
        _make_intent(
            "beta",
            "build db",
            provides=[_make_spec("db_connect", tags=["db", "sql"])],
            evidence=[Evidence.code_committed("initial")],
        )
    )
    b.close()
    return db_path


# ---------------------------------------------------------------------------
# Demo subcommand
# ---------------------------------------------------------------------------


class TestDemo:
    def test_demo_calls_run_demo(self):
        with patch("convergent.__main__._cmd_demo") as mock_demo:
            main(["demo"])
            mock_demo.assert_called_once()


# ---------------------------------------------------------------------------
# Inspect — format outputs
# ---------------------------------------------------------------------------


class TestInspectTable:
    def test_table_output(self, db_with_data, capsys):
        main(["inspect", db_with_data])
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out
        assert "handler" in captured.out

    def test_table_with_evidence(self, db_with_data, capsys):
        main(["inspect", db_with_data, "--show-evidence"])
        captured = capsys.readouterr()
        assert "test_pass" in captured.out


class TestInspectDot:
    def test_dot_output(self, db_with_data, capsys):
        main(["inspect", db_with_data, "--format", "dot"])
        captured = capsys.readouterr()
        assert "digraph" in captured.out
        assert "alpha" in captured.out


class TestInspectHtml:
    def test_html_output(self, db_with_data, capsys):
        main(["inspect", db_with_data, "--format", "html"])
        captured = capsys.readouterr()
        assert "<!DOCTYPE html>" in captured.out
        assert "alpha" in captured.out


class TestInspectMatrix:
    def test_matrix_output(self, db_with_data, capsys):
        main(["inspect", db_with_data, "--format", "matrix"])
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFilters:
    def test_agent_filter(self, db_with_data, capsys):
        main(["inspect", db_with_data, "--agent", "alpha"])
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" not in captured.out

    def test_agent_filter_no_results(self, db_with_data):
        with pytest.raises(SystemExit, match="1"):
            main(["inspect", db_with_data, "--agent", "nobody"])

    def test_min_stability_filter(self, db_with_data, capsys):
        main(["inspect", db_with_data, "--min-stability", "0.99"])
        captured = capsys.readouterr()
        # High threshold should filter all intents — empty graph
        assert "empty graph" in captured.out


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


class TestFileOutput:
    def test_output_to_file(self, db_with_data, tmp_path):
        out_file = str(tmp_path / "output.txt")
        main(["inspect", db_with_data, "--output", out_file])
        assert os.path.exists(out_file)
        with open(out_file) as f:
            content = f.read()
        assert "alpha" in content


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_db(self):
        with pytest.raises(SystemExit, match="1"):
            main(["inspect", "/nonexistent/path.db"])

    def test_no_command_shows_help(self):
        with pytest.raises(SystemExit, match="0"):
            main([])
