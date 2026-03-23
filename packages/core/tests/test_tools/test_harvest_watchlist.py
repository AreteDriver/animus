"""Tests for harvest watchlist feature."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from animus.harvest import HarvestResult
from animus.harvest_watchlist import (
    _load_watchlist,
    _save_watchlist,
    add_to_watchlist,
    get_changes_report,
    get_due_repos,
    get_watchlist,
    remove_from_watchlist,
    run_watchlist_scan,
    update_scan_result,
)
from animus.harvest_watchlist_tools import (
    WATCHLIST_ADD_TOOL,
    WATCHLIST_LIST_TOOL,
    WATCHLIST_REMOVE_TOOL,
    WATCHLIST_SCAN_TOOL,
    _tool_watchlist_add,
    _tool_watchlist_list,
    _tool_watchlist_remove,
)


@pytest.fixture(autouse=True)
def _isolate_watchlist(tmp_path, monkeypatch):
    """Point WATCHLIST_FILE to tmp_path so tests don't touch real data."""
    test_file = tmp_path / "harvest_watchlist.json"
    monkeypatch.setattr("animus.harvest_watchlist.WATCHLIST_FILE", test_file)


# ---------------------------------------------------------------------------
# add_to_watchlist
# ---------------------------------------------------------------------------


class TestAddToWatchlist:
    def test_add_repo(self):
        entry = add_to_watchlist("user/repo", tags=["competitor"], notes="Test repo")
        assert entry["target"] == "user/repo"
        assert entry["tags"] == ["competitor"]
        assert entry["notes"] == "Test repo"
        assert entry["last_scanned"] is None
        assert entry["last_score"] is None
        assert entry["added_at"] is not None

    def test_add_full_url(self):
        entry = add_to_watchlist("https://github.com/user/repo2")
        assert entry["target"] == "user/repo2"

    def test_add_duplicate_raises(self):
        add_to_watchlist("user/repo")
        with pytest.raises(ValueError, match="already on the watchlist"):
            add_to_watchlist("user/repo")

    def test_add_no_tags_or_notes(self):
        entry = add_to_watchlist("user/repo")
        assert entry["tags"] == []
        assert entry["notes"] == ""

    def test_add_persists_to_disk(self, tmp_path):
        add_to_watchlist("user/repo")
        # Read the file directly
        watchlist_file = tmp_path / "harvest_watchlist.json"
        data = json.loads(watchlist_file.read_text())
        assert len(data["repos"]) == 1
        assert data["repos"][0]["target"] == "user/repo"


# ---------------------------------------------------------------------------
# remove_from_watchlist
# ---------------------------------------------------------------------------


class TestRemoveFromWatchlist:
    def test_remove_existing(self):
        add_to_watchlist("user/repo")
        assert remove_from_watchlist("user/repo") is True
        assert get_watchlist() == []

    def test_remove_nonexistent(self):
        assert remove_from_watchlist("user/nope") is False

    def test_remove_keeps_others(self):
        add_to_watchlist("user/repo1")
        add_to_watchlist("user/repo2")
        remove_from_watchlist("user/repo1")
        repos = get_watchlist()
        assert len(repos) == 1
        assert repos[0]["target"] == "user/repo2"


# ---------------------------------------------------------------------------
# get_watchlist
# ---------------------------------------------------------------------------


class TestGetWatchlist:
    def test_empty(self):
        assert get_watchlist() == []

    def test_returns_all(self):
        add_to_watchlist("user/a")
        add_to_watchlist("user/b")
        repos = get_watchlist()
        assert len(repos) == 2
        targets = [r["target"] for r in repos]
        assert "user/a" in targets
        assert "user/b" in targets


# ---------------------------------------------------------------------------
# get_due_repos
# ---------------------------------------------------------------------------


class TestGetDueRepos:
    def test_never_scanned_is_due(self):
        add_to_watchlist("user/repo")
        due = get_due_repos()
        assert len(due) == 1
        assert due[0]["target"] == "user/repo"

    def test_recently_scanned_not_due(self):
        add_to_watchlist("user/repo")
        update_scan_result("user/repo", score=80)
        due = get_due_repos(interval_hours=168)
        assert len(due) == 0

    def test_old_scan_is_due(self):
        add_to_watchlist("user/repo")
        # Manually set old scan time
        data = _load_watchlist()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=200)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        data["repos"][0]["last_scanned"] = old_time
        _save_watchlist(data)

        due = get_due_repos(interval_hours=168)
        assert len(due) == 1

    def test_custom_interval(self):
        add_to_watchlist("user/repo")
        update_scan_result("user/repo", score=80)
        # With 0-hour interval, everything is due
        due = get_due_repos(interval_hours=0)
        assert len(due) == 1

    def test_empty_watchlist(self):
        assert get_due_repos() == []


# ---------------------------------------------------------------------------
# update_scan_result
# ---------------------------------------------------------------------------


class TestUpdateScanResult:
    def test_updates_score(self):
        add_to_watchlist("user/repo")
        entry = update_scan_result("user/repo", score=85)
        assert entry is not None
        assert entry["last_score"] == 85
        assert entry["last_scanned"] is not None

    def test_updates_findings(self):
        add_to_watchlist("user/repo")
        findings = {"notable_patterns": ["pytest testing"]}
        entry = update_scan_result("user/repo", score=90, findings=findings)
        assert entry["last_findings"] == findings

    def test_not_found_returns_none(self):
        result = update_scan_result("user/nope", score=50)
        assert result is None

    def test_persists_to_disk(self, tmp_path):
        add_to_watchlist("user/repo")
        update_scan_result("user/repo", score=75)
        watchlist_file = tmp_path / "harvest_watchlist.json"
        data = json.loads(watchlist_file.read_text())
        assert data["repos"][0]["last_score"] == 75


# ---------------------------------------------------------------------------
# get_changes_report
# ---------------------------------------------------------------------------


class TestGetChangesReport:
    def test_initial_scan(self):
        add_to_watchlist("user/repo")
        current = {"score": 80, "notable_patterns": ["pytest"], "tools_worth_adopting": ["celery"]}
        report = get_changes_report("user/repo", current)
        assert "Initial" in report["score_change"]

    def test_score_change(self):
        add_to_watchlist("user/repo")
        update_scan_result(
            "user/repo",
            score=70,
            findings={"notable_patterns": ["pytest"], "tools_worth_adopting": ["celery"]},
        )
        current = {"score": 85, "notable_patterns": ["pytest"], "tools_worth_adopting": ["celery"]}
        report = get_changes_report("user/repo", current)
        assert "+15" in report["score_change"]
        assert "70" in report["score_change"]
        assert "85" in report["score_change"]

    def test_new_patterns_detected(self):
        add_to_watchlist("user/repo")
        update_scan_result(
            "user/repo",
            score=70,
            findings={"notable_patterns": ["pytest"], "tools_worth_adopting": []},
        )
        current = {
            "score": 72,
            "notable_patterns": ["pytest", "GraphQL client"],
            "tools_worth_adopting": [],
        }
        report = get_changes_report("user/repo", current)
        assert "GraphQL client" in report["new_patterns"]

    def test_removed_patterns_detected(self):
        add_to_watchlist("user/repo")
        update_scan_result(
            "user/repo",
            score=70,
            findings={"notable_patterns": ["REST API", "pytest"], "tools_worth_adopting": []},
        )
        current = {"score": 72, "notable_patterns": ["pytest"], "tools_worth_adopting": []}
        report = get_changes_report("user/repo", current)
        assert "REST API" in report["removed_patterns"]

    def test_new_dependencies(self):
        add_to_watchlist("user/repo")
        update_scan_result(
            "user/repo",
            score=70,
            findings={"notable_patterns": [], "tools_worth_adopting": ["celery"]},
        )
        current = {
            "score": 70,
            "notable_patterns": [],
            "tools_worth_adopting": ["celery", "graphql-core"],
        }
        report = get_changes_report("user/repo", current)
        assert "graphql-core" in report["new_dependencies"]

    def test_significant_score_generates_alert(self):
        add_to_watchlist("user/repo")
        update_scan_result(
            "user/repo",
            score=60,
            findings={"notable_patterns": [], "tools_worth_adopting": []},
        )
        current = {"score": 75, "notable_patterns": [], "tools_worth_adopting": []}
        report = get_changes_report("user/repo", current)
        assert report["alert"] is not None
        assert "improved" in report["alert"]

    def test_repo_not_on_watchlist(self):
        current = {"score": 80, "notable_patterns": [], "tools_worth_adopting": []}
        report = get_changes_report("user/nope", current)
        assert report["alert"] == "Repo not on watchlist"


# ---------------------------------------------------------------------------
# run_watchlist_scan
# ---------------------------------------------------------------------------


class TestRunWatchlistScan:
    @patch("animus.harvest.harvest_repo")
    def test_scans_due_repos(self, mock_harvest):
        mock_harvest.return_value = HarvestResult(
            repo="user/repo",
            score=80,
            notable_patterns=["pytest testing"],
            tools_worth_adopting=["celery"],
        )
        add_to_watchlist("user/repo")
        report = asyncio.run(run_watchlist_scan())
        assert report["scanned"] == 1
        mock_harvest.assert_called_once()

    @patch("animus.harvest.harvest_repo")
    def test_reports_changes(self, mock_harvest):
        # First scan
        add_to_watchlist("user/repo")
        update_scan_result(
            "user/repo",
            score=70,
            findings={"notable_patterns": ["old pattern"], "tools_worth_adopting": []},
        )

        # Force it due by setting old timestamp
        data = _load_watchlist()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=200)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        data["repos"][0]["last_scanned"] = old_time
        _save_watchlist(data)

        mock_harvest.return_value = HarvestResult(
            repo="user/repo",
            score=85,
            notable_patterns=["new pattern"],
            tools_worth_adopting=["graphql-core"],
        )

        report = asyncio.run(run_watchlist_scan())
        assert report["scanned"] == 1
        assert len(report["changes"]) == 1
        assert report["changes"][0]["repo"] == "user/repo"

    @patch("animus.harvest.harvest_repo")
    def test_handles_scan_error(self, mock_harvest):
        mock_harvest.side_effect = RuntimeError("Clone failed")
        add_to_watchlist("user/broken")
        report = asyncio.run(run_watchlist_scan())
        assert report["scanned"] == 0
        assert len(report["errors"]) == 1
        assert "Clone failed" in report["errors"][0]["error"]

    def test_empty_watchlist(self):
        report = asyncio.run(run_watchlist_scan())
        assert report["scanned"] == 0
        assert report["changes"] == []
        assert report["no_changes"] == []


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_watchlist_add_spec(self):
        assert WATCHLIST_ADD_TOOL.name == "animus_watchlist_add"
        assert "target" in WATCHLIST_ADD_TOOL.parameters["properties"]
        assert WATCHLIST_ADD_TOOL.parameters["required"] == ["target"]
        assert WATCHLIST_ADD_TOOL.category == "analysis"

    def test_watchlist_remove_spec(self):
        assert WATCHLIST_REMOVE_TOOL.name == "animus_watchlist_remove"
        assert "target" in WATCHLIST_REMOVE_TOOL.parameters["properties"]
        assert WATCHLIST_REMOVE_TOOL.parameters["required"] == ["target"]

    def test_watchlist_list_spec(self):
        assert WATCHLIST_LIST_TOOL.name == "animus_watchlist_list"
        assert WATCHLIST_LIST_TOOL.parameters["properties"] == {}

    def test_watchlist_scan_spec(self):
        assert WATCHLIST_SCAN_TOOL.name == "animus_watchlist_scan"
        assert "interval_hours" in WATCHLIST_SCAN_TOOL.parameters["properties"]

    def test_all_registered_in_default_registry(self):
        from animus.tools import create_default_registry

        registry = create_default_registry()
        assert registry.get("animus_watchlist_add") is not None
        assert registry.get("animus_watchlist_remove") is not None
        assert registry.get("animus_watchlist_list") is not None
        assert registry.get("animus_watchlist_scan") is not None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_add_handler_missing_target(self):
        result = _tool_watchlist_add({})
        assert not result.success
        assert "target" in result.error

    def test_add_handler_success(self):
        result = _tool_watchlist_add({"target": "user/repo", "tags": "competitor,test"})
        assert result.success
        output = json.loads(result.output)
        assert output["target"] == "user/repo"
        assert "competitor" in output["tags"]

    def test_add_handler_duplicate(self):
        _tool_watchlist_add({"target": "user/repo"})
        result = _tool_watchlist_add({"target": "user/repo"})
        assert not result.success
        assert "already" in result.error

    def test_remove_handler_missing_target(self):
        result = _tool_watchlist_remove({})
        assert not result.success
        assert "target" in result.error

    def test_remove_handler_success(self):
        _tool_watchlist_add({"target": "user/repo"})
        result = _tool_watchlist_remove({"target": "user/repo"})
        assert result.success

    def test_remove_handler_not_found(self):
        result = _tool_watchlist_remove({"target": "user/nope"})
        assert not result.success

    def test_list_handler_empty(self):
        result = _tool_watchlist_list({})
        assert result.success
        output = json.loads(result.output)
        assert output == []

    def test_list_handler_with_repos(self):
        _tool_watchlist_add({"target": "user/a"})
        _tool_watchlist_add({"target": "user/b"})
        result = _tool_watchlist_list({})
        assert result.success
        output = json.loads(result.output)
        assert len(output) == 2
