"""Tests for harvest watchlist tool handlers."""

from __future__ import annotations

import json
from unittest.mock import patch

from animus.harvest_watchlist_tools import (
    _tool_watchlist_add,
    _tool_watchlist_list,
    _tool_watchlist_remove,
    _tool_watchlist_scan,
)


class TestWatchlistAddTool:
    def test_add_success(self):
        entry = {"target": "test/repo", "added": "2026-03-25"}
        with patch("animus.harvest_watchlist.add_to_watchlist", return_value=entry):
            result = _tool_watchlist_add({"target": "test/repo", "tags": "ai,ml"})
            assert result.success
            assert "test/repo" in result.output

    def test_add_missing_target(self):
        result = _tool_watchlist_add({})
        assert not result.success
        assert "Missing required" in result.error

    def test_add_value_error(self):
        with patch(
            "animus.harvest_watchlist.add_to_watchlist",
            side_effect=ValueError("duplicate"),
        ):
            result = _tool_watchlist_add({"target": "test/repo"})
            assert not result.success
            assert "duplicate" in result.error

    def test_add_unexpected_error(self):
        with patch(
            "animus.harvest_watchlist.add_to_watchlist",
            side_effect=OSError("disk full"),
        ):
            result = _tool_watchlist_add({"target": "test/repo"})
            assert not result.success
            assert "Failed to add" in result.error


class TestWatchlistRemoveTool:
    def test_remove_success(self):
        with patch("animus.harvest_watchlist.remove_from_watchlist", return_value=True):
            result = _tool_watchlist_remove({"target": "test/repo"})
            assert result.success

    def test_remove_not_found(self):
        with patch("animus.harvest_watchlist.remove_from_watchlist", return_value=False):
            result = _tool_watchlist_remove({"target": "test/repo"})
            assert not result.success
            assert "not found" in result.error

    def test_remove_missing_target(self):
        result = _tool_watchlist_remove({})
        assert not result.success

    def test_remove_exception(self):
        with patch(
            "animus.harvest_watchlist.remove_from_watchlist",
            side_effect=OSError("fail"),
        ):
            result = _tool_watchlist_remove({"target": "test/repo"})
            assert not result.success
            assert "Failed to remove" in result.error


class TestWatchlistListTool:
    def test_list_success(self):
        repos = [{"target": "a/b"}]
        with patch("animus.harvest_watchlist.get_watchlist", return_value=repos):
            result = _tool_watchlist_list({})
            assert result.success
            data = json.loads(result.output)
            assert len(data) == 1

    def test_list_exception(self):
        with patch(
            "animus.harvest_watchlist.get_watchlist",
            side_effect=OSError("db locked"),
        ):
            result = _tool_watchlist_list({})
            assert not result.success
            assert "Failed to list" in result.error


class TestWatchlistScanTool:
    def test_scan_success(self):
        report = {"scanned": 2}

        async def fake_scan(**kwargs):
            return report

        with patch(
            "animus.harvest_watchlist.run_watchlist_scan", side_effect=fake_scan
        ):
            result = _tool_watchlist_scan({})
            assert result.success
            data = json.loads(result.output)
            assert data["scanned"] == 2

    def test_scan_exception(self):
        async def fail_scan(**kwargs):
            raise RuntimeError("network")

        with patch(
            "animus.harvest_watchlist.run_watchlist_scan", side_effect=fail_scan
        ):
            result = _tool_watchlist_scan({})
            assert not result.success
            assert "Scan failed" in result.error
