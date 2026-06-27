"""Tests for animus_bounty.tools.bounty_tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import animus_bounty.tools.bounty_tools as tools_module
from animus_bounty.config import BountyConfig
from animus_bounty.matcher import SkillMatcher
from animus_bounty.tracker import BountyTracker


@pytest.fixture(autouse=True)
def reset_tool_state():
    """Reset global tool state between tests."""
    tools_module._tracker = None
    tools_module._config = None
    tools_module._matcher = None
    yield
    tools_module._tracker = None
    tools_module._config = None
    tools_module._matcher = None


class TestEnsureInitialized:
    def test_creates_instances(self, tmp_data_dir):
        with patch.dict("os.environ", {"BOUNTY_DATA_DIR": str(tmp_data_dir)}):
            tracker, config, matcher = tools_module._ensure_initialized()
            assert tracker is not None
            assert config is not None
            assert matcher is not None

    def test_idempotent(self, tmp_data_dir):
        with patch.dict("os.environ", {"BOUNTY_DATA_DIR": str(tmp_data_dir)}):
            t1, _, _ = tools_module._ensure_initialized()
            t2, _, _ = tools_module._ensure_initialized()
            assert t1 is t2

    def test_loads_config_file(self, tmp_data_dir):
        import yaml

        config_file = tmp_data_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"scan_interval_hours": 12}, f)

        with patch.dict("os.environ", {"BOUNTY_DATA_DIR": str(tmp_data_dir)}):
            _, config, _ = tools_module._ensure_initialized()
            assert config.scan_interval_hours == 12


class TestRunScan:
    @pytest.mark.asyncio
    async def test_run_scan(self, tmp_data_dir, sample_bounty):
        from animus_bounty.platforms.base import ScanResult

        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        config.filters.platforms = ["github"]

        mock_scan_result = ScanResult(
            platform_name="github",
            bounties=[sample_bounty],
            scan_duration_seconds=0.1,
        )

        with patch.object(
            tools_module.GitHubPlatform,
            "scan",
            new=AsyncMock(return_value=mock_scan_result),
        ):
            report = await tools_module._run_scan(config, tracker)
            assert report["total_scanned"] == 1
            assert report["total_new"] == 1
            assert len(report["platforms"]) == 1
            assert report["platforms"][0]["platform"] == "github"

    @pytest.mark.asyncio
    async def test_run_scan_empty(self, tmp_data_dir):
        from animus_bounty.platforms.base import ScanResult

        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        config.filters.platforms = ["github"]

        mock_scan_result = ScanResult(platform_name="github")

        with patch.object(
            tools_module.GitHubPlatform,
            "scan",
            new=AsyncMock(return_value=mock_scan_result),
        ):
            report = await tools_module._run_scan(config, tracker)
            assert report["total_scanned"] == 0
            assert report["total_new"] == 0


class TestBountyScan:
    def test_scan(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        mock_run = AsyncMock(
            return_value={
                "total_scanned": 1,
                "total_new": 1,
                "platforms": [{"platform": "github", "found": 1, "new": 1}],
            }
        )
        with patch("animus_bounty.tools.bounty_tools._run_scan", mock_run):
            result = tools_module.bounty_scan({})
            assert result["success"]
            assert result["report"]["total_new"] == 1


class TestBountyStatus:
    def test_status(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_status({})
        assert result["success"]
        assert "summary" in result
        assert result["summary"]["total_discovered"] == 0


class TestBountyMatch:
    def test_match_by_id(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty])
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_match({"bounty_id": sample_bounty.bounty_id})
        assert result["success"]
        assert "match" in result
        assert "auto_claim_recommended" in result

    def test_match_by_url(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_match(
            {
                "url": "https://example.com/bounty",
                "title": "Test bounty",
                "platform": "generic",
                "description": "A test",
                "skills_required": ["python"],
            }
        )
        assert result["success"]

    def test_match_no_params(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_match({})
        assert not result["success"]
        assert "error" in result

    def test_match_not_found(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_match({"bounty_id": "nonexistent"})
        assert not result["success"]


class TestBountyClaim:
    def test_claim_success(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty])
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_claim(
            {
                "bounty_id": sample_bounty.bounty_id,
                "notes": "Claiming it",
            }
        )
        assert result["success"]
        assert result["claim"]["status"] == "claimed"

    def test_claim_no_id(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_claim({})
        assert not result["success"]

    def test_claim_not_found(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        result = tools_module.bounty_claim({"bounty_id": "nonexistent"})
        assert not result["success"]

    def test_claim_duplicate(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty])
        config = BountyConfig(data_dir=tmp_data_dir)
        matcher = SkillMatcher(config)
        tools_module._tracker = tracker
        tools_module._config = config
        tools_module._matcher = matcher

        tools_module.bounty_claim({"bounty_id": sample_bounty.bounty_id})
        result = tools_module.bounty_claim({"bounty_id": sample_bounty.bounty_id})
        assert not result["success"]
        assert "already claimed" in result["error"]


class TestGetScanners:
    def test_default_all(self, sample_config):
        scanners = tools_module._get_scanners(sample_config)
        names = [s.name for s in scanners]
        assert "github" in names
        assert "gitcoin" in names

    def test_filtered(self, tmp_data_dir):
        config = BountyConfig(data_dir=tmp_data_dir)
        config.filters.platforms = ["github"]
        scanners = tools_module._get_scanners(config)
        names = [s.name for s in scanners]
        assert "github" in names
        assert "gitcoin" not in names


class TestBountyToolsList:
    def test_all_tools_defined(self):
        from animus_bounty.tools.bounty_tools import BOUNTY_TOOLS

        assert len(BOUNTY_TOOLS) == 4
        names = {t["name"] for t in BOUNTY_TOOLS}
        assert names == {"bounty_scan", "bounty_status", "bounty_match", "bounty_claim"}

    def test_all_have_category(self):
        from animus_bounty.tools.bounty_tools import BOUNTY_TOOLS

        for tool in BOUNTY_TOOLS:
            assert tool["category"] == "bounty"

    def test_all_have_handler(self):
        from animus_bounty.tools.bounty_tools import BOUNTY_TOOLS

        for tool in BOUNTY_TOOLS:
            assert callable(tool["handler"])
