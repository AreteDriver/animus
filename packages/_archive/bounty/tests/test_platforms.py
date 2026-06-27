"""Tests for animus_bounty.platforms."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from animus_bounty.config import BountyConfig
from animus_bounty.models import Difficulty, Platform
from animus_bounty.platforms.base import BasePlatform, ScanResult
from animus_bounty.platforms.generic import GenericPlatform, parse_generic_bounty
from animus_bounty.platforms.gitcoin import GitcoinPlatform, _parse_grant
from animus_bounty.platforms.github import (
    GitHubPlatform,
    _estimate_difficulty,
    _extract_reward,
    _parse_issue,
)


class TestScanResult:
    def test_empty(self):
        r = ScanResult(platform_name="test")
        assert r.count == 0
        assert r.errors == []

    def test_with_bounties(self, sample_bounty):
        r = ScanResult(platform_name="test", bounties=[sample_bounty])
        assert r.count == 1

    def test_to_dict(self):
        r = ScanResult(platform_name="test", errors=["err"], scan_duration_seconds=1.5)
        d = r.to_dict()
        assert d["platform_name"] == "test"
        assert d["count"] == 0
        assert d["errors"] == ["err"]
        assert d["scan_duration_seconds"] == 1.5


class TestBasePlatform:
    def test_scan_error_handling(self, sample_config):
        class BadPlatform(BasePlatform):
            name = "bad"

            async def _scan(self):
                raise RuntimeError("boom")

        platform = BadPlatform(sample_config)
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(platform.scan())
        assert len(result.errors) == 1
        assert "boom" in result.errors[0]
        assert result.scan_duration_seconds > 0

    def test_scan_success(self, sample_config, sample_bounty):
        class GoodPlatform(BasePlatform):
            name = "good"

            async def _scan(self):
                return ScanResult(platform_name=self.name, bounties=[sample_bounty])

        platform = GoodPlatform(sample_config)
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(platform.scan())
        assert result.count == 1
        assert result.scan_duration_seconds > 0

    def test_health_check_default(self, sample_config):
        class TestPlatform(BasePlatform):
            name = "test"

            async def _scan(self):
                return ScanResult(platform_name=self.name)

        platform = TestPlatform(sample_config)
        import asyncio

        assert asyncio.get_event_loop().run_until_complete(platform.health_check())


class TestGitHubPlatform:
    def test_estimate_difficulty_easy(self):
        assert _estimate_difficulty(["good first issue"]) == Difficulty.EASY
        assert _estimate_difficulty(["beginner", "easy"]) == Difficulty.EASY

    def test_estimate_difficulty_medium(self):
        assert _estimate_difficulty(["medium"]) == Difficulty.MEDIUM
        assert _estimate_difficulty(["intermediate"]) == Difficulty.MEDIUM

    def test_estimate_difficulty_hard(self):
        assert _estimate_difficulty(["hard"]) == Difficulty.HARD
        assert _estimate_difficulty(["complex", "advanced"]) == Difficulty.HARD

    def test_estimate_difficulty_default(self):
        assert _estimate_difficulty(["unknown"]) == Difficulty.MEDIUM
        assert _estimate_difficulty([]) == Difficulty.MEDIUM

    def test_extract_reward_dollar(self):
        assert _extract_reward("Fix bug $50", "", []) == 50.0
        assert _extract_reward("", "Bounty: $100.00", []) == 100.0

    def test_extract_reward_usd_text(self):
        assert _extract_reward("200 USD reward", "", []) == 200.0

    def test_extract_reward_bounty_keyword(self):
        assert _extract_reward("", "bounty: 75", []) == 75.0

    def test_extract_reward_comma(self):
        assert _extract_reward("$1,000 bounty", "", []) == 1000.0

    def test_extract_reward_none(self):
        assert _extract_reward("Fix typo", "No reward mentioned", []) == 0.0

    def test_parse_issue(self):
        issue = {
            "title": "Fix bug",
            "html_url": "https://github.com/org/repo/issues/1",
            "body": "There is a bug in the login flow",
            "labels": [{"name": "bounty"}, {"name": "good first issue"}],
            "repository_url": "https://api.github.com/repos/org/repo",
            "number": 1,
            "state": "open",
        }
        bounty = _parse_issue(issue)
        assert bounty.title == "Fix bug"
        assert bounty.platform == Platform.GITHUB
        assert "bounty" in bounty.labels
        assert bounty.raw_data["issue_number"] == 1

    def test_parse_issue_minimal(self):
        issue = {"title": "", "labels": [], "body": None}
        bounty = _parse_issue(issue)
        assert bounty.platform == Platform.GITHUB
        assert bounty.labels == []

    def test_parse_issue_string_labels(self):
        issue = {"title": "Test", "labels": ["bounty", "easy"], "body": "desc"}
        bounty = _parse_issue(issue)
        assert "bounty" in bounty.labels

    @pytest.mark.asyncio
    async def test_scan_success(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Bounty: Fix tests",
                    "html_url": "https://github.com/org/repo/issues/1",
                    "body": "$50 bounty",
                    "labels": [{"name": "bounty"}],
                    "number": 1,
                    "state": "open",
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.github.httpx.AsyncClient", return_value=mock_client):
            platform = GitHubPlatform(sample_config)
            result = await platform.scan()
            assert result.count > 0
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_scan_api_error(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 403

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.github.httpx.AsyncClient", return_value=mock_client):
            platform = GitHubPlatform(sample_config)
            result = await platform.scan()
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_scan_http_error(self, sample_config):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.github.httpx.AsyncClient", return_value=mock_client):
            platform = GitHubPlatform(sample_config)
            result = await platform.scan()
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_health_check_ok(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.github.httpx.AsyncClient", return_value=mock_client):
            platform = GitHubPlatform(sample_config)
            assert await platform.health_check()

    @pytest.mark.asyncio
    async def test_health_check_fail(self, sample_config):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.github.httpx.AsyncClient", return_value=mock_client):
            platform = GitHubPlatform(sample_config)
            assert not await platform.health_check()

    def test_headers_with_token(self, tmp_data_dir):
        config = BountyConfig(data_dir=tmp_data_dir)
        config.credentials.github_token = "ghp_test"
        platform = GitHubPlatform(config)
        headers = platform._headers()
        assert headers["Authorization"] == "Bearer ghp_test"

    def test_headers_without_token(self, sample_config):
        platform = GitHubPlatform(sample_config)
        headers = platform._headers()
        assert "Authorization" not in headers


class TestGitcoinPlatform:
    def test_parse_grant(self):
        data = {
            "title": "My Grant",
            "url": "https://gitcoin.co/grant/1",
            "description": "A cool project",
            "amountUSD": 500,
            "token": "ETH",
            "chain": "ethereum",
            "tags": ["defi"],
            "id": "grant-1",
        }
        bounty = _parse_grant(data)
        assert bounty.title == "My Grant"
        assert bounty.platform == Platform.GITCOIN
        assert bounty.reward_usd == 500.0

    def test_parse_grant_minimal(self):
        data = {}
        bounty = _parse_grant(data)
        assert bounty.title == "Untitled"
        assert bounty.reward_usd == 0.0

    def test_parse_grant_name_fallback(self):
        data = {"name": "Named Grant"}
        bounty = _parse_grant(data)
        assert bounty.title == "Named Grant"

    @pytest.mark.asyncio
    async def test_scan_with_api_key(self, tmp_data_dir):
        config = BountyConfig(data_dir=tmp_data_dir)
        config.credentials.gitcoin_api_key = "test_key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"rounds": []}}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.gitcoin.httpx.AsyncClient", return_value=mock_client):
            platform = GitcoinPlatform(config)
            result = await platform.scan()
            assert result.errors == []
            # Verify Authorization header was included
            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert headers.get("Authorization") == "Bearer test_key"

    @pytest.mark.asyncio
    async def test_scan_success(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"rounds": [{"id": "1", "roundMetadata": {"name": "Test Round"}}]}
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.gitcoin.httpx.AsyncClient", return_value=mock_client):
            platform = GitcoinPlatform(sample_config)
            result = await platform.scan()
            assert result.count == 1

    @pytest.mark.asyncio
    async def test_scan_api_error(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.gitcoin.httpx.AsyncClient", return_value=mock_client):
            platform = GitcoinPlatform(sample_config)
            result = await platform.scan()
            assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_scan_http_error(self, sample_config):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.gitcoin.httpx.AsyncClient", return_value=mock_client):
            platform = GitcoinPlatform(sample_config)
            result = await platform.scan()
            assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_health_check_ok(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.gitcoin.httpx.AsyncClient", return_value=mock_client):
            platform = GitcoinPlatform(sample_config)
            assert await platform.health_check()

    @pytest.mark.asyncio
    async def test_health_check_fail(self, sample_config):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.gitcoin.httpx.AsyncClient", return_value=mock_client):
            platform = GitcoinPlatform(sample_config)
            assert not await platform.health_check()

    @pytest.mark.asyncio
    async def test_scan_null_metadata(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"rounds": [{"id": "2", "roundMetadata": None}]}}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.gitcoin.httpx.AsyncClient", return_value=mock_client):
            platform = GitcoinPlatform(sample_config)
            result = await platform.scan()
            assert result.count == 1


class TestGenericPlatform:
    def test_parse_generic_bounty(self):
        data = {
            "title": "Generic Task",
            "url": "https://example.com/task",
            "reward_usd": 100,
            "description": "Do the thing",
            "skills": ["python"],
            "labels": ["easy"],
        }
        bounty = parse_generic_bounty(data)
        assert bounty.title == "Generic Task"
        assert bounty.platform == Platform.GENERIC
        assert bounty.reward_usd == 100.0

    def test_parse_generic_bounty_minimal(self):
        bounty = parse_generic_bounty({})
        assert bounty.title == "Untitled"
        assert bounty.reward_usd == 0.0

    @pytest.mark.asyncio
    async def test_scan_no_endpoint(self, sample_config):
        platform = GenericPlatform(sample_config)
        result = await platform.scan()
        assert len(result.errors) == 1
        assert "No endpoint" in result.errors[0]

    @pytest.mark.asyncio
    async def test_scan_success_array(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"title": "Task 1", "url": "https://example.com/1"},
            {"title": "Task 2", "url": "https://example.com/2"},
        ]

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.generic.httpx.AsyncClient", return_value=mock_client):
            platform = GenericPlatform(sample_config, endpoint="https://api.example.com/bounties")
            result = await platform.scan()
            assert result.count == 2

    @pytest.mark.asyncio
    async def test_scan_success_object(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bounties": [{"title": "Task", "url": "https://example.com/t"}]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.generic.httpx.AsyncClient", return_value=mock_client):
            platform = GenericPlatform(sample_config, endpoint="https://api.example.com/bounties")
            result = await platform.scan()
            assert result.count == 1

    @pytest.mark.asyncio
    async def test_scan_api_error(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.generic.httpx.AsyncClient", return_value=mock_client):
            platform = GenericPlatform(sample_config, endpoint="https://api.example.com/bounties")
            result = await platform.scan()
            assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_scan_http_error(self, sample_config):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.generic.httpx.AsyncClient", return_value=mock_client):
            platform = GenericPlatform(sample_config, endpoint="https://api.example.com/bounties")
            result = await platform.scan()
            assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_health_check_no_endpoint(self, sample_config):
        platform = GenericPlatform(sample_config)
        assert not await platform.health_check()

    @pytest.mark.asyncio
    async def test_health_check_ok(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.generic.httpx.AsyncClient", return_value=mock_client):
            platform = GenericPlatform(sample_config, endpoint="https://api.example.com/bounties")
            assert await platform.health_check()

    @pytest.mark.asyncio
    async def test_health_check_fail(self, sample_config):
        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.generic.httpx.AsyncClient", return_value=mock_client):
            platform = GenericPlatform(sample_config, endpoint="https://api.example.com/bounties")
            assert not await platform.health_check()

    @pytest.mark.asyncio
    async def test_health_check_server_error(self, sample_config):
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("animus_bounty.platforms.generic.httpx.AsyncClient", return_value=mock_client):
            platform = GenericPlatform(sample_config, endpoint="https://api.example.com/bounties")
            assert not await platform.health_check()
