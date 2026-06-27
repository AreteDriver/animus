"""Tests for Animus MCP tools."""

from __future__ import annotations

import pytest

from animus_validator.config import ValidatorConfig
from animus_validator.models import Network, RewardRecord, StakePosition
from animus_validator.tools import validator_tools
from animus_validator.tools.validator_tools import (
    VALIDATOR_TOOLS,
    validator_compound,
    validator_discover,
    validator_positions,
    validator_rewards,
)


@pytest.fixture(autouse=True)
def reset_globals(tmp_path):
    """Reset tool module globals before each test."""
    validator_tools._config = ValidatorConfig(data_dir=tmp_path / "data")
    validator_tools._config.ensure_dirs()
    validator_tools._tracker = None
    yield
    validator_tools._config = None
    validator_tools._tracker = None


class TestValidatorPositionsTool:
    def test_empty_positions(self):
        result = validator_positions({})
        assert result["success"] is True
        assert result["positions"] == []
        assert result["total"] == 0

    def test_with_positions(self, tmp_path):
        config = validator_tools._config
        from animus_validator.store import PositionStore

        store = PositionStore(config.data_dir)
        store.record(StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0))
        store.record(StakePosition(network=Network.ETHEREUM, validator_address="0xv2", amount=32.0))

        result = validator_positions({})
        assert result["total"] == 2

    def test_filter_by_network(self, tmp_path):
        config = validator_tools._config
        from animus_validator.store import PositionStore

        store = PositionStore(config.data_dir)
        store.record(StakePosition(network=Network.SUI, validator_address="0xv1", amount=100.0))
        store.record(StakePosition(network=Network.ETHEREUM, validator_address="0xv2", amount=32.0))

        result = validator_positions({"network": "sui"})
        assert result["total"] == 1
        assert result["positions"][0]["network"] == "sui"


class TestValidatorRewardsTool:
    def test_empty_rewards(self):
        result = validator_rewards({})
        assert result["success"] is True
        assert result["report"]["total_rewards"] == 0.0
        assert result["recent_rewards"] == []

    def test_with_rewards(self, tmp_path):
        config = validator_tools._config
        from animus_validator.store import RewardStore

        store = RewardStore(config.data_dir)
        store.record(RewardRecord(network=Network.SUI, amount=1.5))

        result = validator_rewards({"days": 30})
        assert result["report"]["total_rewards"] == 1.5
        assert result["period_days"] == 30

    def test_default_days(self):
        result = validator_rewards({})
        assert result["period_days"] == 30


class TestValidatorCompoundTool:
    def test_dry_run_empty(self):
        result = validator_compound({})
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["eligible_positions"] == []
        assert result["total_compoundable"] == 0.0

    def test_dry_run_with_eligible(self, tmp_path):
        config = validator_tools._config
        from animus_validator.store import PositionStore

        store = PositionStore(config.data_dir)
        store.record(
            StakePosition(
                network=Network.SUI,
                validator_address="0xv1",
                amount=100.0,
                rewards_pending=5.0,
                auto_compound=True,
            )
        )

        result = validator_compound({"dry_run": True})
        assert result["dry_run"] is True
        assert len(result["eligible_positions"]) == 1
        assert result["total_compoundable"] == 5.0

    def test_non_dry_run(self, tmp_path):
        result = validator_compound({"dry_run": False})
        assert result["success"] is True
        assert result["dry_run"] is False

    def test_default_is_dry_run(self):
        result = validator_compound({})
        assert result["dry_run"] is True


class TestValidatorDiscoverTool:
    def test_discover_default(self):
        result = validator_discover({})
        assert result["success"] is True
        assert isinstance(result["networks"], list)
        assert result["total"] >= 0

    def test_discover_high_apy_filter(self):
        result = validator_discover({"min_apy": 100.0})
        assert result["total"] == 0

    def test_discover_low_apy_filter(self):
        result = validator_discover({"min_apy": 1.0})
        assert result["total"] >= 1


class TestValidatorToolsRegistry:
    def test_tools_count(self):
        assert len(VALIDATOR_TOOLS) == 4

    def test_tool_names(self):
        names = {t["name"] for t in VALIDATOR_TOOLS}
        assert names == {
            "validator_positions",
            "validator_rewards",
            "validator_compound",
            "validator_discover",
        }

    def test_all_have_category(self):
        for tool in VALIDATOR_TOOLS:
            assert tool["category"] == "validator"

    def test_all_have_handlers(self):
        for tool in VALIDATOR_TOOLS:
            assert callable(tool["handler"])

    def test_all_have_descriptions(self):
        for tool in VALIDATOR_TOOLS:
            assert len(tool["description"]) > 0

    def test_all_have_parameters(self):
        for tool in VALIDATOR_TOOLS:
            assert "type" in tool["parameters"]
            assert tool["parameters"]["type"] == "object"


class TestEnsureInitialized:
    def test_initializes_from_scratch(self):
        validator_tools._config = None
        validator_tools._tracker = None

        # This will create a default config
        result = validator_positions({})
        assert result["success"] is True
        assert validator_tools._config is not None
        assert validator_tools._tracker is not None

    def test_reuses_existing(self, tmp_path):
        # First call initializes
        validator_positions({})
        config1 = validator_tools._config
        tracker1 = validator_tools._tracker

        # Second call reuses
        validator_positions({})
        assert validator_tools._config is config1
        assert validator_tools._tracker is tracker1
