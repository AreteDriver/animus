"""Tests for package initialization."""

from __future__ import annotations

import animus_validator


class TestPackageInit:
    def test_version(self):
        assert animus_validator.__version__ == "0.1.0"

    def test_exports(self):
        assert hasattr(animus_validator, "ValidatorConfig")
        assert hasattr(animus_validator, "Network")
        assert hasattr(animus_validator, "StakeStatus")
        assert hasattr(animus_validator, "RewardType")
        assert hasattr(animus_validator, "Node")
        assert hasattr(animus_validator, "StakePosition")
        assert hasattr(animus_validator, "RewardRecord")
        assert hasattr(animus_validator, "CompoundAction")
        assert hasattr(animus_validator, "NetworkInfo")

    def test_all_list(self):
        expected = {
            "CompoundAction",
            "Network",
            "NetworkInfo",
            "Node",
            "RewardRecord",
            "RewardType",
            "StakePosition",
            "StakeStatus",
            "ValidatorConfig",
            "__version__",
        }
        assert set(animus_validator.__all__) == expected
