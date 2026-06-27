"""Tests for animus_bounty package init."""

from __future__ import annotations


class TestPackageInit:
    def test_version(self):
        from animus_bounty import __version__

        assert __version__ == "0.1.0"

    def test_exports(self):
        from animus_bounty import (
            Bounty,
            BountyConfig,
            BountyStatus,
            BountyStore,
            BountyTracker,
            ClaimResult,
            Platform,
            SkillMatch,
            SkillMatcher,
        )

        # Verify they're the right types
        assert Bounty is not None
        assert BountyConfig is not None
        assert BountyStatus is not None
        assert BountyStore is not None
        assert BountyTracker is not None
        assert ClaimResult is not None
        assert Platform is not None
        assert SkillMatch is not None
        assert SkillMatcher is not None

    def test_all_list(self):
        import animus_bounty

        assert "__version__" in animus_bounty.__all__
        assert "Bounty" in animus_bounty.__all__
        assert "BountyConfig" in animus_bounty.__all__
        assert "Platform" in animus_bounty.__all__
