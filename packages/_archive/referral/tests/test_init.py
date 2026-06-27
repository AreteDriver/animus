"""Tests for package init and version."""

from __future__ import annotations

import animus_referral
from animus_referral import (
    Account,
    BonusStatus,
    BonusType,
    ProgramType,
    ReferralBonus,
    ReferralConfig,
    ReferralLink,
    ReferralManager,
    ReferralProgram,
    ReferralTracker,
)


class TestPackageInit:
    def test_version(self):
        assert animus_referral.__version__ == "0.1.0"

    def test_exports(self):
        assert ReferralConfig is not None
        assert ReferralManager is not None
        assert ReferralTracker is not None
        assert ReferralProgram is not None
        assert Account is not None
        assert ReferralLink is not None
        assert ReferralBonus is not None
        assert ProgramType is not None
        assert BonusType is not None
        assert BonusStatus is not None

    def test_all_exports(self):
        for name in animus_referral.__all__:
            assert hasattr(animus_referral, name)
