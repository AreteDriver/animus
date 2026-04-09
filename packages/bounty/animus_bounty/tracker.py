"""Track bounty lifecycle from discovery to payment."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animus_bounty.models import Bounty, BountyStatus, ClaimResult, SkillMatch
from animus_bounty.store import BountyStore, ClaimStore, MatchStore

logger = logging.getLogger("animus_bounty.tracker")


class BountyTracker:
    """Orchestrates bounty lifecycle tracking.

    Manages the flow: discover -> match -> claim -> submit -> paid.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.bounty_store = BountyStore(data_dir)
        self.match_store = MatchStore(data_dir)
        self.claim_store = ClaimStore(data_dir)

    def record_bounties(self, bounties: list[Bounty]) -> int:
        """Record discovered bounties. Returns count of new ones."""
        return self.bounty_store.record_batch(bounties)

    def record_match(self, match: SkillMatch) -> None:
        """Record a skill match evaluation."""
        self.match_store.record(match)

    def claim_bounty(self, bounty_id: str, notes: str = "") -> ClaimResult:
        """Mark a bounty as claimed."""
        claim = ClaimResult(
            bounty_id=bounty_id,
            status=BountyStatus.CLAIMED,
            notes=notes,
        )
        self.claim_store.record(claim)
        return claim

    def submit_bounty(self, bounty_id: str, notes: str = "") -> ClaimResult | None:
        """Mark a claimed bounty as submitted."""
        claim = self.claim_store.get_for_bounty(bounty_id)
        if claim is None:
            logger.warning("Cannot submit bounty %s: not claimed", bounty_id)
            return None
        claim.status = BountyStatus.SUBMITTED
        claim.submitted_at = datetime.now(UTC)
        if notes:
            claim.notes = notes
        self.claim_store.update(claim)
        return claim

    def mark_paid(
        self, bounty_id: str, actual_reward: float = 0.0, notes: str = ""
    ) -> ClaimResult | None:
        """Mark a bounty as paid."""
        claim = self.claim_store.get_for_bounty(bounty_id)
        if claim is None:
            logger.warning("Cannot mark bounty %s as paid: not claimed", bounty_id)
            return None
        claim.status = BountyStatus.PAID
        claim.paid_at = datetime.now(UTC)
        claim.actual_reward = actual_reward
        if notes:
            claim.notes = notes
        self.claim_store.update(claim)
        return claim

    def mark_rejected(self, bounty_id: str, notes: str = "") -> ClaimResult | None:
        """Mark a bounty as rejected."""
        claim = self.claim_store.get_for_bounty(bounty_id)
        if claim is None:
            logger.warning("Cannot reject bounty %s: not claimed", bounty_id)
            return None
        claim.status = BountyStatus.REJECTED
        if notes:
            claim.notes = notes
        self.claim_store.update(claim)
        return claim

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of bounty tracking status."""
        all_bounties = self.bounty_store.load_all()
        all_claims = self.claim_store.load_all()
        all_matches = self.match_store.load_all()

        status_counts: dict[str, int] = {}
        for claim in all_claims:
            status_counts[claim.status.value] = status_counts.get(claim.status.value, 0) + 1

        platform_counts: dict[str, int] = {}
        for bounty in all_bounties:
            platform_counts[bounty.platform.value] = (
                platform_counts.get(bounty.platform.value, 0) + 1
            )

        total_earned = sum(c.actual_reward for c in all_claims if c.status == BountyStatus.PAID)
        total_pending = sum(
            b.reward_usd
            for b in all_bounties
            if any(
                c.bounty_id == b.bounty_id
                and c.status
                in (BountyStatus.CLAIMED, BountyStatus.IN_PROGRESS, BountyStatus.SUBMITTED)
                for c in all_claims
            )
        )

        avg_match = (
            sum(m.match_score for m in all_matches) / len(all_matches) if all_matches else 0.0
        )

        return {
            "total_discovered": len(all_bounties),
            "total_claims": len(all_claims),
            "total_matches": len(all_matches),
            "status_breakdown": status_counts,
            "platform_breakdown": platform_counts,
            "total_earned_usd": round(total_earned, 2),
            "total_pending_usd": round(total_pending, 2),
            "avg_match_score": round(avg_match, 3),
        }
