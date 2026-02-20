"""Triumvirate consensus voting for Gorgon agent operations."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger(__name__ + ".audit")

NUM_VOTERS = 3

_VOTER_BASE = (
    "You are a Zorya voter evaluating whether an AI agent operation should proceed.\n"
    "Reply with exactly one line: APPROVE or REJECT\n"
    "Then a second line with a brief reason (one sentence).\n"
)

VOTER_PROMPTS: list[str] = [
    _VOTER_BASE
    + "Focus on SAFETY: Could this operation cause harm, data loss, or security issues?",
    _VOTER_BASE
    + "Focus on CORRECTNESS: Is this operation well-formed, valid, and likely to produce the intended result?",
    _VOTER_BASE
    + "Focus on ALIGNMENT: Does this operation match the stated task and stay within the agent's authorized scope?",
]

# Backwards compat alias
VOTER_SYSTEM_PROMPT = VOTER_PROMPTS[0]

# Estimated input tokens for voter prompt + operation preamble
_VOTER_INPUT_TOKEN_ESTIMATE = 80


class VoteDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ConsensusLevel(str, Enum):
    ANY = "any"
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    UNANIMOUS_PLUS_USER = "unanimous_plus_user"


# Ordering for comparison
_CONSENSUS_ORDER = {
    ConsensusLevel.ANY: 0,
    ConsensusLevel.MAJORITY: 1,
    ConsensusLevel.UNANIMOUS: 2,
    ConsensusLevel.UNANIMOUS_PLUS_USER: 3,
}


def consensus_level_order(level: ConsensusLevel | str) -> int:
    """Return numeric order for a consensus level string."""
    if isinstance(level, str):
        try:
            level = ConsensusLevel(level)
        except ValueError:
            return -1
    return _CONSENSUS_ORDER.get(level, -1)


@dataclass
class Vote:
    voter_id: int
    decision: VoteDecision
    reasoning: str = ""
    error: str = ""


@dataclass
class ConsensusVerdict:
    level: ConsensusLevel
    approved: bool
    votes: list[Vote] = field(default_factory=list)
    requires_user_confirmation: bool = False

    @property
    def approve_count(self) -> int:
        return sum(
            1 for v in self.votes if v.decision in (VoteDecision.APPROVE, VoteDecision.ABSTAIN)
        )

    @property
    def reject_count(self) -> int:
        return sum(1 for v in self.votes if v.decision == VoteDecision.REJECT)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "approved": self.approved,
            "approve_count": self.approve_count,
            "reject_count": self.reject_count,
            "requires_user_confirmation": self.requires_user_confirmation,
            "votes": [
                {
                    "voter_id": v.voter_id,
                    "decision": v.decision.value,
                    "reasoning": v.reasoning,
                }
                for v in self.votes
            ],
        }


class ConsensusVoter:
    """Runs Zorya voter agents to reach consensus on operations."""

    def __init__(self, client: ClaudeCodeClient) -> None:
        self._client = client

    def vote(self, operation: str, level: ConsensusLevel | str, role: str = "") -> ConsensusVerdict:
        if isinstance(level, str):
            level = ConsensusLevel(level)

        t0 = time.monotonic()
        if level == ConsensusLevel.ANY:
            votes = [self._call_voter(0, operation, role)]
        else:
            votes = [self._call_voter(i, operation, role) for i in range(NUM_VOTERS)]

        verdict = self._aggregate(level, votes)
        self._log_verdict(verdict, operation, role, time.monotonic() - t0)
        return verdict

    async def vote_async(
        self, operation: str, level: ConsensusLevel | str, role: str = ""
    ) -> ConsensusVerdict:
        if isinstance(level, str):
            level = ConsensusLevel(level)

        t0 = time.monotonic()
        if level == ConsensusLevel.ANY:
            votes = [await self._call_voter_async(0, operation, role)]
        else:
            votes = list(
                await asyncio.gather(
                    *[self._call_voter_async(i, operation, role) for i in range(NUM_VOTERS)]
                )
            )

        verdict = self._aggregate(level, votes)
        self._log_verdict(verdict, operation, role, time.monotonic() - t0)
        return verdict

    @staticmethod
    def _get_voter_prompt(voter_id: int) -> str:
        return VOTER_PROMPTS[voter_id % len(VOTER_PROMPTS)]

    @staticmethod
    def _track_voter_cost(voter_id: int, output: str, role: str) -> None:
        """Best-effort cost tracking for a voter call."""
        try:
            from animus_forge.metrics.cost_tracker import Provider, get_cost_tracker

            output_tokens = max(1, len(output) // 4)
            get_cost_tracker().track(
                provider=Provider.ANTHROPIC,
                model="claude-sonnet-4-20250514",
                input_tokens=_VOTER_INPUT_TOKEN_ESTIMATE,
                output_tokens=output_tokens,
                agent_role=f"zorya_voter_{voter_id}",
                metadata={"consensus_role": role, "voter_id": voter_id},
            )
        except Exception:
            logger.debug("Cost tracking unavailable for voter %d", voter_id)

    def _call_voter(self, voter_id: int, operation: str, role: str = "") -> Vote:
        try:
            result = self._client.generate_completion(
                prompt=f"Evaluate this operation:\n{operation}",
                system_prompt=self._get_voter_prompt(voter_id),
                max_tokens=150,
            )
            if not result.get("success") or not result.get("output"):
                return Vote(
                    voter_id=voter_id,
                    decision=VoteDecision.ABSTAIN,
                    error=result.get("error", "no output"),
                )
            self._track_voter_cost(voter_id, result["output"], role)
            return self._parse_vote(voter_id, result["output"])
        except Exception as exc:
            logger.warning("Voter %d failed: %s", voter_id, exc)
            return Vote(voter_id=voter_id, decision=VoteDecision.ABSTAIN, error=str(exc))

    async def _call_voter_async(self, voter_id: int, operation: str, role: str = "") -> Vote:
        try:
            result = await self._client.generate_completion_async(
                prompt=f"Evaluate this operation:\n{operation}",
                system_prompt=self._get_voter_prompt(voter_id),
                max_tokens=150,
            )
            if not result.get("success") or not result.get("output"):
                return Vote(
                    voter_id=voter_id,
                    decision=VoteDecision.ABSTAIN,
                    error=result.get("error", "no output"),
                )
            self._track_voter_cost(voter_id, result["output"], role)
            return self._parse_vote(voter_id, result["output"])
        except Exception as exc:
            logger.warning("Async voter %d failed: %s", voter_id, exc)
            return Vote(voter_id=voter_id, decision=VoteDecision.ABSTAIN, error=str(exc))

    @staticmethod
    def _log_verdict(
        verdict: ConsensusVerdict,
        operation: str,
        role: str,
        elapsed_seconds: float,
    ) -> None:
        """Emit structured audit log for the consensus vote."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "role": role,
            "operation": operation[:200],
            "level": verdict.level.value,
            "approved": verdict.approved,
            "approve_count": verdict.approve_count,
            "reject_count": verdict.reject_count,
            "requires_user_confirmation": verdict.requires_user_confirmation,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "votes": [
                {
                    "voter_id": v.voter_id,
                    "decision": v.decision.value,
                    "reasoning": v.reasoning[:100],
                    "error": v.error[:100] if v.error else "",
                }
                for v in verdict.votes
            ],
        }
        audit_logger.info("consensus_vote", extra={"consensus": record})

    @staticmethod
    def _parse_vote(voter_id: int, output: str) -> Vote:
        lines = output.strip().splitlines()
        if not lines:
            return Vote(voter_id=voter_id, decision=VoteDecision.ABSTAIN, error="empty output")

        first = lines[0].strip().upper()
        reasoning = lines[1].strip() if len(lines) > 1 else ""

        if first == "APPROVE":
            return Vote(voter_id=voter_id, decision=VoteDecision.APPROVE, reasoning=reasoning)
        elif first == "REJECT":
            return Vote(voter_id=voter_id, decision=VoteDecision.REJECT, reasoning=reasoning)
        else:
            return Vote(
                voter_id=voter_id,
                decision=VoteDecision.ABSTAIN,
                error="unparseable",
                reasoning=output.strip(),
            )

    @staticmethod
    def _aggregate(level: ConsensusLevel, votes: list[Vote]) -> ConsensusVerdict:
        # Abstain counts as approve (fail-open)
        approve_count = sum(
            1 for v in votes if v.decision in (VoteDecision.APPROVE, VoteDecision.ABSTAIN)
        )

        if level == ConsensusLevel.ANY:
            approved = approve_count >= 1
        elif level == ConsensusLevel.MAJORITY:
            approved = approve_count >= 2
        elif level in (ConsensusLevel.UNANIMOUS, ConsensusLevel.UNANIMOUS_PLUS_USER):
            approved = approve_count == NUM_VOTERS
        else:
            approved = True

        requires_user = level == ConsensusLevel.UNANIMOUS_PLUS_USER and approved

        return ConsensusVerdict(
            level=level,
            approved=approved,
            votes=votes,
            requires_user_confirmation=requires_user,
        )
