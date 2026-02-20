"""Tests for convergent.protocol and convergent.coordination_config."""

from __future__ import annotations

import json

import pytest
from convergent.coordination_config import CoordinationConfig
from convergent.protocol import (
    AgentIdentity,
    ConsensusRequest,
    Decision,
    DecisionOutcome,
    QuorumLevel,
    Signal,
    StigmergyMarker,
    Vote,
    VoteChoice,
)

# --- Enum tests ---


class TestQuorumLevel:
    def test_values(self) -> None:
        assert QuorumLevel.ANY == "any"
        assert QuorumLevel.MAJORITY == "majority"
        assert QuorumLevel.UNANIMOUS == "unanimous"
        assert QuorumLevel.UNANIMOUS_HUMAN == "unanimous_human"

    def test_from_string(self) -> None:
        assert QuorumLevel("majority") is QuorumLevel.MAJORITY

    def test_all_members(self) -> None:
        assert len(QuorumLevel) == 4


class TestVoteChoice:
    def test_values(self) -> None:
        assert VoteChoice.APPROVE == "approve"
        assert VoteChoice.REJECT == "reject"
        assert VoteChoice.ABSTAIN == "abstain"
        assert VoteChoice.ESCALATE == "escalate"

    def test_from_string(self) -> None:
        assert VoteChoice("escalate") is VoteChoice.ESCALATE

    def test_all_members(self) -> None:
        assert len(VoteChoice) == 4


class TestDecisionOutcome:
    def test_values(self) -> None:
        assert DecisionOutcome.APPROVED == "approved"
        assert DecisionOutcome.REJECTED == "rejected"
        assert DecisionOutcome.DEADLOCK == "deadlock"
        assert DecisionOutcome.ESCALATED == "escalated"

    def test_from_string(self) -> None:
        assert DecisionOutcome("deadlock") is DecisionOutcome.DEADLOCK

    def test_all_members(self) -> None:
        assert len(DecisionOutcome) == 4


# --- AgentIdentity tests ---


class TestAgentIdentity:
    def test_creation(self) -> None:
        agent = AgentIdentity(
            agent_id="agent-1",
            role="builder",
            model="claude:sonnet",
        )
        assert agent.agent_id == "agent-1"
        assert agent.role == "builder"
        assert agent.model == "claude:sonnet"
        assert agent.phi_score == 0.5  # default

    def test_custom_phi_score(self) -> None:
        agent = AgentIdentity("a", "r", "m", phi_score=0.8)
        assert agent.phi_score == 0.8

    def test_frozen(self) -> None:
        agent = AgentIdentity("a", "r", "m")
        with pytest.raises(AttributeError):
            agent.phi_score = 0.9  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        original = AgentIdentity("agent-1", "tester", "ollama:qwen", phi_score=0.7)
        restored = AgentIdentity.from_json(original.to_json())
        assert restored == original

    def test_json_structure(self) -> None:
        agent = AgentIdentity("a", "r", "m", phi_score=0.5)
        d = json.loads(agent.to_json())
        assert d == {
            "agent_id": "a",
            "role": "r",
            "model": "m",
            "phi_score": 0.5,
        }


# --- Vote tests ---


class TestVote:
    @pytest.fixture()
    def agent(self) -> AgentIdentity:
        return AgentIdentity("agent-1", "reviewer", "claude:sonnet", phi_score=0.8)

    def test_creation(self, agent: AgentIdentity) -> None:
        vote = Vote(
            agent=agent,
            choice=VoteChoice.APPROVE,
            confidence=0.9,
            reasoning="Code looks clean",
        )
        assert vote.agent == agent
        assert vote.choice is VoteChoice.APPROVE
        assert vote.confidence == 0.9
        assert vote.reasoning == "Code looks clean"
        assert vote.weighted_score == 0.0  # default, set by PhiScorer later
        assert vote.timestamp  # auto-populated

    def test_frozen(self, agent: AgentIdentity) -> None:
        vote = Vote(agent, VoteChoice.REJECT, 0.5, "bugs found")
        with pytest.raises(AttributeError):
            vote.confidence = 1.0  # type: ignore[misc]

    def test_json_roundtrip(self, agent: AgentIdentity) -> None:
        original = Vote(
            agent=agent,
            choice=VoteChoice.ESCALATE,
            confidence=0.3,
            reasoning="Not my domain",
            weighted_score=0.24,
        )
        restored = Vote.from_json(original.to_json())
        assert restored == original

    def test_json_preserves_enum(self, agent: AgentIdentity) -> None:
        vote = Vote(agent, VoteChoice.ABSTAIN, 0.0, "no opinion")
        d = json.loads(vote.to_json())
        assert d["choice"] == "abstain"

    def test_json_preserves_nested_agent(self, agent: AgentIdentity) -> None:
        vote = Vote(agent, VoteChoice.APPROVE, 1.0, "lgtm")
        d = json.loads(vote.to_json())
        assert d["agent"]["agent_id"] == "agent-1"
        assert d["agent"]["phi_score"] == 0.8


# --- ConsensusRequest tests ---


class TestConsensusRequest:
    def test_creation(self) -> None:
        req = ConsensusRequest(
            request_id="req-1",
            task_id="task-42",
            question="Should we merge this PR?",
            context="PR #123 adds auth middleware",
            quorum=QuorumLevel.MAJORITY,
        )
        assert req.request_id == "req-1"
        assert req.task_id == "task-42"
        assert req.quorum is QuorumLevel.MAJORITY
        assert req.artifacts == []  # default
        assert req.timeout_seconds == 300  # default
        assert req.requested_at  # auto-populated

    def test_with_artifacts(self) -> None:
        req = ConsensusRequest(
            "r",
            "t",
            "q",
            "c",
            QuorumLevel.UNANIMOUS,
            artifacts=["src/auth.py", "tests/test_auth.py"],
            timeout_seconds=600,
        )
        assert req.artifacts == ["src/auth.py", "tests/test_auth.py"]
        assert req.timeout_seconds == 600

    def test_frozen(self) -> None:
        req = ConsensusRequest("r", "t", "q", "c", QuorumLevel.ANY)
        with pytest.raises(AttributeError):
            req.question = "new question"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        original = ConsensusRequest(
            "req-2",
            "task-7",
            "Deploy to prod?",
            "All tests pass",
            QuorumLevel.UNANIMOUS_HUMAN,
            artifacts=["/deploy.yml"],
            timeout_seconds=120,
        )
        restored = ConsensusRequest.from_json(original.to_json())
        assert restored == original

    def test_json_preserves_enum(self) -> None:
        req = ConsensusRequest("r", "t", "q", "c", QuorumLevel.UNANIMOUS)
        d = json.loads(req.to_json())
        assert d["quorum"] == "unanimous"


# --- Decision tests ---


class TestDecision:
    @pytest.fixture()
    def consensus_request(self) -> ConsensusRequest:
        return ConsensusRequest(
            "req-1",
            "task-1",
            "Merge PR?",
            "context",
            QuorumLevel.MAJORITY,
        )

    @pytest.fixture()
    def votes(self) -> list[Vote]:
        agents = [
            AgentIdentity("a1", "builder", "claude:sonnet", 0.8),
            AgentIdentity("a2", "tester", "claude:haiku", 0.6),
            AgentIdentity("a3", "reviewer", "ollama:qwen", 0.5),
        ]
        return [
            Vote(agents[0], VoteChoice.APPROVE, 0.9, "looks good", weighted_score=0.72),
            Vote(agents[1], VoteChoice.APPROVE, 0.7, "tests pass", weighted_score=0.42),
            Vote(agents[2], VoteChoice.REJECT, 0.4, "style issues", weighted_score=0.20),
        ]

    def test_creation(
        self,
        consensus_request: ConsensusRequest,
        votes: list[Vote],
    ) -> None:
        decision = Decision(
            request=consensus_request,
            votes=votes,
            outcome=DecisionOutcome.APPROVED,
            total_weighted_approve=1.14,
            total_weighted_reject=0.20,
        )
        assert decision.outcome is DecisionOutcome.APPROVED
        assert len(decision.votes) == 3
        assert decision.total_weighted_approve == 1.14
        assert decision.human_override is None
        assert decision.reasoning_summary == ""

    def test_mutable(
        self,
        consensus_request: ConsensusRequest,
        votes: list[Vote],
    ) -> None:
        decision = Decision(consensus_request, votes, DecisionOutcome.DEADLOCK)
        decision.reasoning_summary = "Updated after evaluation"
        assert decision.reasoning_summary == "Updated after evaluation"
        decision.human_override = "Approved by admin"
        assert decision.human_override == "Approved by admin"

    def test_json_roundtrip(
        self,
        consensus_request: ConsensusRequest,
        votes: list[Vote],
    ) -> None:
        original = Decision(
            request=consensus_request,
            votes=votes,
            outcome=DecisionOutcome.APPROVED,
            total_weighted_approve=1.14,
            total_weighted_reject=0.20,
            reasoning_summary="Majority approved",
        )
        restored = Decision.from_json(original.to_json())
        assert restored.outcome == original.outcome
        assert restored.request == original.request
        assert len(restored.votes) == len(original.votes)
        assert restored.total_weighted_approve == original.total_weighted_approve
        assert restored.reasoning_summary == original.reasoning_summary
        for orig, rest in zip(original.votes, restored.votes, strict=True):
            assert orig == rest

    def test_json_preserves_enums(
        self,
        consensus_request: ConsensusRequest,
        votes: list[Vote],
    ) -> None:
        decision = Decision(consensus_request, votes, DecisionOutcome.ESCALATED)
        d = json.loads(decision.to_json())
        assert d["outcome"] == "escalated"
        assert d["request"]["quorum"] == "majority"
        assert d["votes"][0]["choice"] == "approve"


# --- StigmergyMarker tests ---


class TestStigmergyMarker:
    def test_creation(self) -> None:
        marker = StigmergyMarker(
            marker_id="m-1",
            agent_id="agent-1",
            marker_type="file_modified",
            target="src/auth.py",
            content="Added JWT validation",
        )
        assert marker.marker_id == "m-1"
        assert marker.marker_type == "file_modified"
        assert marker.strength == 1.0  # default
        assert marker.expires_at is None  # default
        assert marker.created_at  # auto-populated

    def test_custom_strength(self) -> None:
        marker = StigmergyMarker("m", "a", "known_issue", "t", "c", strength=0.5)
        assert marker.strength == 0.5

    def test_with_expiry(self) -> None:
        marker = StigmergyMarker(
            "m",
            "a",
            "pattern_found",
            "t",
            "c",
            expires_at="2026-12-31T23:59:59+00:00",
        )
        assert marker.expires_at == "2026-12-31T23:59:59+00:00"

    def test_frozen(self) -> None:
        marker = StigmergyMarker("m", "a", "dependency", "t", "c")
        with pytest.raises(AttributeError):
            marker.strength = 0.1  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        original = StigmergyMarker(
            "m-2",
            "agent-3",
            "quality_signal",
            "tests/test_auth.py",
            "Flaky test — intermittent timeout failures",
            strength=0.75,
            expires_at="2026-03-01T00:00:00+00:00",
        )
        restored = StigmergyMarker.from_json(original.to_json())
        assert restored == original


# --- Signal tests ---


class TestSignal:
    def test_broadcast_creation(self) -> None:
        signal = Signal(
            signal_type="task_complete",
            source_agent="agent-1",
        )
        assert signal.signal_type == "task_complete"
        assert signal.source_agent == "agent-1"
        assert signal.target_agent is None  # broadcast
        assert signal.payload == ""
        assert signal.timestamp  # auto-populated

    def test_targeted_signal(self) -> None:
        signal = Signal(
            signal_type="blocked",
            source_agent="agent-2",
            target_agent="agent-1",
            payload='{"blocked_on": "task-42"}',
        )
        assert signal.target_agent == "agent-1"
        assert json.loads(signal.payload) == {"blocked_on": "task-42"}

    def test_frozen(self) -> None:
        signal = Signal("conflict", "a1")
        with pytest.raises(AttributeError):
            signal.payload = "new"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        original = Signal(
            "resource_available",
            "agent-3",
            target_agent="agent-1",
            payload='{"resource": "gpu-0"}',
        )
        restored = Signal.from_json(original.to_json())
        assert restored == original

    def test_json_roundtrip_broadcast(self) -> None:
        original = Signal("task_complete", "agent-1")
        restored = Signal.from_json(original.to_json())
        assert restored == original
        assert restored.target_agent is None


# --- CoordinationConfig tests ---


class TestCoordinationConfig:
    def test_defaults(self) -> None:
        config = CoordinationConfig()
        assert config.db_path == "./convergent_coordination.db"
        assert config.default_quorum is QuorumLevel.MAJORITY
        assert config.phi_decay_rate == 0.05
        assert config.stigmergy_evaporation_rate == 0.1
        assert config.signal_bus_type == "sqlite"
        assert config.vote_timeout_seconds == 300

    def test_custom_values(self) -> None:
        config = CoordinationConfig(
            db_path="/tmp/test.db",
            default_quorum=QuorumLevel.UNANIMOUS,
            phi_decay_rate=0.1,
            stigmergy_evaporation_rate=0.2,
            signal_bus_type="redis",
            vote_timeout_seconds=60,
        )
        assert config.db_path == "/tmp/test.db"
        assert config.default_quorum is QuorumLevel.UNANIMOUS
        assert config.phi_decay_rate == 0.1

    def test_mutable(self) -> None:
        config = CoordinationConfig()
        config.vote_timeout_seconds = 600
        assert config.vote_timeout_seconds == 600

    def test_json_roundtrip(self) -> None:
        original = CoordinationConfig(
            db_path="/data/convergent.db",
            default_quorum=QuorumLevel.UNANIMOUS_HUMAN,
            phi_decay_rate=0.08,
        )
        restored = CoordinationConfig.from_json(original.to_json())
        assert restored.db_path == original.db_path
        assert restored.default_quorum == original.default_quorum
        assert restored.phi_decay_rate == original.phi_decay_rate

    def test_json_preserves_enum(self) -> None:
        config = CoordinationConfig(default_quorum=QuorumLevel.ANY)
        d = json.loads(config.to_json())
        assert d["default_quorum"] == "any"


# --- Cross-module import tests ---


class TestPublicAPI:
    """Verify all Phase 3 types are importable from the top-level package."""

    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "AgentIdentity")
        assert hasattr(convergent, "Vote")
        assert hasattr(convergent, "VoteChoice")
        assert hasattr(convergent, "ConsensusRequest")
        assert hasattr(convergent, "Decision")
        assert hasattr(convergent, "DecisionOutcome")
        assert hasattr(convergent, "QuorumLevel")
        assert hasattr(convergent, "Signal")
        assert hasattr(convergent, "StigmergyMarker")
        assert hasattr(convergent, "CoordinationConfig")

    def test_all_exports_listed(self) -> None:
        import convergent

        phase3_types = [
            "AgentIdentity",
            "ConsensusRequest",
            "CoordinationConfig",
            "Decision",
            "DecisionOutcome",
            "QuorumLevel",
            "Signal",
            "StigmergyMarker",
            "Vote",
            "VoteChoice",
        ]
        for name in phase3_types:
            assert name in convergent.__all__, f"{name} not in __all__"
