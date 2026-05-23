"""Integration: verify mutation sites emit EventLog events.

Closes Constitutional Principle P3 (Transparency) on four
coordination mutation sites:

    INTENT_PUBLISHED — PythonGraphBackend.publish (resolver.py)
    VOTE_CAST       — Triumvirate.submit_vote   (triumvirate.py)
    DECISION_MADE   — Triumvirate.evaluate      (triumvirate.py)
    MARKER_LEFT     — StigmergyField.leave_marker (stigmergy.py)

The fifth and sixth (SCORE_UPDATED, INTENT_RESOLVED) defer to
the Week 3-4 active-inference scorer registry, which provides the
hook point Intent.add_evidence and stability transitions need.
See ``docs/specs/quorum_v2_week3-4_active_inference_resolver.md``.
"""

from __future__ import annotations

from convergent.coordination_config import CoordinationConfig
from convergent.event_log import EventLog, EventType
from convergent.intent import Intent
from convergent.protocol import AgentIdentity, QuorumLevel, Vote, VoteChoice
from convergent.resolver import PythonGraphBackend
from convergent.score_store import ScoreStore
from convergent.scoring import PhiScorer
from convergent.stigmergy import StigmergyField
from convergent.triumvirate import Triumvirate


def _agent(name: str = "agent-1") -> AgentIdentity:
    return AgentIdentity(name, "reviewer", "claude:sonnet", phi_score=0.5)


def _vote(
    agent: AgentIdentity,
    choice: VoteChoice = VoteChoice.APPROVE,
) -> Vote:
    return Vote(agent=agent, choice=choice, confidence=0.8, reasoning="t")


class TestIntentPublishedEmits:
    def test_publish_emits_intent_published(self) -> None:
        log = EventLog(":memory:")
        backend = PythonGraphBackend(event_log=log)
        intent = Intent(agent_id="agent-1", intent="build_auth")
        backend.publish(intent)
        events = log.query(event_type=EventType.INTENT_PUBLISHED)
        assert len(events) == 1
        assert events[0].agent_id == "agent-1"
        assert events[0].correlation_id == intent.id
        assert events[0].payload["intent"] == "build_auth"
        assert events[0].payload["intent_id"] == intent.id

    def test_publish_without_event_log_unchanged(self) -> None:
        backend = PythonGraphBackend()  # no event_log
        intent = Intent(agent_id="agent-1", intent="x")
        # Returns stability, doesn't raise
        score = backend.publish(intent)
        assert 0.0 <= score <= 1.0


class TestVoteCastEmits:
    def test_submit_vote_emits_vote_cast(self) -> None:
        log = EventLog(":memory:")
        scorer = PhiScorer(ScoreStore(":memory:"))
        tri = Triumvirate(scorer, CoordinationConfig(), event_log=log)
        req = tri.create_request("task-1", "Q?", "ctx")
        tri.submit_vote(req.request_id, _vote(_agent("voter-7")))
        events = log.query(event_type=EventType.VOTE_CAST)
        assert len(events) == 1
        ev = events[0]
        assert ev.agent_id == "voter-7"
        assert ev.correlation_id == req.request_id
        assert ev.payload["choice"] == "approve"
        assert "weight" in ev.payload

    def test_submit_vote_without_event_log_unchanged(self) -> None:
        scorer = PhiScorer(ScoreStore(":memory:"))
        tri = Triumvirate(scorer, CoordinationConfig())
        req = tri.create_request("task-1", "Q?", "ctx")
        tri.submit_vote(req.request_id, _vote(_agent("voter-7")))
        assert len(tri._votes[req.request_id]) == 1


class TestDecisionMadeEmits:
    def test_evaluate_emits_decision_made(self) -> None:
        log = EventLog(":memory:")
        scorer = PhiScorer(ScoreStore(":memory:"))
        tri = Triumvirate(scorer, CoordinationConfig(), event_log=log)
        req = tri.create_request("task-1", "Q?", "ctx", quorum=QuorumLevel.MAJORITY)
        tri.submit_vote(req.request_id, _vote(_agent("a")))
        tri.submit_vote(req.request_id, _vote(_agent("b")))
        decision = tri.evaluate(req.request_id)
        events = log.query(event_type=EventType.DECISION_MADE)
        assert len(events) == 1
        ev = events[0]
        assert ev.agent_id == "triumvirate"
        assert ev.correlation_id == req.request_id
        assert ev.payload["outcome"] == decision.outcome.value
        assert ev.payload["task_id"] == "task-1"
        assert ev.payload["vote_count"] == 2

    def test_deadlock_decision_also_emits(self) -> None:
        log = EventLog(":memory:")
        scorer = PhiScorer(ScoreStore(":memory:"))
        tri = Triumvirate(scorer, CoordinationConfig(), event_log=log)
        req = tri.create_request("task-1", "Q?", "ctx")
        # Don't submit any votes → DEADLOCK path
        tri.evaluate(req.request_id)
        events = log.query(event_type=EventType.DECISION_MADE)
        assert len(events) == 1
        assert events[0].payload["outcome"] == "deadlock"


class TestMarkerLeftEmits:
    def test_leave_marker_emits_marker_left(self) -> None:
        log = EventLog(":memory:")
        field = StigmergyField(":memory:", event_log=log)
        marker = field.leave_marker(
            agent_id="agent-1",
            marker_type="file_modified",
            target="src/auth.py",
            content="changed login flow",
            strength=0.9,
        )
        events = log.query(event_type=EventType.MARKER_LEFT)
        assert len(events) == 1
        ev = events[0]
        assert ev.agent_id == "agent-1"
        assert ev.correlation_id == "src/auth.py"
        assert ev.payload["marker_type"] == "file_modified"
        assert ev.payload["target"] == "src/auth.py"
        assert ev.payload["strength"] == 0.9
        assert ev.payload["marker_id"] == marker.marker_id

    def test_leave_marker_without_event_log_unchanged(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker(
            agent_id="agent-1",
            marker_type="t",
            target="src/x.py",
            content="c",
        )
        assert marker.agent_id == "agent-1"


class TestEventsAreBitemporal:
    def test_emitted_events_have_valid_from_and_recorded_at(self) -> None:
        log = EventLog(":memory:")
        backend = PythonGraphBackend(event_log=log)
        backend.publish(Intent(agent_id="agent-1", intent="x"))
        events = log.query(event_type=EventType.INTENT_PUBLISHED)
        assert events[0].valid_from is not None
        assert events[0].recorded_at is not None
        assert events[0].valid_from == events[0].recorded_at
