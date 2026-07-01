"""Tests for convergent.gorgon_bridge integration bridge."""

from __future__ import annotations

import pytest
from convergent.coordination_config import CoordinationConfig
from convergent.gorgon_bridge import GorgonBridge
from convergent.protocol import DecisionOutcome


@pytest.fixture()
def bridge() -> GorgonBridge:
    """Create an in-memory GorgonBridge for testing."""
    config = CoordinationConfig(db_path=":memory:")
    return GorgonBridge(config)


class TestEnrichPrompt:
    def test_empty_when_no_context(self, bridge: GorgonBridge) -> None:
        result = bridge.enrich_prompt("agent-1", "implement auth", [])
        assert result == ""

    def test_includes_stigmergy_markers(self, bridge: GorgonBridge) -> None:
        bridge.leave_marker("agent-1", "known_issue", "auth.py", "Race condition here")
        result = bridge.enrich_prompt("agent-2", "fix auth", ["auth.py"])
        assert "Race condition" in result
        assert "Stigmergy" in result

    def test_includes_flocking_separation(self, bridge: GorgonBridge) -> None:
        bridge.leave_marker("agent-1", "file_modified", "auth.py", "Editing auth")
        result = bridge.enrich_prompt("agent-2", "fix auth", ["auth.py"], current_work="fix auth")
        assert "Separation" in result

    def test_includes_trust_scores(self, bridge: GorgonBridge) -> None:
        bridge.record_task_outcome("agent-1", "review", "approved")
        result = bridge.enrich_prompt("agent-1", "review code", [])
        assert "Trust Scores" in result
        assert "review" in result

    def test_combines_multiple_sections(self, bridge: GorgonBridge) -> None:
        bridge.leave_marker("agent-1", "known_issue", "a.py", "Bug here")
        bridge.record_task_outcome("agent-2", "coding", "approved")
        result = bridge.enrich_prompt("agent-2", "fix bug", ["a.py"])
        assert "Stigmergy" in result
        assert "Trust Scores" in result


class TestConsensusWorkflow:
    def test_full_lifecycle(self, bridge: GorgonBridge) -> None:
        # Create request
        request_id = bridge.request_consensus(
            task_id="task-1",
            question="Should we merge this PR?",
            context="All tests pass",
        )
        assert request_id  # Non-empty UUID

        # Submit votes
        bridge.submit_agent_vote(
            request_id,
            "agent-1",
            "reviewer",
            "claude:sonnet",
            "approve",
            0.9,
            "Looks good",
        )
        bridge.submit_agent_vote(
            request_id,
            "agent-2",
            "tester",
            "claude:haiku",
            "approve",
            0.8,
            "Tests pass",
        )
        bridge.submit_agent_vote(
            request_id,
            "agent-3",
            "reviewer",
            "claude:sonnet",
            "reject",
            0.5,
            "Minor issues",
        )

        # Evaluate
        decision = bridge.evaluate(request_id)
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_custom_quorum(self, bridge: GorgonBridge) -> None:
        request_id = bridge.request_consensus(
            task_id="task-1",
            question="Deploy to production?",
            context="Critical change",
            quorum="unanimous",
        )
        bridge.submit_agent_vote(
            request_id,
            "agent-1",
            "reviewer",
            "m",
            "approve",
            0.9,
            "ok",
        )
        bridge.submit_agent_vote(
            request_id,
            "agent-2",
            "reviewer",
            "m",
            "reject",
            0.5,
            "no",
        )
        decision = bridge.evaluate(request_id)
        assert decision.outcome is DecisionOutcome.REJECTED

    def test_get_decision_after_evaluate(self, bridge: GorgonBridge) -> None:
        request_id = bridge.request_consensus("t", "q", "c")
        bridge.submit_agent_vote(request_id, "a", "r", "m", "approve", 0.9, "ok")
        bridge.evaluate(request_id)
        decision = bridge.get_decision(request_id)
        assert decision is not None
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_get_decision_before_evaluate(self, bridge: GorgonBridge) -> None:
        request_id = bridge.request_consensus("t", "q", "c")
        assert bridge.get_decision(request_id) is None

    def test_vote_history(self, bridge: GorgonBridge) -> None:
        r1 = bridge.request_consensus("task-1", "q1", "c1")
        r2 = bridge.request_consensus("task-1", "q2", "c2")
        bridge.submit_agent_vote(r1, "a", "r", "m", "approve", 0.9, "ok")
        bridge.submit_agent_vote(r2, "a", "r", "m", "reject", 0.5, "no")
        bridge.evaluate(r1)
        bridge.evaluate(r2)
        history = bridge.get_vote_history("task-1")
        assert len(history) == 2

    def test_with_artifacts(self, bridge: GorgonBridge) -> None:
        request_id = bridge.request_consensus(
            "task-1",
            "Review PR",
            "context",
            artifacts=["PR #42", "src/auth.py"],
        )
        bridge.submit_agent_vote(request_id, "a", "r", "m", "approve", 0.9, "ok")
        decision = bridge.evaluate(request_id)
        assert decision.request.artifacts == ["PR #42", "src/auth.py"]


class TestRecordOutcome:
    def test_updates_phi_score(self, bridge: GorgonBridge) -> None:
        score = bridge.record_task_outcome("agent-1", "review", "approved")
        assert score > 0.5  # Approval increases score from 0.5 prior

    def test_leaves_file_markers(self, bridge: GorgonBridge) -> None:
        bridge.record_task_outcome("agent-1", "coding", "approved", file_paths=["a.py", "b.py"])
        markers_a = bridge.stigmergy.get_markers("a.py")
        markers_b = bridge.stigmergy.get_markers("b.py")
        assert len(markers_a) == 1
        assert len(markers_b) == 1
        assert markers_a[0].marker_type == "file_modified"

    def test_no_markers_without_files(self, bridge: GorgonBridge) -> None:
        bridge.record_task_outcome("agent-1", "review", "approved")
        # No files = no markers (just score update)
        assert bridge.stigmergy.count() == 0

    def test_rejection_decreases_score(self, bridge: GorgonBridge) -> None:
        score = bridge.record_task_outcome("agent-1", "review", "rejected")
        assert score < 0.5


class TestGetAgentScore:
    def test_new_agent_gets_prior(self, bridge: GorgonBridge) -> None:
        assert bridge.get_agent_score("new-agent", "review") == 0.5

    def test_score_updates_after_outcomes(self, bridge: GorgonBridge) -> None:
        bridge.record_task_outcome("agent-1", "review", "approved")
        bridge.record_task_outcome("agent-1", "review", "approved")
        score = bridge.get_agent_score("agent-1", "review")
        assert score > 0.5


class TestLeaveMarker:
    def test_leaves_marker(self, bridge: GorgonBridge) -> None:
        bridge.leave_marker("agent-1", "pattern_found", "api.py", "Use REST conventions")
        markers = bridge.stigmergy.get_markers("api.py")
        assert len(markers) == 1
        assert markers[0].content == "Use REST conventions"


class TestEvaporate:
    def test_evaporate_returns_count(self, bridge: GorgonBridge) -> None:
        removed = bridge.evaporate_markers()
        assert removed == 0  # Nothing to evaporate


class TestProperties:
    def test_scorer_accessible(self, bridge: GorgonBridge) -> None:
        assert bridge.scorer is not None

    def test_triumvirate_accessible(self, bridge: GorgonBridge) -> None:
        assert bridge.triumvirate is not None

    def test_stigmergy_accessible(self, bridge: GorgonBridge) -> None:
        assert bridge.stigmergy is not None

    def test_flocking_accessible(self, bridge: GorgonBridge) -> None:
        assert bridge.flocking is not None

    def test_signal_bus_none_for_memory(self, bridge: GorgonBridge) -> None:
        # In-memory mode has no signal bus (no filesystem)
        assert bridge.signal_bus is None


class TestFilePersistence:
    def test_file_backed_bridge(self, tmp_path: object) -> None:
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "coord.db")
        config = CoordinationConfig(db_path=db_path)
        b1 = GorgonBridge(config)
        b1.record_task_outcome("agent-1", "review", "approved")
        b1.leave_marker("agent-1", "known_issue", "a.py", "Bug")
        b1.close()

        b2 = GorgonBridge(config)
        assert b2.get_agent_score("agent-1", "review") > 0.5
        assert len(b2.stigmergy.get_markers("a.py")) == 1
        assert b2.signal_bus is not None
        b2.close()


class TestSignalBusBackendSelection:
    def test_default_uses_sqlite_backend(self, tmp_path: object) -> None:
        import pathlib

        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        db_path = str(pathlib.Path(str(tmp_path)) / "coord.db")
        config = CoordinationConfig(db_path=db_path)
        bridge = GorgonBridge(config)
        assert bridge.signal_bus is not None
        assert isinstance(bridge.signal_bus.backend, SQLiteSignalBackend)
        bridge.close()

    def test_explicit_filesystem_backend(self, tmp_path: object) -> None:
        import pathlib

        from convergent.signal_backend import FilesystemSignalBackend

        db_path = str(pathlib.Path(str(tmp_path)) / "coord.db")
        config = CoordinationConfig(db_path=db_path, signal_bus_type="filesystem")
        bridge = GorgonBridge(config)
        assert bridge.signal_bus is not None
        assert isinstance(bridge.signal_bus.backend, FilesystemSignalBackend)
        bridge.close()

    def test_memory_has_no_signal_bus(self) -> None:
        config = CoordinationConfig(db_path=":memory:")
        bridge = GorgonBridge(config)
        assert bridge.signal_bus is None

    def test_sqlite_signal_bus_stores_signals(self, tmp_path: object) -> None:
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "coord.db")
        config = CoordinationConfig(db_path=db_path)
        bridge = GorgonBridge(config)
        bridge.record_task_outcome("agent-1", "review", "approved")
        signals = bridge.signal_bus.get_signals()
        assert len(signals) == 1
        assert signals[0].signal_type == "task_outcome"
        bridge.close()

    def test_close_stops_signal_bus(self, tmp_path: object) -> None:
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "coord.db")
        config = CoordinationConfig(db_path=db_path)
        bridge = GorgonBridge(config)
        bridge.signal_bus.start_polling()
        assert bridge.signal_bus.is_polling
        bridge.close()
        assert not bridge.signal_bus.is_polling

    def test_sqlite_signal_db_path(self, tmp_path: object) -> None:
        import pathlib

        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        db_path = str(pathlib.Path(str(tmp_path)) / "coord.db")
        config = CoordinationConfig(db_path=db_path)
        bridge = GorgonBridge(config)
        backend = bridge.signal_bus.backend
        assert isinstance(backend, SQLiteSignalBackend)
        expected = str(pathlib.Path(db_path).with_suffix(".signals.db"))
        assert backend.db_path == expected
        bridge.close()


class TestDecisionHistoryViabridge:
    def test_get_decision_history(self, bridge: GorgonBridge) -> None:
        request_id = bridge.request_consensus("task-1", "q", "c")
        bridge.submit_agent_vote(request_id, "a1", "r", "m", "approve", 0.9, "ok")
        bridge.evaluate(request_id)
        history = bridge.get_decision_history()
        assert len(history) == 1
        assert history[0]["task_id"] == "task-1"
        assert history[0]["outcome"] == "approved"

    def test_get_decision_history_filtered(self, bridge: GorgonBridge) -> None:
        r1 = bridge.request_consensus("task-1", "q1", "c")
        bridge.submit_agent_vote(r1, "a1", "r", "m", "approve", 0.9, "ok")
        bridge.evaluate(r1)
        r2 = bridge.request_consensus("task-2", "q2", "c")
        bridge.submit_agent_vote(r2, "a1", "r", "m", "approve", 0.9, "ok")
        bridge.evaluate(r2)
        history = bridge.get_decision_history(task_id="task-1")
        assert len(history) == 1

    def test_get_agent_vote_stats(self, bridge: GorgonBridge) -> None:
        r1 = bridge.request_consensus("t1", "q", "c")
        bridge.submit_agent_vote(r1, "a1", "r", "m", "approve", 0.9, "ok")
        bridge.evaluate(r1)
        r2 = bridge.request_consensus("t2", "q", "c")
        bridge.submit_agent_vote(r2, "a1", "r", "m", "reject", 0.5, "no")
        bridge.evaluate(r2)
        stats = bridge.get_agent_vote_stats("a1")
        assert stats["total"] == 2
        assert stats["approve_count"] == 1
        assert stats["reject_count"] == 1

    def test_get_agent_vote_stats_empty(self, bridge: GorgonBridge) -> None:
        stats = bridge.get_agent_vote_stats("nonexistent")
        assert stats["total"] == 0


class TestPayloadSafety:
    def test_outcome_payload_is_valid_json(self, tmp_path: object) -> None:
        """Signal payload must be valid JSON even with special characters."""
        import json
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "coord.db")
        config = CoordinationConfig(db_path=db_path)
        bridge = GorgonBridge(config)
        # Use skill_domain with characters that would break f-string JSON
        bridge.record_task_outcome("agent-1", 'code "review"', "approved")
        signals = bridge.signal_bus.get_signals()
        assert len(signals) == 1
        payload = json.loads(signals[0].payload)
        assert payload["skill_domain"] == 'code "review"'
        assert payload["outcome"] == "approved"
        bridge.close()


class TestClose:
    def test_close_does_not_raise(self, bridge: GorgonBridge) -> None:
        bridge.close()  # Should not raise


class TestDefaultConfig:
    def test_no_config_uses_defaults(self) -> None:
        bridge = GorgonBridge()
        assert bridge.get_agent_score("agent-1", "review") == 0.5
        bridge.close()


class TestPublicAPI:
    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "GorgonBridge")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "GorgonBridge" in convergent.__all__

    def test_resolution_types_exported(self) -> None:
        import convergent

        assert "Adjustment" in convergent.__all__
        assert "ConflictReport" in convergent.__all__
        assert "ResolutionResult" in convergent.__all__
