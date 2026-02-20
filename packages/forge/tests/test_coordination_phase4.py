"""Tests for Convergent Phase 4 integration (health, cycles, event log)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.agents.convergence import (
    HAS_CONVERGENT,
    check_dependency_cycles,
    create_event_log,
    get_coordination_health,
    get_execution_order,
)

# ---------------------------------------------------------------------------
# create_event_log
# ---------------------------------------------------------------------------


class TestCreateEventLog:
    """Test the create_event_log factory function."""

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_returns_event_log_when_convergent_available(self, tmp_path):
        db_path = str(tmp_path / "test.events.db")
        event_log = create_event_log(db_path=db_path)
        assert event_log is not None
        event_log.close()

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_in_memory_mode(self):
        event_log = create_event_log(db_path=":memory:")
        assert event_log is not None
        event_log.close()

    def test_returns_none_when_convergent_unavailable(self, monkeypatch):
        import animus_forge.agents.convergence as conv_mod

        monkeypatch.setattr(conv_mod, "HAS_CONVERGENT", False)
        assert create_event_log() is None

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_returns_none_on_exception(self):
        with patch("convergent.EventLog", side_effect=RuntimeError("boom")):
            result = create_event_log(db_path=":memory:")
            assert result is None

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_default_path_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event_log = create_event_log()
        assert event_log is not None
        assert (tmp_path / ".gorgon").exists()
        event_log.close()


# ---------------------------------------------------------------------------
# get_coordination_health
# ---------------------------------------------------------------------------


class TestGetCoordinationHealth:
    """Test the get_coordination_health helper."""

    def test_returns_empty_when_bridge_is_none(self):
        assert get_coordination_health(None) == {}

    def test_returns_empty_when_convergent_unavailable(self, monkeypatch):
        import animus_forge.agents.convergence as conv_mod

        monkeypatch.setattr(conv_mod, "HAS_CONVERGENT", False)
        assert get_coordination_health(MagicMock()) == {}

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_returns_health_dict(self):
        from convergent import CoordinationConfig, GorgonBridge

        bridge = GorgonBridge(CoordinationConfig(db_path=":memory:"))
        try:
            health = get_coordination_health(bridge)
            assert isinstance(health, dict)
            assert "grade" in health
            assert "issues" in health
            assert health["grade"] == "A"  # Empty system = healthy
        finally:
            bridge.close()

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_handles_exception_gracefully(self):
        bridge = MagicMock()
        # HealthChecker.from_bridge will try to access attrs, mock will provide them
        # but we'll break it at the check() level
        with patch(
            "convergent.HealthChecker.from_bridge",
            side_effect=RuntimeError("broken"),
        ):
            result = get_coordination_health(bridge)
            assert result == {}


# ---------------------------------------------------------------------------
# check_dependency_cycles
# ---------------------------------------------------------------------------


class TestCheckDependencyCycles:
    """Test the check_dependency_cycles helper."""

    def test_returns_empty_when_resolver_is_none(self):
        assert check_dependency_cycles(None) == []

    def test_returns_empty_when_convergent_unavailable(self, monkeypatch):
        import animus_forge.agents.convergence as conv_mod

        monkeypatch.setattr(conv_mod, "HAS_CONVERGENT", False)
        assert check_dependency_cycles(MagicMock()) == []

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_no_cycles_returns_empty(self):
        from convergent import (
            Evidence,
            EvidenceKind,
            Intent,
            IntentResolver,
            InterfaceKind,
            InterfaceSpec,
            PythonGraphBackend,
        )

        backend = PythonGraphBackend()
        backend.publish(
            Intent(
                agent_id="a1",
                intent="auth",
                provides=[InterfaceSpec(name="Auth", kind=InterfaceKind.FUNCTION, signature="")],
                evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="ok")],
            )
        )
        resolver = IntentResolver(backend=backend)
        assert check_dependency_cycles(resolver) == []

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_detects_cycles(self):
        from convergent import (
            Evidence,
            EvidenceKind,
            Intent,
            IntentResolver,
            InterfaceKind,
            InterfaceSpec,
            PythonGraphBackend,
        )

        backend = PythonGraphBackend()
        backend.publish(
            Intent(
                agent_id="a1",
                intent="A",
                provides=[
                    InterfaceSpec(name="AService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                requires=[
                    InterfaceSpec(name="BService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="ok")],
            )
        )
        backend.publish(
            Intent(
                agent_id="a2",
                intent="B",
                provides=[
                    InterfaceSpec(name="BService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                requires=[
                    InterfaceSpec(name="AService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="ok")],
            )
        )
        resolver = IntentResolver(backend=backend)
        cycles = check_dependency_cycles(resolver)
        assert len(cycles) >= 1
        assert "intent_ids" in cycles[0]
        assert "display" in cycles[0]


# ---------------------------------------------------------------------------
# get_execution_order
# ---------------------------------------------------------------------------


class TestGetExecutionOrder:
    """Test the get_execution_order helper."""

    def test_returns_empty_when_resolver_is_none(self):
        assert get_execution_order(None) == []

    def test_returns_empty_when_convergent_unavailable(self, monkeypatch):
        import animus_forge.agents.convergence as conv_mod

        monkeypatch.setattr(conv_mod, "HAS_CONVERGENT", False)
        assert get_execution_order(MagicMock()) == []

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_returns_ordered_list(self):
        from convergent import (
            Evidence,
            EvidenceKind,
            Intent,
            IntentResolver,
            InterfaceKind,
            InterfaceSpec,
            PythonGraphBackend,
        )

        backend = PythonGraphBackend()
        backend.publish(
            Intent(
                agent_id="a1",
                intent="db",
                provides=[
                    InterfaceSpec(name="DBService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="ok")],
            )
        )
        backend.publish(
            Intent(
                agent_id="a2",
                intent="api",
                requires=[
                    InterfaceSpec(name="DBService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                provides=[InterfaceSpec(name="API", kind=InterfaceKind.FUNCTION, signature="")],
                evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="ok")],
            )
        )
        resolver = IntentResolver(backend=backend)
        order = get_execution_order(resolver)
        assert order.index("db") < order.index("api")

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_returns_empty_on_cycle(self):
        from convergent import (
            Evidence,
            EvidenceKind,
            Intent,
            IntentResolver,
            InterfaceKind,
            InterfaceSpec,
            PythonGraphBackend,
        )

        backend = PythonGraphBackend()
        backend.publish(
            Intent(
                agent_id="a1",
                intent="A",
                provides=[
                    InterfaceSpec(name="AService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                requires=[
                    InterfaceSpec(name="BService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="ok")],
            )
        )
        backend.publish(
            Intent(
                agent_id="a2",
                intent="B",
                provides=[
                    InterfaceSpec(name="BService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                requires=[
                    InterfaceSpec(name="AService", kind=InterfaceKind.FUNCTION, signature="")
                ],
                evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="ok")],
            )
        )
        resolver = IntentResolver(backend=backend)
        # Cycles cause ValueError in topological_order, caught and returns []
        result = get_execution_order(resolver)
        assert result == []


# ---------------------------------------------------------------------------
# Coordination API routes
# ---------------------------------------------------------------------------


class TestCoordinationAPI:
    """Test coordination API endpoints."""

    def test_health_no_convergent(self, monkeypatch):
        from animus_forge.api_routes.coordination import coordination_health

        monkeypatch.setattr(
            "animus_forge.api_routes.coordination.state",
            MagicMock(coordination_bridge=None),
        )
        with patch("animus_forge.agents.convergence.HAS_CONVERGENT", False):
            result = coordination_health()
        assert result["available"] is False

    def test_health_no_bridge(self, monkeypatch):
        from animus_forge.api_routes.coordination import coordination_health

        mock_state = MagicMock()
        mock_state.coordination_bridge = None
        monkeypatch.setattr("animus_forge.api_routes.coordination.state", mock_state)

        if not HAS_CONVERGENT:
            pytest.skip("convergent not installed")

        result = coordination_health()
        assert "no active coordination bridge" in result.get("reason", "")

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_health_with_bridge(self, monkeypatch):
        from convergent import CoordinationConfig, GorgonBridge

        from animus_forge.api_routes.coordination import coordination_health

        bridge = GorgonBridge(CoordinationConfig(db_path=":memory:"))
        mock_state = MagicMock()
        mock_state.coordination_bridge = bridge
        monkeypatch.setattr("animus_forge.api_routes.coordination.state", mock_state)

        try:
            result = coordination_health()
            assert result["available"] is True
            assert "grade" in result
        finally:
            bridge.close()

    def test_events_no_convergent(self, monkeypatch):
        from animus_forge.api_routes.coordination import coordination_events

        monkeypatch.setattr(
            "animus_forge.api_routes.coordination.state",
            MagicMock(coordination_event_log=None),
        )
        with patch("animus_forge.agents.convergence.HAS_CONVERGENT", False):
            result = coordination_events()
        assert result["available"] is False

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_events_with_log(self, monkeypatch):
        from convergent import EventLog, EventType

        from animus_forge.api_routes.coordination import coordination_events

        log = EventLog(":memory:")
        log.record(EventType.INTENT_PUBLISHED, agent_id="a1", payload={"task": "test"})

        mock_state = MagicMock()
        mock_state.coordination_event_log = log
        monkeypatch.setattr("animus_forge.api_routes.coordination.state", mock_state)

        result = coordination_events(event_type=None, agent=None, limit=50)
        assert result["available"] is True
        assert result["count"] == 1
        assert result["events"][0]["agent_id"] == "a1"
        log.close()

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_events_invalid_type(self, monkeypatch):
        from convergent import EventLog

        from animus_forge.api_routes.coordination import coordination_events

        log = EventLog(":memory:")
        mock_state = MagicMock()
        mock_state.coordination_event_log = log
        monkeypatch.setattr("animus_forge.api_routes.coordination.state", mock_state)

        result = coordination_events(event_type="nonexistent_type", agent=None, limit=50)
        assert "error" in result
        assert "valid_types" in result
        log.close()

    def test_events_no_log(self, monkeypatch):
        from animus_forge.api_routes.coordination import coordination_events

        if not HAS_CONVERGENT:
            pytest.skip("convergent not installed")

        mock_state = MagicMock()
        mock_state.coordination_event_log = None
        monkeypatch.setattr("animus_forge.api_routes.coordination.state", mock_state)

        result = coordination_events(event_type=None, agent=None, limit=50)
        assert result["available"] is True
        assert result["events"] == []


# ---------------------------------------------------------------------------
# Coordination CLI
# ---------------------------------------------------------------------------


class TestCoordinationCLI:
    """Test coordination CLI commands."""

    def test_health_no_convergent(self, monkeypatch):
        from typer.testing import CliRunner

        from animus_forge.cli.commands.coordination import coordination_app

        monkeypatch.setattr("animus_forge.agents.convergence.HAS_CONVERGENT", False)
        runner = CliRunner()
        result = runner.invoke(coordination_app, ["health"])
        assert result.exit_code == 1
        assert "not installed" in result.output

    def test_health_no_db_file(self, tmp_path):
        if not HAS_CONVERGENT:
            pytest.skip("convergent not installed")

        from typer.testing import CliRunner

        from animus_forge.cli.commands.coordination import coordination_app

        runner = CliRunner()
        result = runner.invoke(
            coordination_app, ["health", "--db", str(tmp_path / "nonexistent.db")]
        )
        assert result.exit_code == 1
        assert "No coordination database" in result.output

    def test_events_no_convergent(self, monkeypatch):
        from typer.testing import CliRunner

        from animus_forge.cli.commands.coordination import coordination_app

        monkeypatch.setattr("animus_forge.agents.convergence.HAS_CONVERGENT", False)
        runner = CliRunner()
        result = runner.invoke(coordination_app, ["events"])
        assert result.exit_code == 1
        assert "not installed" in result.output

    def test_cycles_no_convergent(self, monkeypatch):
        from typer.testing import CliRunner

        from animus_forge.cli.commands.coordination import coordination_app

        monkeypatch.setattr("animus_forge.agents.convergence.HAS_CONVERGENT", False)
        runner = CliRunner()
        result = runner.invoke(coordination_app, ["cycles"])
        assert result.exit_code == 1
        assert "not installed" in result.output

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_events_with_db(self, tmp_path):
        from convergent import EventLog, EventType
        from typer.testing import CliRunner

        from animus_forge.cli.commands.coordination import coordination_app

        # Create an events DB with data
        db_path = str(tmp_path / "test.events.db")
        log = EventLog(db_path)
        log.record(EventType.INTENT_PUBLISHED, agent_id="a1", payload={"task": "test"})
        log.close()

        runner = CliRunner()
        result = runner.invoke(coordination_app, ["events", "--db", db_path])
        assert result.exit_code == 0
        assert "intent_published" in result.output


# ---------------------------------------------------------------------------
# Supervisor event recording
# ---------------------------------------------------------------------------


class TestSupervisorEventRecording:
    """Test that SupervisorAgent records coordination events."""

    def test_supervisor_accepts_event_log_kwarg(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        event_log = MagicMock()
        sup = SupervisorAgent(provider=provider, event_log=event_log)
        assert sup._event_log is event_log

    def test_supervisor_works_without_event_log(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        sup = SupervisorAgent(provider=provider)
        assert sup._event_log is None

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_coherence_events_recorded(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        event_log = MagicMock()

        # Create a real checker that will publish intents
        from animus_forge.agents.convergence import create_checker

        checker = create_checker()

        sup = SupervisorAgent(
            provider=provider,
            convergence_checker=checker,
            event_log=event_log,
        )

        asyncio.run(
            sup._execute_delegations(
                [
                    {"agent": "builder", "task": "Build it"},
                    {"agent": "tester", "task": "Test it"},
                ],
                [],
                lambda _: None,
            )
        )

        # Should have recorded INTENT_PUBLISHED for each delegation
        record_calls = event_log.record.call_args_list
        event_types = [str(c.args[0]) for c in record_calls if c.args]
        assert any("INTENT_PUBLISHED" in et for et in event_types)

    def test_outcome_events_recorded(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        bridge = MagicMock()
        bridge.record_task_outcome.return_value = 0.6
        event_log = MagicMock()

        sup = SupervisorAgent(
            provider=provider,
            coordination_bridge=bridge,
            event_log=event_log,
        )

        asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )

        # Should have recorded SCORE_UPDATED
        record_calls = event_log.record.call_args_list
        event_types = [str(c.args[0]) for c in record_calls if c.args]
        assert any("SCORE_UPDATED" in et for et in event_types)

    def test_decision_events_recorded(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="result text")
        bridge = MagicMock()

        mock_decision = MagicMock()
        mock_decision.outcome.value = "approved"
        mock_decision.reasoning_summary = "looks good"
        bridge.evaluate.return_value = mock_decision
        bridge.request_consensus.return_value = "req-1"
        bridge.record_task_outcome.return_value = 0.7

        event_log = MagicMock()

        sup = SupervisorAgent(
            provider=provider,
            coordination_bridge=bridge,
            event_log=event_log,
        )

        asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                        "_skill_name": "code-builder",
                    }
                ],
                [],
                lambda _: None,
            )
        )

        record_calls = event_log.record.call_args_list
        event_types = [str(c.args[0]) for c in record_calls if c.args]
        assert any("DECISION_MADE" in et for et in event_types)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestEventLogGracefulDegradation:
    """Test that broken event log never breaks the supervisor pipeline."""

    def test_broken_event_log_does_not_break_coherence(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        event_log = MagicMock()
        event_log.record.side_effect = RuntimeError("DB locked")

        # Mock checker that reports conflicts
        checker = MagicMock()
        checker.enabled = True
        convergence_result = MagicMock()
        convergence_result.has_conflicts = True
        convergence_result.conflicts = [{"agent": "builder", "description": "overlap"}]
        convergence_result.dropped_agents = set()
        convergence_result.adjustments = []
        checker.check_delegations.return_value = convergence_result

        sup = SupervisorAgent(
            provider=provider,
            convergence_checker=checker,
            event_log=event_log,
        )

        result = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert "builder" in result
        assert result["builder"] == "done"

    def test_broken_event_log_does_not_break_outcome_recording(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        bridge = MagicMock()
        bridge.record_task_outcome.return_value = 0.6
        event_log = MagicMock()
        event_log.record.side_effect = RuntimeError("DB locked")

        sup = SupervisorAgent(
            provider=provider,
            coordination_bridge=bridge,
            event_log=event_log,
        )

        result = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert result["builder"] == "done"
        # Bridge outcome recording should still succeed
        bridge.record_task_outcome.assert_called_once()
