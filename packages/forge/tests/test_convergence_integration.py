"""Tests for the Convergent ↔ Gorgon integration adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.agents.convergence import (
    HAS_CONVERGENT,
    ConvergenceResult,
    DelegationConvergenceChecker,
    create_bridge,
    create_checker,
    format_convergence_alert,
)


class TestConvergenceResult:
    """Test the ConvergenceResult dataclass."""

    def test_clean_result(self):
        result = ConvergenceResult()
        assert not result.has_conflicts
        assert result.adjustments == []
        assert result.conflicts == []
        assert result.dropped_agents == set()

    def test_result_with_conflicts(self):
        result = ConvergenceResult(
            conflicts=[{"agent": "builder", "description": "overlap with tester"}],
        )
        assert result.has_conflicts


class TestDelegationConvergenceChecker:
    """Test the adapter between Gorgon delegations and Convergent."""

    def test_disabled_without_resolver(self):
        checker = DelegationConvergenceChecker(resolver=None)
        assert not checker.enabled
        result = checker.check_delegations([{"agent": "builder", "task": "Build it"}])
        assert not result.has_conflicts
        assert result.dropped_agents == set()

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_enabled_with_convergent(self):
        from convergent import IntentResolver

        resolver = IntentResolver(min_stability=0.0)
        checker = DelegationConvergenceChecker(resolver=resolver)
        assert checker.enabled

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_independent_agents_no_conflicts(self):
        from convergent import IntentResolver

        resolver = IntentResolver(min_stability=0.0)
        checker = DelegationConvergenceChecker(resolver=resolver)

        delegations = [
            {"agent": "builder", "task": "Build the auth module"},
            {"agent": "tester", "task": "Write tests for the API"},
            {"agent": "documenter", "task": "Document the endpoints"},
        ]
        result = checker.check_delegations(delegations)
        # Independent roles with different tags should not conflict
        assert result.dropped_agents == set()

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_overlapping_delegations_detected(self):
        from convergent import IntentResolver

        resolver = IntentResolver(min_stability=0.0)
        checker = DelegationConvergenceChecker(resolver=resolver)

        # planner and architect share "architecture" and "design" tags (2+ overlap)
        delegations = [
            {"agent": "planner", "task": "Design the auth architecture"},
            {"agent": "architect", "task": "Design the system architecture"},
        ]
        result = checker.check_delegations(delegations)
        # Overlapping tags should produce adjustments or conflicts
        assert len(result.adjustments) > 0 or len(result.conflicts) > 0

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_delegation_to_intent_structure(self):
        checker = DelegationConvergenceChecker(resolver=MagicMock())
        intent = checker._delegation_to_intent({"agent": "builder", "task": "Build auth"})
        assert intent.agent_id == "builder"
        assert intent.intent == "Build auth"
        assert len(intent.provides) == 1
        assert "implementation" in intent.provides[0].tags


class TestCreateChecker:
    """Test the create_checker factory function."""

    def test_returns_checker(self):
        checker = create_checker()
        assert isinstance(checker, DelegationConvergenceChecker)

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_enabled_when_convergent_available(self):
        checker = create_checker()
        assert checker.enabled is True

    def test_disabled_when_convergent_unavailable(self, monkeypatch):
        import animus_forge.agents.convergence as conv_mod

        monkeypatch.setattr(conv_mod, "HAS_CONVERGENT", False)
        checker = create_checker()
        assert checker.enabled is False

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_checker_can_process_delegations(self):
        checker = create_checker()
        result = checker.check_delegations(
            [
                {"agent": "builder", "task": "Build it"},
                {"agent": "tester", "task": "Test it"},
            ]
        )
        assert isinstance(result, ConvergenceResult)


class TestFormatConvergenceAlert:
    """Test the format_convergence_alert formatter."""

    def test_empty_result(self):
        result = ConvergenceResult()
        alert = format_convergence_alert(result)
        assert alert == ""

    def test_conflicts_formatted(self):
        result = ConvergenceResult(
            conflicts=[
                {"agent": "builder", "description": "overlap with tester"},
                {"agent": "planner", "description": "design conflict"},
            ]
        )
        alert = format_convergence_alert(result)
        assert "Conflicts (2):" in alert
        assert "builder: overlap with tester" in alert
        assert "planner: design conflict" in alert

    def test_dropped_agents_formatted(self):
        result = ConvergenceResult(dropped_agents={"builder", "tester"})
        alert = format_convergence_alert(result)
        assert "Dropped agents (2):" in alert
        assert "builder" in alert
        assert "tester" in alert

    def test_adjustments_formatted(self):
        result = ConvergenceResult(
            adjustments=[
                {"agent": "builder", "description": "consume from tester"},
            ]
        )
        alert = format_convergence_alert(result)
        assert "Adjustments (1):" in alert
        assert "builder: consume from tester" in alert

    def test_combined_alert(self):
        result = ConvergenceResult(
            conflicts=[{"agent": "a", "description": "c1"}],
            dropped_agents={"b"},
            adjustments=[{"agent": "c", "description": "a1"}],
        )
        alert = format_convergence_alert(result)
        assert "Conflicts" in alert
        assert "Dropped agents" in alert
        assert "Adjustments" in alert


class TestCreateBridge:
    """Test the create_bridge factory function."""

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_returns_bridge_when_convergent_available(self, tmp_path):
        db_path = str(tmp_path / "coord.db")
        bridge = create_bridge(db_path=db_path)
        assert bridge is not None
        bridge.close()

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_in_memory_mode(self):
        bridge = create_bridge(db_path=":memory:")
        assert bridge is not None
        bridge.close()

    def test_returns_none_when_convergent_unavailable(self, monkeypatch):
        import animus_forge.agents.convergence as conv_mod

        monkeypatch.setattr(conv_mod, "HAS_CONVERGENT", False)
        assert create_bridge() is None

    @pytest.mark.skipif(not HAS_CONVERGENT, reason="convergent not installed")
    def test_returns_none_on_exception(self):
        with patch("convergent.GorgonBridge", side_effect=RuntimeError("boom")):
            result = create_bridge(db_path=":memory:")
            assert result is None


class TestSupervisorBridgeIntegration:
    """Test that SupervisorAgent properly uses the coordination bridge."""

    def test_supervisor_accepts_bridge_kwarg(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        bridge = MagicMock()
        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)
        assert sup._bridge is bridge

    def test_supervisor_works_without_bridge(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        sup = SupervisorAgent(provider=provider)
        assert sup._bridge is None

    def test_enrichment_adds_to_prompt(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="agent result")
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = "## Coordination Context\nTest enrichment"

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)

        asyncio.run(sup._run_agent("builder", "Build auth", []))

        bridge.enrich_prompt.assert_called_once_with(
            agent_id="builder",
            task_description="Build auth",
            file_paths=[],
            current_work="Build auth",
        )
        # Verify the system prompt sent to provider includes enrichment
        call_args = provider.complete.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "Coordination Context" in system_msg

    def test_enrichment_exception_does_not_break_agent(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="agent result")
        bridge = MagicMock()
        bridge.enrich_prompt.side_effect = RuntimeError("stigmergy DB locked")

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)

        result = asyncio.run(sup._run_agent("builder", "Build auth", []))
        assert result == "agent result"

    def test_outcome_recording_after_delegations(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        bridge = MagicMock()
        bridge.record_task_outcome.return_value = 0.6

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)

        results = asyncio.run(
            sup._execute_delegations(
                [
                    {"agent": "builder", "task": "Build it"},
                    {"agent": "tester", "task": "Test it"},
                ],
                [],
                lambda _: None,
            )
        )
        assert "builder" in results
        assert "tester" in results

        # Both outcomes should be recorded
        assert bridge.record_task_outcome.call_count == 2
        calls = bridge.record_task_outcome.call_args_list
        agents_recorded = {c.kwargs["agent_id"] for c in calls}
        assert agents_recorded == {"builder", "tester"}

    def test_outcome_recording_detects_errors(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=[RuntimeError("provider down"), "success result"])
        bridge = MagicMock()
        bridge.record_task_outcome.return_value = 0.5

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)

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

        # Check that the error agent got "failed" outcome
        calls = bridge.record_task_outcome.call_args_list
        outcomes = {c.kwargs["agent_id"]: c.kwargs["outcome"] for c in calls}
        assert outcomes["builder"] == "failed"
        assert outcomes["tester"] == "approved"

    def test_outcome_recording_exception_does_not_break_results(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        bridge = MagicMock()
        bridge.record_task_outcome.side_effect = RuntimeError("DB error")

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)

        result = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert "builder" in result
        assert result["builder"] == "done"


class TestConsensusVoting:
    """Test consensus voting integration in SupervisorAgent."""

    def test_consensus_skipped_when_bridge_is_none(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")

        sup = SupervisorAgent(provider=provider)
        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        assert results["builder"] == "done"

    def test_consensus_skipped_when_quorum_any(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        bridge = MagicMock()

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)
        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "any",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        assert results["builder"] == "done"
        bridge.request_consensus.assert_not_called()

    def test_approved_result_passes_through(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="built successfully")
        bridge = MagicMock()

        mock_decision = MagicMock()
        mock_decision.outcome.value = "approved"
        mock_decision.reasoning_summary = "All good"
        bridge.evaluate.return_value = mock_decision
        bridge.request_consensus.return_value = "req-1"

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)
        results = asyncio.run(
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
        assert results["builder"] == "built successfully"
        bridge.request_consensus.assert_called_once()
        assert bridge.submit_agent_vote.call_count == 2

    def test_rejected_result_replaced(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="bad code output")
        bridge = MagicMock()

        mock_decision = MagicMock()
        mock_decision.outcome.value = "rejected"
        mock_decision.reasoning_summary = "Output violates safety constraints"
        bridge.evaluate.return_value = mock_decision
        bridge.request_consensus.return_value = "req-2"

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)
        results = asyncio.run(
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
        assert "CONSENSUS REJECTED" in results["builder"]
        assert "safety constraints" in results["builder"]

    def test_deadlock_adds_warning(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="agent output here")
        bridge = MagicMock()

        mock_decision = MagicMock()
        mock_decision.outcome.value = "deadlock"
        mock_decision.reasoning_summary = "Split vote"
        bridge.evaluate.return_value = mock_decision
        bridge.request_consensus.return_value = "req-3"

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)
        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "unanimous",
                        "_skill_name": "code-builder",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        assert "agent output here" in results["builder"]
        assert "CONSENSUS DEADLOCK" in results["builder"]
        assert "degraded confidence" in results["builder"]

    def test_voting_exception_does_not_break_pipeline(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="completed")
        bridge = MagicMock()
        bridge.request_consensus.side_effect = RuntimeError("DB locked")

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)
        results = asyncio.run(
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
        assert results["builder"] == "completed"

    def test_consensus_skipped_for_error_results(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("provider down"))
        bridge = MagicMock()

        sup = SupervisorAgent(provider=provider, coordination_bridge=bridge)
        asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        bridge.request_consensus.assert_not_called()
