"""Tests for the agent autonomy loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_forge.agents.autonomy import (
    AutonomyLoop,
    AutonomyResult,
    LoopIteration,
    LoopPhase,
    StopReason,
)
from animus_forge.agents.message_bus import AgentMessageBus


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=MagicMock(content="Mock response"))
    return provider


@pytest.fixture
def message_bus():
    return AgentMessageBus()


@pytest.fixture
def loop(mock_provider):
    return AutonomyLoop(
        provider=mock_provider,
        max_iterations=5,
        max_errors=2,
    )


class TestLoopPhase:
    def test_phase_values(self):
        assert LoopPhase.OBSERVE == "observe"
        assert LoopPhase.PLAN == "plan"
        assert LoopPhase.ACT == "act"
        assert LoopPhase.REFLECT == "reflect"
        assert LoopPhase.STOPPED == "stopped"


class TestStopReason:
    def test_stop_reason_values(self):
        assert StopReason.GOAL_ACHIEVED == "goal_achieved"
        assert StopReason.MAX_ITERATIONS == "max_iterations"
        assert StopReason.BUDGET_EXHAUSTED == "budget_exhausted"
        assert StopReason.ERROR_THRESHOLD == "error_threshold"
        assert StopReason.MANUAL_STOP == "manual_stop"
        assert StopReason.REFLECTION_HALT == "reflection_halt"


class TestLoopIteration:
    def test_defaults(self):
        it = LoopIteration(iteration=1, phase=LoopPhase.OBSERVE)
        assert it.iteration == 1
        assert it.observation == ""
        assert it.plan == ""
        assert it.action_result == ""
        assert it.reflection == ""
        assert it.tokens_used == 0
        assert it.duration_ms == 0


class TestAutonomyResult:
    def test_to_dict(self):
        result = AutonomyResult(
            goal="Test goal",
            stop_reason=StopReason.GOAL_ACHIEVED,
            final_output="Done!",
            total_tokens=500,
            total_duration_ms=1234,
            iterations=[
                LoopIteration(iteration=1, phase=LoopPhase.REFLECT),
            ],
        )
        d = result.to_dict()
        assert d["goal"] == "Test goal"
        assert d["stop_reason"] == "goal_achieved"
        assert d["iteration_count"] == 1
        assert d["total_tokens"] == 500

    def test_to_dict_truncates_output(self):
        result = AutonomyResult(
            goal="g",
            stop_reason=StopReason.MAX_ITERATIONS,
            final_output="x" * 5000,
        )
        d = result.to_dict()
        assert len(d["final_output"]) == 2000


class TestAutonomyLoopProperties:
    def test_max_iterations(self, loop):
        assert loop.max_iterations == 5

    def test_stop_sets_flag(self, loop):
        loop.stop()
        assert loop._stopped is True


class TestAutonomyLoopRun:
    @pytest.mark.asyncio
    async def test_goal_achieved_via_json(self, mock_provider):
        """Loop stops when reflection says goal_achieved=true."""
        responses = [
            "I see the state",  # observe
            "I'll do X",  # plan
            "Did X successfully",  # act
            '{"assessment": "done", "goal_achieved": true, "should_continue": false, "reason": "done"}',
        ]
        mock_provider.complete = AsyncMock(side_effect=[MagicMock(content=r) for r in responses])

        loop = AutonomyLoop(provider=mock_provider, max_iterations=5)
        result = await loop.run("Achieve the goal")

        assert result.stop_reason == StopReason.GOAL_ACHIEVED
        assert len(result.iterations) == 1
        assert result.final_output == "Did X successfully"
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, mock_provider):
        """Loop stops after max_iterations."""
        mock_provider.complete = AsyncMock(
            return_value=MagicMock(
                content='{"assessment": "ongoing", "goal_achieved": false, "should_continue": true}'
            )
        )

        loop = AutonomyLoop(provider=mock_provider, max_iterations=2)
        result = await loop.run("Never-ending goal")

        assert result.stop_reason == StopReason.MAX_ITERATIONS
        assert len(result.iterations) == 2

    @pytest.mark.asyncio
    async def test_budget_exhausted(self, mock_provider):
        """Loop stops when budget manager says no."""
        bm = MagicMock()
        bm.can_allocate.return_value = False

        loop = AutonomyLoop(
            provider=mock_provider,
            max_iterations=5,
            budget_manager=bm,
        )
        result = await loop.run("Goal")

        assert result.stop_reason == StopReason.BUDGET_EXHAUSTED
        assert len(result.iterations) == 0

    @pytest.mark.asyncio
    async def test_manual_stop(self, mock_provider):
        """Loop respects manual stop request."""
        loop = AutonomyLoop(provider=mock_provider, max_iterations=5)
        loop.stop()

        result = await loop.run("Goal")

        assert result.stop_reason == StopReason.MANUAL_STOP
        assert len(result.iterations) == 0

    @pytest.mark.asyncio
    async def test_error_threshold(self, mock_provider):
        """Loop stops after too many consecutive errors."""
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        loop = AutonomyLoop(
            provider=mock_provider,
            max_iterations=10,
            max_errors=2,
        )
        result = await loop.run("Goal")

        assert result.stop_reason == StopReason.ERROR_THRESHOLD
        assert len(result.iterations) == 2

    @pytest.mark.asyncio
    async def test_reflection_halt(self, mock_provider):
        """Loop stops when reflection says should_continue=false."""
        responses = [
            "Observation",
            "Plan",
            "Action result",
            '{"assessment": "stuck", "goal_achieved": false, "should_continue": false, "reason": "stuck"}',
        ]
        mock_provider.complete = AsyncMock(side_effect=[MagicMock(content=r) for r in responses])

        loop = AutonomyLoop(provider=mock_provider, max_iterations=5)
        result = await loop.run("Goal")

        assert result.stop_reason == StopReason.REFLECTION_HALT
        assert len(result.iterations) == 1

    @pytest.mark.asyncio
    async def test_custom_act_fn(self, mock_provider):
        """act_fn is used instead of provider for the act phase."""
        responses = [
            "Observation",
            "Plan to act",
            '{"goal_achieved": true, "should_continue": false}',
        ]
        mock_provider.complete = AsyncMock(side_effect=[MagicMock(content=r) for r in responses])

        async def custom_act(plan: str) -> str:
            return f"Executed: {plan}"

        loop = AutonomyLoop(provider=mock_provider, max_iterations=5)
        result = await loop.run("Goal", act_fn=custom_act)

        assert result.stop_reason == StopReason.GOAL_ACHIEVED
        assert "Executed:" in result.final_output

    @pytest.mark.asyncio
    async def test_initial_state_provided(self, mock_provider):
        """Initial state is passed to the observe prompt."""
        call_args = []

        async def track_complete(prompt):
            call_args.append(prompt)
            return MagicMock(content='{"goal_achieved": true}')

        mock_provider.complete = track_complete

        loop = AutonomyLoop(provider=mock_provider, max_iterations=1)
        await loop.run("Goal", initial_state="Starting context")

        # First call is observe — should contain the initial state
        assert "Starting context" in call_args[0]

    @pytest.mark.asyncio
    async def test_history_accumulates(self, mock_provider):
        """History from prior iterations is passed to observe."""
        call_count = 0

        async def counting_complete(prompt):
            nonlocal call_count
            call_count += 1
            if call_count <= 8:  # 4 calls per iteration, 2 iterations
                return MagicMock(content='{"goal_achieved": false, "should_continue": true}')
            return MagicMock(content='{"goal_achieved": true}')

        mock_provider.complete = counting_complete

        loop = AutonomyLoop(provider=mock_provider, max_iterations=3)
        result = await loop.run("Goal")
        assert len(result.iterations) >= 2

    @pytest.mark.asyncio
    async def test_final_output_from_last_iteration(self, mock_provider):
        """If no explicit final_output, uses last action_result."""
        mock_provider.complete = AsyncMock(
            return_value=MagicMock(content='{"goal_achieved": false, "should_continue": true}')
        )

        loop = AutonomyLoop(provider=mock_provider, max_iterations=1)
        result = await loop.run("Goal")

        assert result.final_output != ""  # Should have something from observe or act

    @pytest.mark.asyncio
    async def test_provider_returns_string(self, mock_provider):
        """Handle providers that return plain strings instead of objects."""
        mock_provider.complete = AsyncMock(return_value='{"goal_achieved": true}')

        loop = AutonomyLoop(provider=mock_provider, max_iterations=5)
        result = await loop.run("Goal")

        assert result.stop_reason == StopReason.GOAL_ACHIEVED


class TestParseReflection:
    def setup_method(self):
        self.loop = AutonomyLoop(provider=MagicMock(), max_iterations=1)

    def test_json_goal_achieved(self):
        stop, reason = self.loop._parse_reflection(
            '{"goal_achieved": true, "should_continue": false}'
        )
        assert stop is True
        assert reason == StopReason.GOAL_ACHIEVED

    def test_json_should_not_continue(self):
        stop, reason = self.loop._parse_reflection(
            '{"goal_achieved": false, "should_continue": false}'
        )
        assert stop is True
        assert reason == StopReason.REFLECTION_HALT

    def test_json_should_continue(self):
        stop, reason = self.loop._parse_reflection(
            '{"goal_achieved": false, "should_continue": true}'
        )
        assert stop is False

    def test_keyword_goal_achieved(self):
        stop, reason = self.loop._parse_reflection("The goal achieved, we can stop now.")
        assert stop is True
        assert reason == StopReason.GOAL_ACHIEVED

    def test_keyword_goal_is_achieved(self):
        stop, reason = self.loop._parse_reflection("The goal is achieved.")
        assert stop is True

    def test_keyword_should_stop(self):
        stop, reason = self.loop._parse_reflection(
            "We should stop because there's nothing left to do."
        )
        assert stop is True
        assert reason == StopReason.REFLECTION_HALT

    def test_keyword_no_further_action(self):
        stop, reason = self.loop._parse_reflection("No further action is needed.")
        assert stop is True

    def test_no_stop_signal(self):
        stop, reason = self.loop._parse_reflection(
            "We need to keep working on this. More progress needed."
        )
        assert stop is False

    def test_malformed_json_fallback(self):
        stop, reason = self.loop._parse_reflection('{"goal_achieved": maybe}')
        assert stop is False  # Falls to keyword, no keyword match

    def test_json_embedded_in_text(self):
        stop, reason = self.loop._parse_reflection(
            'Here is my assessment: {"goal_achieved": true} That is all.'
        )
        assert stop is True


class TestMessageBusIntegration:
    @pytest.mark.asyncio
    async def test_publishes_start_and_complete(self, mock_provider, message_bus):
        responses = [
            "Observe",
            "Plan",
            "Act",
            '{"goal_achieved": true}',
        ]
        mock_provider.complete = AsyncMock(side_effect=[MagicMock(content=r) for r in responses])

        loop = AutonomyLoop(
            provider=mock_provider,
            max_iterations=5,
            message_bus=message_bus,
        )
        await loop.run("Goal")

        start_msgs = message_bus.get_messages("autonomy.started")
        assert len(start_msgs) == 1
        assert start_msgs[0].payload["goal"] == "Goal"

        complete_msgs = message_bus.get_messages("autonomy.completed")
        assert len(complete_msgs) == 1
        assert complete_msgs[0].payload["stop_reason"] == "goal_achieved"

        iter_msgs = message_bus.get_messages("autonomy.iteration")
        assert len(iter_msgs) == 1

    @pytest.mark.asyncio
    async def test_no_crash_without_bus(self, mock_provider):
        """Loop works fine without message bus."""
        mock_provider.complete = AsyncMock(
            return_value=MagicMock(content='{"goal_achieved": true}')
        )

        loop = AutonomyLoop(provider=mock_provider, max_iterations=1)
        result = await loop.run("Goal")
        assert result.stop_reason == StopReason.GOAL_ACHIEVED

    @pytest.mark.asyncio
    async def test_bus_error_swallowed(self, mock_provider):
        """Broken message bus doesn't crash the loop."""
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("Bus exploded")

        mock_provider.complete = AsyncMock(
            return_value=MagicMock(content='{"goal_achieved": true}')
        )

        loop = AutonomyLoop(
            provider=mock_provider,
            max_iterations=1,
            message_bus=bus,
        )
        result = await loop.run("Goal")
        assert result.stop_reason == StopReason.GOAL_ACHIEVED


class TestEstimateTokens:
    def test_normal_text(self):
        assert AutonomyLoop._estimate_tokens("Hello world") >= 1

    def test_empty_text(self):
        assert AutonomyLoop._estimate_tokens("") == 1  # minimum 1

    def test_long_text(self):
        tokens = AutonomyLoop._estimate_tokens("x" * 4000)
        assert tokens == 1000


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_error_resets_on_success(self, mock_provider):
        """Consecutive error counter resets after a successful iteration."""
        call_count = 0

        async def alternating(prompt):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:  # First iteration fails at observe
                raise RuntimeError("Temporary failure")
            # After that, succeed
            return MagicMock(content='{"goal_achieved": true}')

        mock_provider.complete = alternating

        loop = AutonomyLoop(
            provider=mock_provider,
            max_iterations=5,
            max_errors=3,
        )
        result = await loop.run("Goal")
        # Should have 1 error iteration, then succeed
        assert any(it.reflection.startswith("Error:") for it in result.iterations)

    @pytest.mark.asyncio
    async def test_total_duration_tracked(self, mock_provider):
        mock_provider.complete = AsyncMock(
            return_value=MagicMock(content='{"goal_achieved": true}')
        )

        loop = AutonomyLoop(provider=mock_provider, max_iterations=1)
        result = await loop.run("Goal")
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_budget_checked_each_iteration(self, mock_provider):
        """Budget is checked at the start of each iteration."""
        bm = MagicMock()
        call_count = 0

        def budget_check(tokens):
            nonlocal call_count
            call_count += 1
            return call_count <= 1  # Allow first, deny second

        bm.can_allocate.side_effect = budget_check

        mock_provider.complete = AsyncMock(
            return_value=MagicMock(content='{"goal_achieved": false, "should_continue": true}')
        )

        loop = AutonomyLoop(
            provider=mock_provider,
            max_iterations=5,
            budget_manager=bm,
        )
        result = await loop.run("Goal")

        assert result.stop_reason == StopReason.BUDGET_EXHAUSTED
        assert len(result.iterations) == 1
