"""Tests for SupervisorAgent message bus and process registry wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_forge.agents.message_bus import AgentMessageBus
from animus_forge.agents.process_registry import ProcessRegistry
from animus_forge.agents.supervisor import SupervisorAgent


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=MagicMock(content="Direct response"))
    return provider


@pytest.fixture
def message_bus():
    return AgentMessageBus()


@pytest.fixture
def supervisor(mock_provider, message_bus):
    return SupervisorAgent(
        provider=mock_provider,
        message_bus=message_bus,
    )


class TestMessageBusProperty:
    """Test message_bus property on SupervisorAgent."""

    def test_message_bus_none_by_default(self, mock_provider):
        sup = SupervisorAgent(provider=mock_provider)
        assert sup.message_bus is None

    def test_message_bus_set_via_init(self, supervisor, message_bus):
        assert supervisor.message_bus is message_bus

    def test_message_bus_is_functional(self, supervisor, message_bus):
        received = []

        def on_msg(msg):
            received.append(msg)

        message_bus.subscribe("delegation.*", on_msg)
        message_bus.publish("delegation.test", sender="test", payload="hello")
        assert len(received) == 1


class TestProcessRegistryProperty:
    """Test process_registry property on SupervisorAgent."""

    def test_process_registry_none_without_managers(self, mock_provider):
        sup = SupervisorAgent(provider=mock_provider)
        assert sup.process_registry is None

    def test_process_registry_with_subagent_manager(self, mock_provider):
        sam = MagicMock()
        sam.list_runs.return_value = []
        sup = SupervisorAgent(
            provider=mock_provider,
            subagent_manager=sam,
        )
        reg = sup.process_registry
        assert reg is not None
        assert isinstance(reg, ProcessRegistry)

    def test_process_registry_with_budget_manager(self, mock_provider):
        bm = MagicMock()
        sup = SupervisorAgent(
            provider=mock_provider,
            budget_manager=bm,
        )
        reg = sup.process_registry
        assert reg is not None

    def test_process_registry_aggregates_subagent_manager(self, mock_provider):
        sam = MagicMock()
        sam.list_runs.return_value = []
        sup = SupervisorAgent(
            provider=mock_provider,
            subagent_manager=sam,
        )
        reg = sup.process_registry
        procs = reg.list_all()
        assert isinstance(procs, list)
        sam.list_runs.assert_called_once()


class TestDelegationMessagePublishing:
    """Test that delegation events are published to message bus."""

    @pytest.mark.asyncio
    async def test_delegation_start_published(self, mock_provider, message_bus):
        """Verify delegation.started published for each agent."""
        mock_provider.complete = AsyncMock(return_value=MagicMock(content="Agent result"))
        sup = SupervisorAgent(
            provider=mock_provider,
            message_bus=message_bus,
        )

        delegations = [
            {"agent": "planner", "task": "Plan the feature"},
            {"agent": "builder", "task": "Build the feature"},
        ]

        await sup._execute_delegations(delegations, [], None)

        start_msgs = message_bus.get_messages("delegation.started")
        assert len(start_msgs) == 2
        agents = {m.payload["agent"] for m in start_msgs}
        assert agents == {"planner", "builder"}

    @pytest.mark.asyncio
    async def test_delegation_completion_published(self, mock_provider, message_bus):
        """Verify delegation.completed published after execution."""
        # _run_agent returns the response directly — must be a string
        mock_provider.complete = AsyncMock(return_value="Success result")
        sup = SupervisorAgent(
            provider=mock_provider,
            message_bus=message_bus,
        )

        delegations = [{"agent": "tester", "task": "Run tests"}]
        await sup._execute_delegations(delegations, [], None)

        completed_msgs = message_bus.get_messages("delegation.completed")
        assert len(completed_msgs) == 1
        assert completed_msgs[0].payload["agent"] == "tester"
        assert completed_msgs[0].payload["success"] is True

    @pytest.mark.asyncio
    async def test_delegation_failure_published(self, mock_provider, message_bus):
        """Verify delegation.failed published on error."""
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
        sup = SupervisorAgent(
            provider=mock_provider,
            message_bus=message_bus,
        )

        delegations = [{"agent": "builder", "task": "Build it"}]
        await sup._execute_delegations(delegations, [], None)

        failed_msgs = message_bus.get_messages("delegation.failed")
        assert len(failed_msgs) == 1
        assert failed_msgs[0].payload["agent"] == "builder"
        assert failed_msgs[0].payload["success"] is False

    @pytest.mark.asyncio
    async def test_no_publish_without_bus(self, mock_provider):
        """No crash when message_bus is None."""
        mock_provider.complete = AsyncMock(return_value=MagicMock(content="Result"))
        sup = SupervisorAgent(provider=mock_provider)

        delegations = [{"agent": "planner", "task": "Plan"}]
        results = await sup._execute_delegations(delegations, [], None)
        assert "planner" in results

    @pytest.mark.asyncio
    async def test_parallel_delegation_with_bus(self, mock_provider, message_bus):
        """Message bus works with parallel SubAgentManager path."""
        from animus_forge.agents.subagent_manager import RunStatus

        sam = MagicMock()

        mock_run = MagicMock()
        mock_run.status = RunStatus.COMPLETED
        mock_run.agent = "builder"
        mock_run.result = "Built successfully"
        mock_run.config = MagicMock(timeout_seconds=300)

        sam.spawn_batch = AsyncMock(return_value=[mock_run])

        sup = SupervisorAgent(
            provider=mock_provider,
            subagent_manager=sam,
            message_bus=message_bus,
        )

        delegations = [{"agent": "builder", "task": "Build it"}]
        await sup._execute_delegations(delegations, [], None)

        start_msgs = message_bus.get_messages("delegation.started")
        assert len(start_msgs) == 1

    @pytest.mark.asyncio
    async def test_task_truncated_in_start_message(self, mock_provider, message_bus):
        """Long task descriptions are truncated in published messages."""
        mock_provider.complete = AsyncMock(return_value=MagicMock(content="Done"))
        sup = SupervisorAgent(
            provider=mock_provider,
            message_bus=message_bus,
        )

        long_task = "x" * 500
        delegations = [{"agent": "planner", "task": long_task}]
        await sup._execute_delegations(delegations, [], None)

        start_msgs = message_bus.get_messages("delegation.started")
        assert len(start_msgs[0].payload["task"]) == 200
