"""End-to-end smoke test with live Ollama.

Validates the full pipeline: SupervisorAgent → delegation → message bus →
autonomy loop → process registry. Requires Ollama running locally.

Excluded from CI via SMOKE_TEST env var guard.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SMOKE_TEST") != "1",
    reason="Set SMOKE_TEST=1 to run live Ollama smoke tests",
)

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"


@pytest.fixture
def agent_provider():
    """Create a real AgentProvider wrapping OllamaProvider."""
    from animus_forge.agents.provider_wrapper import AgentProvider
    from animus_forge.providers.ollama_provider import OllamaProvider

    raw = OllamaProvider(model=OLLAMA_MODEL, host=OLLAMA_HOST)
    return AgentProvider(raw)


@pytest.fixture
def autonomy_provider():
    """Wrapper for autonomy loop — accepts a prompt string, returns string."""
    from animus_forge.providers.base import CompletionRequest
    from animus_forge.providers.ollama_provider import OllamaProvider

    raw = OllamaProvider(model=OLLAMA_MODEL, host=OLLAMA_HOST)
    raw.initialize()

    class PromptProvider:
        """Adapts OllamaProvider for the autonomy loop interface."""

        async def complete(self, prompt: str) -> str:
            request = CompletionRequest(
                prompt=prompt,
                system_prompt="You are a helpful autonomous agent.",
                max_tokens=512,
                temperature=0.3,
            )
            response = await raw.complete_async(request)
            return response.content

    return PromptProvider()


class TestSupervisorSmoke:
    """Smoke test SupervisorAgent with live Ollama."""

    @pytest.mark.asyncio
    async def test_direct_response(self, agent_provider):
        """Supervisor can generate a direct response (no delegation)."""
        from animus_forge.agents.message_bus import AgentMessageBus
        from animus_forge.agents.supervisor import SupervisorAgent

        bus = AgentMessageBus()
        sup = SupervisorAgent(
            provider=agent_provider,
            message_bus=bus,
        )

        result = await sup.process_message(
            "What is 2 + 2? Answer with just the number.",
            context=[],
            progress_callback=lambda stage, detail: None,
        )

        assert result is not None
        assert len(result) > 0
        assert "4" in result


class TestAutonomySmoke:
    """Smoke test the autonomy loop with live Ollama."""

    @pytest.mark.asyncio
    async def test_simple_goal(self, autonomy_provider):
        """Autonomy loop completes a simple reasoning goal."""
        from animus_forge.agents.autonomy import AutonomyLoop
        from animus_forge.agents.message_bus import AgentMessageBus

        bus = AgentMessageBus()
        loop = AutonomyLoop(
            provider=autonomy_provider,
            max_iterations=3,
            message_bus=bus,
        )

        result = await loop.run(
            goal="Determine what 15 * 7 equals. State the answer clearly.",
            initial_state="You need to calculate 15 * 7.",
        )

        assert len(result.iterations) >= 1
        assert result.total_tokens > 0
        assert result.total_duration_ms > 0
        assert result.final_output != ""

        # Message bus should have events
        assert bus.message_count > 0
        start_msgs = bus.get_messages("autonomy.started")
        assert len(start_msgs) == 1
        complete_msgs = bus.get_messages("autonomy.completed")
        assert len(complete_msgs) == 1


class TestMessageBusSmoke:
    """Smoke test message bus with real agent interactions."""

    @pytest.mark.asyncio
    async def test_delegation_events_published(self, agent_provider):
        """Message bus captures delegation events from real execution."""
        from animus_forge.agents.message_bus import AgentMessageBus
        from animus_forge.agents.supervisor import SupervisorAgent

        bus = AgentMessageBus()
        received_topics = []

        def on_any(msg):
            received_topics.append(msg.topic)

        bus.subscribe("delegation.*", on_any)

        sup = SupervisorAgent(
            provider=agent_provider,
            message_bus=bus,
        )

        delegations = [
            {"agent": "planner", "task": "List 3 steps to make coffee"},
        ]
        results = await sup._execute_delegations(delegations, [], None)

        assert "planner" in results
        assert len(results["planner"]) > 0
        assert "delegation.started" in received_topics
        assert any(t in received_topics for t in ("delegation.completed", "delegation.failed"))


class TestProcessRegistrySmoke:
    """Smoke test process registry with SubAgentManager."""

    @pytest.mark.asyncio
    async def test_registry_tracks_runs(self, agent_provider):
        """Process registry reflects runs from SubAgentManager."""
        from animus_forge.agents.message_bus import AgentMessageBus
        from animus_forge.agents.process_registry import ProcessType
        from animus_forge.agents.subagent_manager import SubAgentManager
        from animus_forge.agents.supervisor import SupervisorAgent

        bus = AgentMessageBus()
        sam = SubAgentManager(max_concurrent=2)

        sup = SupervisorAgent(
            provider=agent_provider,
            subagent_manager=sam,
            message_bus=bus,
        )

        reg = sup.process_registry
        assert reg is not None

        initial = reg.list_all(process_type=ProcessType.AGENT)
        initial_count = len(initial)

        async def quick_agent(agent, task, config):
            return await agent_provider.complete(
                [
                    {"role": "system", "content": f"You are a {agent}. Be brief."},
                    {"role": "user", "content": task},
                ]
            )

        run = await sam.spawn("analyst", "Count to 5", quick_agent)
        assert run.task_handle is not None
        await run.task_handle

        agents = reg.list_all(process_type=ProcessType.AGENT)
        assert len(agents) > initial_count


class TestRunStoreSmoke:
    """Smoke test agent run persistence."""

    def test_save_and_retrieve(self):
        """AgentRunStore round-trips data through SQLite."""
        import sqlite3
        import time

        from animus_forge.agents.agent_config import AgentConfig
        from animus_forge.agents.run_store import AgentRunStore
        from animus_forge.agents.subagent_manager import AgentRun, RunStatus

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(open("migrations/017_agent_runs.sql").read())

        class SimpleBackend:
            placeholder = "?"

            def adapt_query(self, q):
                return q

            def execute(self, q, p=()):
                c = conn.execute(q, p)
                conn.commit()
                return c

            def fetchone(self, q, p=()):
                r = conn.execute(q, p).fetchone()
                return dict(r) if r else None

            def fetchall(self, q, p=()):
                return [dict(r) for r in conn.execute(q, p).fetchall()]

        store = AgentRunStore(SimpleBackend())

        run = AgentRun(
            run_id="smoke-run-1",
            agent="builder",
            task="Build a hello world",
            config=AgentConfig(role="builder"),
            status=RunStatus.COMPLETED,
            result="print('hello world')",
            started_at=time.time() - 5,
            completed_at=time.time(),
        )

        store.save_run(run)
        loaded = store.get_run("smoke-run-1")

        assert loaded is not None
        assert loaded["agent"] == "builder"
        assert loaded["status"] == "completed"
        assert loaded["result"] == "print('hello world')"
        assert store.count() == 1
