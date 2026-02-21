"""Tests for the IntelligentRouter — memory, tools, and automation integration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_bootstrap.gateway.cognitive_types import CognitiveResponse, ToolCall
from animus_bootstrap.gateway.models import (
    GatewayMessage,
    GatewayResponse,
    create_message,
)
from animus_bootstrap.gateway.session import SessionManager
from animus_bootstrap.intelligence.automations.models import AutomationResult
from animus_bootstrap.intelligence.memory import MemoryContext
from animus_bootstrap.intelligence.router import IntelligentRouter

# ======================================================================
# Helpers
# ======================================================================


def _make_message(text: str = "hello", channel: str = "webchat") -> GatewayMessage:
    """Create a simple test message."""
    return create_message(channel, "user1", "Alice", text)


def _make_mock_cognitive(response: str = "I'm Animus.") -> AsyncMock:
    """Create a mock CognitiveBackend."""
    cog = AsyncMock()
    cog.generate_response = AsyncMock(return_value=response)
    return cog


def _make_mock_memory(
    memory_context: MemoryContext | None = None,
) -> MagicMock:
    """Create a mock MemoryManager with async methods."""
    mem = MagicMock()
    mem.recall = AsyncMock(return_value=memory_context or MemoryContext())
    mem.store_conversation = AsyncMock()
    mem.close = MagicMock()
    return mem


def _make_mock_automations(
    results: list[AutomationResult] | None = None,
) -> MagicMock:
    """Create a mock AutomationEngine."""
    engine = MagicMock()
    engine.evaluate_message = AsyncMock(return_value=results or [])
    return engine


def _make_mock_tools(schemas: list[dict] | None = None) -> MagicMock:
    """Create a mock ToolExecutor."""
    tools = MagicMock()
    tools.get_schemas = MagicMock(return_value=schemas or [])
    tools.execute = AsyncMock()
    tools.execute_batch = AsyncMock(return_value=[])
    return tools


# ======================================================================
# TestIntelligentRouter — core handle_message flow
# ======================================================================


class TestIntelligentRouter:
    """Tests for the main handle_message pipeline."""

    @pytest.fixture()
    def cognitive(self) -> AsyncMock:
        return _make_mock_cognitive()

    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "irouter-test.db")
        yield mgr
        mgr.close()

    @pytest.fixture()
    def router(self, cognitive: AsyncMock, session_mgr: SessionManager) -> IntelligentRouter:
        return IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)

    # ------------------------------------------------------------------
    # Basic flow (no memory/tools/automations)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio()
    async def test_handle_message_returns_response(self, router: IntelligentRouter) -> None:
        msg = _make_message("hello")
        resp = await router.handle_message(msg)
        assert isinstance(resp, GatewayResponse)
        assert resp.text == "I'm Animus."
        assert resp.channel == "webchat"

    @pytest.mark.asyncio()
    async def test_handle_message_stores_user_and_assistant(
        self, router: IntelligentRouter, session_mgr: SessionManager
    ) -> None:
        msg = _make_message("hello")
        await router.handle_message(msg)

        # Re-fetch session to check stored messages
        msg2 = _make_message("check")
        session = await session_mgr.get_or_create_session(msg2)
        context = await session_mgr.get_context(session)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "hello"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "I'm Animus."

    @pytest.mark.asyncio()
    async def test_handle_message_calls_cognitive(
        self, cognitive: AsyncMock, router: IntelligentRouter
    ) -> None:
        msg = _make_message("hi")
        await router.handle_message(msg)
        cognitive.generate_response.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_without_memory_tools_automations_behaves_like_base(
        self, router: IntelligentRouter
    ) -> None:
        """Without optional components, IntelligentRouter delegates to parent flow."""
        msg = _make_message("plain")
        resp = await router.handle_message(msg)
        assert resp.text == "I'm Animus."

    @pytest.mark.asyncio()
    async def test_system_prompt_passed_to_cognitive(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            system_prompt="You are Animus.",
        )
        msg = _make_message("hi")
        await router.handle_message(msg)
        call_kwargs = cognitive.generate_response.call_args
        assert call_kwargs.kwargs["system_prompt"] == "You are Animus."

    # ------------------------------------------------------------------
    # Memory recall
    # ------------------------------------------------------------------

    @pytest.mark.asyncio()
    async def test_memory_recall_enriches_system_prompt(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        mem_ctx = MemoryContext(
            episodic=["You discussed Python yesterday"],
            semantic=["Alice is a developer"],
        )
        memory = _make_mock_memory(mem_ctx)
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            memory=memory,
        )
        msg = _make_message("hello")
        await router.handle_message(msg)

        memory.recall.assert_awaited_once_with("hello")
        call_kwargs = cognitive.generate_response.call_args
        system = call_kwargs.kwargs["system_prompt"]
        assert "Relevant Past Conversations" in system
        assert "You discussed Python yesterday" in system
        assert "Known Facts" in system
        assert "Alice is a developer" in system

    @pytest.mark.asyncio()
    async def test_memory_recall_failure_is_caught(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        memory = _make_mock_memory()
        memory.recall = AsyncMock(side_effect=RuntimeError("DB error"))
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            memory=memory,
        )
        msg = _make_message("hello")
        with caplog.at_level(logging.ERROR):
            resp = await router.handle_message(msg)
        assert resp.text == "I'm Animus."
        assert "Memory recall failed" in caplog.text

    @pytest.mark.asyncio()
    async def test_memory_recall_with_empty_context(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        memory = _make_mock_memory(MemoryContext())
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            memory=memory,
        )
        msg = _make_message("hello")
        resp = await router.handle_message(msg)
        assert resp.text == "I'm Animus."
        # System prompt should be empty since no base prompt and empty memory
        call_kwargs = cognitive.generate_response.call_args
        assert call_kwargs.kwargs["system_prompt"] == ""

    # ------------------------------------------------------------------
    # Memory store
    # ------------------------------------------------------------------

    @pytest.mark.asyncio()
    async def test_memory_store_called_after_response(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        memory = _make_mock_memory()
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            memory=memory,
        )
        msg = _make_message("hello")
        await router.handle_message(msg)
        memory.store_conversation.assert_awaited_once()
        call_args = memory.store_conversation.call_args
        # First arg is session_id (str), second is context (list[dict])
        assert isinstance(call_args.args[0], str)
        assert isinstance(call_args.args[1], list)

    @pytest.mark.asyncio()
    async def test_memory_store_failure_is_caught(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        memory = _make_mock_memory()
        memory.store_conversation = AsyncMock(side_effect=RuntimeError("Write error"))
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            memory=memory,
        )
        msg = _make_message("hello")
        with caplog.at_level(logging.ERROR):
            resp = await router.handle_message(msg)
        assert resp.text == "I'm Animus."
        assert "Memory store failed" in caplog.text

    @pytest.mark.asyncio()
    async def test_memory_store_receives_updated_context(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """store_conversation should be called with context that includes the assistant msg."""
        memory = _make_mock_memory()
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            memory=memory,
        )
        msg = _make_message("hello")
        await router.handle_message(msg)
        # The stored context should contain both user + assistant messages
        call_args = memory.store_conversation.call_args
        context_list = call_args.args[1]
        assert len(context_list) == 2
        roles = [m["role"] for m in context_list]
        assert "user" in roles
        assert "assistant" in roles

    # ------------------------------------------------------------------
    # Automations
    # ------------------------------------------------------------------

    @pytest.mark.asyncio()
    async def test_automation_runs_before_response(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        auto_result = AutomationResult(
            rule_id="r1",
            rule_name="greeting-rule",
            triggered=True,
            actions_executed=["reply: Hello!"],
            timestamp=datetime.now(UTC),
        )
        automations = _make_mock_automations([auto_result])
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            automations=automations,
        )
        msg = _make_message("hello")
        resp = await router.handle_message(msg)
        automations.evaluate_message.assert_awaited_once_with(msg)
        # Response still comes from cognitive, not automation
        assert resp.text == "I'm Animus."

    @pytest.mark.asyncio()
    async def test_automation_no_matching_rules(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        automations = _make_mock_automations([])
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            automations=automations,
        )
        msg = _make_message("hello")
        resp = await router.handle_message(msg)
        assert resp.text == "I'm Animus."

    @pytest.mark.asyncio()
    async def test_automation_triggered_but_no_actions(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        auto_result = AutomationResult(
            rule_id="r2",
            rule_name="log-only-rule",
            triggered=True,
            actions_executed=[],
            timestamp=datetime.now(UTC),
        )
        automations = _make_mock_automations([auto_result])
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            automations=automations,
        )
        msg = _make_message("hello")
        resp = await router.handle_message(msg)
        assert resp.text == "I'm Animus."

    @pytest.mark.asyncio()
    async def test_automation_fires_logs_info(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        auto_result = AutomationResult(
            rule_id="r1",
            rule_name="greeting-rule",
            triggered=True,
            actions_executed=["reply: Hello!"],
            timestamp=datetime.now(UTC),
        )
        automations = _make_mock_automations([auto_result])
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            automations=automations,
        )
        msg = _make_message("hello")
        with caplog.at_level(logging.INFO):
            await router.handle_message(msg)
        assert "greeting-rule" in caplog.text

    @pytest.mark.asyncio()
    async def test_automation_evaluation_failure_is_caught(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        automations = _make_mock_automations()
        automations.evaluate_message = AsyncMock(side_effect=RuntimeError("Automation DB error"))
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            automations=automations,
        )
        msg = _make_message("hello")
        with caplog.at_level(logging.ERROR):
            resp = await router.handle_message(msg)
        assert resp.text == "I'm Animus."
        assert "Automation evaluation failed" in caplog.text

    @pytest.mark.asyncio()
    async def test_multiple_automations_fire(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        results = [
            AutomationResult(
                rule_id="r1",
                rule_name="rule-a",
                triggered=True,
                actions_executed=["action1"],
                timestamp=datetime.now(UTC),
            ),
            AutomationResult(
                rule_id="r2",
                rule_name="rule-b",
                triggered=True,
                actions_executed=["action2"],
                timestamp=datetime.now(UTC),
            ),
        ]
        automations = _make_mock_automations(results)
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            automations=automations,
        )
        msg = _make_message("hello")
        with caplog.at_level(logging.INFO):
            await router.handle_message(msg)
        assert "rule-a" in caplog.text
        assert "rule-b" in caplog.text

    # ------------------------------------------------------------------
    # Full pipeline (memory + automations together)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio()
    async def test_full_pipeline_memory_and_automations(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        mem_ctx = MemoryContext(semantic=["Alice likes Python"])
        memory = _make_mock_memory(mem_ctx)
        auto_result = AutomationResult(
            rule_id="r1",
            rule_name="full-rule",
            triggered=True,
            actions_executed=["logged"],
            timestamp=datetime.now(UTC),
        )
        automations = _make_mock_automations([auto_result])
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            memory=memory,
            automations=automations,
            system_prompt="You are Animus.",
        )
        msg = _make_message("hello")
        resp = await router.handle_message(msg)

        assert resp.text == "I'm Animus."
        memory.recall.assert_awaited_once()
        memory.store_conversation.assert_awaited_once()
        automations.evaluate_message.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_assistant_message_metadata(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """Verify the stored assistant message has correct fields."""
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        msg = _make_message("hello", channel="telegram")
        await router.handle_message(msg)

        # Fetch context to verify assistant message
        session = await session_mgr.get_or_create_session(
            _make_message("check", channel="telegram")
        )
        context = await session_mgr.get_context(session)
        assert context[-1]["role"] == "assistant"
        assert context[-1]["content"] == "I'm Animus."

    @pytest.mark.asyncio()
    async def test_different_cognitive_response(
        self,
        session_mgr: SessionManager,
    ) -> None:
        cog = _make_mock_cognitive("Custom response text")
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr)
        msg = _make_message("hello")
        resp = await router.handle_message(msg)
        assert resp.text == "Custom response text"

    @pytest.mark.asyncio()
    async def test_inherits_channel_methods(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """IntelligentRouter inherits register_channel, broadcast, etc."""
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        adapter = AsyncMock()
        router.register_channel("discord", adapter)
        assert "discord" in router.channels
        router.unregister_channel("discord")
        assert "discord" not in router.channels

    @pytest.mark.asyncio()
    async def test_max_tool_iterations_stored(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            max_tool_iterations=10,
        )
        assert router._max_tool_iterations == 10


# ======================================================================
# TestBuildSystemPrompt
# ======================================================================


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt method."""

    @pytest.fixture()
    def cognitive(self) -> AsyncMock:
        return _make_mock_cognitive()

    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "prompt-test.db")
        yield mgr
        mgr.close()

    def test_empty_prompt_and_no_memory_returns_empty(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        result = router._build_system_prompt(None)
        assert result == ""

    def test_system_prompt_only(self, cognitive: AsyncMock, session_mgr: SessionManager) -> None:
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            system_prompt="You are a helpful AI.",
        )
        result = router._build_system_prompt(None)
        assert result == "You are a helpful AI."

    def test_memory_episodic_adds_section(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        ctx = MemoryContext(episodic=["Past conversation about Python"])
        result = router._build_system_prompt(ctx)
        assert "## Relevant Past Conversations" in result
        assert "- Past conversation about Python" in result

    def test_memory_semantic_adds_section(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        ctx = MemoryContext(semantic=["Alice is a developer"])
        result = router._build_system_prompt(ctx)
        assert "## Known Facts" in result
        assert "- Alice is a developer" in result

    def test_memory_procedural_adds_section(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        ctx = MemoryContext(procedural=["To deploy, run make deploy"])
        result = router._build_system_prompt(ctx)
        assert "## How-To Knowledge" in result
        assert "- To deploy, run make deploy" in result

    def test_memory_user_prefs_adds_section(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        ctx = MemoryContext(user_prefs={"theme": "dark", "language": "en"})
        result = router._build_system_prompt(ctx)
        assert "## User Preferences" in result
        assert "- theme: dark" in result
        assert "- language: en" in result

    def test_full_memory_context_all_sections(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        ctx = MemoryContext(
            episodic=["conv about Rust"],
            semantic=["Bob is a designer"],
            procedural=["Use pytest for tests"],
            user_prefs={"editor": "vim"},
        )
        result = router._build_system_prompt(ctx)
        assert "## Relevant Past Conversations" in result
        assert "## Known Facts" in result
        assert "## How-To Knowledge" in result
        assert "## User Preferences" in result
        assert "- conv about Rust" in result
        assert "- Bob is a designer" in result
        assert "- Use pytest for tests" in result
        assert "- editor: vim" in result

    def test_system_prompt_combined_with_memory(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            system_prompt="You are Animus, a personal AI.",
        )
        ctx = MemoryContext(semantic=["User prefers concise answers"])
        result = router._build_system_prompt(ctx)
        assert result.startswith("You are Animus, a personal AI.")
        assert "## Known Facts" in result
        assert "- User prefers concise answers" in result

    def test_empty_memory_context_no_sections(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        ctx = MemoryContext()
        result = router._build_system_prompt(ctx)
        assert result == ""

    def test_empty_memory_context_with_system_prompt(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            system_prompt="Base prompt.",
        )
        ctx = MemoryContext()
        result = router._build_system_prompt(ctx)
        assert result == "Base prompt."

    def test_multiple_episodic_entries(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        ctx = MemoryContext(episodic=["conv1", "conv2", "conv3"])
        result = router._build_system_prompt(ctx)
        assert "- conv1" in result
        assert "- conv2" in result
        assert "- conv3" in result


# ======================================================================
# TestCognitiveLoop
# ======================================================================


class TestCognitiveLoop:
    """Tests for the _cognitive_loop method."""

    @pytest.fixture()
    def cognitive(self) -> AsyncMock:
        return _make_mock_cognitive("loop response")

    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "loop-test.db")
        yield mgr
        mgr.close()

    @pytest.mark.asyncio()
    async def test_simple_call_delegates_to_cognitive(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        messages = [{"role": "user", "content": "hi"}]
        result = await router._cognitive_loop(messages, "system prompt")
        assert result == "loop response"
        cognitive.generate_response.assert_awaited_once_with(
            messages, system_prompt="system prompt"
        )

    @pytest.mark.asyncio()
    async def test_empty_system_prompt(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        messages = [{"role": "user", "content": "hello"}]
        result = await router._cognitive_loop(messages, "")
        assert result == "loop response"
        cognitive.generate_response.assert_awaited_once_with(messages, system_prompt="")

    @pytest.mark.asyncio()
    async def test_multi_message_conversation(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ]
        result = await router._cognitive_loop(messages, "be friendly")
        assert result == "loop response"
        call_args = cognitive.generate_response.call_args
        assert len(call_args.args[0]) == 3


# ======================================================================
# TestGetToolSchemas
# ======================================================================


class TestGetToolSchemas:
    """Tests for the get_tool_schemas method."""

    @pytest.fixture()
    def cognitive(self) -> AsyncMock:
        return _make_mock_cognitive()

    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "schema-test.db")
        yield mgr
        mgr.close()

    def test_returns_schemas_from_tool_executor(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        schemas = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        tools = _make_mock_tools(schemas)
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr, tools=tools)
        result = router.get_tool_schemas()
        assert result == schemas
        tools.get_schemas.assert_called_once()

    def test_returns_empty_list_when_no_tool_executor(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr)
        result = router.get_tool_schemas()
        assert result == []

    def test_returns_multiple_schemas(
        self, cognitive: AsyncMock, session_mgr: SessionManager
    ) -> None:
        schemas = [
            {"name": "tool_a", "description": "A", "input_schema": {}},
            {"name": "tool_b", "description": "B", "input_schema": {}},
            {"name": "tool_c", "description": "C", "input_schema": {}},
        ]
        tools = _make_mock_tools(schemas)
        router = IntelligentRouter(cognitive=cognitive, session_manager=session_mgr, tools=tools)
        result = router.get_tool_schemas()
        assert len(result) == 3
        names = [s["name"] for s in result]
        assert names == ["tool_a", "tool_b", "tool_c"]


# ======================================================================
# TestCognitiveLoopWithTools — structured tool_use loop
# ======================================================================


def _make_structured_cognitive(
    responses: list[CognitiveResponse],
) -> AsyncMock:
    """Create a mock cognitive backend that supports generate_structured.

    Returns responses in order on successive calls. Also has generate_response
    for backward compat.
    """
    cog = AsyncMock()
    cog.generate_response = AsyncMock(return_value="fallback text")
    cog.generate_structured = AsyncMock(side_effect=responses)
    return cog


class TestCognitiveLoopWithTools:
    """Tests for the _cognitive_loop with native tool_use support."""

    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "toolloop-test.db")
        yield mgr
        mgr.close()

    @pytest.fixture()
    def tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

    @pytest.fixture()
    def tools(self, tool_schemas: list[dict]) -> MagicMock:
        t = _make_mock_tools(tool_schemas)
        t.execute = AsyncMock(return_value=MagicMock(output="Search result: found 3 items"))
        return t

    @pytest.mark.asyncio()
    async def test_tool_use_triggers_execution(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """When LLM returns tool_use, the tool is executed."""
        responses = [
            CognitiveResponse(
                text="Let me search.",
                tool_calls=[
                    ToolCall(id="toolu_01", name="web_search", arguments={"query": "test"})
                ],
                stop_reason="tool_use",
            ),
            CognitiveResponse(text="I found 3 items.", stop_reason="end_turn"),
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "search for test"}]
        result = await router._cognitive_loop(messages, "system prompt")

        assert result == "I found 3 items."
        tools.execute.assert_awaited_once_with("web_search", {"query": "test"})
        assert cog.generate_structured.await_count == 2

    @pytest.mark.asyncio()
    async def test_tool_results_fed_back_to_llm(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """Tool results are appended to messages and sent back to the LLM."""
        responses = [
            CognitiveResponse(
                text="Checking.",
                tool_calls=[ToolCall(id="toolu_01", name="web_search", arguments={"query": "q"})],
                stop_reason="tool_use",
            ),
            CognitiveResponse(text="Done.", stop_reason="end_turn"),
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "go"}]
        await router._cognitive_loop(messages, "sys")

        # Second call should include assistant content blocks + tool results
        second_call = cog.generate_structured.call_args_list[1]
        call_messages = second_call.args[0]
        # Original user message + assistant content + user tool_result
        assert len(call_messages) == 3
        # Assistant message has content blocks
        assistant_msg = call_messages[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        # Tool result message
        tool_result_msg = call_messages[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_01"
        assert tool_result_msg["content"][0]["content"] == "Search result: found 3 items"

    @pytest.mark.asyncio()
    async def test_multiple_tool_iterations(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """Supports multiple back-and-forth tool call rounds."""
        responses = [
            CognitiveResponse(
                text="Step 1.",
                tool_calls=[ToolCall(id="t1", name="web_search", arguments={"query": "a"})],
                stop_reason="tool_use",
            ),
            CognitiveResponse(
                text="Step 2.",
                tool_calls=[ToolCall(id="t2", name="web_search", arguments={"query": "b"})],
                stop_reason="tool_use",
            ),
            CognitiveResponse(text="All done.", stop_reason="end_turn"),
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "multi-step"}]
        result = await router._cognitive_loop(messages, "sys")

        assert result == "All done."
        assert cog.generate_structured.await_count == 3
        assert tools.execute.await_count == 2

    @pytest.mark.asyncio()
    async def test_max_iterations_reached(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """When max iterations is reached, returns last available text."""
        # All responses have tool calls — loop will hit max
        responses = [
            CognitiveResponse(
                text=f"Iteration {i}.",
                tool_calls=[ToolCall(id=f"t{i}", name="web_search", arguments={"query": "x"})],
                stop_reason="tool_use",
            )
            for i in range(10)
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(
            cognitive=cog,
            session_manager=session_mgr,
            tools=tools,
            max_tool_iterations=3,
        )

        messages = [{"role": "user", "content": "loop forever"}]
        result = await router._cognitive_loop(messages, "sys")

        # Should return text from last response (iteration 2, zero-indexed)
        assert result == "Iteration 2."
        assert cog.generate_structured.await_count == 3

    @pytest.mark.asyncio()
    async def test_max_iterations_no_text_fallback(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """When max iterations reached with empty text, returns fallback message."""
        responses = [
            CognitiveResponse(
                text="",
                tool_calls=[ToolCall(id=f"t{i}", name="web_search", arguments={})],
                stop_reason="tool_use",
            )
            for i in range(5)
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(
            cognitive=cog,
            session_manager=session_mgr,
            tools=tools,
            max_tool_iterations=2,
        )

        messages = [{"role": "user", "content": "loop"}]
        result = await router._cognitive_loop(messages, "sys")

        assert result == "I've reached my tool call limit for this turn."

    @pytest.mark.asyncio()
    async def test_tool_execution_error_doesnt_break_loop(
        self, session_mgr: SessionManager
    ) -> None:
        """Tool execution failures are passed back to LLM as error output."""
        tools = _make_mock_tools(
            [{"name": "failing_tool", "description": "Fails", "input_schema": {}}]
        )
        tools.execute = AsyncMock(
            return_value=MagicMock(output="Tool 'failing_tool' failed: connection error")
        )

        responses = [
            CognitiveResponse(
                text="Trying tool.",
                tool_calls=[ToolCall(id="t1", name="failing_tool", arguments={})],
                stop_reason="tool_use",
            ),
            CognitiveResponse(
                text="The tool failed, but I can help anyway.",
                stop_reason="end_turn",
            ),
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "try it"}]
        result = await router._cognitive_loop(messages, "sys")

        assert result == "The tool failed, but I can help anyway."
        # Error output was passed back to LLM
        second_call = cog.generate_structured.call_args_list[1]
        tool_result_content = second_call.args[0][2]["content"][0]["content"]
        assert "failed" in tool_result_content

    @pytest.mark.asyncio()
    async def test_no_tools_uses_simple_path(self, session_mgr: SessionManager) -> None:
        """Without tools, falls back to simple generate_response."""
        cog = AsyncMock()
        cog.generate_response = AsyncMock(return_value="simple response")
        cog.generate_structured = AsyncMock()  # Should NOT be called
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr)

        messages = [{"role": "user", "content": "hello"}]
        result = await router._cognitive_loop(messages, "sys")

        assert result == "simple response"
        cog.generate_response.assert_awaited_once()
        cog.generate_structured.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_tools_with_empty_schemas_uses_simple_path(
        self, session_mgr: SessionManager
    ) -> None:
        """Tools executor with no registered tools falls back to simple path."""
        tools = _make_mock_tools([])  # Empty schemas
        cog = AsyncMock()
        cog.generate_response = AsyncMock(return_value="no schemas")
        cog.generate_structured = AsyncMock()
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "hello"}]
        result = await router._cognitive_loop(messages, "sys")

        assert result == "no schemas"
        cog.generate_response.assert_awaited_once()
        cog.generate_structured.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_backend_without_generate_structured_uses_simple_path(
        self, session_mgr: SessionManager
    ) -> None:
        """Backend without generate_structured method falls back to simple path."""
        cog = MagicMock(spec=[])  # No attributes at all
        cog.generate_response = AsyncMock(return_value="old backend")
        tools = _make_mock_tools([{"name": "t", "description": "T", "input_schema": {}}])
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "hello"}]
        result = await router._cognitive_loop(messages, "sys")

        assert result == "old backend"
        cog.generate_response.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_assistant_content_blocks_include_text_and_tool_use(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """Assistant content includes both text and tool_use blocks."""
        responses = [
            CognitiveResponse(
                text="Here's what I'll do:",
                tool_calls=[ToolCall(id="t1", name="web_search", arguments={"query": "q"})],
                stop_reason="tool_use",
            ),
            CognitiveResponse(text="Final.", stop_reason="end_turn"),
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "go"}]
        await router._cognitive_loop(messages, "sys")

        second_call = cog.generate_structured.call_args_list[1]
        assistant_content = second_call.args[0][1]["content"]
        # Should have text block + tool_use block
        assert len(assistant_content) == 2
        assert assistant_content[0] == {"type": "text", "text": "Here's what I'll do:"}
        assert assistant_content[1]["type"] == "tool_use"
        assert assistant_content[1]["id"] == "t1"
        assert assistant_content[1]["name"] == "web_search"
        assert assistant_content[1]["input"] == {"query": "q"}

    @pytest.mark.asyncio()
    async def test_empty_text_omitted_from_assistant_content(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """When LLM returns empty text with tool calls, text block is omitted."""
        responses = [
            CognitiveResponse(
                text="",
                tool_calls=[ToolCall(id="t1", name="web_search", arguments={"query": "q"})],
                stop_reason="tool_use",
            ),
            CognitiveResponse(text="Done.", stop_reason="end_turn"),
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "go"}]
        await router._cognitive_loop(messages, "sys")

        second_call = cog.generate_structured.call_args_list[1]
        assistant_content = second_call.args[0][1]["content"]
        # Only tool_use block, no text block
        assert len(assistant_content) == 1
        assert assistant_content[0]["type"] == "tool_use"

    @pytest.mark.asyncio()
    async def test_first_response_no_tools_returns_immediately(
        self, session_mgr: SessionManager, tools: MagicMock
    ) -> None:
        """If LLM doesn't use tools on first call, returns text immediately."""
        responses = [
            CognitiveResponse(text="No tools needed.", stop_reason="end_turn"),
        ]
        cog = _make_structured_cognitive(responses)
        router = IntelligentRouter(cognitive=cog, session_manager=session_mgr, tools=tools)

        messages = [{"role": "user", "content": "hello"}]
        result = await router._cognitive_loop(messages, "sys")

        assert result == "No tools needed."
        assert cog.generate_structured.await_count == 1
        tools.execute.assert_not_awaited()


# ======================================================================
# TestPersonaIntegration — persona engine with IntelligentRouter
# ======================================================================


class TestPersonaIntegration:
    """Tests for persona engine integration in IntelligentRouter."""

    @pytest.fixture()
    def cognitive(self) -> AsyncMock:
        return _make_mock_cognitive()

    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "persona-test.db")
        yield mgr
        mgr.close()

    @pytest.mark.asyncio()
    async def test_handle_message_with_persona_engine(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """When persona_engine is set, it selects a persona for the message."""
        from animus_bootstrap.personas.engine import PersonaEngine, PersonaProfile
        from animus_bootstrap.personas.voice import VoiceConfig

        engine = PersonaEngine()
        persona = PersonaProfile(
            name="TestBot",
            system_prompt="You are TestBot.",
            voice=VoiceConfig(tone="formal"),
            is_default=True,
        )
        engine.register_persona(persona)

        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            persona_engine=engine,
        )
        msg = _make_message("hello")
        resp = await router.handle_message(msg)
        assert isinstance(resp, GatewayResponse)
        assert resp.text == "I'm Animus."

        # System prompt should use persona's system_prompt
        call_kwargs = cognitive.generate_response.call_args
        system = call_kwargs.kwargs["system_prompt"]
        assert "You are TestBot." in system

    def test_build_system_prompt_with_persona(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """_build_system_prompt uses persona.system_prompt when no context_adapter."""
        from animus_bootstrap.personas.engine import PersonaProfile
        from animus_bootstrap.personas.voice import VoiceConfig

        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            system_prompt="Fallback prompt.",
        )
        persona = PersonaProfile(
            name="CodeBot",
            system_prompt="You are CodeBot, a coding assistant.",
            voice=VoiceConfig(tone="technical"),
        )
        result = router._build_system_prompt(None, persona=persona)
        assert "You are CodeBot, a coding assistant." in result
        # Should NOT contain the fallback prompt
        assert "Fallback prompt." not in result

    def test_build_system_prompt_with_persona_and_context_adapter(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """_build_system_prompt uses context_adapter when both persona and adapter present."""
        from animus_bootstrap.personas.context import ContextAdapter
        from animus_bootstrap.personas.engine import PersonaProfile
        from animus_bootstrap.personas.voice import VoiceConfig

        adapter = ContextAdapter()
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            system_prompt="Fallback.",
            context_adapter=adapter,
        )
        persona = PersonaProfile(
            name="MentorBot",
            system_prompt="You are MentorBot.",
            voice=VoiceConfig(tone="mentor"),
            knowledge_domains=["python", "testing"],
        )
        message = _make_message("teach me", channel="slack")
        result = router._build_system_prompt(None, persona=persona, message=message)
        # Should use the adapted prompt (includes voice + channel norms)
        assert "You are MentorBot." in result
        assert "python" in result
        # Should NOT contain fallback
        assert "Fallback." not in result

    def test_build_system_prompt_no_persona_uses_system_prompt(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """Without persona, _build_system_prompt uses the system_prompt fallback."""
        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
            system_prompt="Base system prompt.",
        )
        result = router._build_system_prompt(None, persona=None)
        assert result == "Base system prompt."

    def test_build_system_prompt_persona_with_memory(
        self,
        cognitive: AsyncMock,
        session_mgr: SessionManager,
    ) -> None:
        """Persona prompt combines with memory context sections."""
        from animus_bootstrap.personas.engine import PersonaProfile

        router = IntelligentRouter(
            cognitive=cognitive,
            session_manager=session_mgr,
        )
        persona = PersonaProfile(
            name="MemBot",
            system_prompt="You are MemBot.",
        )
        mem_ctx = MemoryContext(
            semantic=["User is a Python developer"],
        )
        result = router._build_system_prompt(mem_ctx, persona=persona)
        assert "You are MemBot." in result
        assert "## Known Facts" in result
        assert "User is a Python developer" in result
