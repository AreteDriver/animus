"""Intelligent message router — extends MessageRouter with memory, tools, and automations."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from animus_bootstrap.gateway.cognitive import CognitiveBackend
from animus_bootstrap.gateway.models import GatewayMessage, GatewayResponse
from animus_bootstrap.gateway.router import MessageRouter
from animus_bootstrap.gateway.session import SessionManager
from animus_bootstrap.intelligence.automations.engine import AutomationEngine
from animus_bootstrap.intelligence.memory import MemoryContext, MemoryManager
from animus_bootstrap.intelligence.tools.executor import ToolExecutor
from animus_bootstrap.personas.context import ContextAdapter
from animus_bootstrap.personas.engine import PersonaEngine, PersonaProfile

logger = logging.getLogger(__name__)


class IntelligentRouter(MessageRouter):
    """Extends MessageRouter with memory, tools, and automation support.

    Pipeline:
        1. Evaluate automations (side effects, logging)
        2. Get/create session + store user message
        3. Build conversation context
        4. Recall relevant memories
        5. Build enriched system prompt
        6. Cognitive loop (with tool call handling)
        7. Store assistant message
        8. Store conversation in memory
        9. Return response
    """

    def __init__(
        self,
        cognitive: CognitiveBackend,
        session_manager: SessionManager,
        memory: MemoryManager | None = None,
        tools: ToolExecutor | None = None,
        automations: AutomationEngine | None = None,
        system_prompt: str = "",
        max_tool_iterations: int = 5,
        persona_engine: PersonaEngine | None = None,
        context_adapter: ContextAdapter | None = None,
        identity_manager: Any | None = None,
    ) -> None:
        super().__init__(cognitive, session_manager)
        self._memory = memory
        self._tools = tools
        self._automations = automations
        self._system_prompt = system_prompt
        self._max_tool_iterations = max_tool_iterations
        self._persona_engine = persona_engine
        self._context_adapter = context_adapter
        self._identity_manager = identity_manager
        self._interaction_count: int = 0

    async def handle_message(self, message: GatewayMessage) -> GatewayResponse:
        """Enhanced message handling with memory + tools + automations."""
        self._interaction_count += 1

        # 1. Check automations first (may short-circuit)
        if self._automations:
            try:
                results = await self._automations.evaluate_message(message)
                for result in results:
                    if result.triggered and result.actions_executed:
                        logger.info("Automation '%s' fired", result.rule_name)
            except Exception:
                logger.exception("Automation evaluation failed")

        # 2. Get/create session
        session = await self._session_manager.get_or_create_session(message)
        await self._session_manager.add_message(session, message)

        # 3. Build context from session history
        conversation = await self._session_manager.get_context(session)

        # 4. Recall relevant memories
        memory_context: MemoryContext | None = None
        if self._memory:
            try:
                memory_context = await self._memory.recall(message.text)
            except Exception:
                logger.exception("Memory recall failed")

        # 4b. Select persona for this message
        persona: PersonaProfile | None = None
        if self._persona_engine:
            persona = self._persona_engine.get_persona_for_message(message)

        # 5. Build enriched system prompt
        session_history = conversation if conversation else None
        system = self._build_system_prompt(
            memory_context, persona=persona, message=message, session_history=session_history
        )

        # 6. Cognitive loop (handles tool calls)
        response_text = await self._cognitive_loop(conversation, system)

        # 7. Store assistant message
        assistant_msg = GatewayMessage(
            id=str(uuid.uuid4()),
            channel=message.channel,
            channel_message_id="",
            sender_id="animus",
            sender_name="Animus",
            text=response_text,
            timestamp=datetime.now(UTC),
            role="assistant",
        )
        await self._session_manager.add_message(session, assistant_msg)

        # 8. Store conversation in memory
        if self._memory:
            try:
                updated_context = await self._session_manager.get_context(session)
                await self._memory.store_conversation(session.id, updated_context)
            except Exception:
                logger.exception("Memory store failed")

        return GatewayResponse(text=response_text, channel=message.channel)

    def _build_system_prompt(
        self,
        memory_context: MemoryContext | None,
        persona: PersonaProfile | None = None,
        message: GatewayMessage | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        """Build system prompt enriched with identity, memory context, and persona."""
        parts: list[str] = []

        # 1. Identity files (prepended first — foundational context)
        if self._identity_manager:
            try:
                identity_prompt = self._identity_manager.get_identity_prompt()
                if identity_prompt:
                    parts.append(identity_prompt)
            except Exception:
                logger.exception("Failed to load identity prompt")

        # 2. Determine base prompt from persona or fallback to system_prompt
        if persona and self._context_adapter and message:
            base = self._context_adapter.adapt_prompt(persona, message, session_history)
        elif persona:
            base = persona.system_prompt
        else:
            base = self._system_prompt

        if base:
            parts.append(base)

        if memory_context:
            if memory_context.episodic:
                parts.append("\n## Relevant Past Conversations")
                parts.extend(f"- {m}" for m in memory_context.episodic)
            if memory_context.semantic:
                parts.append("\n## Known Facts")
                parts.extend(f"- {m}" for m in memory_context.semantic)
            if memory_context.procedural:
                parts.append("\n## How-To Knowledge")
                parts.extend(f"- {m}" for m in memory_context.procedural)
            if memory_context.user_prefs:
                parts.append("\n## User Preferences")
                for k, v in memory_context.user_prefs.items():
                    parts.append(f"- {k}: {v}")

        return "\n".join(parts) if parts else ""

    async def _cognitive_loop(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> str:
        """LLM generation loop with tool call handling.

        If the cognitive backend supports structured responses and tools are
        available, runs a multi-turn loop: call LLM -> execute tool calls ->
        feed results back -> repeat until final text or max iterations.
        """
        tool_schemas = self.get_tool_schemas() if self._tools else None

        # Check if backend supports structured responses
        has_structured = hasattr(self._cognitive, "generate_structured")

        if not has_structured or not tool_schemas:
            # Simple path: single LLM call, no tool loop
            return await self._cognitive.generate_response(messages, system_prompt=system_prompt)

        # Structured path: tool_use loop
        current_messages: list[dict] = list(messages)
        response = None

        for _ in range(self._max_tool_iterations):
            response = await self._cognitive.generate_structured(
                current_messages,
                system_prompt=system_prompt,
                tools=tool_schemas,
            )

            if not response.has_tool_calls:
                return response.text

            # Execute tool calls
            results: list[dict] = []
            for tool_call in response.tool_calls:
                result = await self._tools.execute(tool_call.name, tool_call.arguments)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result.output,
                    }
                )

            # Append assistant message with tool_use blocks + tool results
            assistant_content: list[dict] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

            current_messages.append({"role": "assistant", "content": assistant_content})
            current_messages.append({"role": "user", "content": results})

        # Max iterations reached -- return whatever text we have
        if response and response.text:
            return response.text
        return "I've reached my tool call limit for this turn."

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas for injection into LLM requests."""
        if self._tools:
            return self._tools.get_schemas()
        return []
