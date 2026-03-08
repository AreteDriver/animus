"""High-level agent task runner.

Provides a unified interface for submitting agent tasks with tool execution,
streaming progress, and persistent results. Bridges SubAgentManager,
AgentProvider, and ForgeToolRegistry into a single entry point.

Usage:
    runner = AgentTaskRunner(provider=agent_provider, tool_registry=registry)
    result = await runner.run("builder", "Implement login feature", tools=True)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.agents.agent_config import AgentConfig
    from animus_forge.agents.provider_wrapper import AgentProvider
    from animus_forge.agents.subagent_manager import SubAgentManager
    from animus_forge.state.agent_memory import AgentMemory
    from animus_forge.tools.registry import ForgeToolRegistry
    from animus_forge.websocket.broadcaster import Broadcaster

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of an agent task execution."""

    task_id: str
    agent: str
    task: str
    output: str = ""
    status: str = "completed"
    error: str | None = None
    tool_calls: int = 0
    tokens_used: int = 0
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "task": self.task[:200],
            "output": self.output[:2000] if self.output else "",
            "status": self.status,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
        }


class AgentTaskRunner:
    """Unified agent task execution with tools, streaming, and memory.

    Combines:
    - AgentProvider for LLM completions
    - ForgeToolRegistry for tool execution
    - SubAgentManager for concurrent agent spawning
    - Broadcaster for real-time progress
    - AgentMemory for persistent context
    """

    def __init__(
        self,
        provider: AgentProvider,
        tool_registry: ForgeToolRegistry | None = None,
        subagent_manager: SubAgentManager | None = None,
        broadcaster: Broadcaster | None = None,
        agent_memory: AgentMemory | None = None,
        budget_manager: Any = None,
    ):
        self.provider = provider
        self.tool_registry = tool_registry
        self.subagent_manager = subagent_manager
        self.broadcaster = broadcaster
        self.agent_memory = agent_memory
        self.budget_manager = budget_manager
        self._results: dict[str, TaskResult] = {}

    async def run(
        self,
        agent: str,
        task: str,
        use_tools: bool = True,
        config: AgentConfig | None = None,
        context: str = "",
        max_iterations: int = 8,
    ) -> TaskResult:
        """Execute an agent task.

        Args:
            agent: Agent role name (e.g., 'builder', 'tester').
            task: Task description.
            use_tools: Whether to enable tool execution.
            config: Optional agent config override.
            context: Additional context to inject.
            max_iterations: Max tool loop iterations.

        Returns:
            TaskResult with output and metadata.
        """
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        start = time.time()

        result = TaskResult(task_id=task_id, agent=agent, task=task)
        self._results[task_id] = result

        self._emit_status(task_id, "running", agent=agent, task=task)

        try:
            # Build messages
            messages = self._build_messages(agent, task, context)

            # Inject memory context
            if self.agent_memory is not None:
                try:
                    memories = self.agent_memory.recall_context(agent, max_entries=10)
                    if memories:
                        mem_text = self.agent_memory.format_context(memories)
                        messages.insert(
                            1, {"role": "user", "content": f"Context from memory:\n{mem_text}"}
                        )
                except Exception as e:
                    logger.debug("Failed to recall memory: %s", e)

            # Execute with or without tools
            tool_count = 0
            if use_tools and self.tool_registry:

                def progress_cb(stage: str, detail: str) -> None:
                    nonlocal tool_count
                    if stage == "tools":
                        tool_count += 1
                    self._emit_log(task_id, "info", f"[{stage}] {detail}")

                output = await self.provider.complete_with_tools(
                    messages=messages,
                    tool_registry=self.tool_registry,
                    max_iterations=max_iterations,
                    progress_callback=progress_cb,
                    agent_id=agent,
                )
            else:
                output = await self.provider.complete(messages)

            result.output = output
            result.status = "completed"
            result.tool_calls = tool_count
            result.duration_ms = int((time.time() - start) * 1000)

            # Store in memory
            if self.agent_memory is not None:
                try:
                    self.agent_memory.store(
                        agent_id=agent,
                        content=output[:500],
                        memory_type="learned",
                        metadata={"task": task[:200], "task_id": task_id},
                        importance=0.6,
                    )
                except Exception as e:
                    logger.debug("Failed to store memory: %s", e)

            self._emit_status(task_id, "completed", agent=agent)

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.duration_ms = int((time.time() - start) * 1000)
            self._emit_status(task_id, "failed", agent=agent, error=str(e))
            logger.error("Agent %s task failed: %s", agent, e)

        return result

    async def spawn(
        self,
        agent: str,
        task: str,
        use_tools: bool = True,
        parent_id: str | None = None,
    ) -> str:
        """Spawn an agent task in the background via SubAgentManager.

        Args:
            agent: Agent role name.
            task: Task description.
            use_tools: Whether to enable tools.
            parent_id: Parent run ID for nesting.

        Returns:
            Run ID for tracking.

        Raises:
            RuntimeError: If no SubAgentManager configured.
        """
        if self.subagent_manager is None:
            raise RuntimeError("SubAgentManager not configured for background spawning")

        async def execute_fn(agent_name: str, task_desc: str, config: AgentConfig) -> str:
            result = await self.run(agent_name, task_desc, use_tools=use_tools, config=config)
            if result.error:
                raise RuntimeError(result.error)
            return result.output

        run = await self.subagent_manager.spawn(
            agent=agent,
            task=task,
            execute_fn=execute_fn,
            parent_id=parent_id,
        )
        return run.run_id

    def get_result(self, task_id: str) -> TaskResult | None:
        """Get a task result by ID."""
        return self._results.get(task_id)

    def list_results(self, status: str | None = None) -> list[TaskResult]:
        """List all task results, optionally filtered."""
        results = list(self._results.values())
        if status:
            results = [r for r in results if r.status == status]
        return results

    def _build_messages(self, agent: str, task: str, context: str = "") -> list[dict[str, str]]:
        """Build conversation messages for the agent."""
        system = f"You are a {agent} agent. Be thorough and precise."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        if context:
            messages.insert(1, {"role": "user", "content": f"Additional context:\n{context}"})

        return messages

    def _emit_status(self, task_id: str, status: str, **kwargs: Any) -> None:
        """Emit a status update via broadcaster."""
        if self.broadcaster is None:
            return
        try:
            self.broadcaster.on_status_change(
                execution_id=task_id,
                status=status,
                current_step=kwargs.get("agent"),
                error=kwargs.get("error"),
            )
        except Exception:
            pass

    def _emit_log(self, task_id: str, level: str, message: str) -> None:
        """Emit a log entry via broadcaster."""
        if self.broadcaster is None:
            return
        try:
            self.broadcaster.on_log(
                execution_id=task_id,
                level=level,
                message=message,
            )
        except Exception:
            pass
