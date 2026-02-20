"""Agent Context Manager for Workflow Execution.

Provides memory-aware context for AI agents during workflow execution.
Integrates AgentMemory and ContextWindow for seamless context injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .memory import AgentMemory, ContextWindow, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory behavior during workflow execution."""

    # Memory retrieval settings
    max_facts: int = 10
    max_preferences: int = 5
    max_recent: int = 5
    min_importance: float = 0.3

    # Memory storage settings
    store_outputs: bool = True
    output_importance: float = 0.6
    store_errors: bool = True
    error_importance: float = 0.8

    # Context injection settings
    inject_facts: bool = True
    inject_preferences: bool = True
    inject_workflow_context: bool = True
    inject_recent: bool = True


@dataclass
class AgentContext:
    """Memory-aware context for an agent during workflow execution.

    Manages:
    - Long-term memory (AgentMemory) for facts, preferences, learned info
    - Short-term context (ContextWindow) for current conversation
    - Context injection for AI prompts
    - Memory storage for outputs and learnings
    """

    agent_id: str
    workflow_id: str | None = None
    memory: AgentMemory | None = None
    context_window: ContextWindow | None = None
    config: MemoryConfig = field(default_factory=MemoryConfig)
    _cached_context: str | None = field(default=None, repr=False)
    _context_dirty: bool = field(default=True, repr=False)

    def __post_init__(self):
        """Initialize context window with memory if not provided."""
        if self.memory and not self.context_window:
            self.context_window = ContextWindow(
                memory=self.memory,
                agent_id=self.agent_id,
            )

    def load_context(self) -> str:
        """Load relevant memories and format as context string.

        Returns:
            Formatted context string for prompt injection
        """
        if not self.memory:
            return ""

        if not self._context_dirty and self._cached_context:
            return self._cached_context

        # Calculate max entries based on config
        max_entries = (
            (self.config.max_facts if self.config.inject_facts else 0)
            + (self.config.max_preferences if self.config.inject_preferences else 0)
            + (self.config.max_recent if self.config.inject_recent else 0)
            + (5 if self.config.inject_workflow_context else 0)
        )

        memories = self.memory.recall_context(
            agent_id=self.agent_id,
            workflow_id=self.workflow_id,
            include_facts=self.config.inject_facts,
            include_preferences=self.config.inject_preferences,
            max_entries=max_entries,
        )

        self._cached_context = self.memory.format_context(memories)
        self._context_dirty = False
        return self._cached_context

    def inject_into_prompt(self, prompt: str) -> str:
        """Inject memory context into a prompt.

        Args:
            prompt: Original prompt

        Returns:
            Prompt with memory context prepended (if available)
        """
        context = self.load_context()
        if not context:
            return prompt

        return f"""## Prior Context

{context}

## Current Task

{prompt}"""

    def store_output(
        self,
        step_id: str,
        output: dict,
        memory_type: str = "learned",
    ) -> list[int]:
        """Store step outputs as memories.

        Args:
            step_id: Step identifier
            output: Step output dictionary
            memory_type: Type of memory to store

        Returns:
            List of stored memory IDs
        """
        if not self.memory or not self.config.store_outputs:
            return []

        memory_ids = []

        # Store main response if present
        if "response" in output:
            response = output["response"]
            # Truncate very long responses for memory storage
            if len(response) > 2000:
                response = response[:2000] + "..."

            memory_id = self.memory.store(
                agent_id=self.agent_id,
                content=f"Output from {step_id}: {response}",
                memory_type=memory_type,
                workflow_id=self.workflow_id,
                metadata={"step_id": step_id, "output_type": "response"},
                importance=self.config.output_importance,
            )
            memory_ids.append(memory_id)
            self._context_dirty = True

        # Store any structured outputs
        for key, value in output.items():
            if key == "response":
                continue
            if isinstance(value, (str, int, float, bool)):
                memory_id = self.memory.store(
                    agent_id=self.agent_id,
                    content=f"{step_id}.{key} = {value}",
                    memory_type="fact",
                    workflow_id=self.workflow_id,
                    metadata={"step_id": step_id, "output_key": key},
                    importance=self.config.output_importance * 0.8,
                )
                memory_ids.append(memory_id)
                self._context_dirty = True

        return memory_ids

    def store_error(self, step_id: str, error: str) -> int | None:
        """Store an error as a memory for learning.

        Args:
            step_id: Step that failed
            error: Error message

        Returns:
            Memory ID if stored
        """
        if not self.memory or not self.config.store_errors:
            return None

        memory_id = self.memory.store(
            agent_id=self.agent_id,
            content=f"Error in {step_id}: {error}",
            memory_type="learned",
            workflow_id=self.workflow_id,
            metadata={"step_id": step_id, "is_error": True},
            importance=self.config.error_importance,
        )
        self._context_dirty = True
        return memory_id

    def store_fact(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> int:
        """Store a fact as a memory.

        Args:
            content: Fact content
            importance: Importance score (0.0 to 1.0)
            metadata: Optional metadata

        Returns:
            Memory ID
        """
        if not self.memory:
            raise RuntimeError("No memory backend configured")

        memory_id = self.memory.store(
            agent_id=self.agent_id,
            content=content,
            memory_type="fact",
            workflow_id=self.workflow_id,
            metadata=metadata or {},
            importance=importance,
        )
        self._context_dirty = True
        return memory_id

    def store_preference(
        self,
        content: str,
        importance: float = 0.7,
        metadata: dict | None = None,
    ) -> int:
        """Store a preference as a memory.

        Args:
            content: Preference content
            importance: Importance score
            metadata: Optional metadata

        Returns:
            Memory ID
        """
        if not self.memory:
            raise RuntimeError("No memory backend configured")

        memory_id = self.memory.store(
            agent_id=self.agent_id,
            content=content,
            memory_type="preference",
            workflow_id=self.workflow_id,
            metadata=metadata or {},
            importance=importance,
        )
        self._context_dirty = True
        return memory_id

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the context window.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
        """
        if self.context_window:
            self.context_window.add_message(MessageRole(role), content)

    def get_conversation_messages(self) -> list[dict]:
        """Get conversation messages for API call.

        Returns:
            List of message dicts
        """
        if self.context_window:
            return self.context_window.get_messages()
        return []

    def save_to_memory(self) -> None:
        """Save current context window to long-term memory."""
        if self.context_window:
            self.context_window.save_to_memory(self.workflow_id)

    def load_from_memory(self, workflow_id: str | None = None) -> bool:
        """Load previous context from memory.

        Args:
            workflow_id: Optional workflow ID to load context for

        Returns:
            True if context was loaded
        """
        if self.context_window:
            return self.context_window.load_from_memory(workflow_id or self.workflow_id)
        return False

    def get_stats(self) -> dict:
        """Get memory statistics for this agent.

        Returns:
            Dictionary with memory stats
        """
        stats = {
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id,
            "has_memory": self.memory is not None,
            "has_context_window": self.context_window is not None,
        }

        if self.memory:
            stats["memory_stats"] = self.memory.get_stats(self.agent_id)

        if self.context_window:
            cw_stats = self.context_window.get_stats()
            stats["context_window"] = {
                "total_tokens": cw_stats.total_tokens,
                "message_count": cw_stats.message_count,
                "utilization_percent": cw_stats.utilization_percent,
            }

        return stats

    def clear_cache(self) -> None:
        """Clear cached context (forces reload on next access)."""
        self._cached_context = None
        self._context_dirty = True


class WorkflowMemoryManager:
    """Manages agent contexts for an entire workflow execution.

    Creates and tracks AgentContext instances for each agent role
    used during workflow execution.
    """

    def __init__(
        self,
        memory: AgentMemory | None = None,
        workflow_id: str | None = None,
        config: MemoryConfig | None = None,
    ):
        """Initialize workflow memory manager.

        Args:
            memory: Shared AgentMemory backend
            workflow_id: Workflow execution ID
            config: Default memory configuration
        """
        self.memory = memory
        self.workflow_id = workflow_id
        self.config = config or MemoryConfig()
        self._contexts: dict[str, AgentContext] = {}

    def get_context(self, agent_id: str) -> AgentContext:
        """Get or create an agent context.

        Args:
            agent_id: Agent identifier (typically the role)

        Returns:
            AgentContext for the agent
        """
        if agent_id not in self._contexts:
            self._contexts[agent_id] = AgentContext(
                agent_id=agent_id,
                workflow_id=self.workflow_id,
                memory=self.memory,
                config=self.config,
            )
        return self._contexts[agent_id]

    def inject_context(self, agent_id: str, prompt: str) -> str:
        """Inject memory context into a prompt for an agent.

        Args:
            agent_id: Agent identifier
            prompt: Original prompt

        Returns:
            Prompt with context injected
        """
        context = self.get_context(agent_id)
        return context.inject_into_prompt(prompt)

    def store_output(self, agent_id: str, step_id: str, output: dict) -> list[int]:
        """Store step output as memories.

        Args:
            agent_id: Agent identifier
            step_id: Step identifier
            output: Step output

        Returns:
            List of memory IDs
        """
        context = self.get_context(agent_id)
        return context.store_output(step_id, output)

    def store_error(self, agent_id: str, step_id: str, error: str) -> int | None:
        """Store an error as a memory.

        Args:
            agent_id: Agent identifier
            step_id: Step that failed
            error: Error message

        Returns:
            Memory ID if stored
        """
        context = self.get_context(agent_id)
        return context.store_error(step_id, error)

    def save_all(self) -> None:
        """Save all agent contexts to memory."""
        for context in self._contexts.values():
            context.save_to_memory()

    def get_all_stats(self) -> dict:
        """Get statistics for all agent contexts.

        Returns:
            Dictionary with stats for each agent
        """
        return {
            "workflow_id": self.workflow_id,
            "agent_count": len(self._contexts),
            "agents": {
                agent_id: context.get_stats() for agent_id, context in self._contexts.items()
            },
        }


def create_workflow_memory(
    workflow_id: str | None = None,
    db_path: str = "gorgon-memory.db",
    config: MemoryConfig | None = None,
) -> WorkflowMemoryManager:
    """Create a workflow memory manager with a fresh memory backend.

    Args:
        workflow_id: Workflow execution ID
        db_path: Path to SQLite database
        config: Memory configuration

    Returns:
        Configured WorkflowMemoryManager
    """
    memory = AgentMemory(db_path=db_path)
    return WorkflowMemoryManager(
        memory=memory,
        workflow_id=workflow_id,
        config=config,
    )
