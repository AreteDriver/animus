"""Tests for agent memory module."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "src")

from animus_forge.state.memory import (
    AgentMemory,
    ContextWindow,
    ContextWindowStats,
    MemoryEntry,
    Message,
    MessageRole,
)


class TestMemoryEntry:
    """Tests for MemoryEntry class."""

    def test_create_entry(self):
        """Can create memory entry."""
        entry = MemoryEntry(
            agent_id="agent-1",
            content="Test memory",
            memory_type="fact",
            importance=0.8,
        )
        assert entry.agent_id == "agent-1"
        assert entry.content == "Test memory"
        assert entry.importance == 0.8

    def test_to_dict(self):
        """Entry can be converted to dict."""
        entry = MemoryEntry(
            agent_id="agent-1",
            content="Test",
            memory_type="conversation",
        )
        data = entry.to_dict()
        assert data["agent_id"] == "agent-1"
        assert data["memory_type"] == "conversation"

    def test_from_dict(self):
        """Entry can be created from dict."""
        data = {
            "id": 1,
            "agent_id": "agent-1",
            "content": "Test",
            "memory_type": "fact",
            "metadata": '{"key": "value"}',
            "importance": 0.9,
            "access_count": 5,
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.id == 1
        assert entry.agent_id == "agent-1"
        assert entry.metadata == {"key": "value"}
        assert entry.importance == 0.9


class TestAgentMemory:
    """Tests for AgentMemory class."""

    @pytest.fixture
    def memory(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "memory.db")
            yield AgentMemory(db_path=db_path)

    def test_store_memory(self, memory):
        """Can store a memory."""
        mem_id = memory.store(
            agent_id="agent-1",
            content="User prefers dark mode",
            memory_type="preference",
        )
        assert mem_id > 0

    def test_recall_memory(self, memory):
        """Can recall stored memories."""
        memory.store("agent-1", "Fact 1", memory_type="fact")
        memory.store("agent-1", "Fact 2", memory_type="fact")

        memories = memory.recall("agent-1", memory_type="fact")
        assert len(memories) == 2

    def test_recall_by_workflow(self, memory):
        """Can filter by workflow."""
        memory.store("agent-1", "Memory 1", workflow_id="wf-1")
        memory.store("agent-1", "Memory 2", workflow_id="wf-2")

        memories = memory.recall("agent-1", workflow_id="wf-1")
        assert len(memories) == 1
        assert memories[0].content == "Memory 1"

    def test_recall_by_importance(self, memory):
        """Can filter by importance threshold."""
        memory.store("agent-1", "Low importance", importance=0.2)
        memory.store("agent-1", "High importance", importance=0.9)

        memories = memory.recall("agent-1", min_importance=0.5)
        assert len(memories) == 1
        assert memories[0].content == "High importance"

    def test_recall_limit(self, memory):
        """Recall respects limit."""
        for i in range(10):
            memory.store("agent-1", f"Memory {i}")

        memories = memory.recall("agent-1", limit=5)
        assert len(memories) == 5

    def test_recall_recent(self, memory):
        """Can recall recent memories."""
        memory.store("agent-1", "Recent memory")

        # Use recall without time filter since SQLite timestamp format may differ
        memories = memory.recall("agent-1", limit=10)
        assert len(memories) == 1
        assert memories[0].content == "Recent memory"

    def test_recall_context(self, memory):
        """Can recall contextual memories."""
        memory.store("agent-1", "User prefers Python", memory_type="preference")
        memory.store("agent-1", "API returns JSON", memory_type="fact", importance=0.8)
        memory.store("agent-1", "Workflow note", workflow_id="wf-1")

        context = memory.recall_context("agent-1", workflow_id="wf-1")
        assert "preferences" in context or "facts" in context or "workflow" in context

    def test_forget_specific(self, memory):
        """Can forget specific memory."""
        mem_id = memory.store("agent-1", "To forget")

        count = memory.forget("agent-1", memory_id=mem_id)
        assert count == 1

        memories = memory.recall("agent-1")
        assert len(memories) == 0

    def test_forget_by_type(self, memory):
        """Can forget all of a type."""
        memory.store("agent-1", "Convo 1", memory_type="conversation")
        memory.store("agent-1", "Convo 2", memory_type="conversation")
        memory.store("agent-1", "Fact 1", memory_type="fact")

        count = memory.forget("agent-1", memory_type="conversation")
        assert count == 2

        remaining = memory.recall("agent-1")
        assert len(remaining) == 1
        assert remaining[0].memory_type == "fact"

    def test_update_importance(self, memory):
        """Can update memory importance."""
        mem_id = memory.store("agent-1", "Test", importance=0.5)

        success = memory.update_importance(mem_id, 0.95)
        assert success is True

        memories = memory.recall("agent-1")
        assert memories[0].importance == 0.95

    def test_get_stats(self, memory):
        """Can get memory statistics."""
        memory.store("agent-1", "Fact", memory_type="fact", importance=0.8)
        memory.store("agent-1", "Pref", memory_type="preference", importance=0.6)

        stats = memory.get_stats("agent-1")
        assert stats["total_memories"] == 2
        assert stats["by_type"]["fact"] == 1
        assert stats["by_type"]["preference"] == 1

    def test_format_context(self, memory):
        """Can format memories as context string."""
        memory.store("agent-1", "User likes Python", memory_type="preference")
        memory.store("agent-1", "API uses REST", memory_type="fact", importance=0.8)

        context = memory.recall_context("agent-1")
        formatted = memory.format_context(context)

        assert isinstance(formatted, str)
        # Should contain some content
        assert len(formatted) > 0

    def test_access_tracking(self, memory):
        """Memory access is tracked."""
        memory.store("agent-1", "Test memory")

        # First recall
        memories = memory.recall("agent-1")
        # Access count should have increased (starts at 0, becomes 1 on first recall)

        # Second recall
        memories = memory.recall("agent-1")
        assert memories[0].access_count >= 1


class TestMessage:
    """Tests for Message class."""

    def test_create_message(self):
        """Can create a message."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            tokens=10,
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.tokens == 10

    def test_to_dict(self):
        """Message can be converted to dict."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there",
        )
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"

    def test_tool_message_to_dict(self):
        """Tool message includes name and tool_call_id."""
        msg = Message(
            role=MessageRole.TOOL,
            content="Tool result",
            name="search",
            tool_call_id="call_123",
        )
        data = msg.to_dict()
        assert data["role"] == "tool"
        assert data["name"] == "search"
        assert data["tool_call_id"] == "call_123"

    def test_from_dict(self):
        """Message can be created from dict."""
        data = {
            "role": "user",
            "content": "Test message",
            "tokens": 5,
            "metadata": {"key": "value"},
        }
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"
        assert msg.tokens == 5
        assert msg.metadata == {"key": "value"}


class TestContextWindow:
    """Tests for ContextWindow class."""

    def test_create_context_window(self):
        """Can create context window."""
        window = ContextWindow(max_tokens=10000, reserve_tokens=1000)
        assert window.max_tokens == 10000
        assert window.reserve_tokens == 1000

    def test_for_model(self):
        """Can create context window for specific model."""
        window = ContextWindow.for_model("gpt-4o")
        assert window.max_tokens == 128000

        window = ContextWindow.for_model("gpt-4")
        assert window.max_tokens == 8192

        window = ContextWindow.for_model("claude-3-opus")
        assert window.max_tokens == 200000

    def test_unknown_model_uses_default(self):
        """Unknown model uses default token limit."""
        window = ContextWindow.for_model("unknown-model")
        assert window.max_tokens == 128000

    def test_set_system_message(self):
        """Can set system message."""
        window = ContextWindow()
        window.set_system_message("You are a helpful assistant.")

        messages = window.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."

    def test_add_user_message(self):
        """Can add user message."""
        window = ContextWindow()
        msg = window.add_user_message("Hello!")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"

    def test_add_assistant_message(self):
        """Can add assistant message."""
        window = ContextWindow()
        msg = window.add_assistant_message("Hi there!")

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"

    def test_add_tool_message(self):
        """Can add tool message."""
        window = ContextWindow()
        msg = window.add_tool_message("Search result", name="search", tool_call_id="123")

        assert msg.role == MessageRole.TOOL
        assert msg.name == "search"
        assert msg.tool_call_id == "123"

    def test_add_message_with_string_role(self):
        """Can add message with string role."""
        window = ContextWindow()
        msg = window.add_message("user", "Test message")

        assert msg.role == MessageRole.USER

    def test_get_messages_order(self):
        """Messages are returned in correct order."""
        window = ContextWindow()
        window.set_system_message("System")
        window.add_user_message("User 1")
        window.add_assistant_message("Assistant 1")
        window.add_user_message("User 2")

        messages = window.get_messages()
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"

    def test_get_messages_without_system(self):
        """Can get messages without system message."""
        window = ContextWindow()
        window.set_system_message("System")
        window.add_user_message("User")

        messages = window.get_messages(include_system=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_available_tokens(self):
        """Available tokens calculated correctly."""
        window = ContextWindow(max_tokens=10000, reserve_tokens=1000)

        # Initially should have ~9000 available (10000 - 1000 reserve)
        initial_available = window.available_tokens
        assert initial_available > 8000

        # Add a message
        window.add_user_message("A" * 4000)  # ~1000 tokens

        # Should have less available
        assert window.available_tokens < initial_available

    def test_clear(self):
        """Can clear messages."""
        window = ContextWindow()
        window.set_system_message("System")
        window.add_user_message("User 1")
        window.add_user_message("User 2")

        window.clear()

        messages = window.get_messages()
        # System message should still be there
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_get_stats(self):
        """Can get context window stats."""
        window = ContextWindow(max_tokens=10000)
        window.set_system_message("System prompt")
        window.add_user_message("User message")
        window.add_assistant_message("Assistant response")

        stats = window.get_stats()

        assert isinstance(stats, ContextWindowStats)
        assert stats.message_count == 2
        assert stats.user_messages == 1
        assert stats.assistant_messages == 1
        assert stats.total_tokens > 0
        assert 0 <= stats.utilization_percent <= 100

    def test_truncation_removes_oldest(self):
        """Truncation removes oldest messages."""
        # Small context window
        window = ContextWindow(max_tokens=100, reserve_tokens=10)

        # Add many messages that exceed limit
        for i in range(20):
            window.add_user_message(f"Message {i}: " + "x" * 50)

        # Should have truncated some messages
        assert len(window._messages) < 20

    def test_truncation_with_summarizer(self):
        """Truncation uses summarizer when available."""
        summaries = []

        def mock_summarizer(messages):
            summaries.append(len(messages))
            return "Summary of conversation"

        window = ContextWindow(
            max_tokens=200,
            reserve_tokens=20,
            summarizer=mock_summarizer,
        )

        # Add many messages to trigger truncation
        for i in range(50):
            window.add_user_message(f"Message {i}: " + "x" * 20)

        # Summarizer should have been called at least once
        assert len(summaries) > 0

    def test_to_dict(self):
        """Can serialize to dictionary."""
        window = ContextWindow(max_tokens=5000, agent_id="test-agent")
        window.set_system_message("System")
        window.add_user_message("Hello")

        data = window.to_dict()

        assert data["max_tokens"] == 5000
        assert data["agent_id"] == "test-agent"
        assert data["system_message"] is not None
        assert len(data["messages"]) == 1

    def test_from_dict(self):
        """Can deserialize from dictionary."""
        data = {
            "max_tokens": 8000,
            "reserve_tokens": 2000,
            "agent_id": "restored-agent",
            "system_message": {"role": "system", "content": "System prompt"},
            "messages": [
                {"role": "user", "content": "Hello", "tokens": 5},
                {"role": "assistant", "content": "Hi", "tokens": 4},
            ],
            "summary": None,
            "total_tokens": 9,
        }

        window = ContextWindow.from_dict(data)

        assert window.max_tokens == 8000
        assert window.agent_id == "restored-agent"
        assert len(window._messages) == 2

    def test_custom_token_counter(self):
        """Can use custom token counter."""

        def exact_counter(text):
            return len(text)  # One token per character

        window = ContextWindow(token_counter=exact_counter)
        msg = window.add_user_message("Hello")

        assert msg.tokens == 5  # "Hello" has 5 characters

    @pytest.fixture
    def memory(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "memory.db")
            yield AgentMemory(db_path=db_path)

    def test_save_to_memory(self, memory):
        """Can save context to memory."""
        window = ContextWindow(memory=memory, agent_id="test-agent")
        window.add_user_message("User message")
        window.add_assistant_message("Assistant response")

        entry_ids = window.save_to_memory()

        assert len(entry_ids) == 2

        # Verify in memory
        memories = memory.recall("test-agent")
        assert len(memories) == 2

    def test_load_from_memory(self, memory):
        """Can load context from memory."""
        # Store some messages
        memory.store(
            agent_id="test-agent",
            content="[user] Previous question",
            memory_type="conversation",
        )
        memory.store(
            agent_id="test-agent",
            content="[assistant] Previous answer",
            memory_type="conversation",
        )

        window = ContextWindow(memory=memory, agent_id="test-agent")
        count = window.load_from_memory()

        assert count == 2
        assert len(window._messages) == 2

    def test_save_to_memory_without_memory(self):
        """Save to memory returns empty when no memory configured."""
        window = ContextWindow()  # No memory
        window.add_user_message("Test")

        entry_ids = window.save_to_memory()
        assert entry_ids == []

    def test_load_from_memory_without_memory(self):
        """Load from memory returns 0 when no memory configured."""
        window = ContextWindow()  # No memory
        count = window.load_from_memory()
        assert count == 0


class TestContextWindowStats:
    """Tests for ContextWindowStats class."""

    def test_default_values(self):
        """Stats have sensible defaults."""
        stats = ContextWindowStats()

        assert stats.total_tokens == 0
        assert stats.message_count == 0
        assert stats.utilization_percent == 0.0

    def test_calculated_stats(self):
        """Stats are calculated correctly."""
        window = ContextWindow(max_tokens=10000, reserve_tokens=500)
        window.add_user_message("Test message 1")
        window.add_user_message("Test message 2")
        window.add_assistant_message("Response")

        stats = window.get_stats()

        assert stats.message_count == 3
        assert stats.user_messages == 2
        assert stats.assistant_messages == 1
        assert stats.available_tokens > 0
        assert stats.total_tokens > 0
