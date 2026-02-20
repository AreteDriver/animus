"""Tests for AgentContext and WorkflowMemoryManager."""

import os
import tempfile

import pytest

from animus_forge.state import (
    AgentContext,
    AgentMemory,
    MemoryConfig,
    WorkflowMemoryManager,
    create_workflow_memory,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def memory(temp_db):
    """Create an AgentMemory instance."""
    return AgentMemory(db_path=temp_db)


@pytest.fixture
def agent_context(memory):
    """Create an AgentContext instance."""
    return AgentContext(
        agent_id="test-agent",
        workflow_id="test-workflow",
        memory=memory,
    )


class TestAgentContext:
    """Tests for AgentContext class."""

    def test_create_agent_context(self, memory):
        """Test creating an agent context."""
        ctx = AgentContext(
            agent_id="builder",
            workflow_id="wf-123",
            memory=memory,
        )

        assert ctx.agent_id == "builder"
        assert ctx.workflow_id == "wf-123"
        assert ctx.memory is memory
        assert ctx.context_window is not None

    def test_load_context_empty(self, agent_context):
        """Test loading context when no memories exist."""
        context = agent_context.load_context()
        assert context == ""

    def test_store_and_load_fact(self, agent_context):
        """Test storing and loading a fact."""
        memory_id = agent_context.store_fact("Python is a programming language")
        assert memory_id > 0

        # Clear cache and reload
        agent_context.clear_cache()
        context = agent_context.load_context()
        assert "Python is a programming language" in context

    def test_store_preference(self, agent_context):
        """Test storing a preference."""
        memory_id = agent_context.store_preference("User prefers verbose output")
        assert memory_id > 0

        agent_context.clear_cache()
        context = agent_context.load_context()
        assert "verbose output" in context

    def test_store_output(self, agent_context):
        """Test storing step output."""
        output = {
            "response": "The analysis shows positive results.",
            "confidence": 0.95,
        }

        memory_ids = agent_context.store_output("analyze", output)
        assert len(memory_ids) == 2  # response + confidence fact

    def test_store_error(self, agent_context):
        """Test storing an error."""
        memory_id = agent_context.store_error("step1", "Connection timeout")
        assert memory_id > 0

        agent_context.clear_cache()
        agent_context.load_context()
        # Errors stored as "learned" type, may or may not appear in context
        # depending on recall settings

    def test_inject_into_prompt(self, agent_context):
        """Test injecting memory context into a prompt."""
        # Store some context
        agent_context.store_fact("The project uses FastAPI")

        agent_context.clear_cache()
        prompt = "Write a new endpoint"
        injected = agent_context.inject_into_prompt(prompt)

        assert "## Prior Context" in injected
        assert "## Current Task" in injected
        assert "Write a new endpoint" in injected
        assert "FastAPI" in injected

    def test_inject_empty_context(self, memory):
        """Test that empty context doesn't modify prompt."""
        ctx = AgentContext(
            agent_id="new-agent",
            workflow_id="wf-456",
            memory=memory,
        )

        prompt = "Do something"
        injected = ctx.inject_into_prompt(prompt)
        assert injected == "Do something"

    def test_get_stats(self, agent_context):
        """Test getting agent statistics."""
        agent_context.store_fact("Test fact 1")
        agent_context.store_fact("Test fact 2")

        stats = agent_context.get_stats()
        assert stats["agent_id"] == "test-agent"
        assert stats["has_memory"] is True
        assert stats["has_context_window"] is True
        assert stats["memory_stats"]["total_memories"] >= 2

    def test_context_caching(self, agent_context):
        """Test that context is cached until cleared."""
        # Store with high importance to ensure it's recalled
        agent_context.store_fact("Initial fact", importance=0.8)
        context1 = agent_context.load_context()

        # Store more without clearing cache (high importance)
        agent_context.memory.store(
            agent_id="test-agent",
            content="New fact",
            memory_type="fact",
            importance=0.8,
        )

        # Should return cached version
        context2 = agent_context.load_context()
        assert context1 == context2

        # After clearing, should include new fact
        agent_context.clear_cache()
        context3 = agent_context.load_context()
        assert "New fact" in context3

    def test_config_disables_features(self, memory):
        """Test that config can disable memory features."""
        config = MemoryConfig(
            store_outputs=False,
            inject_facts=False,
        )

        ctx = AgentContext(
            agent_id="test",
            memory=memory,
            config=config,
        )

        # Store a fact directly
        memory.store(agent_id="test", content="A fact", memory_type="fact")

        # Should not appear in context since inject_facts=False
        context = ctx.load_context()
        assert "A fact" not in context

        # Should not store outputs
        ids = ctx.store_output("step", {"response": "test"})
        assert ids == []


class TestWorkflowMemoryManager:
    """Tests for WorkflowMemoryManager class."""

    def test_create_manager(self, memory):
        """Test creating a workflow memory manager."""
        manager = WorkflowMemoryManager(
            memory=memory,
            workflow_id="wf-test",
        )

        assert manager.memory is memory
        assert manager.workflow_id == "wf-test"

    def test_get_context(self, memory):
        """Test getting/creating agent contexts."""
        manager = WorkflowMemoryManager(memory=memory)

        ctx1 = manager.get_context("planner")
        ctx2 = manager.get_context("builder")
        ctx3 = manager.get_context("planner")  # Same as ctx1

        assert ctx1.agent_id == "planner"
        assert ctx2.agent_id == "builder"
        assert ctx1 is ctx3  # Same instance

    def test_inject_context(self, memory):
        """Test injecting context through manager."""
        manager = WorkflowMemoryManager(memory=memory)

        # Store a fact for planner (high importance to ensure recall)
        manager.get_context("planner").store_fact("Project uses Python", importance=0.8)

        # Inject context
        prompt = "Create a plan"
        injected = manager.inject_context("planner", prompt)

        assert "Python" in injected
        assert "Create a plan" in injected

    def test_store_output_through_manager(self, memory):
        """Test storing output through manager."""
        manager = WorkflowMemoryManager(
            memory=memory,
            workflow_id="wf-output-test",
        )

        output = {"response": "Plan created successfully"}
        ids = manager.store_output("planner", "plan-step", output)

        assert len(ids) > 0

    def test_store_error_through_manager(self, memory):
        """Test storing error through manager."""
        manager = WorkflowMemoryManager(memory=memory)

        error_id = manager.store_error("builder", "build-step", "Syntax error")
        assert error_id is not None

    def test_get_all_stats(self, memory):
        """Test getting stats for all agents."""
        manager = WorkflowMemoryManager(
            memory=memory,
            workflow_id="wf-stats",
        )

        # Create some contexts and store data
        manager.get_context("planner").store_fact("Fact 1")
        manager.get_context("builder").store_fact("Fact 2")

        stats = manager.get_all_stats()

        assert stats["workflow_id"] == "wf-stats"
        assert stats["agent_count"] == 2
        assert "planner" in stats["agents"]
        assert "builder" in stats["agents"]

    def test_save_all(self, memory):
        """Test saving all agent contexts."""
        manager = WorkflowMemoryManager(memory=memory)

        # Create contexts with messages
        planner_ctx = manager.get_context("planner")
        planner_ctx.add_message("user", "Create a plan")
        planner_ctx.add_message("assistant", "Here is the plan...")

        # Should not raise
        manager.save_all()


class TestCreateWorkflowMemory:
    """Tests for create_workflow_memory helper."""

    def test_create_with_defaults(self, temp_db):
        """Test creating manager with defaults."""
        manager = create_workflow_memory(db_path=temp_db)

        assert manager.memory is not None
        assert manager.workflow_id is None
        assert manager.config is not None

    def test_create_with_workflow_id(self, temp_db):
        """Test creating manager with workflow ID."""
        manager = create_workflow_memory(
            workflow_id="wf-123",
            db_path=temp_db,
        )

        assert manager.workflow_id == "wf-123"

    def test_create_with_config(self, temp_db):
        """Test creating manager with custom config."""
        config = MemoryConfig(
            max_facts=5,
            store_outputs=False,
        )

        manager = create_workflow_memory(
            db_path=temp_db,
            config=config,
        )

        assert manager.config.max_facts == 5
        assert manager.config.store_outputs is False


class TestMemoryConfig:
    """Tests for MemoryConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()

        assert config.max_facts == 10
        assert config.max_preferences == 5
        assert config.max_recent == 5
        assert config.min_importance == 0.3
        assert config.store_outputs is True
        assert config.store_errors is True
        assert config.inject_facts is True
        assert config.inject_preferences is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MemoryConfig(
            max_facts=20,
            store_outputs=False,
            min_importance=0.5,
        )

        assert config.max_facts == 20
        assert config.store_outputs is False
        assert config.min_importance == 0.5


class TestExecutorMemoryIntegration:
    """Integration tests for executor memory support."""

    def test_executor_accepts_memory_manager(self, memory):
        """Test that WorkflowExecutor accepts memory_manager parameter."""
        from animus_forge.workflow import WorkflowExecutor

        manager = WorkflowMemoryManager(memory=memory)

        executor = WorkflowExecutor(
            memory_manager=manager,
            dry_run=True,
        )

        assert executor.memory_manager is manager

    def test_executor_accepts_memory_config(self, memory):
        """Test that WorkflowExecutor accepts memory_config parameter."""
        from animus_forge.workflow import WorkflowExecutor

        config = MemoryConfig(store_outputs=False)

        executor = WorkflowExecutor(
            memory_config=config,
            dry_run=True,
        )

        assert executor.memory_config is config

    def test_executor_creates_memory_manager_on_execute(self, temp_db):
        """Test that executor creates memory manager during execution."""
        from animus_forge.workflow import StepConfig, WorkflowConfig, WorkflowExecutor

        executor = WorkflowExecutor(dry_run=True)
        assert executor.memory_manager is None

        # Create a simple workflow
        workflow = WorkflowConfig(
            name="test",
            version="1.0",
            description="Test workflow",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo hello"},
                )
            ],
        )

        # Execute with memory enabled
        executor.execute(workflow, enable_memory=True)

        # Memory manager should be created
        assert executor.memory_manager is not None

    def test_executor_can_disable_memory(self, temp_db):
        """Test that memory can be disabled during execution."""
        from animus_forge.workflow import StepConfig, WorkflowConfig, WorkflowExecutor

        executor = WorkflowExecutor(dry_run=True)

        workflow = WorkflowConfig(
            name="test",
            version="1.0",
            description="Test workflow",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo hello"},
                )
            ],
        )

        # Execute with memory disabled
        executor.execute(workflow, enable_memory=False)

        # Memory manager should not be created
        assert executor.memory_manager is None
