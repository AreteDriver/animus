"""Performance benchmarks for Gorgon core operations.

Run: pytest tests/test_benchmarks.py --benchmark-only
"""

import asyncio

import pytest

from animus_forge.cache.backends import MemoryCache
from animus_forge.db import TaskStore
from animus_forge.state.backends import SQLiteBackend
from animus_forge.workflow.loader import (
    ConditionConfig,
    WorkflowConfig,
    load_workflow,
)

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def workflow_20_steps():
    """A 20-step workflow dict for parse benchmarks."""
    return {
        "name": "benchmark-workflow",
        "version": "1.0",
        "description": "Benchmark workflow with 20 steps",
        "token_budget": 200000,
        "steps": [
            {
                "id": f"step_{i}",
                "type": "shell",
                "params": {
                    "command": f"echo step {i}",
                    "description": f"Step {i} of the workflow",
                },
                **(
                    {
                        "condition": {
                            "field": f"step_{i - 1}_output",
                            "operator": "not_empty",
                        }
                    }
                    if i > 0
                    else {}
                ),
                "on_failure": "continue_with_default" if i % 3 == 0 else "abort",
                "max_retries": 2,
                "timeout_seconds": 60,
                "outputs": [f"step_{i}_output"],
                "depends_on": [f"step_{i - 1}"] if i > 0 else [],
            }
            for i in range(20)
        ],
        "inputs": {
            "input": {"type": "string", "required": True, "description": "Input text"},
        },
        "outputs": ["step_19_output"],
    }


@pytest.fixture
def workflow_yaml_10(tmp_path):
    """A 10-step workflow YAML file."""
    content = """
name: benchmark-yaml-workflow
version: "1.0"
description: Benchmark YAML workflow
token_budget: 100000
steps:
"""
    for i in range(10):
        content += f"""  - id: step_{i}
    type: shell
    params:
      command: "echo step {i}"
"""
    path = tmp_path / "bench.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def task_store_db(tmp_path):
    """TaskStore with a temporary SQLite backend."""
    db_path = str(tmp_path / "bench_tasks.db")
    backend = SQLiteBackend(db_path)
    # Create required tables
    backend.executescript("""
        CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            status TEXT NOT NULL,
            agent_role TEXT,
            model TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            duration_ms INTEGER DEFAULT 0,
            error TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_scores (
            agent_role TEXT PRIMARY KEY,
            total_tasks INTEGER DEFAULT 0,
            successful_tasks INTEGER DEFAULT 0,
            failed_tasks INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            avg_duration_ms REAL DEFAULT 0.0,
            success_rate REAL DEFAULT 0.0,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS budget_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            agent_role TEXT,
            total_tokens INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            task_count INTEGER DEFAULT 0
        );
    """)
    store = TaskStore(backend)
    yield store
    backend.close()


# ═══════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════


class TestWorkflowParseBenchmark:
    """Workflow config parsing performance."""

    def test_parse_20_step_workflow(self, benchmark, workflow_20_steps):
        """Benchmark: WorkflowConfig.from_dict on 20-step workflow."""
        config = benchmark(WorkflowConfig.from_dict, workflow_20_steps)
        assert len(config.steps) == 20
        assert config.name == "benchmark-workflow"


class TestYAMLLoadBenchmark:
    """YAML workflow loading performance."""

    def test_load_yaml_10_steps(self, benchmark, workflow_yaml_10):
        """Benchmark: load_workflow from 10-step YAML file."""
        config = benchmark(load_workflow, str(workflow_yaml_10), validate_path=False)
        assert len(config.steps) == 10


class TestConditionEvalBenchmark:
    """Condition evaluation throughput."""

    def test_condition_evaluate_1000(self, benchmark):
        """Benchmark: ConditionConfig.evaluate x1000 mixed operations."""
        conditions = [
            ConditionConfig(field="status", operator="equals", value="success"),
            ConditionConfig(field="count", operator="greater_than", value=5),
            ConditionConfig(field="tags", operator="contains", value="urgent"),
            ConditionConfig(field="output", operator="not_empty"),
            ConditionConfig(field="score", operator="less_than", value=100),
        ]
        context = {
            "status": "success",
            "count": 10,
            "tags": ["urgent", "high-priority"],
            "output": "result data",
            "score": 42,
        }

        def evaluate_all():
            for _ in range(200):
                for cond in conditions:
                    cond.evaluate(context)

        benchmark(evaluate_all)


class TestCacheBenchmark:
    """MemoryCache get/set performance."""

    def test_cache_set_get_1000(self, benchmark):
        """Benchmark: MemoryCache set+get x1000."""
        cache = MemoryCache(max_size=2000, default_ttl=300)

        def cache_ops():
            loop = asyncio.new_event_loop()
            try:
                for i in range(1000):
                    loop.run_until_complete(cache.set(f"key_{i}", {"data": i, "nested": {"a": i}}))
                for i in range(1000):
                    loop.run_until_complete(cache.get(f"key_{i}"))
            finally:
                loop.close()

        benchmark(cache_ops)


class TestTaskStoreBenchmark:
    """TaskStore record and query throughput."""

    def test_record_query_100(self, benchmark, task_store_db):
        """Benchmark: TaskStore record+query x100."""

        def record_and_query():
            for i in range(100):
                task_store_db.record_task(
                    job_id=f"job-{i}",
                    workflow_id="bench-workflow",
                    status="completed" if i % 5 != 0 else "failed",
                    agent_role="builder",
                    model="gpt-4",
                    input_tokens=1000 + i,
                    output_tokens=500 + i,
                    total_tokens=1500 + 2 * i,
                    cost_usd=0.05 + i * 0.001,
                    duration_ms=200 + i,
                )
            task_store_db.query_tasks(limit=20)
            task_store_db.get_agent_stats()
            task_store_db.get_summary()

        benchmark(record_and_query)
