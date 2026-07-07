"""Performance benchmarks for Animus Kernel Head operations.

Run: pytest tests/test_benchmarks.py --benchmark-only
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta

import pytest

pytest.importorskip("pytest_benchmark")

from animus_kernel.head.checkpoint import HeadCheckpoint, HeadCheckpointStore
from animus_kernel.head.context_manager import HeadContextManager
from animus_kernel.head.session_controller import (
    SessionController,
    SessionLifecycleEvent,
    SessionPolicy,
)


# ═══════════════════════════════════════════════════════════════════
# Session Lifecycle Benchmarks
# ═══════════════════════════════════════════════════════════════════


class TestSessionControllerBenchmark:
    """Session policy check and event logging throughput."""

    def test_policy_check_latency(self, benchmark):
        """Benchmark: Single policy check under normal load."""
        policy = SessionPolicy(
            wrapup_threshold=0.96,
            session_timer=timedelta(minutes=30),
            auto_restart=True,
        )
        ctrl = SessionController(policy=policy)

        # Seed with 50 events
        for i in range(50):
            ctrl.log_event(
                session_id="bench",
                event=SessionLifecycleEvent.RUNNING,
                utilization_percent=50.0,
                elapsed_seconds=i * 10,
                turns=1,
                message="",
            )

        def check():
            ctrl.should_finalize("bench", 50.0, 500.0, 50)

        benchmark(check)

    def test_event_log_throughput(self, benchmark):
        """Benchmark: Event logging rate (events/sec)."""
        policy = SessionPolicy(
            wrapup_threshold=0.96,
            session_timer=timedelta(minutes=30),
            auto_restart=True,
        )
        ctrl = SessionController(policy=policy)

        def log_100_events():
            for i in range(100):
                ctrl.log_event(
                    session_id="bench",
                    event=SessionLifecycleEvent.RUNNING,
                    utilization_percent=float(i),
                    elapsed_seconds=float(i * 10),
                    turns=1,
                    message="",
                )

        benchmark(log_100_events)


class TestCheckpointBenchmark:
    """Checkpoint save/load throughput."""

    def _make_checkpoint(self, session_id: str, messages: list[dict]) -> HeadCheckpoint:
        """Build a HeadCheckpoint from raw data."""
        return HeadCheckpoint(
            session_id=session_id,
            started_at=datetime.now(),
            last_active_at=datetime.now(),
            messages=messages,
            summary="Benchmark session summary",
            metadata={"model": "qwen2.5:14b", "turns": len(messages)},
        )

    def test_checkpoint_save_latency(self, benchmark, tmp_path):
        """Benchmark: Save a checkpoint with realistic context."""
        store = HeadCheckpointStore(db_path=tmp_path / "checkpoints" / "head.db")

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(100)
        ]
        checkpoint = self._make_checkpoint("bench-session", messages)

        def save():
            store.save(checkpoint)

        benchmark(save)

    def test_checkpoint_load_latency(self, benchmark, tmp_path):
        """Benchmark: Load a recently saved checkpoint."""
        store = HeadCheckpointStore(db_path=tmp_path / "checkpoints" / "head.db")

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(100)
        ]
        checkpoint = self._make_checkpoint("bench-session", messages)
        store.save(checkpoint)

        def load():
            store.load("bench-session")

        benchmark(load)

    def test_checkpoint_list_recent(self, benchmark, tmp_path):
        """Benchmark: List recent checkpoints with 50 entries."""
        store = HeadCheckpointStore(db_path=tmp_path / "checkpoints" / "head.db")

        for i in range(50):
            cp = self._make_checkpoint(
                f"session-{i}",
                [{"role": "user", "content": f"Msg {i}"}],
            )
            store.save(cp)

        def list_recent():
            store.list_recent(limit=10)

        benchmark(list_recent)


class TestContextManagerBenchmark:
    """HeadContextManager operation benchmarks."""

    def test_finalize_summary_latency(self, benchmark):
        """Benchmark: Generate graceful finalize summary from messages."""
        cm = HeadContextManager(model="mock")

        for i in range(50):
            cm.add_message({"role": "user", "content": f"Task {i}: implement feature X"})
            cm.add_message({"role": "assistant", "content": f"Completed task {i}"})

        def finalize():
            cm.graceful_finalize_summary()

        benchmark(finalize)
