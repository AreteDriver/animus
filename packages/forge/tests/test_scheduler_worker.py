"""Tests for RUN-03 worker lifecycle and termination."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from decimal import Decimal
from uuid import uuid4

import pytest

from animus_forge.missions.domain import Mission, MissionStatus, Task, TaskContext, TaskStatus
from animus_forge.missions.store import MissionLedger
from animus_forge.scheduler.containers import ContainerConfig, ContainerManager
from animus_forge.scheduler.cost_enforcer import CostEnforcer
from animus_forge.scheduler.lease import LeaseManager
from animus_forge.scheduler.metrics import SchedulerMetrics
from animus_forge.scheduler.mission_scheduler import MissionScheduler, SchedulerConfig
from animus_forge.scheduler.worker_pool import CitizenWorkerPool, PoolConfig
from animus_forge.scheduler.worker_process import WorkerProcess
from animus_forge.state.backends import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend():
    b = SQLiteBackend(":memory:")
    with b.transaction():
        b.execute("PRAGMA foreign_keys=ON")
    yield b
    b.close()


@pytest.fixture()
def ledger(backend):
    return MissionLedger(backend)


@pytest.fixture()
def lease_manager(backend):
    return LeaseManager(backend, default_ttl_seconds=60)


@pytest.fixture()
def cost_enforcer(backend):
    return CostEnforcer(backend)


@pytest.fixture()
def worker_pool(lease_manager):
    return CitizenWorkerPool(lease_manager, config=PoolConfig(max_workers=2))


@pytest.fixture()
def metrics(backend):
    return SchedulerMetrics(backend)


@pytest.fixture()
def sample_mission():
    return Mission(
        repository="AreteDriver/animus",
        objective="Fix off-by-one pagination bug",
        status=MissionStatus.PROPOSED,
        max_cost_usd=Decimal("5.00"),
    )


@pytest.fixture()
def sample_task(sample_mission):
    return Task(
        mission_id=sample_mission.mission_id,
        citizen_role="planner",
        description="Plan the fix",
        status=TaskStatus.READY,
    )


# ---------------------------------------------------------------------------
# WorkerProcess tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_worker_process_runs_citizen():
    ctx = TaskContext(
        mission_objective="o",
        task_description="d",
        repository="r",
    )
    worker = WorkerProcess(
        task_id=str(uuid4()),
        mission_id=str(uuid4()),
        citizen_role="planner",
        description="Plan something",
        context_json=ctx.model_dump(mode="json"),
        timeout_seconds=30.0,
    )
    assert await worker.start()
    result = await worker.wait()
    assert result.ok
    assert result.data is not None
    assert result.data["status"] == "completed"
    assert result.returncode == 0


@pytest.mark.asyncio()
async def test_worker_process_catches_citizen_exception():
    """An unknown role causes the worker entry point to return a failure dict."""
    ctx = TaskContext(
        mission_objective="o",
        task_description="d",
        repository="r",
    )
    worker = WorkerProcess(
        task_id=str(uuid4()),
        mission_id=str(uuid4()),
        citizen_role="nonexistent_role",
        description="d",
        context_json=ctx.model_dump(mode="json"),
        timeout_seconds=30.0,
    )
    assert await worker.start()
    result = await worker.wait()
    assert result.data is not None
    assert result.data["status"] == "failed"
    assert "nonexistent_role" in result.data.get("summary", "")


@pytest.mark.asyncio()
async def test_worker_process_terminates_hung_worker():
    """A worker that ignores SIGTERM must be SIGKILLed after the grace period."""
    slow_cmd = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
    worker = WorkerProcess(
        task_id=str(uuid4()),
        mission_id=str(uuid4()),
        citizen_role="planner",
        description="d",
        context_json={},
        timeout_seconds=300.0,
        grace_period_seconds=0.5,
        command=slow_cmd,
    )
    assert await worker.start()
    pid = worker.pid
    assert pid is not None

    start = time.time()
    await worker.terminate()
    elapsed = time.time() - start

    # Should be killed well before the 300s timeout.
    assert elapsed < 5.0
    # Process should no longer exist.
    assert not _pid_exists(pid)


@pytest.mark.asyncio()
async def test_worker_process_timeout_enforced():
    """WorkerProcess.wait() kills the worker after timeout_seconds."""
    slow_cmd = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
    worker = WorkerProcess(
        task_id=str(uuid4()),
        mission_id=str(uuid4()),
        citizen_role="planner",
        description="d",
        context_json={},
        timeout_seconds=1.0,
        grace_period_seconds=0.5,
        command=slow_cmd,
    )
    assert await worker.start()
    pid = worker.pid
    result = await worker.wait()
    assert result.timed_out
    assert result.killed
    assert not _pid_exists(pid)


# ---------------------------------------------------------------------------
# Worker pool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_pool_submit_and_result(lease_manager):
    pool = CitizenWorkerPool(lease_manager, config=PoolConfig(max_workers=2))
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        task_id = str(uuid4())
        lease_id = await pool.submit(task_id, "planner", ctx, mission_id=str(uuid4()), ttl_seconds=30)
        assert lease_id is not None
        assert pool.active_count() == 1

        returned_task_id, result = await asyncio.wait_for(
            (await pool.results()).get(),
            timeout=10.0,
        )
        assert returned_task_id == task_id
        assert result["status"] == "completed"
        assert pool.active_count() == 0
        assert pool.free_count() == 2
    finally:
        await pool.stop()


@pytest.mark.asyncio()
async def test_pool_timeout_kills_worker(lease_manager):
    """A task that exceeds its ttl is terminated and a timeout result is enqueued."""
    slow_cmd = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
    pool = CitizenWorkerPool(
        lease_manager,
        config=PoolConfig(max_workers=1, worker_timeout_seconds=1, worker_command=slow_cmd),
    )
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        task_id = str(uuid4())
        lease_id = await pool.submit(task_id, "planner", ctx, mission_id=str(uuid4()), ttl_seconds=1)
        assert lease_id is not None

        returned_task_id, result = await asyncio.wait_for(
            (await pool.results()).get(),
            timeout=10.0,
        )
        assert returned_task_id == task_id
        assert result.get("_timed_out") is True
        assert result.get("_killed") is True
        assert pool.active_count() == 0
    finally:
        await pool.stop()


@pytest.mark.asyncio()
async def test_pool_kill_slot_terminates_process(lease_manager):
    slow_cmd = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
    pool = CitizenWorkerPool(
        lease_manager,
        config=PoolConfig(max_workers=1, worker_command=slow_cmd),
    )
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        task_id = str(uuid4())
        lease_id = await pool.submit(task_id, "planner", ctx, mission_id=str(uuid4()), ttl_seconds=300)
        assert lease_id is not None
        await asyncio.sleep(0.3)  # let worker start

        slot = pool._slots["0"]
        pid = slot.pid
        assert pid is not None and _pid_exists(pid)

        assert await pool.kill_slot("0") is True
        await asyncio.sleep(0.5)

        assert pool.active_count() == 0
        assert not _pid_exists(pid)
    finally:
        await pool.stop()


@pytest.mark.asyncio()
async def test_pool_stop_cancels_active_workers(lease_manager):
    slow_cmd = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
    pool = CitizenWorkerPool(
        lease_manager,
        config=PoolConfig(max_workers=1, shutdown_behavior="cancel", worker_command=slow_cmd),
    )
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        await pool.submit(str(uuid4()), "planner", ctx, mission_id=str(uuid4()), ttl_seconds=300)
        await asyncio.sleep(0.3)
        slot = pool._slots["0"]
        pid = slot.pid
        assert pid is not None and _pid_exists(pid)

        await pool.stop()

        assert pool.active_count() == 0
        assert not _pid_exists(pid)
    finally:
        if pool.is_running:
            await pool.stop()


@pytest.mark.asyncio()
async def test_pool_stop_drain_waits_for_completion(lease_manager):
    pool = CitizenWorkerPool(
        lease_manager,
        config=PoolConfig(max_workers=1, shutdown_behavior="drain", drain_timeout_seconds=5.0),
    )
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        await pool.submit(str(uuid4()), "planner", ctx, mission_id=str(uuid4()), ttl_seconds=30)
        await asyncio.sleep(0.3)

        await pool.stop()

        # Task should have completed naturally before drain timeout.
        assert pool.active_count() == 0
    finally:
        if pool.is_running:
            await pool.stop()


@pytest.mark.asyncio()
async def test_pool_no_double_result_on_timeout_and_completion(lease_manager):
    """Forcing a kill on a running slot does not also enqueue the supervisor's late result."""
    slow_cmd = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
    pool = CitizenWorkerPool(
        lease_manager,
        config=PoolConfig(max_workers=1, worker_timeout_seconds=10, worker_command=slow_cmd),
    )
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        await pool.submit(str(uuid4()), "planner", ctx, mission_id=str(uuid4()), ttl_seconds=300)
        # Manually kill the slot; the supervisor should not also enqueue a result.
        await asyncio.sleep(0.3)
        assert await pool.kill_slot("0") is True

        # Wait a bit and make sure no more than one result was queued.
        await asyncio.sleep(0.5)
        queue = await pool.results()
        assert queue.qsize() <= 1
    finally:
        await pool.stop()


@pytest.mark.asyncio()
async def test_pool_isolation_status_reported(lease_manager):
    pool = CitizenWorkerPool(lease_manager, config=PoolConfig(max_workers=2, isolation_mode="process"))
    await pool.start()
    try:
        status = pool.isolation_status()
        assert status["mode"] == "process"
        assert status["max_workers"] == 2
        assert status["runtime_available"] is False
        assert len(status["slots"]) == 2
    finally:
        await pool.stop()


# ---------------------------------------------------------------------------
# Container mode tests
# ---------------------------------------------------------------------------


class FakeContainerProcess:
    """Asyncio-compatible process stand-in for container tests."""

    def __init__(self, sleep_seconds: float, result_payload: dict):
        self._sleep = sleep_seconds
        self._result = json.dumps(result_payload).encode()
        self._killed = asyncio.Event()
        self.returncode: int | None = None

    async def communicate(self, input=None):
        try:
            await asyncio.wait_for(self._killed.wait(), timeout=self._sleep)
            self.returncode = -9
            return b"", b"killed"
        except TimeoutError:
            self.returncode = 0
            return self._result, b""

    def kill(self):
        self._killed.set()


class FakeContainerTask:
    def __init__(self, container_id: str, process: FakeContainerProcess):
        self.container_id = container_id
        self.process = process


class FakeContainerManager(ContainerManager):
    def __init__(self, sleep_seconds: float = 0.1):
        super().__init__(ContainerConfig(runtime="fake"))
        self.sleep_seconds = sleep_seconds
        self.calls: list[dict] = []
        self.killed: set[str] = set()
        self._counter = 0
        self._tasks: dict[str, FakeContainerTask] = {}

    def is_available(self):
        return True

    async def run_task_async(self, **kwargs):
        self._counter += 1
        self.calls.append(kwargs)
        result_payload = {
            "status": "completed",
            "summary": "fake container",
            "changed_files": [],
            "evidence": [],
            "risks": [],
            "confidence": 0.9,
        }
        task = FakeContainerTask(f"fake-cid-{self._counter}", FakeContainerProcess(self.sleep_seconds, result_payload))
        self._tasks[task.container_id] = task
        return task

    async def kill_container(self, container_id: str) -> bool:
        self.killed.add(container_id)
        task = self._tasks.get(container_id)
        if task is not None:
            task.process.kill()
        return True


@pytest.mark.asyncio()
async def test_pool_container_mode_dispatches_and_reports_cid(lease_manager):
    fake = FakeContainerManager(sleep_seconds=0.2)
    pool = CitizenWorkerPool(
        lease_manager,
        config=PoolConfig(max_workers=1, isolation_mode="container"),
        container_manager=fake,
    )
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        task_id = str(uuid4())
        lease_id = await pool.submit(task_id, "planner", ctx, mission_id=str(uuid4()), ttl_seconds=30)
        assert lease_id is not None

        returned_task_id, result = await asyncio.wait_for(
            (await pool.results()).get(),
            timeout=10.0,
        )
        assert returned_task_id == task_id
        meta = result.get("_scheduler_meta", {})
        assert meta.get("container_id", "").startswith("fake-cid-")
        assert result["status"] == "completed"
    finally:
        await pool.stop()


@pytest.mark.asyncio()
async def test_pool_kill_slot_terminates_container(lease_manager):
    fake = FakeContainerManager(sleep_seconds=10.0)
    pool = CitizenWorkerPool(
        lease_manager,
        config=PoolConfig(max_workers=1, isolation_mode="container"),
        container_manager=fake,
    )
    await pool.start()
    try:
        ctx = TaskContext(
            mission_objective="o",
            task_description="d",
            repository="r",
        )
        await pool.submit(str(uuid4()), "planner", ctx, mission_id=str(uuid4()), ttl_seconds=300)
        await asyncio.sleep(0.2)

        cid = pool._slots["0"].container_id
        assert cid is not None

        assert await pool.kill_slot("0") is True
        await asyncio.sleep(0.3)

        assert pool.active_count() == 0
        assert cid in fake.killed
    finally:
        await pool.stop()


# ---------------------------------------------------------------------------
# Integration: MissionScheduler status includes isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_scheduler_status_includes_isolation(
    ledger, lease_manager, worker_pool, cost_enforcer, metrics
):
    scheduler = MissionScheduler(
        ledger=ledger,
        lease_manager=lease_manager,
        worker_pool=worker_pool,
        cost_enforcer=cost_enforcer,
        metrics=metrics,
        config=SchedulerConfig(poll_interval_seconds=1.0),
    )
    await scheduler.start()
    try:
        status = scheduler.status()
        assert "isolation" in status
        assert status["isolation"]["mode"] == "process"
        assert status["isolation"]["max_workers"] == 2
    finally:
        await scheduler.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pid_exists(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
