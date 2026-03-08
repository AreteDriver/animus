"""Tests for AgentMessageBus and ProcessRegistry."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

from animus_forge.agents.agent_config import AgentConfig
from animus_forge.agents.message_bus import (
    AgentMessage,
    AgentMessageBus,
    MessagePriority,
)
from animus_forge.agents.process_registry import (
    ProcessInfo,
    ProcessRegistry,
    ProcessState,
    ProcessType,
)
from animus_forge.agents.subagent_manager import AgentRun, RunStatus

# ---------------------------------------------------------------------------
# AgentMessage tests
# ---------------------------------------------------------------------------


class TestAgentMessage:
    def test_auto_timestamp(self):
        msg = AgentMessage(id="m1", topic="test", sender="a", payload="hi")
        assert msg.timestamp > 0

    def test_not_expired_no_ttl(self):
        msg = AgentMessage(id="m1", topic="test", sender="a", payload="hi")
        assert not msg.is_expired

    def test_expired_with_ttl(self):
        msg = AgentMessage(
            id="m1",
            topic="test",
            sender="a",
            payload="hi",
            timestamp=time.time() - 10,
            ttl_seconds=5,
        )
        assert msg.is_expired

    def test_not_expired_within_ttl(self):
        msg = AgentMessage(
            id="m1",
            topic="test",
            sender="a",
            payload="hi",
            ttl_seconds=300,
        )
        assert not msg.is_expired

    def test_to_dict(self):
        msg = AgentMessage(
            id="m1",
            topic="build.progress",
            sender="builder",
            payload={"progress": 50},
            priority=MessagePriority.HIGH,
        )
        d = msg.to_dict()
        assert d["id"] == "m1"
        assert d["topic"] == "build.progress"
        assert d["sender"] == "builder"
        assert d["priority"] == "high"
        assert d["payload"] == {"progress": 50}


# ---------------------------------------------------------------------------
# AgentMessageBus — publish/subscribe
# ---------------------------------------------------------------------------


class TestMessageBusPubSub:
    def test_publish_returns_message(self):
        bus = AgentMessageBus()
        msg = bus.publish("test.topic", sender="agent1", payload="hello")
        assert msg.id.startswith("msg-")
        assert msg.topic == "test.topic"
        assert msg.sender == "agent1"

    def test_subscriber_receives_message(self):
        bus = AgentMessageBus()
        received = []
        bus.subscribe("test.topic", lambda m: received.append(m))
        bus.publish("test.topic", sender="a", payload="data")
        assert len(received) == 1
        assert received[0].payload == "data"

    def test_wildcard_subscriber(self):
        bus = AgentMessageBus()
        received = []
        bus.subscribe("build.*", lambda m: received.append(m))
        bus.publish("build.progress", sender="a", payload="50%")
        bus.publish("build.complete", sender="a", payload="done")
        bus.publish("review.findings", sender="b", payload="issues")
        assert len(received) == 2

    def test_global_wildcard(self):
        bus = AgentMessageBus()
        received = []
        bus.subscribe("*", lambda m: received.append(m))
        bus.publish("anything", sender="a", payload="x")
        bus.publish("else.here", sender="b", payload="y")
        assert len(received) == 2

    def test_exact_match_only(self):
        bus = AgentMessageBus()
        received = []
        bus.subscribe("build.progress", lambda m: received.append(m))
        bus.publish("build.complete", sender="a", payload="done")
        assert len(received) == 0

    def test_multiple_subscribers(self):
        bus = AgentMessageBus()
        r1, r2 = [], []
        bus.subscribe("topic", lambda m: r1.append(m))
        bus.subscribe("topic", lambda m: r2.append(m))
        bus.publish("topic", sender="a", payload="x")
        assert len(r1) == 1
        assert len(r2) == 1

    def test_subscriber_error_swallowed(self):
        bus = AgentMessageBus()
        good_received = []

        def bad_callback(msg):
            raise RuntimeError("subscriber crash")

        bus.subscribe("topic", bad_callback)
        bus.subscribe("topic", lambda m: good_received.append(m))
        bus.publish("topic", sender="a", payload="x")
        # Good subscriber still got the message despite bad one crashing
        assert len(good_received) == 1

    def test_unsubscribe(self):
        bus = AgentMessageBus()
        received = []

        def callback(m):
            received.append(m)

        bus.subscribe("topic", callback)
        assert bus.unsubscribe("topic", callback) is True
        bus.publish("topic", sender="a", payload="x")
        assert len(received) == 0

    def test_unsubscribe_not_found(self):
        bus = AgentMessageBus()

        def noop(m):
            pass

        assert bus.unsubscribe("topic", noop) is False

    def test_unsubscribe_wrong_pattern(self):
        bus = AgentMessageBus()

        def callback(m):
            pass

        bus.subscribe("topic.a", callback)
        assert bus.unsubscribe("topic.b", callback) is False


# ---------------------------------------------------------------------------
# AgentMessageBus — history and queries
# ---------------------------------------------------------------------------


class TestMessageBusHistory:
    def test_get_messages(self):
        bus = AgentMessageBus()
        bus.publish("t", sender="a", payload="1")
        bus.publish("t", sender="b", payload="2")
        msgs = bus.get_messages("t")
        assert len(msgs) == 2

    def test_get_messages_limit(self):
        bus = AgentMessageBus()
        for i in range(10):
            bus.publish("t", sender="a", payload=str(i))
        msgs = bus.get_messages("t", limit=3)
        assert len(msgs) == 3

    def test_get_messages_since(self):
        bus = AgentMessageBus()
        bus.publish("t", sender="a", payload="old")
        mark = time.time()
        time.sleep(0.01)
        bus.publish("t", sender="a", payload="new")
        msgs = bus.get_messages("t", since=mark)
        assert len(msgs) == 1
        assert msgs[0].payload == "new"

    def test_get_messages_by_sender(self):
        bus = AgentMessageBus()
        bus.publish("t", sender="builder", payload="1")
        bus.publish("t", sender="reviewer", payload="2")
        msgs = bus.get_messages("t", sender="builder")
        assert len(msgs) == 1

    def test_expired_messages_filtered(self):
        bus = AgentMessageBus()
        bus.publish("t", sender="a", payload="old", ttl_seconds=0.01)
        time.sleep(0.02)
        bus.publish("t", sender="a", payload="fresh")
        msgs = bus.get_messages("t")
        assert len(msgs) == 1
        assert msgs[0].payload == "fresh"

    def test_get_messages_empty_topic(self):
        bus = AgentMessageBus()
        msgs = bus.get_messages("nonexistent")
        assert msgs == []

    def test_history_capped(self):
        bus = AgentMessageBus(max_history=5)
        for i in range(10):
            bus.publish("t", sender="a", payload=str(i))
        msgs = bus.get_messages("t")
        assert len(msgs) == 5

    def test_get_thread(self):
        bus = AgentMessageBus()
        original = bus.publish("t", sender="a", payload="question")
        bus.publish("t", sender="b", payload="answer", reply_to=original.id)
        bus.publish("t", sender="c", payload="unrelated")
        thread = bus.get_thread(original.id)
        assert len(thread) == 2

    def test_get_topics(self):
        bus = AgentMessageBus()
        bus.publish("topic.a", sender="x", payload="1")
        bus.publish("topic.b", sender="x", payload="2")
        topics = bus.get_topics()
        assert "topic.a" in topics
        assert "topic.b" in topics

    def test_clear_topic(self):
        bus = AgentMessageBus()
        bus.publish("t", sender="a", payload="1")
        bus.publish("t", sender="a", payload="2")
        removed = bus.clear_topic("t")
        assert removed == 2
        assert bus.get_messages("t") == []

    def test_clear_all(self):
        bus = AgentMessageBus()
        bus.publish("a", sender="x", payload="1")
        bus.publish("b", sender="x", payload="2")
        bus.subscribe("a", lambda m: None)
        total = bus.clear_all()
        assert total == 2
        assert bus.message_count == 0
        assert bus.subscriber_count == 0


# ---------------------------------------------------------------------------
# AgentMessageBus — properties
# ---------------------------------------------------------------------------


class TestMessageBusProperties:
    def test_message_count(self):
        bus = AgentMessageBus()
        assert bus.message_count == 0
        bus.publish("t", sender="a", payload="x")
        assert bus.message_count == 1

    def test_topic_count(self):
        bus = AgentMessageBus()
        assert bus.topic_count == 0
        bus.publish("a", sender="x", payload="1")
        bus.publish("b", sender="x", payload="2")
        assert bus.topic_count == 2

    def test_subscriber_count(self):
        bus = AgentMessageBus()
        assert bus.subscriber_count == 0
        bus.subscribe("a", lambda m: None)
        bus.subscribe("b", lambda m: None)
        assert bus.subscriber_count == 2


# ---------------------------------------------------------------------------
# Topic matching
# ---------------------------------------------------------------------------


class TestTopicMatching:
    def test_exact_match(self):
        assert AgentMessageBus._topic_matches("build", "build")

    def test_no_match(self):
        assert not AgentMessageBus._topic_matches("build", "review")

    def test_wildcard_suffix(self):
        assert AgentMessageBus._topic_matches("build.*", "build.progress")
        assert AgentMessageBus._topic_matches("build.*", "build.complete")
        assert not AgentMessageBus._topic_matches("build.*", "review.findings")

    def test_global_wildcard(self):
        assert AgentMessageBus._topic_matches("*", "anything")

    def test_partial_no_match(self):
        assert not AgentMessageBus._topic_matches("build", "build.progress")


# ---------------------------------------------------------------------------
# ProcessInfo tests
# ---------------------------------------------------------------------------


class TestProcessInfo:
    def test_to_dict(self):
        info = ProcessInfo(
            id="run-abc",
            type=ProcessType.AGENT,
            state=ProcessState.RUNNING,
            name="builder: build it",
            started_at=1000.0,
            duration_ms=500,
        )
        d = info.to_dict()
        assert d["id"] == "run-abc"
        assert d["type"] == "agent"
        assert d["state"] == "running"


# ---------------------------------------------------------------------------
# ProcessRegistry — agent collection
# ---------------------------------------------------------------------------


class TestProcessRegistryAgents:
    def _make_agent_run(self, run_id="run-1", agent="builder", status=RunStatus.RUNNING):
        return AgentRun(
            run_id=run_id,
            agent=agent,
            task="build the thing",
            config=AgentConfig(role=agent),
            status=status,
            started_at=time.time(),
        )

    def test_list_agents(self):
        mgr = MagicMock()
        mgr.list_runs.return_value = [self._make_agent_run()]
        reg = ProcessRegistry(subagent_manager=mgr)
        procs = reg.list_all()
        assert len(procs) == 1
        assert procs[0].type == ProcessType.AGENT

    def test_get_agent(self):
        run = self._make_agent_run()
        mgr = MagicMock()
        mgr.get_run.return_value = run
        reg = ProcessRegistry(subagent_manager=mgr)
        proc = reg.get("run-1")
        assert proc is not None
        assert proc.id == "run-1"

    def test_get_not_found(self):
        mgr = MagicMock()
        mgr.get_run.return_value = None
        reg = ProcessRegistry(subagent_manager=mgr)
        assert reg.get("nonexistent") is None

    def test_filter_by_state(self):
        mgr = MagicMock()
        mgr.list_runs.return_value = [
            self._make_agent_run("r1", status=RunStatus.RUNNING),
            self._make_agent_run("r2", status=RunStatus.COMPLETED),
        ]
        reg = ProcessRegistry(subagent_manager=mgr)
        running = reg.list_all(state=ProcessState.RUNNING)
        assert len(running) == 1

    def test_filter_by_type(self):
        mgr = MagicMock()
        mgr.list_runs.return_value = [self._make_agent_run()]
        reg = ProcessRegistry(subagent_manager=mgr)
        jobs = reg.list_all(process_type=ProcessType.JOB)
        assert len(jobs) == 0
        agents = reg.list_all(process_type=ProcessType.AGENT)
        assert len(agents) == 1

    async def test_cancel_agent(self):
        mgr = MagicMock()
        mgr.cancel = AsyncMock(return_value=True)
        reg = ProcessRegistry(subagent_manager=mgr)
        result = await reg.cancel("run-1")
        assert result is True

    async def test_cancel_not_found(self):
        mgr = MagicMock()
        mgr.cancel = AsyncMock(return_value=False)
        reg = ProcessRegistry(subagent_manager=mgr)
        result = await reg.cancel("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# ProcessRegistry — job collection
# ---------------------------------------------------------------------------


class TestProcessRegistryJobs:
    def test_list_jobs(self):
        job = MagicMock()
        job.id = "job-1"
        job.status = "running"
        job.workflow_id = "my-workflow"
        job.started_at = time.time()
        job.duration_ms = 100
        job.progress = 50

        job_mgr = MagicMock()
        job_mgr.list_jobs.return_value = [job]

        reg = ProcessRegistry(job_manager=job_mgr)
        procs = reg.list_all()
        assert len(procs) == 1
        assert procs[0].type == ProcessType.JOB

    def test_get_job(self):
        job = MagicMock()
        job.id = "job-1"
        job.status = "completed"
        job.workflow_id = "wf"
        job.started_at = 1000.0
        job.duration_ms = 200
        job.progress = 100

        job_mgr = MagicMock()
        job_mgr.get_job.return_value = job

        reg = ProcessRegistry(job_manager=job_mgr)
        proc = reg.get("job-1")
        assert proc is not None
        assert proc.state == ProcessState.COMPLETED

    def test_job_manager_error_handled(self):
        job_mgr = MagicMock()
        job_mgr.list_jobs.side_effect = RuntimeError("db error")
        reg = ProcessRegistry(job_manager=job_mgr)
        procs = reg.list_all()
        assert procs == []

    async def test_cancel_job(self):
        job_mgr = MagicMock()
        job_mgr.cancel_job.return_value = True
        reg = ProcessRegistry(job_manager=job_mgr)
        result = await reg.cancel("job-1")
        assert result is True


# ---------------------------------------------------------------------------
# ProcessRegistry — schedule collection
# ---------------------------------------------------------------------------


class TestProcessRegistrySchedules:
    def test_list_schedules(self):
        schedule = MagicMock()
        schedule.id = "sched-1"
        schedule.status = "active"
        schedule.name = "daily-build"
        schedule.cron = "0 0 * * *"
        schedule.next_run = "2026-03-09T00:00:00"

        sched_mgr = MagicMock()
        sched_mgr.list_schedules.return_value = [schedule]

        reg = ProcessRegistry(schedule_manager=sched_mgr)
        procs = reg.list_all()
        assert len(procs) == 1
        assert procs[0].type == ProcessType.SCHEDULE

    def test_get_schedule(self):
        schedule = MagicMock()
        schedule.id = "sched-1"
        schedule.status = "paused"
        schedule.name = "nightly"
        schedule.cron = "0 2 * * *"
        schedule.next_run = ""

        sched_mgr = MagicMock()
        sched_mgr.get_schedule.return_value = schedule

        reg = ProcessRegistry(schedule_manager=sched_mgr)
        proc = reg.get("sched-1")
        assert proc is not None
        assert proc.state == ProcessState.PAUSED

    async def test_cancel_schedule(self):
        sched_mgr = MagicMock()
        sched_mgr.pause_schedule.return_value = True
        reg = ProcessRegistry(schedule_manager=sched_mgr)
        result = await reg.cancel("sched-1")
        assert result is True


# ---------------------------------------------------------------------------
# ProcessRegistry — summary and properties
# ---------------------------------------------------------------------------


class TestProcessRegistrySummary:
    def test_summary(self):
        run = AgentRun(
            run_id="run-1",
            agent="builder",
            task="build",
            config=AgentConfig(role="builder"),
            status=RunStatus.RUNNING,
            started_at=time.time(),
        )
        mgr = MagicMock()
        mgr.list_runs.return_value = [run]
        reg = ProcessRegistry(subagent_manager=mgr)
        s = reg.summary()
        assert s["total"] == 1
        assert s["by_type"]["agent"] == 1
        assert s["by_state"]["running"] == 1

    def test_active_count(self):
        run = AgentRun(
            run_id="run-1",
            agent="builder",
            task="build",
            config=AgentConfig(role="builder"),
            status=RunStatus.RUNNING,
            started_at=time.time(),
        )
        mgr = MagicMock()
        mgr.list_runs.return_value = [run]
        reg = ProcessRegistry(subagent_manager=mgr)
        assert reg.active_count == 1

    def test_total_count(self):
        mgr = MagicMock()
        mgr.list_runs.return_value = []
        reg = ProcessRegistry(subagent_manager=mgr)
        assert reg.total_count == 0

    def test_empty_registry(self):
        reg = ProcessRegistry()
        assert reg.list_all() == []
        assert reg.get("anything") is None
        assert reg.active_count == 0
        s = reg.summary()
        assert s["total"] == 0


# ---------------------------------------------------------------------------
# ProcessRegistry — error handling edge cases
# ---------------------------------------------------------------------------


class TestProcessRegistryEdgeCases:
    def test_get_job_exception_handled(self):
        """get_job exception falls through to next manager."""
        job_mgr = MagicMock()
        job_mgr.get_job.side_effect = RuntimeError("db error")
        reg = ProcessRegistry(job_manager=job_mgr)
        assert reg.get("job-1") is None

    def test_get_schedule_exception_handled(self):
        """get_schedule exception falls through."""
        sched_mgr = MagicMock()
        sched_mgr.get_schedule.side_effect = RuntimeError("db error")
        reg = ProcessRegistry(schedule_manager=sched_mgr)
        assert reg.get("sched-1") is None

    async def test_cancel_job_exception_handled(self):
        """cancel_job exception falls through."""
        mgr = MagicMock()
        mgr.cancel = AsyncMock(return_value=False)
        job_mgr = MagicMock()
        job_mgr.cancel_job.side_effect = RuntimeError("db error")
        reg = ProcessRegistry(subagent_manager=mgr, job_manager=job_mgr)
        result = await reg.cancel("job-1")
        assert result is False

    async def test_cancel_schedule_exception_handled(self):
        """pause_schedule exception falls through."""
        mgr = MagicMock()
        mgr.cancel = AsyncMock(return_value=False)
        sched_mgr = MagicMock()
        sched_mgr.pause_schedule.side_effect = RuntimeError("error")
        reg = ProcessRegistry(subagent_manager=mgr, schedule_manager=sched_mgr)
        result = await reg.cancel("sched-1")
        assert result is False

    def test_schedule_collect_exception_handled(self):
        """list_schedules exception returns empty list."""
        sched_mgr = MagicMock()
        sched_mgr.list_schedules.side_effect = RuntimeError("error")
        reg = ProcessRegistry(schedule_manager=sched_mgr)
        procs = reg.list_all()
        assert procs == []

    def test_job_status_with_value_attr(self):
        """Job status that has .value attribute (enum-like)."""
        job = MagicMock()
        job.id = "job-1"
        status = MagicMock()
        status.value = "completed"
        job.status = status
        job.workflow_id = "wf"
        started = MagicMock()
        started.timestamp.return_value = 1000.0
        job.started_at = started
        job.duration_ms = 100
        job.progress = 100

        job_mgr = MagicMock()
        job_mgr.list_jobs.return_value = [job]
        reg = ProcessRegistry(job_manager=job_mgr)
        procs = reg.list_all()
        assert len(procs) == 1
        assert procs[0].state == ProcessState.COMPLETED

    def test_schedule_status_with_value_attr(self):
        """Schedule status that has .value attribute (enum-like)."""
        schedule = MagicMock()
        schedule.id = "sched-1"
        status = MagicMock()
        status.value = "active"
        schedule.status = status
        schedule.name = "nightly"
        schedule.cron = "0 0 * * *"
        schedule.next_run = ""

        sched_mgr = MagicMock()
        sched_mgr.list_schedules.return_value = [schedule]
        reg = ProcessRegistry(schedule_manager=sched_mgr)
        procs = reg.list_all()
        assert len(procs) == 1
        assert procs[0].state == ProcessState.RUNNING
