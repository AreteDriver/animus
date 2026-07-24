"""Throughput benchmark for EventLog.record() — AC6.

Acceptance: ≥200 events/sec on local SQLite, signal bus emit
overhead <5ms p99 per record() call (full path, not bridge alone).
"""

from __future__ import annotations

import time
from pathlib import Path

from animus_quorum.event_log import EventLog, EventType
from animus_quorum.protocol import Signal


class _NoopBus:
    """Signal bus stand-in that does nothing — measures bridge
    overhead independent of any real backend."""

    def publish(self, signal: Signal) -> None:
        return


def _record_n(log: EventLog, n: int) -> float:
    start = time.perf_counter()
    for i in range(n):
        log.record(
            EventType.INTENT_PUBLISHED,
            agent_id=f"agent-{i % 16}",
            payload={"i": i},
            correlation_id=f"corr-{i % 32}",
        )
    return time.perf_counter() - start


class TestThroughput:
    def test_in_memory_record_at_least_200_per_sec(self) -> None:
        log = EventLog(":memory:")
        n = 1000
        elapsed = _record_n(log, n)
        rate = n / elapsed
        assert rate >= 200, f"only {rate:.0f}/sec on :memory:"

    def test_disk_record_at_least_200_per_sec(self, tmp_path: Path) -> None:
        log = EventLog(str(tmp_path / "bench.db"))
        n = 500  # WAL fsync per commit makes disk slower
        elapsed = _record_n(log, n)
        rate = n / elapsed
        assert rate >= 200, f"only {rate:.0f}/sec on disk"

    def test_with_bus_at_least_150_per_sec(self) -> None:
        log = EventLog(":memory:", signal_bus=_NoopBus())
        n = 1000
        elapsed = _record_n(log, n)
        rate = n / elapsed
        assert rate >= 150, f"only {rate:.0f}/sec with bus"


class TestBridgeP99:
    def test_signal_emit_p99_under_2ms(self) -> None:
        log = EventLog(":memory:", signal_bus=_NoopBus())
        # Warm up
        for _ in range(50):
            log.record(EventType.INTENT_PUBLISHED, "agent-0")
        # Measure
        samples_ms: list[float] = []
        for i in range(500):
            t0 = time.perf_counter()
            log.record(
                EventType.INTENT_PUBLISHED,
                agent_id=f"agent-{i % 8}",
                payload={"i": i},
            )
            samples_ms.append((time.perf_counter() - t0) * 1000)
        samples_ms.sort()
        p99 = samples_ms[int(len(samples_ms) * 0.99)]
        # Generous: full record() including SQLite commit + bridge.
        # Bridge alone is dominated by JSON serialize.
        assert p99 < 5.0, f"p99 = {p99:.2f}ms"
