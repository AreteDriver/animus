"""Tests for BudgetManager allocate()/release() reservation (roadmap A2).

The parallel executor runs sub-steps on a thread pool. Before A2, each
pre-checked with a bare can_allocate() against the same un-incremented total,
so N concurrent steps could all pass and then collectively overspend. allocate()
now atomically reserves, and can_allocate counts outstanding reservations.
"""

from __future__ import annotations

import threading

from animus_forge.budget import BudgetConfig, BudgetManager


class TestReservationAccounting:
    def test_allocate_reserves_and_can_allocate_sees_it(self):
        mgr = BudgetManager(BudgetConfig(total_budget=1000, reserve_tokens=0))
        assert mgr.allocate(600) is True
        assert mgr.pending == 600
        # 600 reserved → only 400 of headroom remains.
        assert mgr.can_allocate(400) is True
        assert mgr.can_allocate(401) is False

    def test_collective_overspend_is_prevented(self):
        mgr = BudgetManager(BudgetConfig(total_budget=1000, reserve_tokens=0))
        assert mgr.allocate(600) is True
        # Second 600 would push reservations to 1200 > 1000 → refused.
        assert mgr.allocate(600) is False
        # Releasing the first frees room for it.
        mgr.release(600)
        assert mgr.pending == 0
        assert mgr.allocate(600) is True

    def test_record_then_release_converts_reservation_to_usage(self):
        mgr = BudgetManager(BudgetConfig(total_budget=1000, reserve_tokens=0))
        mgr.allocate(500, agent_id="s1")
        # Actual came in under estimate.
        mgr.record_usage("s1", 300)
        mgr.release(500, agent_id="s1")
        assert mgr.pending == 0
        assert mgr.used == 300
        # Room left = 1000 - 300 used - 0 pending = 700.
        assert mgr.can_allocate(700) is True
        assert mgr.can_allocate(701) is False

    def test_release_is_clamped_at_zero(self):
        mgr = BudgetManager(BudgetConfig(total_budget=1000, reserve_tokens=0))
        mgr.allocate(100, agent_id="s1")
        mgr.release(999, agent_id="s1")  # over-release
        assert mgr.pending == 0

    def test_per_agent_limit_counts_reservations(self):
        mgr = BudgetManager(
            BudgetConfig(total_budget=10_000, per_agent_limit=500, reserve_tokens=0)
        )
        assert mgr.allocate(400, agent_id="a") is True
        # a already has 400 reserved; another 200 would breach the 500 cap.
        assert mgr.allocate(200, agent_id="a") is False
        assert mgr.allocate(100, agent_id="a") is True


class TestReservationThreadSafety:
    def test_concurrent_allocate_never_overspends(self):
        # 20 threads each try to reserve 200 against a 2000 budget: at most 10
        # can fit. Without atomic check-and-reserve, more would slip through.
        mgr = BudgetManager(BudgetConfig(total_budget=2000, reserve_tokens=0))
        results: list[bool] = []
        results_lock = threading.Lock()
        start = threading.Barrier(20)

        def worker():
            start.wait()
            ok = mgr.allocate(200)
            with results_lock:
                results.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        granted = sum(results)
        # Exactly the budget's worth was granted, and never more.
        assert granted == 10
        assert granted * 200 <= 2000
        assert mgr.pending == granted * 200
