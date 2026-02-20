"""Tests for ABTestManager."""

import uuid
from datetime import UTC, datetime

from animus_forge.skills.evolver.ab_test import ABTestManager
from animus_forge.skills.evolver.metrics import SkillMetricsAggregator
from animus_forge.state.backends import SQLiteBackend


def _make_backend():
    backend = SQLiteBackend(":memory:")
    backend.executescript("""
        CREATE TABLE IF NOT EXISTS outcome_records (
            step_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            agent_role TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            success INTEGER NOT NULL,
            quality_score REAL NOT NULL,
            cost_usd REAL NOT NULL,
            tokens_used INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            timestamp TEXT NOT NULL,
            skill_name TEXT DEFAULT '',
            skill_version TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS skill_metrics (
            skill_name TEXT NOT NULL,
            skill_version TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            total_invocations INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            avg_quality_score REAL DEFAULT 0.0,
            avg_cost_usd REAL DEFAULT 0.0,
            avg_latency_ms REAL DEFAULT 0.0,
            total_cost_usd REAL DEFAULT 0.0,
            trend TEXT DEFAULT 'stable',
            computed_at TEXT NOT NULL,
            PRIMARY KEY (skill_name, skill_version, period_start)
        );
        CREATE TABLE IF NOT EXISTS skill_experiments (
            id TEXT PRIMARY KEY,
            skill_name TEXT NOT NULL,
            control_version TEXT NOT NULL,
            variant_version TEXT NOT NULL,
            traffic_split REAL DEFAULT 0.5,
            status TEXT DEFAULT 'active',
            min_invocations INTEGER DEFAULT 100,
            start_date TEXT NOT NULL,
            end_date TEXT,
            winner TEXT,
            conclusion_reason TEXT,
            created_at TEXT NOT NULL,
            concluded_at TEXT
        );
    """)
    return backend


def _insert_outcome(backend, skill_name, skill_version, success=1, quality=0.9):
    ts = datetime.now(UTC).isoformat()
    backend.execute(
        "INSERT INTO outcome_records "
        "(step_id, workflow_id, agent_role, provider, model, success, "
        "quality_score, cost_usd, tokens_used, latency_ms, metadata, timestamp, "
        "skill_name, skill_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "w1",
            "builder",
            "openai",
            "gpt-4o",
            success,
            quality,
            0.05,
            1000,
            500,
            "{}",
            ts,
            skill_name,
            skill_version,
        ),
    )


class TestCreateExperiment:
    def test_basic(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1.0.0", "1.0.1")
        assert config.skill_name == "s"
        assert config.control_version == "1.0.0"
        assert config.variant_version == "1.0.1"
        assert config.traffic_split == 0.5

    def test_custom_params(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1", "2", traffic_split=0.3, min_invocations=50)
        assert config.traffic_split == 0.3
        assert config.min_invocations == 50

    def test_persisted(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1", "2")
        rows = backend.fetchall("SELECT * FROM skill_experiments", ())
        assert len(rows) == 1
        assert rows[0]["id"] == config.experiment_id


class TestRouteSkillVersion:
    def test_no_experiment(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        assert mgr.route_skill_version("s", "w1") is None

    def test_deterministic_routing(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        mgr.create_experiment("s", "1.0.0", "1.0.1")

        # Same workflow_id should always get same version
        v1 = mgr.route_skill_version("s", "workflow_abc")
        v2 = mgr.route_skill_version("s", "workflow_abc")
        assert v1 == v2
        assert v1 in ("1.0.0", "1.0.1")

    def test_approximately_splits_traffic(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        mgr.create_experiment("s", "1.0.0", "1.0.1", traffic_split=0.5)

        versions = [mgr.route_skill_version("s", f"w_{i}") for i in range(1000)]
        variant_count = sum(1 for v in versions if v == "1.0.1")
        # Should be roughly 50% ± 10%
        assert 300 < variant_count < 700


class TestEvaluateExperiment:
    def test_insufficient_data(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1.0.0", "1.0.1", min_invocations=100)
        # No outcome data → None
        assert mgr.evaluate_experiment(config.experiment_id) is None

    def test_nonexistent_experiment(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        assert mgr.evaluate_experiment("fake") is None

    def test_variant_wins(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1.0.0", "1.0.1", min_invocations=10)

        # Control: mediocre
        for _ in range(10):
            _insert_outcome(backend, "s", "1.0.0", success=0, quality=0.3)
        # Variant: good
        for _ in range(10):
            _insert_outcome(backend, "s", "1.0.1", success=1, quality=0.9)

        result = mgr.evaluate_experiment(config.experiment_id)
        assert result is not None
        assert result.winner == "1.0.1"
        assert "Variant outperformed" in result.conclusion_reason

    def test_control_wins(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1.0.0", "1.0.1", min_invocations=10)

        for _ in range(10):
            _insert_outcome(backend, "s", "1.0.0", success=1, quality=0.9)
        for _ in range(10):
            _insert_outcome(backend, "s", "1.0.1", success=0, quality=0.3)

        result = mgr.evaluate_experiment(config.experiment_id)
        assert result is not None
        assert result.winner == "1.0.0"


class TestConcludeExperiment:
    def test_conclude(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1", "2")
        mgr.conclude_experiment(config.experiment_id, "2", "variant won")

        row = backend.fetchone(
            "SELECT * FROM skill_experiments WHERE id = ?", (config.experiment_id,)
        )
        assert row["status"] == "concluded"
        assert row["winner"] == "2"


class TestCancelExperiment:
    def test_cancel(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1", "2")
        mgr.cancel_experiment(config.experiment_id, "no longer needed")

        row = backend.fetchone(
            "SELECT * FROM skill_experiments WHERE id = ?", (config.experiment_id,)
        )
        assert row["status"] == "cancelled"


class TestPromoteWinner:
    def test_concluded_experiment(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1", "2")
        mgr.conclude_experiment(config.experiment_id, "2", "winner")

        result = mgr.promote_winner(config.experiment_id)
        assert result is not None
        assert result["winner"] == "2"

    def test_active_experiment(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        config = mgr.create_experiment("s", "1", "2")
        assert mgr.promote_winner(config.experiment_id) is None


class TestGetActiveExperiments:
    def test_empty(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        assert mgr.get_active_experiments() == []

    def test_returns_active_only(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        mgr = ABTestManager(backend, agg)
        mgr.create_experiment("a", "1", "2")
        config_b = mgr.create_experiment("b", "1", "2")
        mgr.cancel_experiment(config_b.experiment_id, "done")

        active = mgr.get_active_experiments()
        assert len(active) == 1
        assert active[0]["skill_name"] == "a"


class TestCompositeScore:
    def test_none_metrics(self):
        assert ABTestManager._composite_score(None) == 0.0

    def test_calculation(self):
        from animus_forge.skills.evolver.models import SkillMetrics

        m = SkillMetrics(
            skill_name="s",
            skill_version="1",
            period_start="a",
            period_end="b",
            success_rate=0.8,
            avg_quality_score=0.9,
            computed_at="now",
        )
        score = ABTestManager._composite_score(m)
        assert abs(score - (0.8 * 0.6 + 0.9 * 0.4)) < 0.001
