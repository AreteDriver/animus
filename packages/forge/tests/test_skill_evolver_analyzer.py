"""Tests for SkillAnalyzer."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from animus_forge.skills.evolver.analyzer import SkillAnalyzer
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
    """)
    return backend


def _insert_outcome(
    backend,
    skill_name="skill_a",
    skill_version="1.0.0",
    success=1,
    quality=0.9,
    cost=0.05,
    latency=500,
    agent_role="builder",
    days_ago=0,
):
    ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    backend.execute(
        "INSERT INTO outcome_records "
        "(step_id, workflow_id, agent_role, provider, model, success, "
        "quality_score, cost_usd, tokens_used, latency_ms, metadata, timestamp, "
        "skill_name, skill_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "w1",
            agent_role,
            "openai",
            "gpt-4o",
            success,
            quality,
            cost,
            1000,
            latency,
            "{}",
            ts,
            skill_name,
            skill_version,
        ),
    )


class TestFindDecliningSkills:
    def test_empty(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        assert analyzer.find_declining_skills() == []

    def test_finds_declining(self):
        backend = _make_backend()
        # Old successes → recent failures
        for _ in range(10):
            _insert_outcome(backend, success=1, days_ago=20)
        for _ in range(10):
            _insert_outcome(backend, success=0, days_ago=1)

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        declining = analyzer.find_declining_skills()
        assert len(declining) >= 1
        assert declining[0][0] == "skill_a"

    def test_ignores_low_volume(self):
        backend = _make_backend()
        for _ in range(3):
            _insert_outcome(backend, success=1, days_ago=20)
        for _ in range(3):
            _insert_outcome(backend, success=0, days_ago=1)

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        assert analyzer.find_declining_skills() == []


class TestFindCostAnomalies:
    def test_empty(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        assert analyzer.find_cost_anomalies() == []

    def test_detects_expensive_skill(self):
        backend = _make_backend()
        # Multiple cheap skills to push fleet average down
        for _ in range(15):
            _insert_outcome(backend, skill_name="cheap_a", cost=0.01)
        for _ in range(15):
            _insert_outcome(backend, skill_name="cheap_b", cost=0.02)
        # Expensive skill (>2x fleet average of ~$0.11)
        for _ in range(15):
            _insert_outcome(backend, skill_name="expensive", cost=0.50)

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        anomalies = analyzer.find_cost_anomalies()
        assert len(anomalies) >= 1
        assert anomalies[0][0] == "expensive"

    def test_no_anomaly_when_similar_costs(self):
        backend = _make_backend()
        for _ in range(15):
            _insert_outcome(backend, skill_name="a", cost=0.05)
        for _ in range(15):
            _insert_outcome(backend, skill_name="b", cost=0.06)

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        assert analyzer.find_cost_anomalies() == []


class TestFindUnderperformers:
    def test_empty(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        assert analyzer.find_underperformers() == []

    def test_detects_low_success(self):
        backend = _make_backend()
        for _ in range(15):
            _insert_outcome(backend, success=0, quality=0.2)
        for _ in range(3):
            _insert_outcome(backend, success=1, quality=0.9)

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        underperformers = analyzer.find_underperformers()
        assert len(underperformers) == 1
        assert underperformers[0][1].success_rate < 0.5


class TestDetectCapabilityGaps:
    def test_no_backend(self):
        agg = MagicMock()
        analyzer = SkillAnalyzer(agg, backend=None)
        assert analyzer.detect_capability_gaps() == []

    def test_detects_gaps(self):
        backend = _make_backend()
        for _ in range(5):
            _insert_outcome(backend, skill_name="", agent_role="reviewer")

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        gaps = analyzer.detect_capability_gaps()
        assert len(gaps) == 1
        assert "reviewer" in gaps[0].description

    def test_ignores_low_count(self):
        backend = _make_backend()
        _insert_outcome(backend, skill_name="", agent_role="reviewer")

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        gaps = analyzer.detect_capability_gaps()
        assert gaps == []


class TestGenerateAnalysisReport:
    def test_empty_report(self):
        backend = _make_backend()
        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        report = analyzer.generate_analysis_report()
        assert report["total_issues"] == 0
        assert "0 issues" in report["summary"]

    def test_report_with_issues(self):
        backend = _make_backend()
        for _ in range(15):
            _insert_outcome(backend, success=0, quality=0.1)

        agg = SkillMetricsAggregator(backend)
        analyzer = SkillAnalyzer(agg, backend)
        report = analyzer.generate_analysis_report()
        assert report["total_issues"] > 0
        assert len(report["underperformers"]) > 0
