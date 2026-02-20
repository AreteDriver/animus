"""Tests for cost tracking module."""

import csv
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from animus_forge.metrics.cost_tracker import (
    CostEntry,
    CostTracker,
    Provider,
    TokenUsage,
    get_cost_tracker,
    initialize_cost_tracker,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_auto_total(self):
        t = TokenUsage(input_tokens=100, output_tokens=50)
        assert t.total_tokens == 150

    def test_explicit_total(self):
        t = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=200)
        assert t.total_tokens == 200

    def test_zero_defaults(self):
        t = TokenUsage()
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.total_tokens == 0


class TestCostEntry:
    """Tests for CostEntry serialization."""

    def _make_entry(self, **overrides):
        defaults = {
            "timestamp": datetime(2026, 1, 15, 10, 30),
            "provider": Provider.OPENAI,
            "model": "gpt-4o-mini",
            "tokens": TokenUsage(input_tokens=1000, output_tokens=500),
            "cost_usd": 0.00045,
        }
        defaults.update(overrides)
        return CostEntry(**defaults)

    def test_to_dict(self):
        entry = self._make_entry(workflow_id="wf-1", agent_role="builder")
        d = entry.to_dict()
        assert d["provider"] == "openai"
        assert d["tokens"]["input"] == 1000
        assert d["tokens"]["output"] == 500
        assert d["workflow_id"] == "wf-1"

    def test_roundtrip(self):
        entry = self._make_entry(
            step_id="step-1",
            agent_role="reviewer",
            metadata={"key": "val"},
        )
        d = entry.to_dict()
        restored = CostEntry.from_dict(d)
        assert restored.provider == Provider.OPENAI
        assert restored.tokens.input_tokens == 1000
        assert restored.agent_role == "reviewer"
        assert restored.metadata == {"key": "val"}


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_known_model(self):
        tracker = CostTracker()
        tokens = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = tracker.calculate_cost("gpt-4o-mini", tokens)
        # input: 0.15, output: 0.60
        assert cost == pytest.approx(0.75, abs=0.001)

    def test_prefix_match(self):
        tracker = CostTracker()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = tracker.calculate_cost("claude-3-sonnet-20240229", tokens)
        # Should match claude-3-sonnet pricing
        assert cost > 0

    def test_unknown_model_fallback(self):
        tracker = CostTracker()
        tokens = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = tracker.calculate_cost("unknown-model-v9", tokens)
        # Fallback: input=1.00, output=2.00
        assert cost == pytest.approx(3.0, abs=0.001)

    def test_zero_tokens(self):
        tracker = CostTracker()
        tokens = TokenUsage()
        cost = tracker.calculate_cost("gpt-4o", tokens)
        assert cost == 0.0


class TestTracking:
    """Tests for the track method and entry storage."""

    def test_track_adds_entry(self):
        tracker = CostTracker()
        entry = tracker.track(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            input_tokens=500,
            output_tokens=200,
            workflow_id="wf-1",
            agent_role="builder",
        )
        assert len(tracker.entries) == 1
        assert entry.provider == Provider.OPENAI
        assert entry.cost_usd > 0
        assert entry.workflow_id == "wf-1"

    def test_track_persists_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "costs.json"
            tracker = CostTracker(storage_path=path)
            tracker.track(
                provider=Provider.ANTHROPIC,
                model="claude-3-haiku",
                input_tokens=100,
                output_tokens=50,
            )
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data["entries"]) == 1

    def test_load_existing_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "costs.json"
            # Write initial data
            tracker1 = CostTracker(storage_path=path)
            tracker1.track(
                provider=Provider.OPENAI,
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
            )
            # Load in new instance
            tracker2 = CostTracker(storage_path=path)
            assert len(tracker2.entries) == 1
            assert tracker2.entries[0].model == "gpt-4o"


class TestBudgetAlerts:
    """Tests for budget tracking and alerts."""

    def test_no_alert_below_threshold(self):
        tracker = CostTracker(budget_limit_usd=100.0, alert_threshold_percent=80.0)
        # Small call, well under budget
        tracker.track(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
        )
        assert len(tracker._alerts) == 0

    def test_alert_at_threshold(self):
        tracker = CostTracker(budget_limit_usd=0.001, alert_threshold_percent=50.0)
        # This should exceed a $0.001 budget
        tracker.track(
            provider=Provider.OPENAI,
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        assert len(tracker._alerts) > 0
        assert tracker._alerts[0]["type"] == "budget_alert"


class TestAggregation:
    """Tests for cost aggregation methods."""

    def _make_tracker_with_entries(self):
        tracker = CostTracker()
        now = datetime.now()
        tracker.entries = [
            CostEntry(
                timestamp=now,
                provider=Provider.OPENAI,
                model="gpt-4o-mini",
                tokens=TokenUsage(input_tokens=1000, output_tokens=500),
                cost_usd=0.05,
                workflow_id="wf-1",
                step_id="step-1",
                agent_role="builder",
            ),
            CostEntry(
                timestamp=now,
                provider=Provider.ANTHROPIC,
                model="claude-3-haiku",
                tokens=TokenUsage(input_tokens=2000, output_tokens=800),
                cost_usd=0.10,
                workflow_id="wf-1",
                step_id="step-2",
                agent_role="reviewer",
            ),
            CostEntry(
                timestamp=now - timedelta(days=60),
                provider=Provider.OPENAI,
                model="gpt-4o",
                tokens=TokenUsage(input_tokens=500, output_tokens=300),
                cost_usd=0.50,
                workflow_id="wf-old",
                agent_role="builder",
            ),
        ]
        return tracker

    def test_get_monthly_cost(self):
        tracker = self._make_tracker_with_entries()
        now = datetime.now()
        cost = tracker.get_monthly_cost(now.year, now.month)
        assert cost == pytest.approx(0.15, abs=0.01)

    def test_get_daily_cost(self):
        tracker = self._make_tracker_with_entries()
        cost = tracker.get_daily_cost()
        assert cost == pytest.approx(0.15, abs=0.01)

    def test_get_workflow_cost(self):
        tracker = self._make_tracker_with_entries()
        result = tracker.get_workflow_cost("wf-1")
        assert result["total_calls"] == 2
        assert result["total_cost"] == pytest.approx(0.15, abs=0.01)
        assert "step-1" in result["by_step"]
        assert "builder" in result["by_agent"]

    def test_get_agent_costs(self):
        tracker = self._make_tracker_with_entries()
        costs = tracker.get_agent_costs(days=30)
        assert "builder" in costs
        assert "reviewer" in costs
        # Old entry should be excluded
        assert costs["builder"]["calls"] == 1

    def test_get_model_costs(self):
        tracker = self._make_tracker_with_entries()
        costs = tracker.get_model_costs(days=30)
        assert "gpt-4o-mini" in costs
        assert "claude-3-haiku" in costs

    def test_get_summary(self):
        tracker = self._make_tracker_with_entries()
        summary = tracker.get_summary(days=30)
        assert summary["total_calls"] == 2
        assert summary["total_cost_usd"] > 0
        assert "by_provider" in summary
        assert "by_model" in summary
        assert summary["budget"]["limit_usd"] is None


class TestExportCSV:
    """Tests for CSV export."""

    def test_export_all(self):
        tracker = CostTracker()
        tracker.entries = [
            CostEntry(
                timestamp=datetime(2026, 1, 15),
                provider=Provider.OPENAI,
                model="gpt-4o",
                tokens=TokenUsage(input_tokens=100, output_tokens=50),
                cost_usd=0.01,
                workflow_id="wf-1",
            ),
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)

        tracker.export_csv(path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2  # header + 1 entry
        assert rows[0][0] == "timestamp"
        assert rows[1][1] == "openai"
        path.unlink()

    def test_export_with_days_filter(self):
        tracker = CostTracker()
        now = datetime.now()
        tracker.entries = [
            CostEntry(
                timestamp=now,
                provider=Provider.OPENAI,
                model="gpt-4o",
                tokens=TokenUsage(input_tokens=100, output_tokens=50),
                cost_usd=0.01,
            ),
            CostEntry(
                timestamp=now - timedelta(days=90),
                provider=Provider.OPENAI,
                model="gpt-4o",
                tokens=TokenUsage(input_tokens=100, output_tokens=50),
                cost_usd=0.01,
            ),
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)

        tracker.export_csv(path, days=30)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + only recent entry
        path.unlink()


class TestClearOldEntries:
    """Tests for clearing old entries."""

    def test_clears_old(self):
        tracker = CostTracker()
        now = datetime.now()
        tracker.entries = [
            CostEntry(
                timestamp=now,
                provider=Provider.OPENAI,
                model="gpt-4o",
                tokens=TokenUsage(),
                cost_usd=0.0,
            ),
            CostEntry(
                timestamp=now - timedelta(days=100),
                provider=Provider.OPENAI,
                model="gpt-4o",
                tokens=TokenUsage(),
                cost_usd=0.0,
            ),
        ]
        removed = tracker.clear_old_entries(days=90)
        assert removed == 1
        assert len(tracker.entries) == 1


class TestGlobalTracker:
    """Tests for module-level tracker functions."""

    def test_get_cost_tracker_default(self):
        import animus_forge.metrics.cost_tracker as mod

        mod._tracker = None
        tracker = get_cost_tracker()
        assert isinstance(tracker, CostTracker)

    def test_initialize_cost_tracker(self):
        tracker = initialize_cost_tracker(budget_limit_usd=50.0)
        assert tracker.budget_limit_usd == 50.0
        assert isinstance(tracker, CostTracker)
