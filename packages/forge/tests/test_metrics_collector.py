"""Tests for metrics collection and export."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from animus_forge.metrics.collector import (
    MetricsCollector,
    StepMetrics,
    WorkflowMetrics,
    get_collector,
)
from animus_forge.metrics.exporters import (
    FileExporter,
    JsonExporter,
    LogExporter,
    PrometheusExporter,
    create_exporter,
)


class TestStepMetrics:
    """Tests for StepMetrics class."""

    def test_create_step_metrics(self):
        """Can create step metrics."""
        step = StepMetrics(step_id="step1", step_type="shell")

        assert step.step_id == "step1"
        assert step.step_type == "shell"
        assert step.status == "pending"
        assert step.started_at is None

    def test_step_start(self):
        """Step can be started."""
        step = StepMetrics(step_id="step1", step_type="shell")
        step.start()

        assert step.status == "running"
        assert step.started_at is not None

    def test_step_complete(self):
        """Step can be completed."""
        step = StepMetrics(step_id="step1", step_type="shell")
        step.start()
        step.complete(tokens=100)

        assert step.status == "success"
        assert step.completed_at is not None
        assert step.tokens_used == 100
        assert step.duration_ms >= 0  # May be 0 if operation completes within same ms

    def test_step_fail(self):
        """Step can be failed."""
        step = StepMetrics(step_id="step1", step_type="shell")
        step.start()
        step.fail("Command failed")

        assert step.status == "failed"
        assert step.error == "Command failed"
        assert step.completed_at is not None

    def test_step_to_dict(self):
        """Step can be converted to dict."""
        step = StepMetrics(step_id="step1", step_type="claude_code")
        step.start()
        step.complete(tokens=50)

        data = step.to_dict()

        assert data["step_id"] == "step1"
        assert data["step_type"] == "claude_code"
        assert data["status"] == "success"
        assert data["tokens_used"] == 50
        assert "started_at" in data
        assert "completed_at" in data

    def test_step_duration_without_start(self):
        """Step duration is 0 without start."""
        step = StepMetrics(step_id="step1", step_type="shell")
        step.complete()

        assert step.duration_ms == 0


class TestWorkflowMetrics:
    """Tests for WorkflowMetrics class."""

    def test_create_workflow_metrics(self):
        """Can create workflow metrics."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test Workflow",
            execution_id="exec1",
        )

        assert wf.workflow_id == "wf1"
        assert wf.workflow_name == "Test Workflow"
        assert wf.status == "pending"

    def test_workflow_start(self):
        """Workflow can be started."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )
        wf.start()

        assert wf.status == "running"
        assert wf.started_at is not None

    def test_workflow_complete(self):
        """Workflow can be completed."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )
        wf.start()

        # Add a step
        step = wf.add_step("step1", "shell")
        step.start()
        step.complete(tokens=100)

        wf.complete()

        assert wf.status == "success"
        assert wf.total_tokens == 100
        assert wf.duration_ms >= 0  # May be 0 if operation completes within same ms

    def test_workflow_fail(self):
        """Workflow can be failed."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )
        wf.start()
        wf.fail("Something went wrong")

        assert wf.status == "failed"
        assert wf.metadata["error"] == "Something went wrong"

    def test_workflow_add_step(self):
        """Can add steps to workflow."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )

        step = wf.add_step("step1", "shell")

        assert step.step_id == "step1"
        assert "step1" in wf.steps

    def test_workflow_get_step(self):
        """Can get step by ID."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )
        wf.add_step("step1", "shell")

        step = wf.get_step("step1")
        assert step is not None

        missing = wf.get_step("nonexistent")
        assert missing is None

    def test_workflow_success_rate(self):
        """Success rate is calculated correctly."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )

        # No steps
        assert wf.success_rate == 0.0

        # Add successful steps
        step1 = wf.add_step("step1", "shell")
        step1.status = "success"

        step2 = wf.add_step("step2", "shell")
        step2.status = "success"

        step3 = wf.add_step("step3", "shell")
        step3.status = "failed"

        assert wf.success_rate == pytest.approx(2 / 3)

    def test_workflow_to_dict(self):
        """Workflow can be converted to dict."""
        wf = WorkflowMetrics(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )
        wf.start()
        wf.add_step("step1", "shell")
        wf.complete()

        data = wf.to_dict()

        assert data["workflow_id"] == "wf1"
        assert data["status"] == "success"
        assert "steps" in data
        assert "step1" in data["steps"]


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a fresh collector."""
        return MetricsCollector()

    def test_start_workflow(self, collector):
        """Can start tracking a workflow."""
        metrics = collector.start_workflow(
            workflow_id="wf1",
            workflow_name="Test",
            execution_id="exec1",
        )

        assert metrics.workflow_id == "wf1"
        assert metrics.status == "running"

    def test_start_step(self, collector):
        """Can start tracking a step."""
        collector.start_workflow("wf1", "Test", "exec1")
        step = collector.start_step("exec1", "step1", "shell")

        assert step is not None
        assert step.status == "running"

    def test_start_step_unknown_workflow(self, collector):
        """Start step returns None for unknown workflow."""
        step = collector.start_step("nonexistent", "step1", "shell")
        assert step is None

    def test_complete_step(self, collector):
        """Can complete a step."""
        collector.start_workflow("wf1", "Test", "exec1")
        collector.start_step("exec1", "step1", "shell")
        collector.complete_step("exec1", "step1", tokens=100)

        metrics = collector.get_active()[0]
        step = metrics.get_step("step1")

        assert step.status == "success"
        assert step.tokens_used == 100

    def test_fail_step(self, collector):
        """Can fail a step."""
        collector.start_workflow("wf1", "Test", "exec1")
        collector.start_step("exec1", "step1", "shell")
        collector.fail_step("exec1", "step1", "Error message")

        metrics = collector.get_active()[0]
        step = metrics.get_step("step1")

        assert step.status == "failed"
        assert step.error == "Error message"

    def test_complete_workflow(self, collector):
        """Can complete a workflow."""
        collector.start_workflow("wf1", "Test", "exec1")
        result = collector.complete_workflow("exec1")

        assert result is not None
        assert result.status == "success"
        assert len(collector.get_active()) == 0

    def test_fail_workflow(self, collector):
        """Can fail a workflow."""
        collector.start_workflow("wf1", "Test", "exec1")
        result = collector.fail_workflow("exec1", "Error")

        assert result is not None
        assert result.status == "failed"

    def test_get_active(self, collector):
        """Can get active workflows."""
        collector.start_workflow("wf1", "Test1", "exec1")
        collector.start_workflow("wf2", "Test2", "exec2")

        active = collector.get_active()
        assert len(active) == 2

    def test_get_history(self, collector):
        """Can get workflow history."""
        collector.start_workflow("wf1", "Test", "exec1")
        collector.complete_workflow("exec1")

        collector.start_workflow("wf2", "Test", "exec2")
        collector.complete_workflow("exec2")

        history = collector.get_history()
        assert len(history) == 2
        # Most recent first
        assert history[0].execution_id == "exec2"

    def test_get_history_limit(self, collector):
        """History respects limit."""
        for i in range(5):
            collector.start_workflow(f"wf{i}", "Test", f"exec{i}")
            collector.complete_workflow(f"exec{i}")

        history = collector.get_history(limit=3)
        assert len(history) == 3

    def test_max_history(self):
        """Collector limits history size."""
        collector = MetricsCollector(max_history=3)

        for i in range(5):
            collector.start_workflow(f"wf{i}", "Test", f"exec{i}")
            collector.complete_workflow(f"exec{i}")

        history = collector.get_history(limit=100)
        assert len(history) == 3

    def test_get_summary(self, collector):
        """Can get metrics summary."""
        collector.start_workflow("wf1", "Test", "exec1")
        collector.start_step("exec1", "step1", "shell")
        collector.complete_step("exec1", "step1", tokens=100)
        collector.complete_workflow("exec1")

        summary = collector.get_summary()

        assert summary["active_workflows"] == 0
        assert summary["total_executions"] == 1
        assert summary["success_rate"] == 100.0
        assert summary["counters"]["workflows_completed"] == 1

    def test_reset(self, collector):
        """Can reset all metrics."""
        collector.start_workflow("wf1", "Test", "exec1")
        collector.complete_workflow("exec1")

        collector.reset()

        assert len(collector.get_active()) == 0
        assert len(collector.get_history()) == 0
        assert collector.get_summary()["total_executions"] == 0

    def test_register_callback(self, collector):
        """Can register callbacks."""
        events = []

        def callback(event_type, metrics):
            events.append((event_type, metrics.execution_id))

        collector.register_callback(callback)
        collector.start_workflow("wf1", "Test", "exec1")
        collector.complete_workflow("exec1")

        assert len(events) == 2
        assert events[0][0] == "workflow_started"
        assert events[1][0] == "workflow_completed"

    def test_callback_exception_handled(self, collector):
        """Callback exceptions don't break collection."""

        def bad_callback(event_type, metrics):
            raise RuntimeError("Callback error")

        collector.register_callback(bad_callback)

        # Should not raise
        collector.start_workflow("wf1", "Test", "exec1")
        collector.complete_workflow("exec1")

    def test_thread_safety(self, collector):
        """Collector is thread-safe."""
        import threading

        def worker(i):
            collector.start_workflow(f"wf{i}", "Test", f"exec{i}")
            collector.start_step(f"exec{i}", "step1", "shell")
            collector.complete_step(f"exec{i}", "step1")
            collector.complete_workflow(f"exec{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(collector.get_history()) == 10


class TestGetCollector:
    """Tests for global collector singleton."""

    def test_get_collector_returns_instance(self):
        """get_collector returns a MetricsCollector."""
        collector = get_collector()
        assert isinstance(collector, MetricsCollector)

    def test_get_collector_returns_same_instance(self):
        """get_collector returns the same instance."""
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    @pytest.fixture
    def collector(self):
        """Create collector with sample data."""
        collector = MetricsCollector()
        collector.start_workflow("wf1", "Test", "exec1")
        collector.start_step("exec1", "step1", "shell")
        collector.complete_step("exec1", "step1", tokens=100)
        collector.complete_workflow("exec1")
        return collector

    def test_export(self, collector):
        """Can export metrics in Prometheus format."""
        exporter = PrometheusExporter(prefix="test")
        output = exporter.export(collector)

        assert "# TYPE" in output
        assert "test_workflows_completed_total" in output
        assert "test_success_rate" in output

    def test_export_workflow(self, collector):
        """Can export single workflow."""
        exporter = PrometheusExporter()
        workflow = collector.get_history()[0]
        output = exporter.export_workflow(workflow)

        assert "workflow_duration_ms" in output
        assert "workflow_tokens" in output

    def test_custom_prefix(self, collector):
        """Exporter uses custom prefix."""
        exporter = PrometheusExporter(prefix="myapp")
        output = exporter.export(collector)

        assert "myapp_" in output


class TestJsonExporter:
    """Tests for JsonExporter."""

    @pytest.fixture
    def collector(self):
        """Create collector with sample data."""
        collector = MetricsCollector()
        collector.start_workflow("wf1", "Test", "exec1")
        collector.complete_workflow("exec1")
        return collector

    def test_export(self, collector):
        """Can export metrics as JSON."""
        exporter = JsonExporter()
        output = exporter.export(collector)

        data = json.loads(output)
        assert "timestamp" in data
        assert "summary" in data
        assert "active" in data
        assert "recent" in data

    def test_export_workflow(self, collector):
        """Can export single workflow as JSON."""
        exporter = JsonExporter()
        workflow = collector.get_history()[0]
        output = exporter.export_workflow(workflow)

        data = json.loads(output)
        assert data["workflow_id"] == "wf1"

    def test_pretty_output(self, collector):
        """Pretty mode adds indentation."""
        pretty = JsonExporter(pretty=True)
        compact = JsonExporter(pretty=False)

        pretty_out = pretty.export(collector)
        compact_out = compact.export(collector)

        # Pretty output is longer (has indentation)
        assert len(pretty_out) > len(compact_out)
        # Both parse to valid JSON
        pretty_data = json.loads(pretty_out)
        compact_data = json.loads(compact_out)
        # Both have same structure (timestamps may differ)
        assert set(pretty_data.keys()) == set(compact_data.keys())
        assert pretty_data["summary"] == compact_data["summary"]


class TestLogExporter:
    """Tests for LogExporter."""

    @pytest.fixture
    def collector(self):
        """Create collector with sample data."""
        collector = MetricsCollector()
        collector.start_workflow("wf1", "Test", "exec1")
        collector.complete_workflow("exec1")
        return collector

    def test_export(self, collector):
        """Can export metrics to logger."""
        mock_logger = MagicMock()
        exporter = LogExporter(logger=mock_logger)

        output = exporter.export(collector)

        mock_logger.log.assert_called_once()
        # Output is JSON
        data = json.loads(output)
        assert "counters" in data

    def test_export_workflow(self, collector):
        """Can export workflow to logger."""
        mock_logger = MagicMock()
        exporter = LogExporter(logger=mock_logger)
        workflow = collector.get_history()[0]

        exporter.export_workflow(workflow)

        mock_logger.log.assert_called_once()
        assert "success" in mock_logger.log.call_args[0][1]

    def test_custom_level(self, collector):
        """Exporter uses custom log level."""
        mock_logger = MagicMock()
        exporter = LogExporter(logger=mock_logger, level=logging.WARNING)

        exporter.export(collector)

        mock_logger.log.assert_called_with(
            logging.WARNING,
            "Metrics summary",
            extra=mock_logger.log.call_args[1]["extra"],
        )


class TestFileExporter:
    """Tests for FileExporter."""

    @pytest.fixture
    def collector(self):
        """Create collector with sample data."""
        collector = MetricsCollector()
        collector.start_workflow("wf1", "Test", "exec1")
        collector.complete_workflow("exec1")
        return collector

    def test_export_json(self, collector):
        """Can export to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileExporter(output_dir=tmpdir, format="json")
            filepath = exporter.export(collector)

            assert Path(filepath).exists()
            assert filepath.endswith(".json")

            with open(filepath) as f:
                data = json.load(f)
            assert "summary" in data

    def test_export_prometheus(self, collector):
        """Can export to Prometheus file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileExporter(output_dir=tmpdir, format="prometheus")
            filepath = exporter.export(collector)

            assert Path(filepath).exists()
            assert filepath.endswith(".prom")

    def test_export_workflow(self, collector):
        """Can export workflow to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileExporter(output_dir=tmpdir, format="json")
            workflow = collector.get_history()[0]

            filepath = exporter.export_workflow(workflow)

            assert Path(filepath).exists()
            assert "workflow_exec1" in filepath

    def test_creates_directory(self, collector):
        """Exporter creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "nested" / "metrics"
            FileExporter(output_dir=new_dir, format="json")

            assert new_dir.exists()

    def test_invalid_format(self):
        """Raises error for invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown format"):
                FileExporter(output_dir=tmpdir, format="invalid")


class TestCreateExporter:
    """Tests for create_exporter factory."""

    def test_create_prometheus(self):
        """Can create Prometheus exporter."""
        exporter = create_exporter("prometheus", prefix="test")
        assert isinstance(exporter, PrometheusExporter)
        assert exporter.prefix == "test"

    def test_create_json(self):
        """Can create JSON exporter."""
        exporter = create_exporter("json", pretty=False)
        assert isinstance(exporter, JsonExporter)
        assert exporter.pretty is False

    def test_create_log(self):
        """Can create Log exporter."""
        exporter = create_exporter("log")
        assert isinstance(exporter, LogExporter)

    def test_create_file(self):
        """Can create File exporter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = create_exporter("file", output_dir=tmpdir)
            assert isinstance(exporter, FileExporter)

    def test_invalid_format(self):
        """Raises error for invalid format."""
        with pytest.raises(ValueError, match="Unknown exporter format"):
            create_exporter("invalid")
