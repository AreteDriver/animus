"""Workflow Scheduler - Cron-like scheduling for YAML workflows."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .executor import ExecutionResult, WorkflowExecutor
from .loader import load_workflow

logger = logging.getLogger(__name__)


class ScheduleStatus(Enum):
    """Schedule status."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled workflow."""

    id: str
    workflow_path: str
    name: str = ""
    description: str = ""

    # Cron expression (e.g., "0 9 * * 1-5" for 9am weekdays)
    cron: str | None = None

    # Or interval in seconds
    interval_seconds: int | None = None

    # Workflow inputs
    inputs: dict = field(default_factory=dict)

    # Execution settings
    dry_run: bool = False
    timeout_seconds: int = 3600

    # Status
    status: ScheduleStatus = ScheduleStatus.ACTIVE

    # Stats
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    last_status: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_path": self.workflow_path,
            "name": self.name,
            "description": self.description,
            "cron": self.cron,
            "interval_seconds": self.interval_seconds,
            "inputs": self.inputs,
            "dry_run": self.dry_run,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScheduleConfig:
        status = data.get("status", "active")
        if isinstance(status, str):
            status = ScheduleStatus(status)

        return cls(
            id=data["id"],
            workflow_path=data["workflow_path"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            cron=data.get("cron"),
            interval_seconds=data.get("interval_seconds"),
            inputs=data.get("inputs", {}),
            dry_run=data.get("dry_run", False),
            timeout_seconds=data.get("timeout_seconds", 3600),
            status=status,
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(UTC),
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
            next_run=datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None,
            run_count=data.get("run_count", 0),
            last_status=data.get("last_status"),
            last_error=data.get("last_error"),
        )


@dataclass
class ExecutionLog:
    """Log entry for a scheduled execution."""

    schedule_id: str
    workflow_path: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"
    tokens_used: int = 0
    steps_completed: int = 0
    steps_total: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "schedule_id": self.schedule_id,
            "workflow_path": self.workflow_path,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "tokens_used": self.tokens_used,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "error": self.error,
        }


# Type for execution callbacks
ExecutionCallback = Callable[[ScheduleConfig, ExecutionResult], None]


class WorkflowScheduler:
    """Cron-like scheduler for YAML workflows.

    Usage:
        scheduler = WorkflowScheduler()

        # Add a scheduled workflow
        scheduler.add(ScheduleConfig(
            id="daily-report",
            workflow_path="workflows/report.yaml",
            cron="0 9 * * *",  # 9am daily
            inputs={"report_type": "daily"}
        ))

        # Start scheduler
        scheduler.start()

        # ... later
        scheduler.shutdown()
    """

    def __init__(
        self,
        data_dir: str | Path = "data/schedules",
        checkpoint_manager=None,
        budget_manager=None,
        on_execution: ExecutionCallback | None = None,
    ):
        """Initialize scheduler.

        Args:
            data_dir: Directory to store schedule configs and logs
            checkpoint_manager: Optional CheckpointManager for state persistence
            budget_manager: Optional BudgetManager for token tracking
            on_execution: Callback invoked after each execution
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.data_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.checkpoint_manager = checkpoint_manager
        self.budget_manager = budget_manager
        self.on_execution = on_execution

        self._scheduler = BackgroundScheduler()
        self._schedules: dict[str, ScheduleConfig] = {}
        self._running = False

    def start(self) -> None:
        """Start the scheduler and load existing schedules."""
        self._load_schedules()

        # Start scheduler first so jobs can get next_run_time
        if not self._scheduler.running:
            self._scheduler.start()
            self._running = True

        # Now register jobs
        for schedule in self._schedules.values():
            if schedule.status == ScheduleStatus.ACTIVE:
                self._register_job(schedule)

        logger.info(f"Workflow scheduler started with {len(self._schedules)} schedule(s)")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            self._running = False
            logger.info("Workflow scheduler shutdown")

    @property
    def is_running(self) -> bool:
        return self._running

    def add(self, config: ScheduleConfig) -> ScheduleConfig:
        """Add a new scheduled workflow.

        Args:
            config: Schedule configuration

        Returns:
            Updated config with next_run populated
        """
        # Validate workflow exists
        try:
            load_workflow(config.workflow_path)
        except Exception as e:
            raise ValueError(f"Invalid workflow: {e}")

        # Validate schedule
        if not config.cron and not config.interval_seconds:
            raise ValueError("Must specify either 'cron' or 'interval_seconds'")

        config.created_at = datetime.now(UTC)
        self._schedules[config.id] = config
        self._save_schedule(config)

        if self._running and config.status == ScheduleStatus.ACTIVE:
            self._register_job(config)

        logger.info(f"Added schedule: {config.id}")
        return config

    def remove(self, schedule_id: str) -> bool:
        """Remove a scheduled workflow."""
        if schedule_id not in self._schedules:
            return False

        # Remove from APScheduler
        job_id = f"wf_{schedule_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        # Remove config file
        config_path = self.data_dir / f"{schedule_id}.json"
        if config_path.exists():
            config_path.unlink()

        del self._schedules[schedule_id]
        logger.info(f"Removed schedule: {schedule_id}")
        return True

    def get(self, schedule_id: str) -> ScheduleConfig | None:
        """Get a schedule by ID."""
        config = self._schedules.get(schedule_id)
        if config:
            self._update_next_run(config)
        return config

    def list(self) -> list[ScheduleConfig]:
        """List all schedules."""
        for config in self._schedules.values():
            self._update_next_run(config)
        return list(self._schedules.values())

    def pause(self, schedule_id: str) -> bool:
        """Pause a schedule."""
        config = self._schedules.get(schedule_id)
        if not config:
            return False

        config.status = ScheduleStatus.PAUSED
        job_id = f"wf_{schedule_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.pause_job(job_id)

        self._save_schedule(config)
        logger.info(f"Paused schedule: {schedule_id}")
        return True

    def resume(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""
        config = self._schedules.get(schedule_id)
        if not config:
            return False

        config.status = ScheduleStatus.ACTIVE
        job_id = f"wf_{schedule_id}"

        if self._scheduler.get_job(job_id):
            self._scheduler.resume_job(job_id)
        else:
            self._register_job(config)

        self._update_next_run(config)
        self._save_schedule(config)
        logger.info(f"Resumed schedule: {schedule_id}")
        return True

    def trigger(self, schedule_id: str) -> ExecutionResult | None:
        """Manually trigger a scheduled workflow immediately."""
        config = self._schedules.get(schedule_id)
        if not config:
            return None

        return self._execute(config)

    def get_history(self, schedule_id: str, limit: int = 20) -> list[ExecutionLog]:
        """Get execution history for a schedule."""
        logs = []
        pattern = f"{schedule_id}_*.json"

        for log_file in sorted(self.logs_dir.glob(pattern), reverse=True)[:limit]:
            try:
                with open(log_file) as f:
                    data = json.load(f)
                logs.append(
                    ExecutionLog(
                        schedule_id=data["schedule_id"],
                        workflow_path=data["workflow_path"],
                        started_at=datetime.fromisoformat(data["started_at"]),
                        completed_at=datetime.fromisoformat(data["completed_at"])
                        if data.get("completed_at")
                        else None,
                        status=data["status"],
                        tokens_used=data.get("tokens_used", 0),
                        steps_completed=data.get("steps_completed", 0),
                        steps_total=data.get("steps_total", 0),
                        error=data.get("error"),
                    )
                )
            except Exception:
                continue

        return logs

    def _register_job(self, config: ScheduleConfig) -> None:
        """Register a schedule with APScheduler."""
        job_id = f"wf_{config.id}"

        # Remove existing job
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        # Create trigger
        if config.cron:
            trigger = CronTrigger.from_crontab(config.cron)
        elif config.interval_seconds:
            trigger = IntervalTrigger(seconds=config.interval_seconds)
        else:
            return

        self._scheduler.add_job(
            self._execute,
            trigger=trigger,
            id=job_id,
            args=[config],
            name=config.name or config.id,
            replace_existing=True,
        )

        self._update_next_run(config)
        logger.debug(f"Registered job: {job_id}")

    def _update_next_run(self, config: ScheduleConfig) -> None:
        """Update next_run from APScheduler."""
        job = self._scheduler.get_job(f"wf_{config.id}")
        if job and job.next_run_time:
            config.next_run = job.next_run_time

    def _execute(self, config: ScheduleConfig) -> ExecutionResult:
        """Execute a scheduled workflow."""
        logger.info(f"Executing scheduled workflow: {config.id} -> {config.workflow_path}")

        started_at = datetime.now(UTC)
        log = ExecutionLog(
            schedule_id=config.id,
            workflow_path=config.workflow_path,
            started_at=started_at,
        )

        try:
            # Load workflow
            workflow = load_workflow(config.workflow_path)
            log.steps_total = len(workflow.steps)

            # Create executor
            executor = WorkflowExecutor(
                checkpoint_manager=self.checkpoint_manager,
                budget_manager=self.budget_manager,
                dry_run=config.dry_run,
            )

            # Execute
            result = executor.execute(workflow, inputs=config.inputs)

            # Update log
            log.completed_at = datetime.now(UTC)
            log.status = result.status
            log.tokens_used = result.total_tokens
            log.steps_completed = len([s for s in result.steps if s.status.value == "success"])

            if result.error:
                log.error = result.error

        except Exception as e:
            logger.error(f"Scheduled execution failed: {e}")
            log.completed_at = datetime.now(UTC)
            log.status = "failed"
            log.error = str(e)
            result = ExecutionResult(
                workflow_name=config.workflow_path, status="failed", error=str(e)
            )

        # Update config stats
        config.last_run = started_at
        config.run_count += 1
        config.last_status = log.status
        config.last_error = log.error
        self._update_next_run(config)
        self._save_schedule(config)

        # Save execution log
        self._save_log(log)

        # Invoke callback
        if self.on_execution:
            try:
                self.on_execution(config, result)
            except Exception as e:
                logger.error(f"Execution callback error: {e}")

        return result

    def _load_schedules(self) -> None:
        """Load schedules from disk."""
        for config_file in self.data_dir.glob("*.json"):
            if config_file.name.startswith("_"):
                continue
            try:
                with open(config_file) as f:
                    data = json.load(f)
                config = ScheduleConfig.from_dict(data)
                self._schedules[config.id] = config
            except Exception as e:
                logger.error(f"Failed to load schedule {config_file}: {e}")

    def _save_schedule(self, config: ScheduleConfig) -> None:
        """Save schedule to disk."""
        config_path = self.data_dir / f"{config.id}.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def _save_log(self, log: ExecutionLog) -> None:
        """Save execution log."""
        timestamp = log.started_at.strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"{log.schedule_id}_{timestamp}.json"
        with open(log_path, "w") as f:
            json.dump(log.to_dict(), f, indent=2)
