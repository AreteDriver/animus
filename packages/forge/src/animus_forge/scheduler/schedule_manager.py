"""Scheduled workflow execution manager."""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, Field

from animus_forge.config import get_settings
from animus_forge.orchestrator import WorkflowEngineAdapter
from animus_forge.state import DatabaseBackend, get_database

logger = logging.getLogger(__name__)


def _parse_datetime(value) -> datetime | None:
    """Parse datetime from database (handles both strings and datetime objects)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class ScheduleType(str, Enum):
    """Types of schedule triggers."""

    CRON = "cron"
    INTERVAL = "interval"


class ScheduleStatus(str, Enum):
    """Schedule status."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


class CronConfig(BaseModel):
    """Cron schedule configuration."""

    minute: str = Field("*", description="Minute (0-59)")
    hour: str = Field("*", description="Hour (0-23)")
    day: str = Field("*", description="Day of month (1-31)")
    month: str = Field("*", description="Month (1-12)")
    day_of_week: str = Field("*", description="Day of week (0-6, mon-sun)")


class IntervalConfig(BaseModel):
    """Interval schedule configuration."""

    seconds: int = Field(0, ge=0, description="Seconds")
    minutes: int = Field(0, ge=0, description="Minutes")
    hours: int = Field(0, ge=0, description="Hours")
    days: int = Field(0, ge=0, description="Days")


class WorkflowSchedule(BaseModel):
    """A scheduled workflow definition."""

    id: str = Field(..., description="Schedule identifier")
    workflow_id: str = Field(..., description="Workflow to execute")
    name: str = Field(..., description="Schedule name")
    description: str = Field("", description="Schedule description")
    schedule_type: ScheduleType = Field(..., description="Type of schedule")
    cron_config: CronConfig | None = Field(None, description="Cron configuration")
    interval_config: IntervalConfig | None = Field(None, description="Interval configuration")
    variables: dict[str, Any] = Field(
        default_factory=dict, description="Workflow variables to pass"
    )
    status: ScheduleStatus = Field(ScheduleStatus.ACTIVE, description="Schedule status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_run: datetime | None = Field(None, description="Last execution time")
    next_run: datetime | None = Field(None, description="Next scheduled execution")
    run_count: int = Field(0, ge=0, description="Total execution count")


class ScheduleExecutionLog(BaseModel):
    """Log entry for a scheduled execution."""

    schedule_id: str
    workflow_id: str
    executed_at: datetime
    status: str
    duration_seconds: float
    error: str | None = None


class ScheduleManager:
    """Manages scheduled workflow execution."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS schedules (
            id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            schedule_type TEXT NOT NULL,
            cron_config TEXT,
            interval_config TEXT,
            variables TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_run TIMESTAMP,
            next_run TIMESTAMP,
            run_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_schedules_status ON schedules(status);
        CREATE INDEX IF NOT EXISTS idx_schedules_workflow ON schedules(workflow_id);

        CREATE TABLE IF NOT EXISTS schedule_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            duration_seconds REAL,
            error TEXT,
            FOREIGN KEY (schedule_id) REFERENCES schedules(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_schedule_logs_schedule
        ON schedule_logs(schedule_id, executed_at DESC);
    """

    def __init__(self, backend: DatabaseBackend | None = None):
        self.settings = get_settings()
        self.backend = backend or get_database()
        self.workflow_engine = WorkflowEngineAdapter()
        self.scheduler = BackgroundScheduler()
        self._schedules: dict[str, WorkflowSchedule] = {}
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.backend.executescript(self.SCHEMA)

    def start(self):
        """Start the scheduler and load existing schedules."""
        self._load_all_schedules()
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")

    def shutdown(self):
        """Shutdown the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler shutdown")

    def _load_all_schedules(self):
        """Load all schedules from database and register active ones."""
        rows = self.backend.fetchall("SELECT * FROM schedules")
        for row in rows:
            try:
                schedule = self._row_to_schedule(row)
                if schedule:
                    self._schedules[schedule.id] = schedule
                    if schedule.status == ScheduleStatus.ACTIVE:
                        self._register_job(schedule)
            except Exception as e:
                logger.error(f"Failed to load schedule from row: {e}")

    def _row_to_schedule(self, row: dict) -> WorkflowSchedule | None:
        """Convert database row to WorkflowSchedule."""
        try:
            cron_config = None
            if row.get("cron_config"):
                cron_config = CronConfig(**json.loads(row["cron_config"]))

            interval_config = None
            if row.get("interval_config"):
                interval_config = IntervalConfig(**json.loads(row["interval_config"]))

            return WorkflowSchedule(
                id=row["id"],
                workflow_id=row["workflow_id"],
                name=row["name"],
                description=row.get("description", ""),
                schedule_type=ScheduleType(row["schedule_type"]),
                cron_config=cron_config,
                interval_config=interval_config,
                variables=json.loads(row["variables"]) if row.get("variables") else {},
                status=ScheduleStatus(row["status"]),
                created_at=_parse_datetime(row.get("created_at")) or datetime.now(),
                last_run=_parse_datetime(row.get("last_run")),
                next_run=_parse_datetime(row.get("next_run")),
                run_count=row.get("run_count", 0),
            )
        except Exception as e:
            logger.error(f"Failed to parse schedule row: {e}")
            return None

    def _save_schedule(self, schedule: WorkflowSchedule) -> bool:
        """Save a schedule to database (insert or update)."""
        try:
            existing = self.backend.fetchone(
                "SELECT id FROM schedules WHERE id = ?", (schedule.id,)
            )
            if existing:
                return self._update_schedule_in_db(schedule)
            else:
                return self._insert_schedule_in_db(schedule)
        except Exception as e:
            logger.error(f"Failed to save schedule {schedule.id}: {e}")
            return False

    def _insert_schedule_in_db(self, schedule: WorkflowSchedule) -> bool:
        """Insert a new schedule into the database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    INSERT INTO schedules
                    (id, workflow_id, name, description, schedule_type, cron_config,
                     interval_config, variables, status, created_at, last_run, next_run, run_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        schedule.id,
                        schedule.workflow_id,
                        schedule.name,
                        schedule.description,
                        schedule.schedule_type.value,
                        json.dumps(schedule.cron_config.model_dump())
                        if schedule.cron_config
                        else None,
                        json.dumps(schedule.interval_config.model_dump())
                        if schedule.interval_config
                        else None,
                        json.dumps(schedule.variables) if schedule.variables else None,
                        schedule.status.value,
                        schedule.created_at.isoformat() if schedule.created_at else None,
                        schedule.last_run.isoformat() if schedule.last_run else None,
                        schedule.next_run.isoformat() if schedule.next_run else None,
                        schedule.run_count,
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to insert schedule {schedule.id}: {e}")
            return False

    def _update_schedule_in_db(self, schedule: WorkflowSchedule) -> bool:
        """Update an existing schedule in the database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    UPDATE schedules
                    SET workflow_id = ?, name = ?, description = ?, schedule_type = ?,
                        cron_config = ?, interval_config = ?, variables = ?, status = ?,
                        last_run = ?, next_run = ?, run_count = ?
                    WHERE id = ?
                    """,
                    (
                        schedule.workflow_id,
                        schedule.name,
                        schedule.description,
                        schedule.schedule_type.value,
                        json.dumps(schedule.cron_config.model_dump())
                        if schedule.cron_config
                        else None,
                        json.dumps(schedule.interval_config.model_dump())
                        if schedule.interval_config
                        else None,
                        json.dumps(schedule.variables) if schedule.variables else None,
                        schedule.status.value,
                        schedule.last_run.isoformat() if schedule.last_run else None,
                        schedule.next_run.isoformat() if schedule.next_run else None,
                        schedule.run_count,
                        schedule.id,
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to update schedule {schedule.id}: {e}")
            return False

    def _register_job(self, schedule: WorkflowSchedule):
        """Register a schedule with APScheduler."""
        job_id = f"schedule_{schedule.id}"

        # Remove existing job if present
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)

        if schedule.status != ScheduleStatus.ACTIVE:
            return

        trigger = self._create_trigger(schedule)
        if not trigger:
            logger.error(f"Failed to create trigger for schedule {schedule.id}")
            return

        self.scheduler.add_job(
            self._execute_scheduled_workflow,
            trigger=trigger,
            id=job_id,
            args=[schedule.id],
            name=schedule.name,
            replace_existing=True,
        )

        # Update next run time
        job = self.scheduler.get_job(job_id)
        if job and job.next_run_time:
            schedule.next_run = job.next_run_time
            self._save_schedule(schedule)

        logger.info(f"Registered job for schedule {schedule.id}")

    def _create_trigger(self, schedule: WorkflowSchedule):
        """Create APScheduler trigger from schedule config."""
        if schedule.schedule_type == ScheduleType.CRON and schedule.cron_config:
            return CronTrigger(
                minute=schedule.cron_config.minute,
                hour=schedule.cron_config.hour,
                day=schedule.cron_config.day,
                month=schedule.cron_config.month,
                day_of_week=schedule.cron_config.day_of_week,
            )
        elif schedule.schedule_type == ScheduleType.INTERVAL and schedule.interval_config:
            cfg = schedule.interval_config
            # Ensure at least some interval is set
            if cfg.seconds == 0 and cfg.minutes == 0 and cfg.hours == 0 and cfg.days == 0:
                cfg.minutes = 1  # Default to 1 minute
            return IntervalTrigger(
                seconds=cfg.seconds,
                minutes=cfg.minutes,
                hours=cfg.hours,
                days=cfg.days,
            )
        return None

    def _execute_scheduled_workflow(self, schedule_id: str):
        """Execute a scheduled workflow."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            logger.error(f"Schedule {schedule_id} not found")
            return

        logger.info(f"Executing scheduled workflow: {schedule.workflow_id}")
        start_time = datetime.now()
        error_msg = None

        try:
            workflow = self.workflow_engine.load_workflow(schedule.workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {schedule.workflow_id} not found")

            if schedule.variables:
                workflow.variables.update(schedule.variables)

            result = self.workflow_engine.execute_workflow(workflow)
            status = result.status

        except Exception as e:
            logger.error(f"Scheduled execution failed: {e}")
            status = "failed"
            error_msg = str(e)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Update schedule stats
        schedule.last_run = start_time
        schedule.run_count += 1

        # Update next run time
        job = self.scheduler.get_job(f"schedule_{schedule_id}")
        if job and job.next_run_time:
            schedule.next_run = job.next_run_time

        self._save_schedule(schedule)
        self._schedules[schedule_id] = schedule

        # Log execution
        self._save_execution_log(
            ScheduleExecutionLog(
                schedule_id=schedule_id,
                workflow_id=schedule.workflow_id,
                executed_at=start_time,
                status=status,
                duration_seconds=duration,
                error=error_msg,
            )
        )

    def _save_execution_log(self, log: ScheduleExecutionLog):
        """Save execution log entry to database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    INSERT INTO schedule_logs
                    (schedule_id, workflow_id, executed_at, status, duration_seconds, error)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log.schedule_id,
                        log.workflow_id,
                        log.executed_at.isoformat(),
                        log.status,
                        log.duration_seconds,
                        log.error,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to save execution log: {e}")

    def create_schedule(self, schedule: WorkflowSchedule) -> bool:
        """Create a new schedule."""
        # Validate workflow exists
        workflow = self.workflow_engine.load_workflow(schedule.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {schedule.workflow_id} not found")

        schedule.created_at = datetime.now()
        if self._save_schedule(schedule):
            self._schedules[schedule.id] = schedule
            if schedule.status == ScheduleStatus.ACTIVE:
                self._register_job(schedule)
            return True
        return False

    def update_schedule(self, schedule: WorkflowSchedule) -> bool:
        """Update an existing schedule."""
        if schedule.id not in self._schedules:
            raise ValueError(f"Schedule {schedule.id} not found")

        # Preserve creation time and run count
        existing = self._schedules[schedule.id]
        schedule.created_at = existing.created_at
        schedule.run_count = existing.run_count
        schedule.last_run = existing.last_run

        if self._save_schedule(schedule):
            self._schedules[schedule.id] = schedule
            self._register_job(schedule)  # Re-register with new config
            return True
        return False

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        if schedule_id not in self._schedules:
            return False

        # Remove from scheduler
        job_id = f"schedule_{schedule_id}"
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)

        # Remove from database (logs cascade due to FK)
        try:
            with self.backend.transaction():
                self.backend.execute(
                    "DELETE FROM schedule_logs WHERE schedule_id = ?", (schedule_id,)
                )
                self.backend.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        except Exception as e:
            logger.error(f"Failed to delete schedule {schedule_id} from database: {e}")

        del self._schedules[schedule_id]
        return True

    def get_schedule(self, schedule_id: str) -> WorkflowSchedule | None:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)

    def list_schedules(self) -> list[dict]:
        """List all schedules."""
        schedules = []
        for schedule in self._schedules.values():
            # Update next run time from scheduler
            job = self.scheduler.get_job(f"schedule_{schedule.id}")
            if job and job.next_run_time:
                schedule.next_run = job.next_run_time

            schedules.append(
                {
                    "id": schedule.id,
                    "name": schedule.name,
                    "workflow_id": schedule.workflow_id,
                    "schedule_type": schedule.schedule_type.value,
                    "status": schedule.status.value,
                    "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                    "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                    "run_count": schedule.run_count,
                }
            )
        return schedules

    def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        schedule.status = ScheduleStatus.PAUSED
        job_id = f"schedule_{schedule_id}"
        if self.scheduler.get_job(job_id):
            self.scheduler.pause_job(job_id)

        self._save_schedule(schedule)
        return True

    def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        schedule.status = ScheduleStatus.ACTIVE
        job_id = f"schedule_{schedule_id}"
        job = self.scheduler.get_job(job_id)

        if job:
            self.scheduler.resume_job(job_id)
        else:
            self._register_job(schedule)

        # Update next run time
        job = self.scheduler.get_job(job_id)
        if job and job.next_run_time:
            schedule.next_run = job.next_run_time

        self._save_schedule(schedule)
        return True

    def trigger_now(self, schedule_id: str) -> bool:
        """Manually trigger a scheduled workflow immediately."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        # Execute in background
        self._execute_scheduled_workflow(schedule_id)
        return True

    def get_execution_history(
        self, schedule_id: str, limit: int = 10
    ) -> list[ScheduleExecutionLog]:
        """Get execution history for a schedule."""
        rows = self.backend.fetchall(
            """
            SELECT * FROM schedule_logs
            WHERE schedule_id = ?
            ORDER BY executed_at DESC
            LIMIT ?
            """,
            (schedule_id, limit),
        )

        logs = []
        for row in rows:
            try:
                logs.append(
                    ScheduleExecutionLog(
                        schedule_id=row["schedule_id"],
                        workflow_id=row["workflow_id"],
                        executed_at=_parse_datetime(row.get("executed_at")) or datetime.now(),
                        status=row["status"],
                        duration_seconds=row.get("duration_seconds", 0),
                        error=row.get("error"),
                    )
                )
            except Exception as e:
                logger.error(f"Failed to parse execution log: {e}")
                continue

        return logs
