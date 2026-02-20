"""Async job execution manager for background workflow runs."""

import json
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from animus_forge.config import get_settings
from animus_forge.orchestrator import WorkflowEngineAdapter, WorkflowResult
from animus_forge.state import DatabaseBackend, get_database

logger = logging.getLogger(__name__)


def _parse_datetime(value) -> datetime | None:
    """Parse datetime from database (handles both strings and datetime objects)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """An async job representing a workflow execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Job ID")
    workflow_id: str = Field(..., description="Workflow being executed")
    status: JobStatus = Field(JobStatus.PENDING, description="Current status")
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation time")
    started_at: datetime | None = Field(None, description="Execution start time")
    completed_at: datetime | None = Field(None, description="Execution end time")
    variables: dict[str, Any] = Field(
        default_factory=dict, description="Variables passed to workflow"
    )
    result: dict[str, Any] | None = Field(None, description="Workflow result")
    error: str | None = Field(None, description="Error message if failed")
    progress: str | None = Field(None, description="Progress message")


class JobManager:
    """Manages async workflow execution with status polling."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            variables TEXT,
            result TEXT,
            error TEXT,
            progress TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_workflow ON jobs(workflow_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
    """

    def __init__(
        self,
        backend: DatabaseBackend | None = None,
        max_workers: int = 4,
        execution_manager=None,
    ):
        self.settings = get_settings()
        self.backend = backend or get_database()
        self.workflow_engine = WorkflowEngineAdapter(execution_manager=execution_manager)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._init_schema()
        self._load_recent_jobs()

    def _init_schema(self):
        """Initialize database schema."""
        self.backend.executescript(self.SCHEMA)

    def _load_recent_jobs(self, limit: int = 100):
        """Load recent jobs from database on startup."""
        rows = self.backend.fetchall(
            """
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            try:
                job = Job(
                    id=row["id"],
                    workflow_id=row["workflow_id"],
                    status=JobStatus(row["status"]),
                    created_at=_parse_datetime(row.get("created_at")),
                    started_at=_parse_datetime(row.get("started_at")),
                    completed_at=_parse_datetime(row.get("completed_at")),
                    variables=json.loads(row["variables"]) if row["variables"] else {},
                    result=json.loads(row["result"]) if row["result"] else None,
                    error=row["error"],
                    progress=row["progress"],
                )
                # Mark running jobs as failed (server restart)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.error = "Server restarted during execution"
                    job.completed_at = datetime.now()
                    self._update_job_in_db(job)
                self._jobs[job.id] = job
            except Exception as e:
                logger.error(f"Failed to load job from row: {e}")

    def _save_job(self, job: Job) -> bool:
        """Save job to database (insert or update)."""
        try:
            existing = self.backend.fetchone("SELECT id FROM jobs WHERE id = ?", (job.id,))
            if existing:
                return self._update_job_in_db(job)
            else:
                return self._insert_job_in_db(job)
        except Exception as e:
            logger.error(f"Failed to save job {job.id}: {e}")
            return False

    def _insert_job_in_db(self, job: Job) -> bool:
        """Insert a new job into the database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    INSERT INTO jobs
                    (id, workflow_id, status, created_at, started_at, completed_at,
                     variables, result, error, progress)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.id,
                        job.workflow_id,
                        job.status.value,
                        job.created_at.isoformat() if job.created_at else None,
                        job.started_at.isoformat() if job.started_at else None,
                        job.completed_at.isoformat() if job.completed_at else None,
                        json.dumps(job.variables) if job.variables else None,
                        json.dumps(job.result) if job.result else None,
                        job.error,
                        job.progress,
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to insert job {job.id}: {e}")
            return False

    def _update_job_in_db(self, job: Job) -> bool:
        """Update an existing job in the database."""
        try:
            with self.backend.transaction():
                self.backend.execute(
                    """
                    UPDATE jobs
                    SET workflow_id = ?, status = ?, started_at = ?, completed_at = ?,
                        variables = ?, result = ?, error = ?, progress = ?
                    WHERE id = ?
                    """,
                    (
                        job.workflow_id,
                        job.status.value,
                        job.started_at.isoformat() if job.started_at else None,
                        job.completed_at.isoformat() if job.completed_at else None,
                        json.dumps(job.variables) if job.variables else None,
                        json.dumps(job.result) if job.result else None,
                        job.error,
                        job.progress,
                        job.id,
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job.id}: {e}")
            return False

    def _execute_workflow(self, job_id: str):
        """Execute workflow in background thread."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status == JobStatus.CANCELLED:
                return

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            job.progress = "Loading workflow..."
            self._save_job(job)

        try:
            workflow = self.workflow_engine.load_workflow(job.workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {job.workflow_id} not found")

            with self._lock:
                job.progress = "Executing workflow..."
                self._save_job(job)

            if job.variables:
                workflow.variables.update(job.variables)

            result: WorkflowResult = self.workflow_engine.execute_workflow(workflow)

            with self._lock:
                job.status = (
                    JobStatus.COMPLETED if result.status == "completed" else JobStatus.FAILED
                )
                job.result = result.model_dump(mode="json")
                job.completed_at = datetime.now()
                job.progress = None
                if result.errors:
                    job.error = "; ".join(result.errors)
                self._save_job(job)

            self._record_task_history(job)

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
                job.progress = None
                self._save_job(job)

            self._record_task_history(job)

    def _record_task_history(self, job: Job) -> None:
        """Record task completion/failure to analytics history."""
        try:
            from animus_forge.db import get_task_store

            result = job.result or {}
            duration_ms = 0
            if job.started_at and job.completed_at:
                duration_ms = int((job.completed_at - job.started_at).total_seconds() * 1000)

            get_task_store().record_task(
                job_id=job.id,
                workflow_id=job.workflow_id,
                status=job.status.value,
                agent_role=result.get("agent_role"),
                model=result.get("model"),
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                cost_usd=result.get("cost_usd", 0.0),
                duration_ms=duration_ms,
                error=job.error,
                created_at=job.created_at,
                completed_at=job.completed_at,
            )
        except Exception as e:
            logger.warning(f"Failed to record task history for job {job.id}: {e}")

    def submit(self, workflow_id: str, variables: dict | None = None) -> Job:
        """Submit a workflow for async execution."""
        # Validate workflow exists
        workflow = self.workflow_engine.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        job = Job(
            workflow_id=workflow_id,
            variables=variables or {},
        )

        with self._lock:
            self._jobs[job.id] = job
            self._save_job(job)

        future = self.executor.submit(self._execute_workflow, job.id)
        self._futures[job.id] = future

        logger.info(f"Submitted job {job.id} for workflow {workflow_id}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        workflow_id: str | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        if workflow_id:
            jobs = [j for j in jobs if j.workflow_id == workflow_id]

        # Sort by creation time, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                return False

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.error = "Cancelled by user"
            self._save_job(job)

        # Try to cancel the future (only works if not yet started)
        future = self._futures.get(job_id)
        if future:
            future.cancel()

        logger.info(f"Cancelled job {job_id}")
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a completed/failed/cancelled job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            # Don't delete running jobs
            if job.status == JobStatus.RUNNING:
                return False

            # Remove from memory
            del self._jobs[job_id]

            # Remove from database
            try:
                with self.backend.transaction():
                    self.backend.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            except Exception as e:
                logger.error(f"Failed to delete job {job_id} from database: {e}")

        return True

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove jobs older than max_age_hours."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        deleted = 0

        with self._lock:
            to_delete = []
            for job_id, job in self._jobs.items():
                if job.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    if job.completed_at and job.completed_at.timestamp() < cutoff:
                        to_delete.append(job_id)

            for job_id in to_delete:
                del self._jobs[job_id]
                deleted += 1

            # Delete from database
            if to_delete:
                try:
                    with self.backend.transaction():
                        self.backend.execute(
                            """
                            DELETE FROM jobs
                            WHERE status IN (?, ?, ?)
                            AND completed_at < ?
                            """,
                            (
                                JobStatus.COMPLETED.value,
                                JobStatus.FAILED.value,
                                JobStatus.CANCELLED.value,
                                cutoff_iso,
                            ),
                        )
                except Exception as e:
                    logger.error(f"Failed to cleanup old jobs from database: {e}")

        logger.info(f"Cleaned up {deleted} old jobs")
        return deleted

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)
        logger.info("Job manager shutdown")

    def get_stats(self) -> dict[str, int]:
        """Get job statistics."""
        stats = {
            "total": len(self._jobs),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for job in self._jobs.values():
            stats[job.status.value] += 1
        return stats
