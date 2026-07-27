"""Tests for the three-citizen runtime (Planner, Builder, Reviewer)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from animus_forge.citizens.builder import BuilderCitizen
from animus_forge.citizens.planner import PlannerCitizen
from animus_forge.citizens.reviewer import ReviewerCitizen
from animus_forge.missions.domain import (
    CitizenOutput,
    Mission,
    Task,
    TaskContext,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_mission():
    return Mission(
        repository="AreteDriver/animus",
        objective="Fix off-by-one pagination bug",
    )


@pytest.fixture()
def sample_task(sample_mission):
    return Task(
        mission_id=sample_mission.mission_id,
        citizen_role="builder",
        description="Implement fix for pagination bug",
    )


@pytest.fixture()
def sample_context(sample_mission):
    return TaskContext(
        mission_objective=sample_mission.objective,
        task_description="Implement fix",
        repository=sample_mission.repository,
        allowed_paths=["src/**/*.py", "tests/**/*.py"],
        protected_paths=[".github/workflows/**", "migrations/**"],
    )


# ---------------------------------------------------------------------------
# Planner Citizen
# ---------------------------------------------------------------------------


class TestPlannerCitizen:
    def test_role_and_capabilities(self):
        p = PlannerCitizen()
        assert p.role == "planner"
        assert "planning" in p.capabilities
        assert p.can_modify_code is False
        assert p.can_approve is False

    def test_run_returns_output(self, sample_task, sample_context):
        p = PlannerCitizen()
        out = p.run(sample_task, sample_context)
        assert isinstance(out, CitizenOutput)
        assert out.status == "completed"
        assert "Planned" in out.summary
        assert out.confidence > 0

    def test_decompose_returns_tasks(self):
        p = PlannerCitizen()
        tasks = p._decompose("Fix bug")
        assert len(tasks) >= 2
        roles = [t["citizen_role"] for t in tasks]
        assert "builder" in roles
        assert "reviewer" in roles

    def test_create_task_graph(self, sample_mission):
        p = PlannerCitizen()
        plan = p._decompose("Fix bug")
        tasks = p.create_task_graph(sample_mission.mission_id, plan)
        assert len(tasks) == len(plan)
        # Builder should have no dependencies
        builder_tasks = [t for t in tasks if t.citizen_role == "builder"]
        assert len(builder_tasks) == 1
        assert builder_tasks[0].dependencies == []
        # Reviewer should depend on builder
        reviewer_tasks = [t for t in tasks if t.citizen_role == "reviewer"]
        assert len(reviewer_tasks) == 1
        assert len(reviewer_tasks[0].dependencies) == 1
        assert reviewer_tasks[0].dependencies[0] == builder_tasks[0].task_id


# ---------------------------------------------------------------------------
# Builder Citizen
# ---------------------------------------------------------------------------


class TestBuilderCitizen:
    def test_role_and_capabilities(self):
        b = BuilderCitizen()
        assert b.role == "builder"
        assert "implementation" in b.capabilities
        assert b.can_modify_code is True
        assert b.can_approve is False

    def test_run_returns_output(self, sample_task, sample_context):
        b = BuilderCitizen()
        out = b.run(sample_task, sample_context)
        assert isinstance(out, CitizenOutput)
        assert out.status == "completed"
        assert len(out.changed_files) > 0
        assert out.evidence[0]["protected_paths_checked"] is True

    def test_path_allowlist_blocks_outside(self, sample_task):
        ctx = TaskContext(
            mission_objective="Fix bug",
            task_description="Build",
            repository="r",
            allowed_paths=["src/**"],
            relevant_files=["src/main.py", "docs/readme.md"],
        )
        b = BuilderCitizen()
        out = b.run(sample_task, ctx)
        assert out.status == "failed"
        assert "docs/readme.md" in out.summary

    def test_path_protected_blocks(self, sample_task):
        ctx = TaskContext(
            mission_objective="Fix bug",
            task_description="Build",
            repository="r",
            protected_paths=[".github/**"],
            relevant_files=[".github/workflows/ci.yml"],
        )
        b = BuilderCitizen()
        out = b.run(sample_task, ctx)
        assert out.status == "failed"
        assert "ci.yml" in out.summary

    def test_empty_allowlist_allows_non_protected(self, sample_task):
        ctx = TaskContext(
            mission_objective="Fix bug",
            task_description="Build",
            repository="r",
            allowed_paths=[],
            protected_paths=["migrations/**"],
            relevant_files=["src/main.py"],
        )
        b = BuilderCitizen()
        out = b.run(sample_task, ctx)
        assert out.status == "completed"

    def test_simulate_changes_fix_bug(self, sample_task, sample_context):
        b = BuilderCitizen()
        files = b._simulate_changes(sample_task, sample_context)
        assert "src/pagination.py" in files
        assert "tests/test_pagination.py" in files

    def test_match_pattern(self):
        b = BuilderCitizen()
        assert b._match_pattern("src/main.py", "src/**") is True
        assert b._match_pattern("src/main.py", "src/*.py") is True
        assert b._match_pattern("src/main.py", "tests/**") is False
        assert b._match_pattern(".github/workflows/ci.yml", ".github/**") is True


# ---------------------------------------------------------------------------
# Reviewer Citizen
# ---------------------------------------------------------------------------


class TestReviewerCitizen:
    def test_role_and_capabilities(self):
        r = ReviewerCitizen()
        assert r.role == "reviewer"
        assert "review" in r.capabilities
        assert r.can_modify_code is False
        assert r.can_approve is False

    def test_approve_clean_submission(self, sample_mission):
        r = ReviewerCitizen()
        task = Task(
            mission_id=sample_mission.mission_id,
            citizen_role="reviewer",
            description="Review",
        )
        ctx = TaskContext(
            mission_objective="Fix bug",
            task_description="Review",
            repository="r",
            prior_attempts=[
                {
                    "type": "build",
                    "changed_files": ["src/pagination.py"],
                }
            ],
            protected_paths=[".github/**"],
        )
        out = r.run(task, ctx)
        assert out.status == "completed"
        assert "passed" in out.summary.lower()

    def test_reject_protected_path(self, sample_mission):
        r = ReviewerCitizen()
        task = Task(
            mission_id=sample_mission.mission_id,
            citizen_role="reviewer",
            description="Review",
        )
        ctx = TaskContext(
            mission_objective="Fix bug",
            task_description="Review",
            repository="r",
            prior_attempts=[
                {
                    "type": "build",
                    "changed_files": [".github/workflows/ci.yml"],
                }
            ],
            protected_paths=[".github/**"],
        )
        out = r.run(task, ctx)
        assert out.status == "needs_repair"
        assert "rejected" in out.summary.lower()
        assert len(out.follow_up_tasks) > 0

    def test_reject_unsafe(self, sample_mission):
        r = ReviewerCitizen()
        task = Task(
            mission_id=sample_mission.mission_id,
            citizen_role="reviewer",
            description="Review unsafe change",
        )
        ctx = TaskContext(
            mission_objective="Fix bug",
            task_description="Review",
            repository="r",
            prior_attempts=[],
            protected_paths=[],
        )
        out = r.run(task, ctx)
        assert out.status == "needs_repair"
        assert "Unsafe" in out.risks[0]["description"]

    def test_reject_too_many_files(self, sample_mission):
        r = ReviewerCitizen()
        task = Task(
            mission_id=sample_mission.mission_id,
            citizen_role="reviewer",
            description="Review",
        )
        files = [f"src/file_{i}.py" for i in range(20)]
        ctx = TaskContext(
            mission_objective="Fix bug",
            task_description="Review",
            repository="r",
            prior_attempts=[{"type": "build", "changed_files": files}],
            protected_paths=[],
        )
        out = r.run(task, ctx)
        assert out.status == "needs_repair"
        assert "Too many files" in out.risks[0]["description"]
