"""Planner Citizen — decomposes a mission objective into a task graph.

The planner never modifies code.  It produces a structured plan with:
- scope
- task graph
- acceptance criteria
- risk classification
- required reviewers
- expected files
- rollback strategy
"""

from __future__ import annotations

from uuid import UUID, uuid4

from animus_forge.citizens.base import Citizen
from animus_forge.missions.domain import CitizenOutput, Task, TaskContext, TaskStatus


class PlannerCitizen(Citizen):
    """Decomposes missions into task graphs."""

    role = "planner"
    capabilities = {"planning", "decomposition", "risk_analysis"}
    can_modify_code = False
    can_approve = False

    def run(self, task: Task, context: TaskContext) -> CitizenOutput:
        """Produce a plan for the mission described in *context*.

        In a production implementation this would invoke a frontier model
        (Claude / Kimi) to reason about scope and decomposition.  For Phase 4
        the logic is deterministic so that evals and tests do not require a
        live provider.
        """
        objective = context.mission_objective

        # Deterministic decomposition based on objective keywords.
        # Real implementation would use an LLM here.
        plan_tasks = self._decompose(objective)

        return CitizenOutput(
            status="completed",
            summary=f"Planned {len(plan_tasks)} tasks for: {objective}",
            artifacts=[],
            evidence=[
                {
                    "type": "plan",
                    "task_count": len(plan_tasks),
                    "tasks": plan_tasks,
                }
            ],
            risks=[],
            follow_up_tasks=[],
            confidence=0.85,
        )

    def _decompose(self, objective: str) -> list[dict]:
        """Return a deterministic task decomposition.

        This is a placeholder.  A real planner would use an LLM to analyse
        the repository and produce an appropriate graph.
        """
        # Default three-citizen pipeline
        return [
            {
                "citizen_role": "builder",
                "description": f"Implement: {objective}",
                "dependencies": [],
            },
            {
                "citizen_role": "reviewer",
                "description": f"Review implementation of: {objective}",
                "dependencies": ["builder"],  # resolved later to UUIDs
            },
        ]

    @staticmethod
    def create_task_graph(
        mission_id: UUID,
        plan_tasks: list[dict],
    ) -> list[Task]:
        """Convert a plan into concrete Task objects with resolved dependencies.

        Args:
            mission_id: Parent mission UUID.
            plan_tasks: List of task dicts from ``_decompose``.

        Returns:
            Ordered list of ``Task`` objects.  Builder tasks come first;
            reviewer tasks depend on them.
        """
        tasks: list[Task] = []
        role_to_id: dict[str, UUID] = {}

        for pt in plan_tasks:
            t = Task(
                mission_id=mission_id,
                citizen_role=pt["citizen_role"],
                description=pt["description"],
                status=TaskStatus.PENDING,
            )
            role_to_id[pt["citizen_role"]] = t.task_id
            tasks.append(t)

        # Resolve dependency references (by role name → UUID)
        for i, pt in enumerate(plan_tasks):
            deps = pt.get("dependencies", [])
            resolved = [role_to_id[d] for d in deps if d in role_to_id]
            tasks[i].dependencies = resolved

        return tasks
