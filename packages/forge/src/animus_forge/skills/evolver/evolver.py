"""SkillEvolver — top-level orchestrator for closed-loop skill improvement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage
from animus_forge.skills.library import SkillLibrary
from animus_forge.state.backends import DatabaseBackend

from .ab_test import ABTestManager
from .analyzer import SkillAnalyzer
from .deprecator import SkillDeprecator
from .generator import SkillGenerator
from .metrics import SkillMetricsAggregator
from .models import ExperimentConfig, SkillChange
from .tuner import SkillTuner
from .versioner import SkillVersioner
from .writer import SkillWriter

logger = logging.getLogger(__name__)


class SkillEvolver:
    """Closed-loop skill improvement orchestrator.

    Composes metrics aggregation, analysis, tuning, generation, deprecation,
    A/B testing, versioning, and filesystem writing into a single entry point.

    Args:
        backend: Database backend for metrics and experiment state.
        skills_dir: Root skills directory on disk.
        approval_gate: Optional approval gate for destructive operations.
        auto_approve: If True, skip approval for non-retirement changes.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        skills_dir: Path,
        approval_gate: ApprovalGate | None = None,
        auto_approve: bool = False,
    ) -> None:
        self._backend = backend
        self._skills_dir = skills_dir
        self._approval_gate = approval_gate or ApprovalGate()
        self._auto_approve = auto_approve

        # Compose sub-components
        self.aggregator = SkillMetricsAggregator(backend)
        self.analyzer = SkillAnalyzer(self.aggregator, backend)
        self.tuner = SkillTuner()
        self.generator = SkillGenerator()
        self.writer = SkillWriter(skills_dir)
        self.deprecator = SkillDeprecator(backend, self._approval_gate)
        self.versioner = SkillVersioner(backend)
        self.ab_manager = ABTestManager(backend, self.aggregator)

        # Load skill library for lookups
        try:
            self._library = SkillLibrary(skills_dir)
        except FileNotFoundError:
            self._library = None

    def run_evolution_cycle(self, days: int = 30) -> dict[str, Any]:
        """Execute a full evolution cycle.

        Steps:
        1. Compute and store metrics
        2. Analyze for problems
        3. Generate tuning proposals for declining skills
        4. Flag underperformers for deprecation
        5. Detect capability gaps
        6. Generate report

        Args:
            days: Look-back window.

        Returns:
            Evolution cycle report dict.
        """
        report: dict[str, Any] = {
            "metrics_computed": 0,
            "tuning_proposals": [],
            "deprecation_flags": [],
            "capability_gaps": [],
            "experiments_checked": [],
        }

        # 1. Compute metrics
        report["metrics_computed"] = self.aggregator.compute_and_store_metrics(days)

        # 2. Analyze
        analysis = self.analyzer.generate_analysis_report(days)
        report["analysis"] = analysis

        # 3. Tune declining skills
        for skill_name, metrics in self.analyzer.find_declining_skills(days):
            change = self.tuner.tune_declining_skill(skill_name, metrics)
            if change:
                report["tuning_proposals"].append(
                    {"skill": skill_name, "change": change.description}
                )

        # 4. Flag underperformers
        for skill_name, metrics in self.analyzer.find_underperformers(days=days):
            record = self.deprecator.flag_for_deprecation(
                skill_name,
                reason=f"Success rate {metrics.success_rate:.0%} below threshold",
                metrics=metrics,
            )
            report["deprecation_flags"].append({"skill": skill_name, "status": record.status})

        # 5. Detect gaps
        gaps = self.analyzer.detect_capability_gaps(days)
        for gap in gaps:
            report["capability_gaps"].append(
                {"description": gap.description, "confidence": gap.confidence}
            )

        # 6. Check active experiments
        report["experiments_checked"] = self.check_experiments()

        return report

    def apply_change(self, change: SkillChange) -> bool:
        """Apply a skill change through the safety pipeline.

        Pipeline: approval → write YAML → record version → update registry.

        Args:
            change: The change to apply.

        Returns:
            True if the change was applied.
        """
        # Approval gate (skip if auto_approve)
        if not self._auto_approve:
            request = self._approval_gate.request_approval(
                stage=ApprovalStage.APPLY,
                title=f"Apply {change.change_type} to {change.skill_name}",
                description=change.description,
                details={
                    "skill_name": change.skill_name,
                    "old_version": change.old_version,
                    "new_version": change.new_version,
                    "modifications": change.modifications,
                },
            )
            status = request.status
            if status.value != "approved":
                logger.info("Change to %s not approved: %s", change.skill_name, status)
                return False

        # Load current skill and apply modifications
        if self._library:
            skill = self._library.get_skill(change.skill_name)
            if skill:
                new_skill = self.tuner.apply_change_to_definition(skill, change)
                category = new_skill.category or "system"

                # Write to disk
                self.writer.write_skill(new_skill, category)
                self.writer.update_registry(new_skill, category)

                # Record version
                yaml_snapshot = self.writer.skill_to_yaml(new_skill)
                self.versioner.record_version(
                    skill_name=change.skill_name,
                    version=change.new_version,
                    previous_version=change.old_version,
                    change_type=change.change_type,
                    description=change.description,
                    yaml_snapshot=yaml_snapshot,
                )

                logger.info(
                    "Applied %s to %s: %s → %s",
                    change.change_type,
                    change.skill_name,
                    change.old_version,
                    change.new_version,
                )
                return True

        logger.warning("Could not apply change: skill %s not found", change.skill_name)
        return False

    def start_ab_test(
        self,
        skill_name: str,
        change: SkillChange,
        traffic_split: float = 0.5,
        min_invocations: int = 100,
    ) -> ExperimentConfig:
        """Start an A/B test for a proposed skill change.

        Writes the variant to disk, records its version, and creates an
        experiment.

        Args:
            skill_name: Skill being tested.
            change: The proposed change (variant).
            traffic_split: Fraction of traffic for variant.
            min_invocations: Minimum calls before conclusion.

        Returns:
            The experiment configuration.
        """
        # Write variant to disk
        if self._library:
            skill = self._library.get_skill(skill_name)
            if skill:
                variant = self.tuner.apply_change_to_definition(skill, change)
                category = variant.category or "system"
                self.writer.write_skill(variant, category)

                yaml_snapshot = self.writer.skill_to_yaml(variant)
                self.versioner.record_version(
                    skill_name=skill_name,
                    version=change.new_version,
                    previous_version=change.old_version,
                    change_type="experiment",
                    description=f"A/B test variant: {change.description}",
                    yaml_snapshot=yaml_snapshot,
                )

        return self.ab_manager.create_experiment(
            skill_name=skill_name,
            control_version=change.old_version,
            variant_version=change.new_version,
            traffic_split=traffic_split,
            min_invocations=min_invocations,
        )

    def check_experiments(self) -> list[dict]:
        """Evaluate all active experiments and conclude those with sufficient data.

        Returns:
            List of concluded experiment summaries.
        """
        concluded: list[dict] = []
        active = self.ab_manager.get_active_experiments()

        for exp in active:
            experiment_id = str(exp["id"])
            result = self.ab_manager.evaluate_experiment(experiment_id)
            if result:
                self.ab_manager.conclude_experiment(
                    experiment_id, result.winner, result.conclusion_reason
                )
                concluded.append(
                    {
                        "experiment_id": experiment_id,
                        "skill": result.skill_name,
                        "winner": result.winner,
                        "reason": result.conclusion_reason,
                    }
                )

        return concluded

    def route_skill(self, skill_name: str, workflow_id: str = "") -> str:
        """Route a skill invocation to the correct version (A/B test aware).

        Args:
            skill_name: Skill being invoked.
            workflow_id: Workflow ID for deterministic routing.

        Returns:
            Version string to use. Falls back to skill's current version.
        """
        # Check for active experiment
        version = self.ab_manager.route_skill_version(skill_name, workflow_id)
        if version:
            return version

        # Fall back to current version from library
        if self._library:
            skill = self._library.get_skill(skill_name)
            if skill:
                return skill.version

        return "1.0.0"

    def get_evolution_status(self) -> dict[str, Any]:
        """Get a summary of the evolution system's current state.

        Returns:
            Dict with active_experiments, flagged_skills, deprecated_skills.
        """
        return {
            "active_experiments": self.ab_manager.get_active_experiments(),
            "flagged_skills": [r.skill_name for r in self.deprecator.get_flagged_skills()],
            "deprecated_skills": [r.skill_name for r in self.deprecator.get_deprecated_skills()],
        }
