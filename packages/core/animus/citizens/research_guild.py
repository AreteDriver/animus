"""Research Guild Orchestrator.

End-to-end pipeline that chains all five Research Guild citizens:

    Harvester → Abstraction → Pattern → First-Principles → Architecture

Usage::

    orchestrator = ResearchGuildOrchestrator(memory_layer=mem, codebase_path=".")
    report = orchestrator.run_pipeline(target="fastapi/fastapi")
    print(report.summary())

Each stage's output is passed as input to the next stage.
Intermediate outputs are optionally stored in Animus memory.
The final output is a concrete Improvement Proposal ready for
human review and Forge execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.citizens.proposal import ImprovementProposal
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.memory import MemoryLayer

logger = get_logger("citizens.research_guild")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    citizen_name: str = ""
    outputs_count: int = 0
    stored_count: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class GuildPipelineReport:
    """Unified report produced after running the full Research Guild pipeline."""

    stages: list[StageResult] = field(default_factory=list)
    final_proposal: ImprovementProposal | None = None
    lineage: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_stages(self) -> int:
        return len(self.stages)

    @property
    def total_outputs(self) -> int:
        return sum(s.outputs_count for s in self.stages)

    @property
    def total_errors(self) -> int:
        return len(self.errors) + sum(len(s.errors) for s in self.stages)

    def summary(self) -> str:
        parts = [
            f"Research Guild Pipeline: {self.total_stages} stage(s), {self.total_outputs} output(s)",
        ]
        for s in self.stages:
            parts.append(
                f"  - {s.citizen_name}: {s.outputs_count} output(s), {s.stored_count} stored, {len(s.errors)} error(s), {s.duration_seconds:.1f}s"
            )
        if self.final_proposal:
            parts.append(f"Final proposal: {self.final_proposal.title} ({self.final_proposal.id})")
        if self.total_errors:
            parts.append(f"Total errors: {self.total_errors}")
        parts.append(f"Total duration: {self.duration_seconds:.1f}s")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [
                {
                    "citizen_name": s.citizen_name,
                    "outputs_count": s.outputs_count,
                    "stored_count": s.stored_count,
                    "errors": s.errors,
                    "duration_seconds": s.duration_seconds,
                }
                for s in self.stages
            ],
            "final_proposal_id": self.final_proposal.id if self.final_proposal else None,
            "final_proposal_title": self.final_proposal.title if self.final_proposal else None,
            "lineage": self.lineage,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════
# Research Guild Orchestrator
# ═══════════════════════════════════════════════════════════════════


class ResearchGuildOrchestrator:
    """Orchestrates the full Research Guild pipeline end-to-end.

    Usage::

        orchestrator = ResearchGuildOrchestrator(memory_layer=mem)
        report = orchestrator.run_pipeline(target="fastapi/fastapi")
        print(report.summary())
    """

    def __init__(
        self,
        memory_layer: MemoryLayer | None = None,
        codebase_path: Path | str = ".",
        evidence_dir: Path | str | None = None,
    ):
        self.memory = memory_layer
        self.codebase_path = Path(codebase_path).expanduser()
        self.evidence_dir = Path(evidence_dir).expanduser() if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(
        self,
        target: str = "",
        skip_harvester: bool = False,
        store_outputs: bool = False,
    ) -> GuildPipelineReport:
        """Run the full Research Guild pipeline.

        Args:
            target: GitHub repo target for Harvester (e.g., "fastapi/fastapi").
                Ignored if skip_harvester=True.
            skip_harvester: If True, skip the Harvester stage and use
                existing sources from memory.
            store_outputs: If True, store all intermediate outputs in memory.

        Returns:
            GuildPipelineReport with stage results and final proposal.
        """
        pipeline_start = time.time()
        report = GuildPipelineReport()
        lineage: list[str] = []

        # Resolve memory for storage
        effective_memory = self.memory if store_outputs else None

        try:
            # ── Stage 1: Harvester ──────────────────────────────────────
            stage1 = self._run_harvester(
                target=target, skip=skip_harvester, memory=effective_memory
            )
            report.stages.append(stage1)
            if stage1.outputs_count > 0:
                lineage.append(f"harvester:{stage1.outputs_count}")
            if stage1.errors:
                report.errors.extend([f"[Harvester] {e}" for e in stage1.errors])

            # ── Stage 2: Abstraction ────────────────────────────────────
            stage2 = self._run_abstraction(memory=effective_memory)
            report.stages.append(stage2)
            if stage2.outputs_count > 0:
                lineage.append(f"abstraction:{stage2.outputs_count}")
            if stage2.errors:
                report.errors.extend([f"[Abstraction] {e}" for e in stage2.errors])

            # ── Stage 3: Pattern ──────────────────────────────────────
            stage3 = self._run_pattern(memory=effective_memory)
            report.stages.append(stage3)
            if stage3.outputs_count > 0:
                lineage.append(f"pattern:{stage3.outputs_count}")
            if stage3.errors:
                report.errors.extend([f"[Pattern] {e}" for e in stage3.errors])

            # ── Stage 4: First-Principles ─────────────────────────────
            stage4 = self._run_first_principles(memory=effective_memory)
            report.stages.append(stage4)
            if stage4.outputs_count > 0:
                lineage.append(f"first_principles:{stage4.outputs_count}")
            if stage4.errors:
                report.errors.extend([f"[First-Principles] {e}" for e in stage4.errors])

            # ── Stage 5: Architecture ───────────────────────────────
            stage5 = self._run_architecture(memory=effective_memory)
            report.stages.append(stage5)
            if stage5.outputs_count > 0:
                lineage.append(f"architecture:{stage5.outputs_count}")
            if stage5.errors:
                report.errors.extend([f"[Architecture] {e}" for e in stage5.errors])

            report.final_proposal = stage5.outputs_count > 0
            report.lineage = lineage

        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            report.errors.append(f"Pipeline fatal error: {e}")

        report.duration_seconds = time.time() - pipeline_start
        logger.info(
            "Research Guild pipeline completed in %.1fs: %s",
            report.duration_seconds,
            report.summary(),
        )
        return report

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_harvester(
        self,
        target: str = "",
        skip: bool = False,
        memory: Any = None,
    ) -> StageResult:
        """Run Harvester stage."""
        from animus.citizens import HarvesterCitizen

        start = time.time()
        result = StageResult(citizen_name="harvester")

        if skip:
            logger.info("Harvester stage skipped per --skip-harvester")
            return result

        try:
            harvester = HarvesterCitizen(memory_layer=memory)
            if target:
                # Harvest a specific repo
                from animus.lugh.repos import harvest_repo

                harvest_result = harvest_repo(target=target, compare=False, depth="quick")
                if harvest_result:
                    source = harvester._source_from_harvest_result(target, harvest_result)
                    if source:
                        harvester.store_source(source)
                        result.outputs_count = 1
                        result.stored_count = 1
            else:
                # Observe codebase and memory for sources
                sources = harvester.observe_codebase()
                if memory:
                    mem_sources = harvester.observe_memory()
                    sources.extend(mem_sources)
                result.outputs_count = len(sources)
                if memory:
                    for src in sources:
                        harvester.store_source(src)
                    result.stored_count = result.outputs_count

        except Exception as e:
            logger.warning("Harvester stage failed: %s", e)
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    def _run_abstraction(self, memory: Any = None) -> StageResult:
        """Run Abstraction stage."""
        from animus.citizens import AbstractionCitizen

        start = time.time()
        result = StageResult(citizen_name="abstraction")

        try:
            abstraction = AbstractionCitizen(memory_layer=memory, codebase_path=self.codebase_path)

            # Gather sources: from memory (harvester outputs) + codebase
            sources: list[dict[str, Any]] = []
            if memory:
                mem_results = abstraction.observe_harvested_sources()
                sources.extend(mem_results)
            codebase_obs = abstraction.observe_codebase()
            sources.extend(codebase_obs)

            # Extract mechanisms from all sources
            mechanisms: list = []
            for src in sources:
                content = src.get("context", {}).get("content", "")
                sid = src.get("context", {}).get("identifier", "")
                if content:
                    mechs = abstraction.extract_mechanisms(content, sid)
                    mechanisms.extend(mechs)

            result.outputs_count = len(mechanisms)

            # Store mechanisms
            if memory:
                for m in mechanisms:
                    abstraction.store_mechanism(m)
                result.stored_count = len(mechanisms)

        except Exception as e:
            logger.warning("Abstraction stage failed: %s", e)
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    def _run_pattern(self, memory: Any = None) -> StageResult:
        """Run Pattern stage."""
        from animus.citizens import PatternCitizen

        start = time.time()
        result = StageResult(citizen_name="pattern")

        try:
            pattern = PatternCitizen(memory_layer=memory, codebase_path=self.codebase_path)

            # Observe mechanisms from memory
            mechanisms = pattern.observe_mechanisms()
            mech_contexts = [m["context"] for m in mechanisms]

            # Discover patterns
            patterns = pattern.discover_patterns(mech_contexts)
            result.outputs_count = len(patterns)

            # Store patterns
            if memory:
                for p in patterns:
                    pattern.store_pattern(p)
                result.stored_count = len(patterns)

        except Exception as e:
            logger.warning("Pattern stage failed: %s", e)
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    def _run_first_principles(self, memory: Any = None) -> StageResult:
        """Run First-Principles stage."""
        from animus.citizens import FirstPrinciplesCitizen

        start = time.time()
        result = StageResult(citizen_name="first_principles")

        try:
            fp = FirstPrinciplesCitizen(memory_layer=memory, codebase_path=self.codebase_path)

            # Observe patterns from memory
            patterns = fp.observe_patterns()
            pattern_contexts = [p["context"] for p in patterns]

            # Reduce to principles
            principles = fp.reduce_to_principles(pattern_contexts)
            result.outputs_count = len(principles)

            # Store principles
            if memory:
                for pr in principles:
                    fp.store_principle(pr)
                result.stored_count = len(principles)

        except Exception as e:
            logger.warning("First-Principles stage failed: %s", e)
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    def _run_architecture(self, memory: Any = None) -> StageResult:
        """Run Architecture stage."""
        from animus.citizens import ArchitectureCitizen

        start = time.time()
        result = StageResult(citizen_name="architecture")

        try:
            arch = ArchitectureCitizen(memory_layer=memory, codebase_path=self.codebase_path)

            # Observe principles from memory
            principles = arch.observe_principles()
            principle_contexts = [p["context"] for p in principles]

            # Analyze gaps
            gaps = arch.analyze_gaps(principle_contexts)
            result.outputs_count = len(gaps)

            # Store gaps and generate proposal
            if memory:
                for g in gaps:
                    arch.store_gap(g)
                result.stored_count = len(gaps)

            # Generate final proposal
            proposal = arch.generate_proposal(gaps)
            if proposal and memory:
                arch.store_proposal(proposal)
                result.stored_count += 1

        except Exception as e:
            logger.warning("Architecture stage failed: %s", e)
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ResearchGuildOrchestrator(codebase={self.codebase_path})"
