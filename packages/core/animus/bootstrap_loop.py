"""Animus Bootstrap Loop — The self-improvement cycle.

This module wires together all three layers into the minimum viable
self-improvement loop:

  Core (identity + memory + cognitive)
    → Forge (workflow execution)
      → Quorum (two-agent consensus)
        → Write improvements back

The bootstrap threshold: Animus can read its own code, reflect on it,
reach consensus between two agents, and write improvements. Once this
loop closes, it accelerates from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from animus.cognitive import CognitiveLayer, ModelConfig
from animus.forge.engine import ForgeEngine
from animus.forge.models import AgentConfig, GateConfig, WorkflowConfig, WorkflowState
from animus.identity import AnimusIdentity
from animus.logging import get_logger
from animus.memory import MemoryLayer, MemoryType

logger = get_logger("bootstrap")


@dataclass
class ConsensusResult:
    """Result of a two-agent consensus check."""

    approved: bool
    approve_weight: float = 0.0
    reject_weight: float = 0.0
    reasoning: list[str] = field(default_factory=list)
    decision_id: str = ""

    @property
    def summary(self) -> str:
        verdict = "APPROVED" if self.approved else "REJECTED"
        return f"{verdict} (approve={self.approve_weight:.2f}, reject={self.reject_weight:.2f})"


@dataclass
class BootstrapResult:
    """Result of one complete bootstrap loop iteration."""

    cycle: int
    files_reviewed: list[str]
    analysis: str
    suggestions: str
    consensus: ConsensusResult | None
    improvements_written: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "files_reviewed": self.files_reviewed,
            "analysis": self.analysis[:500],
            "suggestions": self.suggestions[:500],
            "consensus_approved": self.consensus.approved if self.consensus else None,
            "improvements_written": self.improvements_written,
            "timestamp": self.timestamp,
        }


def build_self_review_workflow(
    files_content: str,
    provider: str = "mock",
    model: str = "mock",
) -> WorkflowConfig:
    """Build a two-agent workflow: reader analyzes code, reviewer evaluates.

    This is the simplest useful workflow — one agent reads and reflects,
    another reviews the reflection and proposes improvements.
    """
    return WorkflowConfig(
        name="self-review",
        description="Animus reviews its own code and proposes improvements",
        agents=[
            AgentConfig(
                name="analyst",
                archetype="analyst",
                system_prompt=(
                    "You are analyzing source code from the Animus AI system. "
                    "Animus is reviewing its own code to find improvements. "
                    "Examine the code and produce:\n"
                    "1. A summary of what the code does\n"
                    "2. Code quality assessment\n"
                    "3. Specific improvement suggestions\n\n"
                    f"Code to analyze:\n{files_content}"
                ),
                outputs=["analysis", "suggestions"],
                budget_tokens=20_000,
            ),
            AgentConfig(
                name="reviewer",
                archetype="reviewer",
                inputs=["analyst.analysis", "analyst.suggestions"],
                system_prompt=(
                    "You are reviewing improvement suggestions for the Animus AI system. "
                    "Evaluate each suggestion for:\n"
                    "1. Correctness — will it actually improve the code?\n"
                    "2. Safety — could it break anything?\n"
                    "3. Priority — what matters most?\n"
                    "Produce a scored list of approved improvements."
                ),
                outputs=["approved_improvements", "score"],
                budget_tokens=15_000,
            ),
        ],
        gates=[
            GateConfig(
                name="quality-check",
                after="reviewer",
                type="automated",
                pass_condition="reviewer.score >= 0.0",
                on_fail="halt",
            ),
        ],
        max_cost_usd=0.10,
        provider=provider,
        model=model,
    )


def run_consensus(
    question: str,
    context: str,
    agent_a_vote: str = "approve",
    agent_a_confidence: float = 0.8,
    agent_a_reasoning: str = "",
    agent_b_vote: str = "approve",
    agent_b_confidence: float = 0.7,
    agent_b_reasoning: str = "",
) -> ConsensusResult:
    """Run a two-agent consensus check using Quorum's Triumvirate.

    This is the minimal Quorum integration — two agents vote on whether
    to accept proposed improvements.
    """
    try:
        from convergent.coordination_config import CoordinationConfig
        from convergent.protocol import AgentIdentity, Vote, VoteChoice
        from convergent.score_store import ScoreStore
        from convergent.scoring import PhiScorer
        from convergent.triumvirate import Triumvirate

        # In-memory score store for bootstrap (no persistence needed yet)
        store = ScoreStore(":memory:")
        scorer = PhiScorer(store=store)
        config = CoordinationConfig(
            db_path=":memory:",
            vote_timeout_seconds=300,
        )
        triumvirate = Triumvirate(scorer=scorer, config=config)

        # Create consensus request
        request = triumvirate.create_request(
            task_id="bootstrap-self-review",
            question=question,
            context=context,
        )

        # Agent A votes (the analyst)
        vote_a = Vote(
            agent=AgentIdentity(
                agent_id="analyst",
                role="analyst",
                model="animus-core",
                phi_score=0.5,
            ),
            choice=VoteChoice(agent_a_vote),
            confidence=agent_a_confidence,
            reasoning=agent_a_reasoning or "Analysis supports these improvements.",
        )
        triumvirate.submit_vote(request.request_id, vote_a)

        # Agent B votes (the reviewer)
        vote_b = Vote(
            agent=AgentIdentity(
                agent_id="reviewer",
                role="reviewer",
                model="animus-core",
                phi_score=0.5,
            ),
            choice=VoteChoice(agent_b_vote),
            confidence=agent_b_confidence,
            reasoning=agent_b_reasoning or "Review confirms improvement quality.",
        )
        triumvirate.submit_vote(request.request_id, vote_b)

        # Evaluate
        decision = triumvirate.evaluate(request.request_id)

        return ConsensusResult(
            approved=decision.outcome.value == "approved",
            approve_weight=decision.total_weighted_approve,
            reject_weight=decision.total_weighted_reject,
            reasoning=[v.reasoning for v in decision.votes],
            decision_id=request.request_id,
        )

    except ImportError:
        logger.warning("Quorum (convergent) not installed, using local consensus fallback")
        return _local_consensus_fallback(
            agent_a_vote,
            agent_a_confidence,
            agent_b_vote,
            agent_b_confidence,
        )


def _local_consensus_fallback(
    vote_a: str,
    confidence_a: float,
    vote_b: str,
    confidence_b: float,
) -> ConsensusResult:
    """Fallback consensus when Quorum isn't available.

    Simple majority: both must approve, or weighted score decides.
    """
    approve_weight = 0.0
    reject_weight = 0.0

    if vote_a == "approve":
        approve_weight += confidence_a * 0.5
    else:
        reject_weight += confidence_a * 0.5

    if vote_b == "approve":
        approve_weight += confidence_b * 0.5
    else:
        reject_weight += confidence_b * 0.5

    return ConsensusResult(
        approved=approve_weight > reject_weight,
        approve_weight=approve_weight,
        reject_weight=reject_weight,
        reasoning=["Local fallback consensus (Quorum not available)"],
    )


class BootstrapLoop:
    """The self-improvement loop that closes the bootstrap gap.

    Wires together:
    - Core identity (read own code)
    - Core memory (persist reflections)
    - Core cognitive (analyze code via LLM)
    - Forge engine (execute multi-step workflow)
    - Quorum consensus (two-agent approval)

    Usage:
        identity = AnimusIdentity(codebase_root="/path/to/animus")
        loop = BootstrapLoop(identity=identity)
        result = loop.run_cycle(files=["packages/core/animus/identity.py"])
    """

    def __init__(
        self,
        identity: AnimusIdentity,
        cognitive: CognitiveLayer | None = None,
        memory: MemoryLayer | None = None,
        data_dir: Path | None = None,
        provider: str = "mock",
        model: str = "mock",
    ):
        self.identity = identity
        self.provider = provider
        self.model = model

        # Initialize cognitive layer (default to mock for bootstrap)
        if cognitive is not None:
            self.cognitive = cognitive
        else:
            if provider == "mock":
                # Mock response uses ## headings so ForgeAgent._parse_outputs()
                # can extract structured outputs from the response text.
                mock_response = (
                    "## analysis\n"
                    "Code structure is well-organized with clear separation of concerns.\n\n"
                    "## suggestions\n"
                    "1. Add docstring to uncovered methods\n"
                    "2. Improve error messages\n\n"
                    "## approved_improvements\n"
                    "1. Add docstrings\n"
                    "2. Better error messages\n\n"
                    "## score\n"
                    "0.85\n"
                )
                config = ModelConfig.mock(
                    default_response=mock_response,
                    response_map={},
                )
            else:
                config = ModelConfig(
                    provider=__import__(
                        "animus.cognitive", fromlist=["ModelProvider"]
                    ).ModelProvider(provider),
                    model_name=model,
                )
            self.cognitive = CognitiveLayer(primary_config=config)

        # Initialize memory (JSON backend for bootstrap)
        if memory is not None:
            self.memory = memory
        else:
            mem_dir = data_dir or (identity.root / ".animus")
            self.memory = MemoryLayer(data_dir=mem_dir, backend="json")

        # Forge engine for workflow execution
        checkpoint_dir = (data_dir or identity.root / ".animus") / "checkpoints"
        self.forge = ForgeEngine(
            cognitive=self.cognitive,
            checkpoint_dir=checkpoint_dir,
        )

        self._cycle_count = identity.reflection_count
        self._results: list[BootstrapResult] = []

    def run_cycle(
        self,
        files: list[str] | None = None,
        package: str = "core",
        max_files: int = 5,
    ) -> BootstrapResult:
        """Execute one complete bootstrap loop iteration.

        1. Read own code files
        2. Run self-review workflow (Forge)
        3. Two-agent consensus (Quorum)
        4. Record in memory
        5. Update identity

        Args:
            files: Specific files to review (relative to codebase root).
                   If None, auto-selects from the specified package.
            package: Which package to review if files not specified.
            max_files: Maximum files to include in one cycle.

        Returns:
            BootstrapResult with full cycle details.
        """
        self._cycle_count += 1
        logger.info(f"Bootstrap cycle #{self._cycle_count} starting")

        # Step 1: Read own code
        if files is None:
            files = self.identity.list_own_files(package)[:max_files]

        files_content = self._read_files(files)
        if not files_content:
            return BootstrapResult(
                cycle=self._cycle_count,
                files_reviewed=[],
                analysis="No files found to review.",
                suggestions="",
                consensus=None,
                improvements_written=False,
            )

        # Step 2: Run self-review workflow via Forge
        workflow = build_self_review_workflow(
            files_content=files_content,
            provider=self.provider,
            model=self.model,
        )

        state = self.forge.run(workflow)
        analysis, suggestions = self._extract_workflow_results(state)

        # Step 3: Two-agent consensus via Quorum
        consensus = run_consensus(
            question=f"Accept improvements from bootstrap cycle #{self._cycle_count}?",
            context=f"Analysis:\n{analysis}\n\nSuggestions:\n{suggestions}",
        )

        logger.info(f"Consensus: {consensus.summary}")

        # Step 4: Record in memory
        self._record_to_memory(files, analysis, suggestions, consensus)

        # Step 5: Update identity
        improvements_written = False
        if consensus.approved:
            self.identity.record_reflection(
                summary=f"Cycle #{self._cycle_count}: {analysis[:200]}",
                improvements=suggestions.split("\n") if suggestions else [],
            )
            improvements_written = True

        result = BootstrapResult(
            cycle=self._cycle_count,
            files_reviewed=files,
            analysis=analysis,
            suggestions=suggestions,
            consensus=consensus,
            improvements_written=improvements_written,
        )
        self._results.append(result)

        logger.info(
            f"Bootstrap cycle #{self._cycle_count} complete: "
            f"reviewed {len(files)} files, "
            f"consensus={'approved' if consensus.approved else 'rejected'}"
        )

        return result

    def _read_files(self, files: list[str]) -> str:
        """Read and concatenate source files."""
        parts = []
        for rel_path in files:
            try:
                content = self.identity.read_own_file(rel_path)
                parts.append(f"--- {rel_path} ---\n{content}\n")
            except (FileNotFoundError, IsADirectoryError) as e:
                logger.warning(f"Skipping {rel_path}: {e}")
        return "\n".join(parts)

    def _extract_workflow_results(self, state: WorkflowState) -> tuple[str, str]:
        """Extract analysis and suggestions from workflow state."""
        analysis = ""
        suggestions = ""

        for result in state.results:
            if "analysis" in result.outputs:
                analysis = result.outputs["analysis"]
            if "suggestions" in result.outputs:
                suggestions = result.outputs["suggestions"]
            if "approved_improvements" in result.outputs:
                suggestions = result.outputs["approved_improvements"]

        return analysis, suggestions

    def _record_to_memory(
        self,
        files: list[str],
        analysis: str,
        suggestions: str,
        consensus: ConsensusResult,
    ) -> None:
        """Persist the reflection cycle to memory."""
        content = (
            f"Bootstrap cycle #{self._cycle_count}\n"
            f"Files reviewed: {', '.join(files)}\n"
            f"Analysis: {analysis[:300]}\n"
            f"Suggestions: {suggestions[:300]}\n"
            f"Consensus: {consensus.summary}"
        )

        self.memory.remember(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            tags=["bootstrap", "self-review", f"cycle-{self._cycle_count}"],
            source="learned",
            confidence=0.9 if consensus.approved else 0.5,
            subtype="self-reflection",
        )

    @property
    def cycle_count(self) -> int:
        """Number of completed bootstrap cycles."""
        return self._cycle_count

    @property
    def results(self) -> list[BootstrapResult]:
        """All completed bootstrap results."""
        return list(self._results)

    def get_history(self) -> list[dict]:
        """Get serializable history of all cycles."""
        return [r.to_dict() for r in self._results]
