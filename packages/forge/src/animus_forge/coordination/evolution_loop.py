"""Autoresearch-style evolution loop for Forge.

Implements an autonomous hypothesis → experiment → evaluate cycle,
gated by budget and constrained by a human-authored definition of
"better" (better.md).

Off by default. Enable with EvolutionConfig(enabled=True).
Requires better.md to exist and be non-empty — hard stop otherwise.

Reference: Karpathy autoresearch pattern mapped onto existing Forge
architecture (consciousness_bridge, budget_gate, workflow_evolution).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from animus_forge.budget.manager import BudgetManager, BudgetStatus
from animus_forge.coordination.identity_anchor import IdentityAnchor

if TYPE_CHECKING:
    from animus_forge.providers.base import Provider

logger = logging.getLogger(__name__)

# Constitutional principles loaded at init, fallback to defaults
_DEFAULT_PRINCIPLES = [
    "P4 Constraint: Forge may not modify Core directly.",
    "P5 Proportionality: Use minimum LLM capability required.",
    "P6 Budget Sovereignty: No LLM call without budget gate.",
    "P8 Jidoka: Any agent may halt the line on violation.",
]

HYPOTHESIS_SYSTEM_PROMPT = """You are the evolution engine of Animus Forge, a sovereign personal AI exocortex.

Your task: given a definition of "better" and prior iteration results, generate a single testable hypothesis for improvement.

Definition of "better" (authored by the human operator):
{better_definition}

Prior iterations (most recent first):
{prior_results}

Current principles:
{principles}

Budget remaining: {budget_remaining} tokens

Respond ONLY with valid JSON:
{{
  "hypothesis": "string — one specific, testable improvement hypothesis",
  "experiment_plan": "string — concrete steps to test this hypothesis",
  "expected_outcome": "string — what success looks like",
  "estimated_tokens": 0
}}"""

EVALUATE_SYSTEM_PROMPT = """You are the evaluation engine of Animus Forge.

Evaluate the experiment result against the definition of "better."

Definition of "better":
{better_definition}

Hypothesis tested:
{hypothesis}

Experiment plan:
{experiment_plan}

Experiment result:
{experiment_result}

Respond ONLY with valid JSON:
{{
  "outcome": "keep|discard",
  "rationale": "string — why this outcome",
  "confidence": 0.0,
  "suggestions_for_next": "string — what to try next based on this result"
}}"""


class EvolutionConfig(BaseModel):
    """Configuration for the evolution loop."""

    enabled: bool = False
    max_iterations: int = 10
    model: str = "claude-sonnet-4-6"
    estimated_tokens_per_iteration: int = 3000
    budget_pause_threshold: float = 0.80
    better_path: Path = Path("forge/better.md")
    audit_log_path: Path = Path("forge/forge_audit.jsonl")
    principles_path: Path | None = None


@dataclass
class IterationRecord:
    """Single iteration result for audit log."""

    iteration: int
    hypothesis: str
    experiment_summary: str
    outcome: str  # "keep" | "discard" | "error"
    rationale: str
    budget_used: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class EvolutionLoop:
    """Autoresearch-style evolution loop.

    Loop structure:
        load_better_md() →
        generate_hypothesis() →
        run_experiment() →
        evaluate_against_better() →
        keep_or_discard() →
        append_to_audit_jsonl() →
        repeat_or_halt()

    Constraints:
    - better.md must exist and be non-empty
    - All LLM calls through budget manager
    - Forge may never write to Core files directly
    - Audit log is append-only
    - Max iterations enforced
    """

    def __init__(
        self,
        provider: Provider,
        budget_manager: BudgetManager,
        config: EvolutionConfig | None = None,
        experiment_runner: Any | None = None,
        identity_anchor: IdentityAnchor | None = None,
    ):
        self._provider = provider
        self._budget = budget_manager
        self._config = config or EvolutionConfig()
        self._experiment_runner = experiment_runner
        self._identity_anchor = identity_anchor
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._iteration_count: int = 0
        self._total_tokens: int = 0
        self._history: list[IterationRecord] = []
        self._principles: list[str] = list(_DEFAULT_PRINCIPLES)
        self._better_definition: str = ""

        if self._config.principles_path and self._config.principles_path.exists():
            self._principles = self._load_principles(self._config.principles_path)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def history(self) -> list[IterationRecord]:
        return list(self._history)

    def start(self) -> None:
        """Start the evolution loop in a background thread."""
        if not self._config.enabled:
            logger.info("Evolution loop is disabled")
            return

        if self.is_running:
            logger.warning("Evolution loop already running")
            return

        # Hard stop: better.md must exist
        self._better_definition = self._load_better()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="evolution-loop",
            daemon=True,
        )
        self._thread.start()
        logger.info("Evolution loop started (max %d iterations)", self._config.max_iterations)

    def stop(self) -> None:
        """Signal the loop to stop and wait for current iteration."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
            self._thread = None
        logger.info(
            "Evolution loop stopped after %d iterations (%d tokens)",
            self._iteration_count,
            self._total_tokens,
        )

    def status(self) -> dict[str, Any]:
        """Return current loop state."""
        return {
            "running": self.is_running,
            "enabled": self._config.enabled,
            "iteration": self._iteration_count,
            "max_iterations": self._config.max_iterations,
            "total_tokens": self._total_tokens,
            "model": self._config.model,
            "has_better_md": bool(self._better_definition),
        }

    def run_one(self) -> IterationRecord:
        """Run a single evolution iteration synchronously.

        Useful for testing or manual triggering.
        """
        if not self._better_definition:
            self._better_definition = self._load_better()
        return self._iterate()

    def _loop(self) -> None:
        """Background loop. Runs iterations until halt condition."""
        while not self._stop_event.is_set():
            if self._iteration_count >= self._config.max_iterations:
                logger.info(
                    "Evolution loop: max iterations reached (%d)", self._config.max_iterations
                )
                break

            if not self._can_continue():
                logger.info("Evolution loop: budget or threshold halt")
                break

            try:
                self._iterate()
            except BetterMdMissing:
                logger.error("Evolution loop: better.md missing or empty — halting")
                break
            except BudgetExhausted:
                logger.warning("Evolution loop: budget exhausted — halting")
                break
            except Exception:
                logger.exception("Evolution loop: iteration %d failed", self._iteration_count)
                # Log the error but don't crash the loop
                self._append_audit(
                    IterationRecord(
                        iteration=self._iteration_count,
                        hypothesis="(failed to generate)",
                        experiment_summary="(iteration error)",
                        outcome="error",
                        rationale="Unhandled exception — see logs",
                        budget_used=0,
                    )
                )
                self._iteration_count += 1

        self._stop_event.set()
        logger.info(
            "Evolution loop exited: %d iterations, %d tokens",
            self._iteration_count,
            self._total_tokens,
        )

    def _can_continue(self) -> bool:
        """Check budget and threshold conditions."""
        if self._budget.status == BudgetStatus.EXCEEDED:
            return False
        if self._budget.usage_percent >= self._config.budget_pause_threshold:
            return False
        if not self._budget.can_allocate(self._config.estimated_tokens_per_iteration):
            return False
        return True

    def _iterate(self) -> IterationRecord:
        """Execute one hypothesis → experiment → evaluate cycle."""
        with self._lock:
            iteration_tokens = 0

            # Phase 1: Generate hypothesis
            self._check_budget(self._config.estimated_tokens_per_iteration)
            hypothesis_data = self._generate_hypothesis()
            iteration_tokens += hypothesis_data.get("_tokens_used", 0)

            # Phase 2: Run experiment
            experiment_result = self._run_experiment(
                hypothesis_data.get("hypothesis", ""),
                hypothesis_data.get("experiment_plan", ""),
            )

            # Phase 2.5: Identity drift check (if anchor configured)
            if self._identity_anchor is not None:
                drift_result = self._identity_anchor.check_drift(
                    hypothesis_data.get("proposed_changes", {})
                )
                if not drift_result.within_bounds:
                    record = IterationRecord(
                        iteration=self._iteration_count,
                        hypothesis=hypothesis_data.get("hypothesis", ""),
                        experiment_summary=experiment_result,
                        outcome="discard",
                        rationale=(
                            f"Identity drift exceeded threshold: "
                            f"score={drift_result.drift_score:.2f}, "
                            f"violations={drift_result.violations}"
                        ),
                        budget_used=iteration_tokens,
                    )
                    self._history.append(record)
                    self._append_audit(record)
                    self._iteration_count += 1
                    self._total_tokens += iteration_tokens
                    return record

            # Phase 3: Evaluate
            self._check_budget(1500)  # evaluation is cheaper
            eval_data = self._evaluate(hypothesis_data, experiment_result)
            iteration_tokens += eval_data.get("_tokens_used", 0)

            # Phase 4: Record
            record = IterationRecord(
                iteration=self._iteration_count,
                hypothesis=hypothesis_data.get("hypothesis", ""),
                experiment_summary=experiment_result,
                outcome=eval_data.get("outcome", "discard"),
                rationale=eval_data.get("rationale", ""),
                budget_used=iteration_tokens,
            )

            self._history.append(record)
            self._append_audit(record)
            self._iteration_count += 1
            self._total_tokens += iteration_tokens

            logger.info(
                "Evolution iteration %d: %s (hypothesis: %.60s...)",
                record.iteration,
                record.outcome,
                record.hypothesis,
            )

            return record

    def _generate_hypothesis(self) -> dict[str, Any]:
        """LLM call: generate a hypothesis based on better.md and history."""
        prior_json = json.dumps(
            [
                {
                    "iteration": r.iteration,
                    "hypothesis": r.hypothesis,
                    "outcome": r.outcome,
                    "rationale": r.rationale,
                }
                for r in reversed(self._history[-5:])
            ],
            indent=2,
        )

        prompt = HYPOTHESIS_SYSTEM_PROMPT.format(
            better_definition=self._better_definition,
            prior_results=prior_json if self._history else "(first iteration — no prior results)",
            principles="\n".join(f"- {p}" for p in self._principles),
            budget_remaining=self._budget.remaining,
        )

        from animus_forge.providers.base import CompletionRequest

        request = CompletionRequest(
            prompt=prompt,
            model=self._config.model,
            temperature=0.7,
            max_tokens=1000,
        )

        response = self._provider.complete(request)

        self._budget.record_usage(
            agent_id="evolution_loop",
            tokens=response.tokens_used,
            operation="generate_hypothesis",
            metadata={"iteration": self._iteration_count, "model": response.model},
        )

        data = self._parse_json(response.content)
        data["_tokens_used"] = response.tokens_used
        return data

    def _run_experiment(self, hypothesis: str, plan: str) -> str:
        """Execute the experiment. Uses runner if provided, else returns plan as result."""
        if self._experiment_runner is not None:
            try:
                return str(self._experiment_runner(hypothesis, plan))
            except Exception as e:
                return f"Experiment runner error: {e}"

        # Default: dry run — return the plan as the result
        # Real experiment runners are injected for production use
        return f"[dry run] Plan executed: {plan}"

    def _evaluate(self, hypothesis_data: dict, experiment_result: str) -> dict[str, Any]:
        """LLM call: evaluate experiment result against better.md."""
        prompt = EVALUATE_SYSTEM_PROMPT.format(
            better_definition=self._better_definition,
            hypothesis=hypothesis_data.get("hypothesis", ""),
            experiment_plan=hypothesis_data.get("experiment_plan", ""),
            experiment_result=experiment_result,
        )

        from animus_forge.providers.base import CompletionRequest

        request = CompletionRequest(
            prompt=prompt,
            model=self._config.model,
            temperature=0.3,
            max_tokens=800,
        )

        response = self._provider.complete(request)

        self._budget.record_usage(
            agent_id="evolution_loop",
            tokens=response.tokens_used,
            operation="evaluate",
            metadata={"iteration": self._iteration_count, "model": response.model},
        )

        data = self._parse_json(response.content)
        data["_tokens_used"] = response.tokens_used
        return data

    def _load_better(self) -> str:
        """Load better.md. Hard stop if missing or empty."""
        path = self._config.better_path
        if not path.exists():
            raise BetterMdMissing(f"better.md not found at {path}")
        content = path.read_text().strip()
        if not content:
            raise BetterMdMissing(f"better.md is empty at {path}")
        return content

    def _check_budget(self, tokens_needed: int) -> None:
        """Pre-call budget gate."""
        if not self._budget.can_allocate(tokens_needed):
            raise BudgetExhausted(
                f"Insufficient budget: need ~{tokens_needed}, have {self._budget.remaining}"
            )

    def _append_audit(self, record: IterationRecord) -> None:
        """Append iteration record to audit JSONL. Append-only, never overwrite."""
        path = self._config.audit_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "iteration": record.iteration,
            "hypothesis": record.hypothesis,
            "experiment_summary": record.experiment_summary,
            "outcome": record.outcome,
            "rationale": record.rationale,
            "budget_used": record.budget_used,
            "timestamp": record.timestamp,
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse LLM JSON response with markdown fence stripping and regex fallback."""
        import re

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        logger.warning("Failed to parse evolution LLM output, using raw text fallback")
        # Fallback: treat the entire response as the hypothesis
        return {"hypothesis": text[:500], "_parse_fallback": True}

    @staticmethod
    def _load_principles(path: Path) -> list[str]:
        """Extract principle lines from a principles markdown file."""
        principles = []
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("### P") and " — " in stripped:
                principles.append(stripped.replace("### ", ""))
        return principles if principles else list(_DEFAULT_PRINCIPLES)


class BetterMdMissing(Exception):
    """Raised when better.md is missing or empty."""


class BudgetExhausted(Exception):
    """Raised when evolution cannot proceed due to budget constraints."""
