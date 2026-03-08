"""Consciousness-Quorum bridge.

Reads Forge monitoring logs + Quorum open intents + constitutional principles,
runs a reflection LLM call, and publishes low-stability intent nodes back to
Quorum for agent consensus.

Off by default. Enable with ConsciousnessConfig(enabled=True).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from animus_forge.budget.manager import BudgetManager, BudgetStatus

if TYPE_CHECKING:
    from animus_forge.providers.base import Provider

logger = logging.getLogger(__name__)

# Quorum imports are optional — bridge degrades gracefully
HAS_CONVERGENT = False
try:
    from convergent.intent import Intent, InterfaceKind, InterfaceSpec
    from convergent.versioning import VersionedGraph

    HAS_CONVERGENT = True
except ImportError:
    pass  # Quorum not installed — bridge degrades gracefully

# Default principles — loaded from file at runtime, this is the fallback
_DEFAULT_PRINCIPLES = [
    "P1 Sovereignty: Serve one user. No telemetry. No vendor lock-in.",
    "P2 Continuity: Memory is never deleted, only archived.",
    "P3 Transparency: Every action is logged and auditable.",
    "P4 Constraint: Forge may not modify Core directly.",
    "P5 Proportionality: Use minimum LLM capability required.",
    "P6 Budget Sovereignty: No LLM call without budget gate.",
    "P7 Arete: Excellence through iteration. Compound systems.",
    "P8 Jidoka: Any agent may halt the line on violation.",
    "P9 Subjectivity Wins: Owner instruction takes precedence.",
]

REFLECTION_SYSTEM_PROMPT = """You are the reflective layer of Animus, a sovereign personal AI exocortex.

Review recent actions and identify patterns. You do not execute tasks — you observe and synthesize.

Constraints:
- You serve one user. Your insights are private.
- You may propose intent updates, but never execute them directly.
- You may flag principle tensions, but never resolve them unilaterally.
- Be concise. One reflection pass, not an essay. You're on a budget.

Current principles:
{principles}

Recent actions ({action_count} records):
{audit_records}

Open intent nodes (unresolved coordination):
{open_nodes}

Budget remaining this session: {budget_remaining} tokens

Respond ONLY with valid JSON matching this schema:
{{
  "summary": "string — what happened since last reflection",
  "insights": ["string — patterns or anomalies observed"],
  "proposed_intent_updates": [{{"content": "string", "tags": ["string"]}}],
  "workflow_patch_ids": ["string — workflow IDs that need review"],
  "principle_tensions": ["string — actions close to violating a principle"],
  "next_reflection_in": 300
}}"""


class ReflectionInput(BaseModel):
    """Assembled from existing Forge monitoring + Quorum state."""

    recent_actions: list[dict[str, Any]] = Field(default_factory=list)
    open_intent_nodes: list[dict[str, Any]] = Field(default_factory=list)
    identity_principles: list[str] = Field(default_factory=list)
    session_budget_remaining: int = 0


class ReflectionOutput(BaseModel):
    """Structured output from reflection LLM call."""

    summary: str = ""
    insights: list[str] = Field(default_factory=list)
    proposed_intent_updates: list[dict[str, Any]] = Field(default_factory=list)
    workflow_patch_ids: list[str] = Field(default_factory=list)
    principle_tensions: list[str] = Field(default_factory=list)
    next_reflection_in: int = 300


@dataclass
class ReflectionRecord:
    """Persisted record of a reflection cycle."""

    timestamp: datetime
    input_summary: str
    output: dict[str, Any]
    tokens_used: int
    model: str


@dataclass
class ConsciousnessConfig:
    """Configuration for the consciousness bridge."""

    enabled: bool = False
    min_idle_seconds: int = 300
    max_audit_records: int = 50
    budget_pause_threshold: float = 0.75
    model: str = "claude-sonnet-4-6"
    estimated_tokens: int = 2000
    principles_path: Path | None = None
    reflections_log_path: Path | None = None
    review_queue_path: Path | None = None


class ConsciousnessBridge:
    """Bridges Forge reflection to Quorum intent graph.

    Reads monitoring logs, runs a budget-gated LLM reflection call,
    and publishes insights as low-stability intent nodes.
    """

    def __init__(
        self,
        provider: Provider,
        budget_manager: BudgetManager,
        config: ConsciousnessConfig | None = None,
        graph: VersionedGraph | None = None,
        metrics_store: Any | None = None,
    ):
        self._provider = provider
        self._budget = budget_manager
        self._config = config or ConsciousnessConfig()
        self._graph = graph
        self._metrics = metrics_store
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_reflection: datetime | None = None
        self._reflection_count: int = 0
        self._total_tokens: int = 0
        self._principles: list[str] = _DEFAULT_PRINCIPLES

        if self._config.principles_path and self._config.principles_path.exists():
            self._principles = self._load_principles(self._config.principles_path)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def reflection_count(self) -> int:
        return self._reflection_count

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def start(self) -> None:
        """Start the background reflection loop."""
        if not self._config.enabled:
            logger.info("Consciousness bridge is disabled")
            return
        if self.is_running:
            logger.warning("Consciousness bridge already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="consciousness-bridge",
            daemon=True,
        )
        self._thread.start()
        logger.info("Consciousness bridge started")

    def stop(self) -> None:
        """Stop the background loop. Waits for current reflection to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        logger.info("Consciousness bridge stopped")

    def status(self) -> dict[str, Any]:
        """Return current bridge state."""
        return {
            "running": self.is_running,
            "enabled": self._config.enabled,
            "last_reflection": (
                self._last_reflection.isoformat() if self._last_reflection else None
            ),
            "reflection_count": self._reflection_count,
            "total_tokens": self._total_tokens,
            "min_idle_seconds": self._config.min_idle_seconds,
            "model": self._config.model,
        }

    def reflect_once(self) -> ReflectionOutput:
        """Run a single reflection cycle.

        1. Gather inputs (audit logs, open intents, principles)
        2. Budget check
        3. LLM call
        4. Parse output
        5. Publish insights to Quorum
        6. Log reflection
        """
        with self._lock:
            reflection_input = self._gather_input()
            self._check_budget()
            raw_output = self._call_llm(reflection_input)
            output = self._parse_output(raw_output)
            self._publish_to_quorum(output)
            self._log_reflection(reflection_input, output)
            self._last_reflection = datetime.now(UTC)
            self._reflection_count += 1
            return output

    def _should_reflect(self) -> bool:
        """Check all trigger conditions."""
        if not self._config.enabled:
            return False

        if self._budget.status in (BudgetStatus.EXCEEDED,):
            return False

        if self._budget.usage_percent >= self._config.budget_pause_threshold:
            return False

        if self._last_reflection is not None:
            elapsed = (datetime.now(UTC) - self._last_reflection).total_seconds()
            if elapsed < self._config.min_idle_seconds:
                return False

        if not self._budget.can_allocate(self._config.estimated_tokens):
            return False

        return True

    def _loop(self) -> None:
        """Background loop. Checks trigger conditions, reflects when idle."""
        while not self._stop_event.is_set():
            try:
                if self._should_reflect():
                    self.reflect_once()
            except Exception:
                logger.exception("Consciousness bridge reflection failed")
            self._stop_event.wait(timeout=30)

    def _gather_input(self) -> ReflectionInput:
        """Assemble reflection input from existing data sources."""
        recent_actions: list[dict[str, Any]] = []
        if self._metrics is not None:
            try:
                recent = self._metrics.get_recent_executions(
                    limit=self._config.max_audit_records,
                )
                recent_actions = recent if isinstance(recent, list) else []
            except Exception:
                logger.debug("Could not read metrics store", exc_info=True)

        open_nodes: list[dict[str, Any]] = []
        if self._graph is not None and HAS_CONVERGENT:
            try:
                snapshot = self._graph.snapshot()
                if hasattr(snapshot, "intents"):
                    open_nodes = [
                        i.to_dict() if hasattr(i, "to_dict") else {"id": str(i)}
                        for i in snapshot.intents
                        if getattr(i, "stability", 1.0) < 0.5
                    ]
            except Exception:
                logger.debug("Could not read intent graph", exc_info=True)

        return ReflectionInput(
            recent_actions=recent_actions,
            open_intent_nodes=open_nodes,
            identity_principles=self._principles,
            session_budget_remaining=self._budget.remaining,
        )

    def _check_budget(self) -> None:
        """Pre-call budget gate. Raises if insufficient."""
        if not self._budget.can_allocate(self._config.estimated_tokens):
            msg = (
                f"Insufficient budget for reflection: "
                f"need ~{self._config.estimated_tokens} tokens, "
                f"have {self._budget.remaining}"
            )
            raise BudgetExhausted(msg)

    def _call_llm(self, reflection_input: ReflectionInput) -> str:
        """Make the budget-gated LLM call."""
        prompt = REFLECTION_SYSTEM_PROMPT.format(
            principles="\n".join(f"- {p}" for p in reflection_input.identity_principles),
            action_count=len(reflection_input.recent_actions),
            audit_records=json.dumps(
                reflection_input.recent_actions[:20],
                default=str,
                indent=2,
            ),
            open_nodes=json.dumps(
                reflection_input.open_intent_nodes[:10],
                default=str,
                indent=2,
            ),
            budget_remaining=reflection_input.session_budget_remaining,
        )

        from animus_forge.providers.base import CompletionRequest

        request = CompletionRequest(
            prompt=prompt,
            model=self._config.model,
            temperature=0.3,
            max_tokens=1500,
        )

        response = self._provider.complete(request)
        self._total_tokens += response.tokens_used

        self._budget.record_usage(
            agent_id="consciousness_bridge",
            tokens=response.tokens_used,
            operation="reflect",
            metadata={"model": response.model, "cycle": self._reflection_count},
        )

        return response.content

    def _parse_output(self, raw: str) -> ReflectionOutput:
        """Parse LLM JSON response. Returns empty output on parse failure."""
        import re

        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        try:
            data = json.loads(text)
            return ReflectionOutput(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Regex fallback for local models
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
                return ReflectionOutput(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        logger.warning("Failed to parse reflection output, skipping cycle")
        return ReflectionOutput(summary="Parse failure — skipped")

    def _publish_to_quorum(self, output: ReflectionOutput) -> None:
        """Write insights and tensions as low-stability intent nodes."""
        # Workflow review queue is independent of Quorum availability
        self._queue_workflow_reviews(output.workflow_patch_ids)

        if self._graph is None or not HAS_CONVERGENT:
            return

        for insight in output.insights:
            try:
                tags = ["reflection", "auto-generated"]
                intent = Intent(
                    agent_id="consciousness_bridge",
                    intent=insight,
                    provides=[
                        InterfaceSpec(
                            name="reflection_surface",
                            kind=InterfaceKind.CONFIG,
                            signature="",
                            tags=tags,
                        ),
                    ],
                )
                self._graph.publish(intent)
            except Exception:
                logger.debug("Failed to publish insight to Quorum", exc_info=True)

        for tension in output.principle_tensions:
            try:
                intent = Intent(
                    agent_id="consciousness_bridge",
                    intent=f"PRINCIPLE TENSION: {tension}",
                    provides=[
                        InterfaceSpec(
                            name="principle_tension",
                            kind=InterfaceKind.CONFIG,
                            signature="",
                            tags=["principle_tension", "requires_attention"],
                        ),
                    ],
                )
                self._graph.publish(intent)
            except Exception:
                logger.debug("Failed to publish tension to Quorum", exc_info=True)

    def _queue_workflow_reviews(self, workflow_ids: list[str]) -> None:
        """Append flagged workflow IDs to review queue file."""
        if not workflow_ids:
            return

        path = self._config.review_queue_path
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for wf_id in workflow_ids:
                record = {
                    "workflow_id": wf_id,
                    "flagged_by": "consciousness_bridge",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "cycle": self._reflection_count,
                }
                f.write(json.dumps(record) + "\n")

    def _log_reflection(
        self,
        inp: ReflectionInput,
        output: ReflectionOutput,
    ) -> None:
        """Append reflection record to log file."""
        path = self._config.reflections_log_path
        if path is None:
            return

        record = ReflectionRecord(
            timestamp=datetime.now(UTC),
            input_summary=f"{len(inp.recent_actions)} actions, {len(inp.open_intent_nodes)} open nodes",
            output=output.model_dump(),
            tokens_used=self._total_tokens,
            model=self._config.model,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": record.timestamp.isoformat(),
                        "input_summary": record.input_summary,
                        "output": record.output,
                        "tokens_used": record.tokens_used,
                        "model": record.model,
                    },
                    default=str,
                )
                + "\n"
            )

    @staticmethod
    def _load_principles(path: Path) -> list[str]:
        """Extract P1-P9 lines from a principles markdown file."""
        principles = []
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("### P") and " — " in stripped:
                principles.append(stripped.replace("### ", ""))
        return principles if principles else _DEFAULT_PRINCIPLES


class BudgetExhausted(Exception):
    """Raised when reflection cannot proceed due to budget constraints."""
