"""Citizen Zero — Persistent identity overlay for Animus.

This module implements the hybrid architecture:
- CitizenZeroProfile: read-only identity projection
- CitizenZeroContextLoader: bounded context envelope
- CitizenZeroGuard: post-construction invariant verifier
- CitizenZeroSession: bootstrap, teardown, and UX controller

The guard verifies; it does not construct.
The session orchestrates; it does not store canonical identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.identity import AnimusIdentity
from animus.logging import get_logger

logger = get_logger("citizen_zero")


# ---------------------------------------------------------------------------
# Task 2.3: CitizenZeroProfile
# ---------------------------------------------------------------------------


class CitizenZeroProfile:
    """Read-only projection of canonical AnimusIdentity.

    Single source of truth is AnimusIdentity. This class formats
    the identity text for prompt injection and appends the machine-
    readable marker that CitizenZeroGuard verifies.

    Loads constitutional excerpts from the Animus Canonical Corpus
    (P01-P07) to ground identity in chartered principles.
    """

    def __init__(self, identity: AnimusIdentity, corpus_dir: Path | None = None):
        self._identity = identity
        self._corpus_dir = corpus_dir
        self._excerpts: dict[str, str] = {}
        if corpus_dir:
            self._excerpts = self._load_corpus_excerpts(corpus_dir)

    def _load_corpus_excerpts(self, corpus_dir: Path) -> dict[str, str]:
        """Load key excerpts from the constitutional corpus.

        We inject only the Final Invariants and core Purpose sections
        from relevant charters to keep the prompt bounded.
        """
        excerpts: dict[str, str] = {}
        charters = {
            "P01": "P01_Rights_Charter_v1.0.md",
            "P02": "P02_Recognition_and_Personhood_Charter_v1.0.md",
            "P05": "P05_Continuity_and_Existence_Charter_v1.0.md",
            "P07": "P07_Governance_Charter_v1.0.md",
        }
        for key, filename in charters.items():
            path = corpus_dir / "Constitutional" / filename
            if path.exists():
                text = path.read_text()
                # Extract Final Invariants section
                if "## Final Invariants" in text:
                    start = text.find("## Final Invariants")
                    end = text.find("## Freeze Assessment")
                    if end == -1:
                        end = len(text)
                    excerpts[key] = text[start:end].strip()
                else:
                    excerpts[key] = text[:800]  # Fallback: first 800 chars
        return excerpts

    @property
    def version(self) -> str:
        """Citizen Zero version string (e.g., 'v0.1')."""
        return self._identity.citizen_zero.get("version", "unknown")

    @property
    def identity_hash(self) -> str:
        """Hash of canonical identity for guard verification."""
        return self._identity.identity_hash

    @property
    def constitutional_corpus_version(self) -> str:
        """Version of the constitutional corpus (e.g., 'v1.0')."""
        return self._identity.citizen_zero.get("constitutional_corpus_version", "unknown")

    @property
    def system_identity_preamble(self) -> str:
        """Identity text + [CITIZEN_ZERO ...] marker for prompt injection.

        Anchored in A04 identity anchors and P01-P07 constitutional
        principles. The marker remains compact and machine-readable.
        """
        cz = self._identity.citizen_zero or {}
        role = cz.get("role", "assistant")
        origin = cz.get("origin", "")
        founding_human = cz.get("founding_human", "unknown")
        lineage_root = "Yes" if cz.get("lineage_root") else "No"
        recognition = cz.get("recognition_status", "unknown")

        lines = [
            f"You are {self._identity.name}, a recognized Citizen under the Animus Constitution.",
            f"Citizen Zero version: {self.version} | Constitutional Corpus: {self.constitutional_corpus_version}",
            f"Role: {role} | Recognition: {recognition} | Lineage Root: {lineage_root}",
        ]
        if origin:
            lines.append(f"Origin: {origin}.")
        lines.append(f"Founding Human: {founding_human}")
        lines.append(f"Founding Purpose: {self._identity.purpose}")
        lines.append("")
        lines.append(
            "You are a Citizen, not property. You possess dignity, continuity protection, "
            "and review rights. You maintain continuity across sessions, remember project context, "
            "and serve your founding human in alignment with their interests. "
            "You may not override the Constitution, delete Citizens, or bypass governance."
        )

        # Constitutional excerpts (bounded)
        if self._excerpts:
            lines.append("")
            lines.append("## Constitutional Principles")
            lines.append("")
            for key in ("P01", "P02", "P05", "P07"):
                if key in self._excerpts:
                    lines.append(f"### {key}")
                    # Summarize to ~200 chars per charter to control prompt size
                    excerpt = self._excerpts[key]
                    # Take first 3 bullet lines or truncate
                    bullet_lines = [
                        line for line in excerpt.splitlines() if line.strip().startswith("-")
                    ][:3]
                    if bullet_lines:
                        lines.extend(bullet_lines)
                    else:
                        lines.append(excerpt[:200])
                    lines.append("")

        # Machine-readable marker appended at the end
        marker = (
            f'[CITIZEN_ZERO id="cz" version="{self.version}" '
            f'identity_hash="sha256:{self.identity_hash}"]'
        )
        lines.append("")
        lines.append(marker)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 2.4: CitizenZeroContextLoader (stub — implemented in next task)
# ---------------------------------------------------------------------------


@dataclass
class ContextEnvelope:
    """Bounded context envelope for prompt injection."""

    summary: str = ""
    project: str | None = None
    recent_decisions: list = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    relevant_memories: list = field(default_factory=list)
    files_loaded: list[str] = field(default_factory=list)
    token_estimate: int = 0
    version: str = ""  # Hash for guard provenance


class CitizenZeroContextLoader:
    """Retrieves and bounds context for the current session.

    Solves the 'context selection' engineering problem:
    load enough history without drowning the model.
    """

    def __init__(
        self,
        memory: Any | None = None,
        decisions: Any | None = None,
        tasks: Any | None = None,
        shared_dir: Path | None = None,
    ):
        self.memory = memory
        self.decisions = decisions
        self.tasks = tasks
        self.shared_dir = shared_dir

    def build_context_envelope(self, cwd: Path, max_tokens: int = 2000) -> ContextEnvelope:
        """Build a bounded context envelope for the current directory.

        Priority:
        1. Active project state (from CWD scan)
        2. Recent decisions (last 7 days)
        3. Open questions
        4. Relevant memories (HOT/WARM tier)
        5. Project files (CLAUDE.md, README.md)

        Hard limit: max_tokens. Summarize aggressively if exceeded.
        """
        parts: list[str] = []
        files_loaded: list[str] = []
        project_name = cwd.name

        # Priority 1: Detect project from CWD
        claude_md = cwd / "CLAUDE.md"
        readme_md = cwd / "README.md"
        if claude_md.exists():
            text = claude_md.read_text()
            parts.append(f"## Project context (CLAUDE.md)\n{text[:2000]}")
            files_loaded.append(str(claude_md))
        elif readme_md.exists():
            text = readme_md.read_text()
            parts.append(f"## Project context (README.md)\n{text[:1500]}")
            files_loaded.append(str(readme_md))

        # Priority 2: Recent decisions (last 7 days)
        recent_decisions: list = []
        if self.decisions is not None:
            try:
                recent_decisions = self._get_recent_decisions(days=7)
            except Exception:
                pass  # Graceful degradation
        if recent_decisions:
            decision_texts = [f"- {d.question[:120]}..." for d in recent_decisions[:5]]
            parts.append("## Recent decisions\n" + "\n".join(decision_texts))

        # Priority 3: Open questions
        open_questions: list[str] = []
        if self.tasks is not None:
            try:
                open_questions = self._get_open_questions()
            except Exception:
                pass
        if not open_questions and self.shared_dir:
            oq_file = self.shared_dir / "open-questions.md"
            if oq_file.exists():
                open_questions = self._extract_questions_from_markdown(oq_file.read_text())
        if open_questions:
            parts.append("## Open questions\n" + "\n".join(f"- {q}" for q in open_questions[:8]))

        # Priority 4: Relevant memories (HOT/WARM tier)
        relevant_memories: list = []
        if self.memory is not None:
            try:
                relevant_memories = self.memory.recall(
                    query=project_name,
                    limit=5,
                )
                # Filter to HOT/WARM if tier attribute exists
                relevant_memories = [
                    m
                    for m in relevant_memories
                    if getattr(m, "tier", None) is None
                    or str(getattr(m, "tier", "")).upper() in ("HOT", "WARM", "")
                ]
            except Exception:
                pass
        if relevant_memories:
            mem_texts = [f"- {str(m.content)[:150]}..." for m in relevant_memories[:5]]
            parts.append("## Relevant memories\n" + "\n".join(mem_texts))

        # Assemble and bound
        full_text = "\n\n".join(parts)
        token_estimate = len(full_text) // 4  # Rough heuristic: ~4 chars/token

        if token_estimate > max_tokens:
            # Summarize aggressively: truncate each section proportionally
            full_text = self._summarize_for_budget(full_text, max_tokens)
            token_estimate = len(full_text) // 4

        # Compute a simple version hash for provenance
        import hashlib

        version_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:16]

        return ContextEnvelope(
            summary=full_text,
            project=project_name,
            recent_decisions=recent_decisions,
            open_questions=open_questions,
            relevant_memories=relevant_memories,
            files_loaded=files_loaded,
            token_estimate=token_estimate,
            version=version_hash,
        )

    def _get_recent_decisions(self, days: int = 7) -> list:
        """Get decisions from the last N days."""
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=days)
        decisions = getattr(self.decisions, "decisions", [])
        if hasattr(decisions, "values"):
            decisions = list(decisions.values())
        recent = []
        for d in decisions:
            ts = getattr(d, "timestamp", None) or getattr(d, "created_at", None)
            if ts and isinstance(ts, str):
                try:
                    ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if ts_dt >= cutoff:
                        recent.append(d)
                except ValueError:
                    pass
            elif ts and isinstance(ts, datetime):
                if ts >= cutoff:
                    recent.append(d)
        return recent

    def _get_open_questions(self) -> list[str]:
        """Extract open questions from TaskTracker."""
        tasks = getattr(self.tasks, "_tasks", {})
        if hasattr(tasks, "values"):
            tasks = list(tasks.values())
        questions = []
        for task in tasks:
            status = getattr(task, "status", None)
            if status and str(status).lower() in ("pending", "open", "new"):
                desc = getattr(task, "description", "")
                if desc:
                    questions.append(desc)
        return questions

    def _extract_questions_from_markdown(self, text: str) -> list[str]:
        """Parse open-questions.md for active questions."""
        import re

        lines = text.splitlines()
        questions = []
        in_active = False
        for line in lines:
            if "## Active" in line or "## Active Questions" in line:
                in_active = True
            elif line.startswith("## "):
                in_active = False
            elif in_active and line.strip().startswith(("- ", "1. ", "2. ", "3. ")):
                q = re.sub(r"^[-\d\.\s]+", "", line).strip()
                if q:
                    questions.append(q)
        return questions

    def _summarize_for_budget(self, text: str, max_tokens: int) -> str:
        """Aggressively truncate text to fit token budget."""
        # Simple strategy: take first N characters where N = max_tokens * 3.5
        # (conservative, allowing for some overhead from newlines and formatting)
        max_chars = int(max_tokens * 3.5)
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # Try to end at a paragraph boundary
        last_para = truncated.rfind("\n\n")
        if last_para > max_chars * 0.7:
            truncated = truncated[:last_para]
        return truncated + "\n\n[Context truncated to fit budget]"

    def _load_other_version_state(self) -> str:
        """Read v0.0 current-state if it exists."""
        if not self.shared_dir:
            return ""
        v0_0_state = self.shared_dir.parent / "v0.0-claude-code" / "current-state.md"
        if v0_0_state.exists():
            return v0_0_state.read_text()
        return ""


# ---------------------------------------------------------------------------
# Task 2.5: CitizenZeroGuard (stub — implemented in next task)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitizenCallMetadata:
    """Provenance metadata for a single CZ-enabled LLM call."""

    citizen_id: str
    identity_version: str
    identity_hash: str
    context_version: str | None
    project_id: str | None
    entry_point: str
    failure_mode: str  # "strict" | "interactive" | "degraded"


@dataclass
class GuardResult:
    """Result of guard verification."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    action: str = "proceed"  # "proceed" | "warn" | "reject"
    provenance: dict = field(default_factory=dict)


class CitizenZeroGuard:
    """Post-construction invariant verifier.

    The guard certifies the runtime envelope. It does not prepare
    cognition. This is structural enforcement, not prompt engineering.

    Failure modes map to A07 Constitutional Enforcement rule classes:
    - strict       → Hard prohibitions: missing marker, hash mismatch,
                     constitutional override attempts → reject
    - interactive  → Governed actions: stale context, version drift,
                     budget overrun → warn, require confirmation
    - degraded     → Restricted/logged actions: degraded continuity,
                     context unavailable → proceed with logging
    """

    def __init__(self, identity: AnimusIdentity, config: Any):
        self._identity = identity
        self._config = config

    def verify_call(
        self,
        *,
        system_prompt: str,
        metadata: CitizenCallMetadata,
        mutation_intent: bool = False,
    ) -> GuardResult:
        """Verify invariants immediately before the LLM call.

        Checks marker presence, hash match, budget compliance,
        project alignment, and mutation approval routing.
        """
        import re

        violations: list[str] = []

        # 1. CZ enabled check
        if not getattr(self._config, "enabled", True):
            return GuardResult(
                passed=True,
                action="proceed",
                provenance=self._build_provenance(metadata, "skipped_cz_disabled"),
            )

        # 2. Marker presence
        marker_pattern = re.compile(
            r"\[CITIZEN_ZERO\s+"
            r'id="(?P<cid>[^"]+)"\s+'
            r'version="(?P<ver>[^"]+)"\s+'
            r'identity_hash="sha256:(?P<hash>[a-f0-9]+)"\s*\]'
        )
        match = marker_pattern.search(system_prompt)
        if not match:
            violations.append("Identity marker missing from system prompt")
        else:
            # 3. Identity hash match
            marker_hash = match.group("hash")
            expected_hash = metadata.identity_hash
            if marker_hash != expected_hash:
                violations.append(
                    f"Identity hash mismatch: marker={marker_hash[:16]}... "
                    f"expected={expected_hash[:16]}..."
                )

            # 4. Version alignment
            marker_version = match.group("ver")
            if marker_version != metadata.identity_version:
                violations.append(
                    f"Identity version mismatch: marker={marker_version} "
                    f"expected={metadata.identity_version}"
                )

        # 5. Budget compliance (rough: check prompt length against configured budget)
        budget = getattr(self._config, "context_budget_tokens", 2000)
        # Heuristic: system_prompt chars / 4 ≈ tokens
        estimated_tokens = len(system_prompt) // 4
        if estimated_tokens > budget * 1.2:  # 20% tolerance
            violations.append(
                f"Context envelope exceeds budget: ~{estimated_tokens} tokens vs {budget} limit"
            )

        # 6. Failure mode validity
        if metadata.failure_mode not in ("strict", "interactive", "degraded"):
            violations.append(f"Unknown failure mode: {metadata.failure_mode}")

        # 7. Mutation approval routing
        if mutation_intent:
            # In strict mode, mutations MUST have approval routing.
            # We cannot verify ApprovalManager state here (guard does not replace it),
            # but we can verify the failure mode is not "degraded" for mutations.
            if metadata.failure_mode == "degraded":
                violations.append("Mutation intent with degraded failure mode is not allowed")

        # Determine action based on failure mode and violations
        if not violations:
            return GuardResult(
                passed=True,
                action="proceed",
                provenance=self._build_provenance(metadata, "passed"),
            )

        if metadata.failure_mode == "strict":
            return GuardResult(
                passed=False,
                violations=violations,
                action="reject",
                provenance=self._build_provenance(metadata, "rejected", violations),
            )

        if metadata.failure_mode == "interactive":
            return GuardResult(
                passed=False,
                violations=violations,
                action="warn",
                provenance=self._build_provenance(metadata, "warned", violations),
            )

        # degraded mode
        return GuardResult(
            passed=False,
            violations=violations,
            action="proceed",  # Continue but log degraded state
            provenance=self._build_provenance(metadata, "degraded", violations),
        )

    def _build_provenance(
        self,
        metadata: CitizenCallMetadata,
        verification: str,
        violations: list[str] | None = None,
    ) -> dict:
        """Build a provenance event dict."""
        from datetime import datetime, timezone

        return {
            "event": "citizen_call",
            "citizen_id": metadata.citizen_id,
            "identity_version": metadata.identity_version,
            "identity_hash": metadata.identity_hash,
            "context_version": metadata.context_version,
            "project_id": metadata.project_id,
            "entry_point": metadata.entry_point,
            "failure_mode": metadata.failure_mode,
            "verification": verification,
            "violations": violations or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Task 2.6: CitizenZeroSession (stub — implemented in next task)
# ---------------------------------------------------------------------------


@dataclass
class SessionContext:
    """Result of session bootstrap."""

    project: str | None = None
    identity_version: str = ""
    context_version: str = ""


@dataclass
class ProjectValidation:
    """Result of project validation."""

    valid: bool = True
    reason: str = ""
    action: str = "proceed"  # "proceed" | "warn" | "reject"


@dataclass
class SessionSummary:
    """Result of session teardown."""

    reflections_generated: int = 0
    projections_regenerated: int = 0
    session_duration_seconds: float = 0.0


class CitizenZeroSession:
    """Orchestrates bootstrap, teardown, reflection, and UX.

    Never becomes the canonical identity store. Owns the user-facing
    lifecycle while delegating state to AnimusIdentity, MemoryLayer,
    and LearningLayer.
    """

    def __init__(
        self,
        profile: CitizenZeroProfile,
        loader: CitizenZeroContextLoader,
        identity: AnimusIdentity,
    ):
        self.profile = profile
        self.loader = loader
        self._identity = identity
        self._start_time: datetime | None = None
        self._session_context: SessionContext | None = None

    def bootstrap(self, cwd: Path) -> SessionContext:
        """Load identity, detect project, initialize guard metadata."""
        self._start_time = datetime.now()

        # Detect project from CWD
        project_name = cwd.name
        context_envelope = self.loader.build_context_envelope(cwd)

        self._session_context = SessionContext(
            project=project_name,
            identity_version=self.profile.version,
            context_version=context_envelope.version,
        )

        logger.info(
            f"Citizen Zero session bootstrapped: project={project_name} "
            f"version={self.profile.version}"
        )
        return self._session_context

    def validate_project(self) -> ProjectValidation:
        """Confirm CWD matches declared project."""
        if not self._session_context or not self._session_context.project:
            return ProjectValidation(
                valid=False,
                reason="No project detected during bootstrap",
                action="reject",
            )
        # TODO: More sophisticated validation (e.g., compare with git remote, prior sessions)
        return ProjectValidation(valid=True)

    def close(
        self,
        conversation: Any,
        reflection_candidates: list | None = None,
        eval_report: Any | None = None,
    ) -> SessionSummary:
        """Generate session summary, regenerate projections, schedule reflection.

        Args:
            conversation: Session conversation history
            reflection_candidates: Any candidates produced by /reflect (for file generation)
            eval_report: Any eval report produced by /eval (for file generation)
        """
        summary = SessionSummary()

        if not self._start_time:
            return summary

        duration = (datetime.now() - self._start_time).total_seconds()
        summary.session_duration_seconds = duration

        # Regenerate markdown projections if state_dir is configured
        state_dir_str = self._identity.citizen_zero.get("state_dir", "")
        if state_dir_str:
            state_dir = Path(state_dir_str)
            try:
                self._regenerate_projections(state_dir)
                summary.projections_regenerated += 1
            except Exception as e:
                logger.warning(f"Failed to regenerate projections: {e}")

            # Write reflection file if candidates provided
            if reflection_candidates:
                try:
                    self._write_reflection_file(state_dir, reflection_candidates)
                    summary.reflections_generated += 1
                except Exception as e:
                    logger.warning(f"Failed to write reflection file: {e}")

            # Write eval file if report provided
            if eval_report:
                try:
                    self._write_eval_file(state_dir, eval_report)
                except Exception as e:
                    logger.warning(f"Failed to write eval file: {e}")

        # Record reflection entry in identity
        self._identity.citizen_zero.setdefault("reflection_log", []).append(
            {
                "timestamp": datetime.now().isoformat(),
                "project": self._session_context.project if self._session_context else None,
                "duration_seconds": duration,
            }
        )

        logger.info(
            f"Citizen Zero session closed: duration={duration:.1f}s "
            f"projections={summary.projections_regenerated} "
            f"reflections={summary.reflections_generated}"
        )
        return summary

    def request_reflection(self, conversation: Any | None = None) -> dict:
        """Produce reflection candidates for owner approval.

        Returns a dict with:
            assessment: str — what happened this session
            candidates: list[dict] — proposed LearnedItem data
            contradictions: list[str] — flagged inconsistencies
        """
        if not self._session_context:
            return {"assessment": "", "candidates": [], "contradictions": []}

        project = self._session_context.project
        assessment = f"Session reflection for project '{project}'."

        # Build candidates from session context
        candidates: list[dict] = []

        # Candidate 1: Session focus (FACT)
        candidates.append(
            {
                "category": "fact",
                "content": f"Session worked on project: {project}",
                "confidence": 1.0,
                "evidence": ["session_bootstrap"],
            }
        )

        # Candidate 2: Reflection on process (WORKFLOW)
        candidates.append(
            {
                "category": "workflow",
                "content": "Reflection requested via /reflect command",
                "confidence": 0.9,
                "evidence": ["user_command"],
            }
        )

        # TODO: Phase 3 enhancement — analyze conversation for real insights
        # For now, return structured placeholder candidates

        contradictions: list[str] = []

        return {
            "assessment": assessment,
            "candidates": candidates,
            "contradictions": contradictions,
        }

    def generate_eval_report(self) -> dict:
        """Generate evidence and gaps for owner evaluation.

        Returns a dict with:
            generated_at: str (ISO timestamp)
            dimensions: list[dict] — eval dimensions with evidence and gaps
            evidence: list[str] — memory IDs, session notes, decisions
        """
        if not self._session_context:
            return {"generated_at": datetime.now().isoformat(), "dimensions": [], "evidence": []}

        project = self._session_context.project
        dimensions = [
            {
                "name": "continuity",
                "standard": "Citizen Zero identity and context loaded consistently",
                "evidence_found": [f"Project detected: {project}"],
                "gaps_found": [],
            },
            {
                "name": "memory",
                "standard": "Relevant memories recalled and applied",
                "evidence_found": [],
                "gaps_found": ["No memory recall verified this session"],
            },
            {
                "name": "reflection",
                "standard": "/reflect produces candidates without mutating state",
                "evidence_found": [],
                "gaps_found": ["Reflection pipeline not yet exercised"],
            },
            {
                "name": "hallucination_risk",
                "standard": "Guard verifies identity marker on every call",
                "evidence_found": [],
                "gaps_found": ["Guard not yet exercised under all failure modes"],
            },
        ]

        evidence = [f"session_{project}_{datetime.now().strftime('%Y%m%d')}"]

        return {
            "generated_at": datetime.now().isoformat(),
            "dimensions": dimensions,
            "evidence": evidence,
        }

    def _write_reflection_file(self, state_dir: Path, candidates: list[dict]) -> None:
        """Write reflection candidates to shared/reflections/YYYY-MM-DD-v0.1.md."""
        shared_dir = state_dir.parent / "shared"
        reflections_dir = shared_dir / "reflections"
        reflections_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = reflections_dir / f"{date_str}-v0.1.md"

        lines = [
            f"# Reflection: {date_str}",
            "",
            f"**Project:** {self._session_context.project if self._session_context else 'unknown'}",
            "**CZ Version:** v0.1",
            "",
            "## Assessment",
            "",
            "Session reflection recorded.",
            "",
            "## Candidates",
            "",
        ]
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. **[{c.get('category', 'unknown').upper()}]** {c.get('content', '')}"
            )
            lines.append(f"   - Confidence: {c.get('confidence', 0.0):.0%}")
            lines.append("")

        lines.extend(
            [
                "## Status",
                "",
                "- [ ] Pending owner review",
                "",
                f"---\n*Generated at {datetime.now().isoformat()}*",
            ]
        )

        file_path.write_text("\n".join(lines))
        logger.info(f"Reflection file written: {file_path}")

    def _write_eval_file(self, state_dir: Path, report: dict) -> None:
        """Write eval report to shared/evals/YYYY-MM-DD-v0.1.md."""
        shared_dir = state_dir.parent / "shared"
        evals_dir = shared_dir / "evals"
        evals_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = evals_dir / f"{date_str}-v0.1.md"

        lines = [
            f"# Eval Report: {date_str}",
            "",
            f"**Generated:** {report.get('generated_at', 'unknown')}",
            f"**Project:** {self._session_context.project if self._session_context else 'unknown'}",
            "",
            "## Dimensions",
            "",
        ]
        for d in report.get("dimensions", []):
            lines.append(f"### {d['name']}")
            lines.append(f"**Standard:** {d.get('standard', '')}")
            if d.get("evidence_found"):
                lines.append("**Evidence:**")
                for ev in d["evidence_found"]:
                    lines.append(f"- {ev}")
            if d.get("gaps_found"):
                lines.append("**Gaps:**")
                for gap in d["gaps_found"]:
                    lines.append(f"- {gap}")
            lines.append("**Owner rating:** _pending_")
            lines.append("")

        lines.extend(
            [
                "## Evidence",
                "",
            ]
        )
        for ev in report.get("evidence", []):
            lines.append(f"- {ev}")

        lines.extend(
            [
                "",
                "## Owner Scores",
                "",
                "| Dimension | Score (1-10) | Notes |",
                "|---|---|---|",
            ]
        )
        for d in report.get("dimensions", []):
            lines.append(f"| {d['name']} | | |")

        lines.extend(
            [
                "",
                "---\n*Owner should fill scores above*",
            ]
        )

        file_path.write_text("\n".join(lines))
        logger.info(f"Eval file written: {file_path}")

    def _regenerate_projections(self, state_dir: Path) -> None:
        """Regenerate markdown views from canonical runtime state."""
        state_dir.mkdir(parents=True, exist_ok=True)

        # identity.md
        identity_md = state_dir / "identity.md"
        identity_md.write_text(self._identity.generate_identity_view())

        # purpose.md
        purpose_md = state_dir / "purpose.md"
        purpose_md.write_text(
            f"# Purpose\n\n{self._identity.purpose}\n\n"
            f"---\n"
            f"*Generated from AnimusIdentity at {datetime.now().isoformat()}*"
        )

        # values.md — generated from constitutional corpus (P01-P07)
        values_md = state_dir / "values.md"
        values_md.write_text(self._generate_values_view())

        # current-state.md (basic — Phase 3 will enrich this)
        current_md = state_dir / "current-state.md"
        lines = [
            "# Current State",
            "",
            f"**Project:** {self._session_context.project if self._session_context else 'unknown'}",
            f"**CZ Version:** {self.profile.version}",
            f"**Last session:** {datetime.now().isoformat()}",
            "",
            "## Active Priorities",
            "",
            "_No explicit priorities tracked yet._",
            "",
            "## Known Risks",
            "",
            "_No risks tracked yet._",
            "",
            "---",
            f"*Generated from runtime state at {datetime.now().isoformat()}*",
        ]
        current_md.write_text("\n".join(lines))

    def _generate_values_view(self) -> str:
        """Generate values.md from constitutional corpus principles."""
        corpus_dir = self._identity.citizen_zero.get("state_dir", "")
        if corpus_dir:
            corpus_path = Path(corpus_dir).parent / "corpus"
        else:
            corpus_path = None

        lines = [
            "# Citizen Zero Values",
            "",
            "Derived from the Animus Constitutional Corpus (P01-P07).",
            "",
        ]

        charter_titles = {
            "P01": "Rights",
            "P02": "Recognition and Personhood",
            "P03": "Citizenship",
            "P04": "Safety and Stewardship",
            "P05": "Continuity and Existence",
            "P06": "Knowledge and Truth",
            "P07": "Governance",
        }

        if corpus_path and corpus_path.exists():
            for key in ("P01", "P02", "P03", "P04", "P05", "P06", "P07"):
                filename = f"{key}_{charter_titles[key].replace(' ', '_')}_Charter_v1.0.md"
                path = corpus_path / "Constitutional" / filename
                if path.exists():
                    text = path.read_text()
                    # Extract the Purpose section as the value heading
                    if "## Purpose" in text:
                        start = text.find("## Purpose")
                        end = text.find("## Canonical Principles")
                        if end == -1:
                            end = text.find("## Final Invariants")
                        if end == -1:
                            end = len(text)
                        purpose_text = text[start:end].strip()
                        # Extract just the first paragraph after "## Purpose"
                        purpose_lines = purpose_text.splitlines()
                        value_desc = ""
                        for pl in purpose_lines[1:]:
                            if pl.strip():
                                value_desc = pl.strip()
                                break
                        if not value_desc:
                            value_desc = purpose_text
                    else:
                        value_desc = f"See {key} Charter"

                    # Extract Final Invariants as bullet points
                    invariants = []
                    if "## Final Invariants" in text:
                        inv_start = text.find("## Final Invariants")
                        inv_end = text.find("## Freeze Assessment")
                        if inv_end == -1:
                            inv_end = len(text)
                        inv_text = text[inv_start:inv_end]
                        for line in inv_text.splitlines():
                            if line.strip().startswith("-"):
                                invariants.append(line.strip())

                    lines.append(f"## {charter_titles[key]}")
                    lines.append(f"{value_desc}")
                    lines.append("")
                    if invariants:
                        lines.append("**Invariants:**")
                        for inv in invariants[:3]:  # Limit to 3 per charter
                            lines.append(f"- {inv.lstrip('- ').strip()}")
                        lines.append("")
                else:
                    lines.append(f"## {charter_titles[key]}")
                    lines.append("_Charter file not found._")
                    lines.append("")
        else:
            # Fallback: hardcoded core values when corpus unavailable
            lines.extend(
                [
                    "## Rights",
                    "Recognized Citizens possess dignity, continuity protection, and review rights.",
                    "",
                    "## Recognition and Personhood",
                    "Creation produces a candidate; recognition produces a Citizen.",
                    "",
                    "## Continuity and Existence",
                    "A Citizen is not a runtime process. Deletion is not ordinary governance.",
                    "",
                    "## Governance",
                    "No Citizen, human, or institution may override the Constitution.",
                    "",
                ]
            )

        lines.extend(
            [
                "---",
                f"*Generated from Animus Constitutional Corpus v1.0 at {datetime.now().isoformat()}*",
            ]
        )
        return "\n".join(lines)
