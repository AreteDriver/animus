"""TerminalAgent — iterative build loop with budget gating and rollback.

Wires SupervisorAgent, FilesystemTools, and CommandRunner into a
read → plan → edit → test → retry cycle.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from animus_kernel.budget.manager import BudgetManager, effective_tokens
from animus_kernel.builder.command_runner import run as run_command
from animus_kernel.sandbox.rollback import RollbackManager
from animus_kernel.tools.filesystem import FilesystemTools

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of a terminal agent build."""

    success: bool = False
    files_changed: list[str] = field(default_factory=list)
    tests_passed: bool = False
    et_consumed: float = 0.0
    iterations_used: int = 0


@dataclass
class BuildCheckpoint:
    """Checkpoint saved after each build step."""

    iteration_count: int = 0
    files_touched: list[str] = field(default_factory=list)
    test_results: dict[str, Any] = field(default_factory=dict)


class TerminalAgent:
    """Iterative builder: reads, plans, edits, tests, and rolls back on failure.

    Integrates the kernel's FilesystemTools, CommandRunner, BudgetManager,
    RollbackManager, and optionally SupervisorAgent into a bounded retry loop.
    """

    def __init__(
        self,
        filesystem_tools: FilesystemTools,
        budget_manager: BudgetManager,
        rollback_manager: RollbackManager,
        supervisor: Any = None,
        test_command: str = "python -m pytest",
        max_iterations: int = 10,
        budget_tokens_per_iteration: int = 1000,
    ) -> None:
        """Initialise the terminal agent.

        Args:
            filesystem_tools: Project-scoped filesystem operations.
            budget_manager: Token budget tracker and gatekeeper.
            rollback_manager: Snapshot / rollback facility.
            supervisor: Optional SupervisorAgent for planning and applying edits.
            test_command: Shell command used to verify the build.
            max_iterations: Upper bound on retry loops.
            budget_tokens_per_iteration: Tokens charged per iteration.
        """
        self.fs = filesystem_tools
        self.budget = budget_manager
        self.rollback = rollback_manager
        self.supervisor = supervisor
        self.test_command = test_command
        self.max_iterations = max_iterations
        self.budget_tokens_per_iteration = budget_tokens_per_iteration
        self._checkpoints: list[BuildCheckpoint] = []

    async def build(self, instruction: str, project_path: Path) -> BuildResult:
        """Run the iterative build loop.

        Steps per iteration (up to *max_iterations*):
        1. Budget gate — allocate tokens or abort.
        2. Read relevant files via FilesystemTools.
        3. Plan edits — delegate to SupervisorAgent when available.
        4. Apply edits — the supervisor's builder agent mutates files via tools.
        5. Run tests via CommandRunner.
        6. Checkpoint state (iteration_count, files_touched, test_results).
        7. On test failure, feed output back into the instruction and retry.

        On ultimate failure the pre-build snapshot is restored via
        RollbackManager.

        Args:
            instruction: High-level build instruction (e.g. "Add OAuth …").
            project_path: Absolute or relative path to the project root.

        Returns:
            BuildResult summarising success, changed files, test status,
            Effective-Tokens consumed, and iterations used.
        """
        project_path = Path(project_path).resolve()

        # Align with the filesystem tools' own root when possible.
        if project_path != getattr(self.fs, "project_root", project_path):
            logger.warning(
                "project_path %s differs from fs root %s",
                project_path,
                self.fs.project_root,
            )

        # --- Pre-build snapshot ---
        all_files = self._discover_files()
        snapshot = self.rollback.create_snapshot(
            files=all_files,
            description=f"pre-build: {instruction[:120]}",
            codebase_path=project_path,
        )

        baseline_hashes = self._hash_files(project_path, all_files)
        files_changed: list[str] = []
        tests_passed = False
        total_et = 0.0
        current_instruction = instruction
        iterations_used = 0

        try:
            for iteration in range(1, self.max_iterations + 1):
                # --- Budget gate ---
                if not self.budget.can_allocate(
                    self.budget_tokens_per_iteration,
                    agent_id="terminal_agent",
                ):
                    logger.warning(
                        "Budget gate closed at iteration %d — cannot proceed",
                        iteration,
                    )
                    break

                reserved = self.budget.allocate(
                    self.budget_tokens_per_iteration,
                    agent_id="terminal_agent",
                )
                if not reserved:
                    logger.warning(
                        "Budget allocation failed at iteration %d",
                        iteration,
                    )
                    break

                try:
                    # 1. Read relevant files
                    structure = self.fs.get_structure()
                    relevant = self._find_relevant_files(current_instruction)
                    for rel_path in relevant:
                        try:
                            self.fs.read_file(rel_path)
                        except Exception:
                            pass  # Best-effort read

                    # 2. Plan edits (supervisor produces a delegation plan)
                    # 3. Apply edits (builder agent inside supervisor uses tools)
                    if self.supervisor is not None:
                        try:
                            await self.supervisor.process_message(
                                current_instruction,
                                progress_callback=lambda stage, detail: logger.info(
                                    "supervisor [%s] %s", stage, detail
                                ),
                            )
                        except Exception as exc:
                            logger.exception(
                                "Supervisor delegation failed at iteration %d: %s",
                                iteration,
                                exc,
                            )

                    # Detect cumulative file changes against the baseline
                    current_hashes = self._hash_files(project_path, all_files)
                    files_changed = [
                        f
                        for f in all_files
                        if baseline_hashes.get(f) != current_hashes.get(f)
                    ]

                    # 4. Run tests
                    test_result = None
                    try:
                        test_result = run_command(
                            self.test_command,
                            cwd=str(project_path),
                        )
                        tests_passed = test_result.exit_code == 0
                    except Exception as exc:
                        logger.exception(
                            "Test command failed at iteration %d: %s",
                            iteration,
                            exc,
                        )
                        tests_passed = False

                    # 5. Checkpoint state
                    checkpoint = BuildCheckpoint(
                        iteration_count=iteration,
                        files_touched=list(files_changed),
                        test_results={
                            "exit_code": test_result.exit_code
                            if test_result
                            else -1,
                            "stdout": (
                                test_result.stdout[:2000]
                                if test_result
                                else ""
                            ),
                            "stderr": (
                                test_result.stderr[:2000]
                                if test_result
                                else ""
                            ),
                            "duration_ms": (
                                test_result.duration_ms
                                if test_result
                                else 0
                            ),
                            "passed": tests_passed,
                        },
                    )
                    self._checkpoint(checkpoint)

                    # 6. Log ET consumption to BudgetManager
                    record = self.budget.record_usage(
                        agent_id="terminal_agent",
                        tokens=self.budget_tokens_per_iteration,
                        operation=f"build_iteration_{iteration}",
                    )
                    total_et += effective_tokens(
                        record, self.budget.config.model_multipliers
                    )

                    iterations_used = iteration

                    if tests_passed:
                        logger.info("Tests passed at iteration %d", iteration)
                        break

                    # 7. Retry preparation
                    stdout_snip = (
                        test_result.stdout[:2000] if test_result else ""
                    )
                    stderr_snip = (
                        test_result.stderr[:2000] if test_result else ""
                    )
                    current_instruction = (
                        f"{instruction}\n\n"
                        f"Previous attempt (iteration {iteration}) "
                        f"failed tests. Fix the issues and try again.\n\n"
                        f"stdout:\n{stdout_snip}\n\n"
                        f"stderr:\n{stderr_snip}"
                    )
                finally:
                    # Release reservation regardless of outcome
                    self.budget.release(
                        self.budget_tokens_per_iteration,
                        agent_id="terminal_agent",
                    )
            else:
                # Max iterations exhausted without passing tests
                iterations_used = self.max_iterations
                logger.error(
                    "Max iterations (%d) reached without success",
                    self.max_iterations,
                )
                tests_passed = False

        except Exception as exc:
            logger.exception("Build loop aborted: %s", exc)
            tests_passed = False
        finally:
            # Rollback on failure to the pre-build snapshot
            if not tests_passed:
                rolled_back = self.rollback.rollback(
                    snapshot.id,
                    codebase_path=project_path,
                )
                if rolled_back:
                    logger.info(
                        "Rolled back to pre-build snapshot %s",
                        snapshot.id,
                    )
                else:
                    logger.error(
                        "Rollback to snapshot %s failed",
                        snapshot.id,
                    )

        return BuildResult(
            success=tests_passed,
            files_changed=files_changed,
            tests_passed=tests_passed,
            et_consumed=round(total_et, 2),
            iterations_used=iterations_used,
        )

    def get_checkpoints(self) -> list[BuildCheckpoint]:
        """Return all checkpoints recorded during the last build."""
        return list(self._checkpoints)

    def _discover_files(self) -> list[str]:
        """Return all tracked files under the project root."""
        try:
            return self.fs.glob_files("**/*")
        except Exception:
            return []

    def _hash_files(
        self,
        project_path: Path,
        files: list[str],
    ) -> dict[str, str]:
        """SHA-256 hashes of file contents to detect changes."""
        hashes: dict[str, str] = {}
        for rel_path in files:
            full = project_path / rel_path
            if full.is_file():
                try:
                    hashes[rel_path] = hashlib.sha256(
                        full.read_bytes()
                    ).hexdigest()
                except Exception:
                    pass
        return hashes

    def _find_relevant_files(self, instruction: str) -> list[str]:
        """Heuristic search for files related to the instruction."""
        keywords = [
            w for w in instruction.lower().split() if len(w) > 3
        ]
        seen: set[str] = set()
        for kw in keywords:
            try:
                result = self.fs.search_code(
                    pattern=kw,
                    max_results=20,
                )
                for match in result.matches:
                    seen.add(match.path)
            except Exception:
                pass
        return list(seen)[:50]

    def _checkpoint(self, checkpoint: BuildCheckpoint) -> None:
        """Persist a checkpoint and emit a log line."""
        self._checkpoints.append(checkpoint)
        logger.info(
            "Checkpoint iter=%d files=%d tests_passed=%s",
            checkpoint.iteration_count,
            len(checkpoint.files_touched),
            checkpoint.test_results.get("passed"),
        )
