"""Self-improvement orchestrator - coordinates the entire self-improvement workflow."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .analyzer import CodebaseAnalyzer, ImprovementSuggestion
from .approval import ApprovalGate, ApprovalStage, ApprovalStatus
from .pr_manager import PRManager, PullRequest
from .rollback import RollbackManager, Snapshot
from .safety import SafetyChecker, SafetyConfig, SafetyViolation
from .sandbox import Sandbox, SandboxResult

if TYPE_CHECKING:
    from animus_forge.agents.provider_wrapper import AgentProvider

logger = logging.getLogger(__name__)


class WorkflowStage(str, Enum):
    """Stages of the self-improvement workflow."""

    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    AWAITING_APPLY_APPROVAL = "awaiting_apply_approval"
    APPLYING = "applying"
    CREATING_PR = "creating_pr"
    AWAITING_MERGE_APPROVAL = "awaiting_merge_approval"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ImprovementPlan:
    """Plan for implementing improvements."""

    id: str
    title: str
    description: str
    suggestions: list[ImprovementSuggestion]
    implementation_steps: list[str]
    estimated_files: list[str]
    estimated_lines: int
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementResult:
    """Result of a self-improvement run."""

    success: bool
    stage_reached: WorkflowStage
    plan: ImprovementPlan | None = None
    snapshot: Snapshot | None = None
    sandbox_result: SandboxResult | None = None
    pull_request: PullRequest | None = None
    error: str | None = None
    violations: list[SafetyViolation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SelfImproveOrchestrator:
    """Orchestrates the complete self-improvement workflow.

    Workflow:
    1. Analyze codebase for improvement opportunities
    2. Create improvement plan
    3. Get human approval for plan
    4. Implement changes in sandbox
    5. Run tests in sandbox
    6. Get human approval to apply changes
    7. Create snapshot for rollback
    8. Apply changes to working tree
    9. Create PR
    10. Get human approval to merge
    """

    def __init__(
        self,
        codebase_path: Path | str = ".",
        provider: AgentProvider | None = None,
        config: SafetyConfig | None = None,
        tool_registry: Any = None,
        db_path: Path | str | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            codebase_path: Path to codebase root.
            provider: Optional AI provider for intelligent improvements.
            config: Safety configuration.
            tool_registry: Optional ForgeToolRegistry for tool-equipped generation.
            db_path: Path to SQLite database for persistent approvals and
                checkpoints. Defaults to ``<codebase>/.gorgon/self_improve.db``.
        """
        self.codebase_path = Path(codebase_path)
        self.provider = provider
        self.config = config or SafetyConfig.load()
        self.tool_registry = tool_registry

        # Persistent state
        self._db_path = Path(db_path) if db_path else self.codebase_path / ".gorgon/self_improve.db"
        approval_backend = self._create_backend()

        # Initialize components
        self.safety_checker = SafetyChecker(self.config)
        self.analyzer = CodebaseAnalyzer(provider, self.codebase_path)
        self.approval_gate = ApprovalGate(backend=approval_backend)
        self.rollback_manager = RollbackManager(
            self.codebase_path / ".gorgon/snapshots",
            self.config.max_snapshots,
        )
        self.pr_manager = PRManager(self.codebase_path, self.config.branch_prefix)

        self._current_stage = WorkflowStage.IDLE
        self._current_plan: ImprovementPlan | None = None

    def _is_local_provider(self) -> bool:
        """Check if the current provider is a local LLM (Ollama)."""
        if not self.provider:
            return True
        provider_name = type(self.provider).__name__.lower()
        if hasattr(self.provider, "provider"):
            provider_name = type(self.provider.provider).__name__.lower()
        return "ollama" in provider_name

    def _create_backend(self):
        """Create a SQLite backend for persistent state."""
        try:
            from animus_forge.state.backends import SQLiteBackend

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            return SQLiteBackend(str(self._db_path))
        except Exception:
            logger.debug("SQLite backend unavailable, using in-memory approvals")
            return None

    @property
    def current_stage(self) -> WorkflowStage:
        """Get current workflow stage."""
        return self._current_stage

    async def run(
        self,
        focus_category: str | None = None,
        auto_approve: bool = False,
    ) -> ImprovementResult:
        """Run the complete self-improvement workflow.

        Args:
            focus_category: Optional category to focus on.
            auto_approve: If True, auto-approve all stages (for testing).

        Returns:
            Result of the improvement run.
        """
        try:
            # Stage 1: Analyze
            self._current_stage = WorkflowStage.ANALYZING
            logger.info("Starting analysis...")

            # Map focus_category string to ImprovementCategory enum
            categories = None
            if focus_category:
                from animus_forge.self_improve.analyzer import ImprovementCategory

                category_map = {c.value: c for c in ImprovementCategory}
                if focus_category.lower() in category_map:
                    categories = [category_map[focus_category.lower()]]
                else:
                    logger.warning(
                        f"Unknown focus category '{focus_category}', analyzing all. "
                        f"Valid: {list(category_map.keys())}"
                    )

            analysis = self.analyzer.analyze(categories=categories)

            if not analysis.suggestions:
                logger.info("No improvement opportunities found")
                self._current_stage = WorkflowStage.COMPLETE
                return ImprovementResult(
                    success=True,
                    stage_reached=WorkflowStage.COMPLETE,
                    metadata={"message": "No improvements needed"},
                )

            # Stage 2: Create plan
            self._current_stage = WorkflowStage.PLANNING
            budget = self.config.max_lines_changed
            selected = []
            budget_used = 0
            for s in analysis.suggestions:
                # Estimate actual change as ~30% of affected code
                change_est = max(s.estimated_lines * 3 // 10, 10)
                if budget_used + change_est <= budget:
                    selected.append(s)
                    budget_used += change_est
                if len(selected) >= 2:
                    break
            plan = self._create_plan(selected or analysis.suggestions[:1])
            self._current_plan = plan

            # Check safety violations (use budget estimate, not raw function lengths)
            violations = self.safety_checker.check_changes(
                files_modified=plan.estimated_files,
                files_added=[],
                files_deleted=[],
                lines_changed=budget_used,
                category=focus_category,
            )

            if self.safety_checker.has_blocking_violations(violations):
                self._current_stage = WorkflowStage.FAILED
                return ImprovementResult(
                    success=False,
                    stage_reached=WorkflowStage.PLANNING,
                    plan=plan,
                    violations=violations,
                    error="Safety violations prevent this improvement",
                )

            # Stage 3: Get plan approval
            if self.config.human_approval_plan:
                self._current_stage = WorkflowStage.AWAITING_PLAN_APPROVAL
                approval = self.approval_gate.request_approval(
                    stage=ApprovalStage.PLAN,
                    title=f"Self-Improvement Plan: {plan.title}",
                    description=plan.description,
                    details={
                        "suggestions": [s.title for s in plan.suggestions],
                        "estimated_files": plan.estimated_files,
                        "estimated_lines": plan.estimated_lines,
                    },
                )

                if auto_approve:
                    self.approval_gate.auto_approve_for_testing(approval.id)
                else:
                    status = await self.approval_gate.wait_for_approval(approval)
                    if status == ApprovalStatus.REJECTED:
                        self._current_stage = WorkflowStage.FAILED
                        return ImprovementResult(
                            success=False,
                            stage_reached=WorkflowStage.AWAITING_PLAN_APPROVAL,
                            plan=plan,
                            error="Plan was rejected",
                        )
                    if status == ApprovalStatus.EXPIRED:
                        self._current_stage = WorkflowStage.FAILED
                        return ImprovementResult(
                            success=False,
                            stage_reached=WorkflowStage.AWAITING_PLAN_APPROVAL,
                            plan=plan,
                            error="Plan approval timed out",
                        )

            # Stage 4: Generate changes via AI
            self._current_stage = WorkflowStage.IMPLEMENTING
            logger.info("Generating code changes...")
            changes = await self._generate_changes(plan)

            if not changes:
                self._current_stage = WorkflowStage.FAILED
                return ImprovementResult(
                    success=False,
                    stage_reached=WorkflowStage.IMPLEMENTING,
                    plan=plan,
                    error="No changes generated",
                )

            # Stage 5: Test in sandbox
            self._current_stage = WorkflowStage.TESTING
            logger.info("Testing changes in sandbox...")
            with Sandbox(self.codebase_path, timeout=self.config.sandbox_timeout) as sandbox:
                applied = await sandbox.apply_changes(changes)
                if not applied:
                    self._current_stage = WorkflowStage.FAILED
                    return ImprovementResult(
                        success=False,
                        stage_reached=WorkflowStage.TESTING,
                        plan=plan,
                        error="Failed to apply changes to sandbox",
                    )
                sandbox_result = await sandbox.validate_changes()

            if not sandbox_result.tests_passed:
                if self.config.auto_rollback_on_test_failure:
                    self._current_stage = WorkflowStage.FAILED
                    return ImprovementResult(
                        success=False,
                        stage_reached=WorkflowStage.TESTING,
                        plan=plan,
                        sandbox_result=sandbox_result,
                        error="Tests failed in sandbox",
                    )

            # Stage 6: Get apply approval
            if self.config.human_approval_apply:
                self._current_stage = WorkflowStage.AWAITING_APPLY_APPROVAL
                approval = self.approval_gate.request_approval(
                    stage=ApprovalStage.APPLY,
                    title=f"Apply Changes: {plan.title}",
                    description="Changes have passed testing. Ready to apply to working tree.",
                    details={
                        "sandbox_status": sandbox_result.status.value,
                        "tests_passed": sandbox_result.tests_passed,
                        "lint_passed": sandbox_result.lint_passed,
                        "files_changed": list(changes.keys()),
                    },
                )

                if auto_approve:
                    self.approval_gate.auto_approve_for_testing(approval.id)
                else:
                    status = await self.approval_gate.wait_for_approval(approval)
                    if status == ApprovalStatus.REJECTED:
                        self._current_stage = WorkflowStage.FAILED
                        return ImprovementResult(
                            success=False,
                            stage_reached=WorkflowStage.AWAITING_APPLY_APPROVAL,
                            plan=plan,
                            sandbox_result=sandbox_result,
                            error="Apply was rejected",
                        )
                    if status == ApprovalStatus.EXPIRED:
                        self._current_stage = WorkflowStage.FAILED
                        return ImprovementResult(
                            success=False,
                            stage_reached=WorkflowStage.AWAITING_APPLY_APPROVAL,
                            plan=plan,
                            sandbox_result=sandbox_result,
                            error="Apply approval timed out",
                        )

            # Stage 7: Create snapshot
            snapshot = self.rollback_manager.create_snapshot(
                files=plan.estimated_files,
                description=f"Before: {plan.title}",
                codebase_path=self.codebase_path,
            )

            # Stage 8: Apply changes to working tree
            self._current_stage = WorkflowStage.APPLYING
            logger.info("Applying changes to working tree...")
            self._apply_changes(changes)

            # Stage 9: Create PR
            self._current_stage = WorkflowStage.CREATING_PR
            branch = self.pr_manager.create_branch(plan.id)
            pr = self.pr_manager.create_pr(
                branch=branch,
                title=plan.title,
                description=plan.description,
                files_changed=plan.estimated_files,
                draft=True,
            )

            # Stage 10: Get merge approval
            if self.config.human_approval_merge:
                self._current_stage = WorkflowStage.AWAITING_MERGE_APPROVAL
                approval = self.approval_gate.request_approval(
                    stage=ApprovalStage.MERGE,
                    title=f"Merge PR: {plan.title}",
                    description="PR is ready for merge.",
                    details={"pr_url": pr.url},
                )

                if not auto_approve:
                    status = await self.approval_gate.wait_for_approval(approval)
                    if status != ApprovalStatus.APPROVED:
                        return ImprovementResult(
                            success=True,
                            stage_reached=WorkflowStage.AWAITING_MERGE_APPROVAL,
                            plan=plan,
                            snapshot=snapshot,
                            sandbox_result=sandbox_result,
                            pull_request=pr,
                            error=f"Merge approval {status.value}",
                        )

            self._current_stage = WorkflowStage.COMPLETE
            return ImprovementResult(
                success=True,
                stage_reached=WorkflowStage.COMPLETE,
                plan=plan,
                snapshot=snapshot,
                sandbox_result=sandbox_result,
                pull_request=pr,
            )

        except Exception as e:
            logger.error(f"Self-improvement failed: {e}", exc_info=True)
            self._current_stage = WorkflowStage.FAILED
            return ImprovementResult(
                success=False,
                stage_reached=self._current_stage,
                error=str(e),
            )

    async def _generate_changes(self, plan: ImprovementPlan) -> dict[str, str]:
        """Generate code changes using the AI provider.

        Builds a prompt from the plan, reads current file contents,
        calls the provider, and parses the response as a JSON dict
        of {file_path: new_content}.

        Args:
            plan: The improvement plan to implement.

        Returns:
            Dict mapping relative file paths to new file contents.
            Empty dict if generation fails.
        """
        if not self.provider:
            logger.warning("No AI provider — cannot generate changes")
            return {}

        # Extract targeted function snippets instead of whole files
        # This gives the LLM manageable context and produces compact responses
        function_snippets: dict[str, list[tuple[str, str, int, int]]] = {}
        file_full_contents: dict[str, str] = {}

        for suggestion in plan.suggestions:
            func_name = ""
            if "Function `" in suggestion.description:
                func_name = suggestion.description.split("Function `")[1].split("`")[0]
            elif suggestion.title.startswith("Long function: "):
                func_name = suggestion.title.replace("Long function: ", "")

            for file_path in suggestion.affected_files:
                full_path = self.codebase_path / file_path
                if not full_path.exists():
                    continue
                try:
                    content = full_path.read_text()
                    file_full_contents[file_path] = content
                except OSError:
                    continue

                if func_name:
                    # Extract the function body
                    lines = content.splitlines()
                    func_start = None
                    func_indent = 0
                    for i, line in enumerate(lines):
                        stripped = line.lstrip()
                        if stripped.startswith(f"def {func_name}(") or stripped.startswith(
                            f"async def {func_name}("
                        ):
                            func_start = i
                            func_indent = len(line) - len(stripped)
                            break
                    if func_start is not None:
                        func_end = func_start + 1
                        for i in range(func_start + 1, len(lines)):
                            line = lines[i]
                            if line.strip() == "":
                                continue
                            indent = len(line) - len(line.lstrip())
                            if indent <= func_indent and line.strip():
                                break
                            func_end = i + 1
                        # Cap at 200 lines to keep LLM response manageable
                        if func_end - func_start > 200:
                            func_end = func_start + 200
                        snippet = "\n".join(lines[func_start:func_end])
                        if file_path not in function_snippets:
                            function_snippets[file_path] = []
                        function_snippets[file_path].append(
                            (func_name, snippet, func_start, func_end)
                        )

        # Build the prompt with targeted snippets
        snippets_text_parts = []
        for file_path, snippets in function_snippets.items():
            for func_name, snippet, start, end in snippets:
                snippets_text_parts.append(
                    f"=== {file_path} :: {func_name}() (lines {start + 1}-{end}) ===\n{snippet}"
                )
        snippets_text = "\n\n".join(snippets_text_parts)

        suggestions_text = "\n".join(
            f"- {s.title}: {s.description}"
            + (f"\n  Hints: {s.implementation_hints}" if s.implementation_hints else "")
            for s in plan.suggestions
        )

        example_file = plan.estimated_files[0] if plan.estimated_files else "example.py"

        # Detect suggestion type to pick the right prompt strategy
        is_documentation = all(
            "docstring" in s.title.lower() or "documentation" in s.category.value.lower()
            for s in plan.suggestions
        )

        if is_documentation:
            # Documentation: generate docstrings programmatically or with minimal LLM help
            return await self._generate_docstring_changes(plan, file_full_contents)

        elif snippets_text:
            # Function refactoring: show targeted function snippets
            prompt = f"""You are refactoring a Python function to be shorter and more readable.

## Task: {plan.title}
{plan.description}

## Suggestions:
{suggestions_text}

## Current function code:
{snippets_text}

## Rules:
1. NEVER rename the function or change its signature.
2. Extract logical blocks into helper methods/functions.
3. Keep the original function as a dispatcher that calls the helpers.
4. Add the helper functions right before the original function.
5. Preserve all imports and type hints.

## Output format:
For each file, output the file path on its own line prefixed with FILE:, then the complete
refactored code in a Python code block. Example:

FILE: {example_file}
```python
def _helper():
    pass

def original_func():
    _helper()
```

Output ONLY the FILE: markers and code blocks. No other text."""

        else:
            # Generic fallback with truncated file contents
            fallback_text = "\n\n".join(
                f"=== {path} ===\n{content[:4000]}"
                for path, content in file_full_contents.items()
            )
            prompt = f"""You are implementing code improvements for a Python project.

## Task: {plan.title}
{plan.description}

## Suggestions:
{suggestions_text}

## Current code:
{fallback_text}

## Rules:
1. Keep changes minimal and focused.
2. Do NOT rename functions or change signatures.

## Output format:
For each file, output FILE: then the path, then a python code block with the COMPLETE file.

FILE: {example_file}
```python
complete file content
```

Output ONLY FILE: markers and code blocks. No other text."""""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a senior Python engineer implementing code improvements. "
                    "Use tools to read files and understand the codebase before making changes. "
                    "Return only valid JSON in your final response.",
                },
                {"role": "user", "content": prompt},
            ]

            if self.tool_registry is not None and hasattr(self.provider, "complete_with_tools"):
                logger.info("Using tool-equipped generation")
                response = await self.provider.complete_with_tools(
                    messages=messages,
                    tool_registry=self.tool_registry,
                    max_iterations=6,
                    max_tokens=16384,
                )
            else:
                response = await self.provider.complete(messages, max_tokens=16384)

            logger.info(f"AI response length: {len(response)} chars")
            changes = self._parse_changes_response(response)

            # If we have function snippets, splice the LLM output back into full files
            # For documentation changes, the LLM returns complete files — use as-is
            if function_snippets and not is_documentation:
                spliced: dict[str, str] = {}
                for file_path, new_code in changes.items():
                    if file_path in function_snippets and file_path in file_full_contents:
                        full_content = file_full_contents[file_path]
                        lines = full_content.splitlines()
                        # Replace the function region with the new code
                        # Use the first (and usually only) function snippet's location
                        _, _, start, end = function_snippets[file_path][0]
                        new_lines = lines[:start] + new_code.splitlines() + lines[end:]
                        spliced[file_path] = "\n".join(new_lines) + "\n"
                    else:
                        spliced[file_path] = new_code
                changes = spliced

            # Validate: filter out critical files
            safe_changes: dict[str, str] = {}
            for file_path, content in changes.items():
                if self.safety_checker.is_protected_file(file_path):
                    logger.warning(f"Skipping protected file: {file_path}")
                    continue
                safe_changes[file_path] = content

            logger.info(f"Generated changes for {len(safe_changes)} files")
            return safe_changes

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {}

    def _parse_changes_response(self, response: str) -> dict[str, str]:
        """Parse the AI response into a file changes dict.

        Supports two formats:
        1. Diff format: {"file": "path", "changes": [{"old": "...", "new": "..."}]}
        2. Legacy full-content: {"path/to/file.py": "complete content"}

        Args:
            response: Raw AI response string.

        Returns:
            Dict mapping file paths to new content (for legacy)
            or to change instructions (for diff format).
        """
        text = response.strip()

        # Strip markdown code fences wrapping the entire response
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Fix trailing commas (common LLM output)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        # Fix double-escaped quotes from local LLMs (\\\" -> \")
        text = text.replace('\\\\"', '\\"')
        text = text.replace("\\\\'", "\\'")

        # Fix single-quoted strings (LLMs mix quotes in JSON)
        # Replace 'value' with "value" when used as JSON values
        text = re.sub(
            r"""(?<=:\s)'([^']*)'""",
            r'"\1"',
            text,
        )

        # Try FILE: marker format first (most reliable for local LLMs)
        file_marker_results = self._parse_file_markers(text)
        if file_marker_results:
            return file_marker_results

        parsed = self._try_parse_json(text)
        if parsed is None:
            # Fallback: extract outermost JSON structure via regex
            match = re.search(r"[\[{][\s\S]*[\]}]", text)
            if match:
                candidate = match.group()
                candidate = re.sub(r",\s*}", "}", candidate)
                candidate = re.sub(r",\s*]", "]", candidate)
                parsed = self._try_parse_json(candidate)

        if parsed is None:
            logger.warning("Failed to parse AI response as JSON")
            return {}

        return self._normalize_parsed_changes(parsed)

    async def _generate_docstring_changes(
        self,
        plan: ImprovementPlan,
        file_full_contents: dict[str, str],
    ) -> dict[str, str]:
        """Generate docstring additions using LLM for content, programmatic for insertion.

        For each missing docstring suggestion, asks the LLM to write just the docstring
        text, then programmatically inserts it at the correct location.
        """
        changes: dict[str, str] = {}

        for suggestion in plan.suggestions:
            for file_path in suggestion.affected_files:
                if file_path not in file_full_contents:
                    continue
                content = file_full_contents[file_path]

                if "module docstring" in suggestion.title.lower():
                    # Module docstring — add at top
                    if file_path not in changes:
                        changes[file_path] = content

                    # Ask LLM for a one-line module docstring
                    if self.provider:
                        resp = await self.provider.complete(
                            [{"role": "user", "content": (
                                f"Write a one-line Python module docstring for a file named "
                                f"'{file_path}'. Return ONLY the docstring text (no quotes), "
                                f"nothing else. Example: Utilities for data processing."
                            )}],
                            max_tokens=100,
                        )
                        docstring_text = resp.strip().strip('"').strip("'")
                    else:
                        # Derive from filename
                        module_name = Path(file_path).stem.replace("_", " ").title()
                        docstring_text = f"{module_name} module."

                    new_content = f'"""{docstring_text}"""\n\n{changes[file_path]}'
                    changes[file_path] = new_content

                elif "Missing docstring:" in suggestion.title:
                    # Function docstring — find the function and insert
                    func_name = suggestion.title.replace("Missing docstring: ", "")
                    if file_path not in changes:
                        changes[file_path] = content

                    current = changes[file_path]
                    lines = current.splitlines()

                    # Find the function definition
                    for i, line in enumerate(lines):
                        stripped = line.lstrip()
                        if stripped.startswith(f"def {func_name}("):
                            indent = len(line) - len(stripped) + 4  # Function body indent

                            # Ask LLM for the docstring
                            if self.provider:
                                # Show the function signature + first 5 body lines
                                func_context = "\n".join(lines[i:i + 6])
                                resp = await self.provider.complete(
                                    [{"role": "user", "content": (
                                        f"Write a concise one-line Python docstring for this function. "
                                        f"Return ONLY the docstring text (no triple quotes), nothing else.\n\n"
                                        f"{func_context}"
                                    )}],
                                    max_tokens=100,
                                )
                                docstring_text = resp.strip().strip('"').strip("'")
                            else:
                                docstring_text = f"{func_name.replace('_', ' ').capitalize()}."

                            # Insert docstring after the def line
                            docstring_line = " " * indent + f'"""{docstring_text}"""'
                            lines.insert(i + 1, docstring_line)
                            changes[file_path] = "\n".join(lines) + "\n"
                            break

        logger.info(f"Generated docstring changes for {len(changes)} files")
        return changes

    def _parse_file_markers(self, text: str) -> dict[str, str]:
        """Parse FILE: marker format from LLM response.

        Format:
            FILE: path/to/file.py
            ```python
            code here
            ```

        Returns:
            Dict mapping file paths to code content, or empty dict if no markers found.
        """
        results: dict[str, str] = {}
        # Split on FILE: markers
        parts = re.split(r"(?:^|\n)FILE:\s*", text)
        for part in parts[1:]:  # Skip everything before first FILE:
            lines = part.strip().splitlines()
            if not lines:
                continue
            file_path = lines[0].strip()
            # Extract code from the code block
            code_lines = lines[1:]
            code_text = "\n".join(code_lines)
            # Strip code fences
            code_text = code_text.strip()
            if code_text.startswith("```"):
                fence_lines = code_text.splitlines()
                fence_lines = fence_lines[1:]  # Remove opening fence
                if fence_lines and fence_lines[-1].strip() == "```":
                    fence_lines = fence_lines[:-1]  # Remove closing fence
                code_text = "\n".join(fence_lines)
            if file_path and code_text.strip():
                results[file_path] = code_text
                logger.info(f"Parsed FILE marker: {file_path} ({len(code_text)} chars)")
        return results

    def _try_parse_json(self, text: str):
        """Attempt JSON parse, return None on failure."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _normalize_parsed_changes(self, parsed) -> dict[str, str]:
        """Normalize parsed JSON into {file_path: content_or_diff} dict."""
        # Handle array of file change objects
        if isinstance(parsed, list):
            result = {}
            for item in parsed:
                if isinstance(item, dict) and "file" in item:
                    result[item["file"]] = self._apply_diff_changes(item)
            return result

        if isinstance(parsed, dict):
            # Diff format: {"file": "path", "changes": [...]}
            if "file" in parsed and "changes" in parsed:
                return {parsed["file"]: self._apply_diff_changes(parsed)}
            # Legacy format: {"path": "content"}
            return {k: str(v) for k, v in parsed.items() if isinstance(k, str)}

        return {}

    def _apply_diff_changes(self, item: dict) -> str:
        """Apply diff changes to the original file content.

        Reads the original file content and applies old→new replacements.
        Returns the modified content.
        """
        file_path = item.get("file", "")
        changes = item.get("changes", [])

        # Read original file
        full_path = self.codebase_path / file_path
        if not full_path.exists():
            logger.warning(f"File not found for diff: {file_path}")
            return ""

        try:
            content = full_path.read_text()
        except OSError as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return ""

        # Apply each old→new replacement
        applied = 0
        for change in changes:
            if not isinstance(change, dict):
                continue
            old = change.get("old", "")
            new = change.get("new", "")
            if not old:
                continue
            if old in content:
                content = content.replace(old, new, 1)
                applied += 1
            else:
                # Fuzzy fallback: try matching with normalized whitespace
                normalized_old = " ".join(old.split())
                for line_idx, line in enumerate(content.splitlines()):
                    normalized_line = " ".join(line.split())
                    if normalized_old.startswith(normalized_line) and len(normalized_line) > 20:
                        # Found approximate start — try block match
                        content_lines = content.splitlines()
                        old_lines = old.splitlines()
                        match_len = 0
                        for j, old_line in enumerate(old_lines):
                            if line_idx + j < len(content_lines):
                                if " ".join(old_line.split()) == " ".join(content_lines[line_idx + j].split()):
                                    match_len += 1
                        if match_len >= len(old_lines) * 0.7:  # 70% line match threshold
                            actual_old = "\n".join(content_lines[line_idx:line_idx + len(old_lines)])
                            content = content.replace(actual_old, new, 1)
                            applied += 1
                            logger.info(f"Fuzzy matched {match_len}/{len(old_lines)} lines in {file_path}")
                            break
                else:
                    logger.warning(f"Could not find match in {file_path}: {old[:60]}...")

        logger.info(f"Applied {applied}/{len(changes)} changes to {file_path}")
        return content

    def _apply_changes(self, changes: dict[str, str]) -> None:
        """Apply generated changes to the working tree.

        Args:
            changes: Dict mapping relative file paths to new content.
        """
        for file_path, content in changes.items():
            full_path = self.codebase_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            logger.info(f"Applied changes to {file_path}")

    def _create_plan(self, suggestions: list[ImprovementSuggestion]) -> ImprovementPlan:
        """Create an improvement plan from suggestions.

        Args:
            suggestions: List of improvement suggestions.

        Returns:
            Created plan.
        """
        import uuid

        # Determine provider capability tier
        is_local = self._is_local_provider()

        if is_local:
            # Local LLMs (Ollama): prefer small, precise fixes
            # Priority: bare excepts > TODOs > missing docstrings > short refactors
            small_fixes = [s for s in suggestions if s.estimated_lines <= 20]
            medium_fixes = [s for s in suggestions if 20 < s.estimated_lines <= 100]

            if small_fixes:
                suggestions = sorted(small_fixes, key=lambda s: s.priority)[:3]
            elif medium_fixes:
                suggestions = sorted(medium_fixes, key=lambda s: s.priority)[:1]
            else:
                # All suggestions are large — take smallest one
                suggestions = sorted(suggestions, key=lambda s: s.estimated_lines)[:1]
        else:
            # Cloud LLMs (Anthropic/OpenAI): can handle larger refactors
            manageable = [s for s in suggestions if s.estimated_lines <= 500]
            if manageable:
                suggestions = manageable[:3]
            else:
                suggestions = sorted(suggestions, key=lambda s: s.estimated_lines)[:1]

        # Aggregate files and estimates
        all_files = set()
        total_lines = 0
        for suggestion in suggestions:
            all_files.update(suggestion.affected_files)
            total_lines += suggestion.estimated_lines

        # Generate title and description
        if len(suggestions) == 1:
            title = suggestions[0].title
            description = suggestions[0].description
        else:
            title = f"Multiple improvements ({len(suggestions)} items)"
            description = "\n".join(f"- {s.title}" for s in suggestions)

        # Generate implementation steps
        steps = [f"Implement: {s.title}" for s in suggestions]

        return ImprovementPlan(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
            suggestions=suggestions,
            implementation_steps=steps,
            estimated_files=list(all_files),
            estimated_lines=total_lines,
        )

    def rollback(self, snapshot_id: str) -> bool:
        """Rollback to a previous snapshot.

        Args:
            snapshot_id: ID of snapshot to rollback to.

        Returns:
            True if rollback succeeded.
        """
        result = self.rollback_manager.rollback(snapshot_id, self.codebase_path)
        if result:
            self._current_stage = WorkflowStage.ROLLED_BACK
        return result

    def get_status(self) -> dict[str, Any]:
        """Get current orchestrator status.

        Returns:
            Status dictionary.
        """
        return {
            "stage": self._current_stage.value,
            "current_plan": self._current_plan.title if self._current_plan else None,
            "pending_approvals": len(self.approval_gate.get_pending()),
            "snapshots_available": len(self.rollback_manager.list_snapshots()),
            "persistent": self._db_path.exists() if self._db_path else False,
        }

    def get_pending_approvals(self, stage: ApprovalStage | None = None) -> list:
        """Get pending approval requests, including those persisted to database.

        This allows resuming after process restarts — pending approvals are
        recovered from SQLite.

        Args:
            stage: Optional filter by approval stage.

        Returns:
            List of pending ApprovalRequest objects.
        """
        return self.approval_gate.get_pending(stage)

    def get_approval_history(self, limit: int = 50) -> list:
        """Get historical approval decisions from persistent storage.

        Args:
            limit: Max items to return.

        Returns:
            List of ApprovalRequest objects.
        """
        return self.approval_gate.get_history(limit=limit)

    def list_snapshots(self) -> list:
        """List available rollback snapshots.

        Returns:
            List of Snapshot objects.
        """
        return self.rollback_manager.list_snapshots()
