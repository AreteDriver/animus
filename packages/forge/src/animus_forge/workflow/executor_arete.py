"""Arete Tools step handler mixin for WorkflowExecutor.

Provides handlers for signal-audit, autopsy-analyze, and verdict-capture
step types. Each integrates with the corresponding Arete Tool via direct
Python import (preferred) or subprocess fallback.
"""

from __future__ import annotations

import json
import logging
import subprocess

from .loader import StepConfig

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation when tools not installed
try:
    from signal_audit.analyzers.quality import run_quality_audit

    HAS_SIGNAL = True
except ImportError:
    HAS_SIGNAL = False

try:
    from autopsy.analyzer import analyze_failure

    HAS_AUTOPSY = True
except ImportError:
    HAS_AUTOPSY = False

try:
    from verdict.store import DecisionStore

    HAS_VERDICT = True
except ImportError:
    HAS_VERDICT = False


def _substitute_context(value: str, context: dict) -> str:
    """Replace ${key} placeholders with context values."""
    for key, val in context.items():
        if isinstance(val, str):
            value = value.replace(f"${{{key}}}", val)
    return value


def _run_subprocess(cmd: list[str]) -> dict:
    """Run a CLI tool as subprocess fallback, return parsed JSON output."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess {cmd[0]} failed (exit {result.returncode}): {result.stderr}"
        )
    return json.loads(result.stdout)


class AreteToolsHandlerMixin:
    """Mixin providing Arete Tool step handlers.

    Expects the following attributes from the host class:
    - dry_run: bool
    """

    def _execute_signal_audit(self, step: StepConfig, context: dict) -> dict:
        """Execute a signal-audit quality gate step.

        Params:
            file: Path to file to audit (supports ${context} substitution)
            min_score: Minimum passing score (0-100, optional)
            output_format: Output format (default: "json")
        """
        file_path = _substitute_context(step.params.get("file", ""), context)
        min_score = step.params.get("min_score")

        if not file_path:
            raise ValueError("signal_audit step requires 'file' param")

        if getattr(self, "dry_run", False):
            return {
                "score": 85,
                "grade": "B",
                "dimensions": {},
                "flags": [],
                "file": file_path,
                "dry_run": True,
            }

        if HAS_SIGNAL:
            result = run_quality_audit(file_path)
        else:
            result = _run_subprocess(["signal-audit", "audit", file_path, "--format", "json"])

        score = result.get("score", 0)
        if min_score is not None and score < min_score:
            raise RuntimeError(f"Signal audit failed: score {score} < min_score {min_score}")

        return {
            "score": score,
            "grade": result.get("grade", ""),
            "dimensions": result.get("dimensions", {}),
            "flags": result.get("flags", []),
            "file": file_path,
        }

    def _execute_autopsy_analyze(self, step: StepConfig, context: dict) -> dict:
        """Execute an autopsy-analyze post-failure forensics step.

        Params:
            error_text: Error text to analyze (supports ${context} substitution)
            workflow_id: ID of the failed workflow (optional)
        """
        error_text = _substitute_context(step.params.get("error_text", ""), context)
        workflow_id = step.params.get("workflow_id", "")

        if not error_text:
            raise ValueError("autopsy_analyze step requires 'error_text' param")

        if getattr(self, "dry_run", False):
            return {
                "failure_type": "unknown",
                "error_chain": [error_text],
                "recommendations": ["Investigate further"],
                "loops_detected": False,
                "workflow_id": workflow_id,
                "dry_run": True,
            }

        if HAS_AUTOPSY:
            result = analyze_failure(error_text, workflow_id=workflow_id)
        else:
            cmd = ["autopsy", "analyze", "--format", "json"]
            proc = subprocess.run(
                cmd,
                input=error_text,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Subprocess autopsy failed (exit {proc.returncode}): {proc.stderr}"
                )
            result = json.loads(proc.stdout)

        return {
            "failure_type": result.get("failure_type", "unknown"),
            "error_chain": result.get("error_chain", []),
            "recommendations": result.get("recommendations", []),
            "loops_detected": result.get("loops_detected", False),
            "workflow_id": workflow_id,
        }

    def _execute_verdict_capture(self, step: StepConfig, context: dict) -> dict:
        """Execute a verdict-capture decision logging step.

        Params:
            title: Decision title (supports ${context} substitution)
            reasoning: Decision reasoning
            alternatives: List of alternatives considered
            category: Decision category (e.g. "architecture", "tooling")
        """
        title = _substitute_context(step.params.get("title", ""), context)
        reasoning = _substitute_context(step.params.get("reasoning", ""), context)
        alternatives = step.params.get("alternatives", [])
        category = step.params.get("category", "general")

        if not title:
            raise ValueError("verdict_capture step requires 'title' param")

        if getattr(self, "dry_run", False):
            return {
                "decision_id": "dry-run-001",
                "title": title,
                "category": category,
                "review_date": "",
                "dry_run": True,
            }

        if HAS_VERDICT:
            store = DecisionStore()
            decision = store.record(
                title=title,
                reasoning=reasoning,
                alternatives=alternatives,
                category=category,
            )
            result = {
                "decision_id": decision.get("id", ""),
                "title": title,
                "category": category,
                "review_date": decision.get("review_date", ""),
            }
        else:
            alt_json = json.dumps(alternatives)
            proc_result = _run_subprocess(
                [
                    "verdict",
                    "record",
                    "--title",
                    title,
                    "--reasoning",
                    reasoning,
                    "--alternatives",
                    alt_json,
                    "--category",
                    category,
                    "--format",
                    "json",
                ]
            )
            result = {
                "decision_id": proc_result.get("id", ""),
                "title": title,
                "category": category,
                "review_date": proc_result.get("review_date", ""),
            }

        return result
