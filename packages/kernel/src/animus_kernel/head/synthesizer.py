"""Synthesizer for Animus Head — wraps raw tool output in natural language.

Converts mechanical tool responses into conversational summaries
so the user gets a human-friendly answer, not raw JSON or shell output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of synthesizing tool output."""

    summary: str
    detail: str = ""  # Optional raw output preserved
    needs_follow_up: bool = False
    suggested_next: list[str] = None  # type: ignore[assignment]


class HeadSynthesizer:
    """Wraps raw tool outputs in natural language summaries.

    Uses lightweight templates — no model call required.
    """

    def synthesize(self, tool_name: str, arguments: dict, result: str) -> SynthesisResult:
        """Synthesize a single tool call result.

        Args:
            tool_name: Name of the tool that was executed
            arguments: Arguments passed to the tool
            result: Raw string result from the tool

        Returns:
            SynthesisResult with human-friendly summary
        """
        if not result or not result.strip():
            return SynthesisResult(
                summary=f"*{tool_name}* returned no output.",
                detail="",
            )

        handler = getattr(self, f"_synthesize_{tool_name}", self._synthesize_generic)
        return handler(arguments, result)

    def synthesize_multi(self, tool_results: list[tuple[str, dict, str]]) -> SynthesisResult:
        """Synthesize multiple tool results into a unified summary.

        Args:
            tool_results: List of (tool_name, arguments, result) tuples

        Returns:
            SynthesisResult combining all outputs
        """
        summaries = []
        for tool_name, args, result in tool_results:
            synth = self.synthesize(tool_name, args, result)
            summaries.append(f"• {synth.summary}")

        return SynthesisResult(
            summary="\n".join(summaries),
            detail="",
            needs_follow_up=False,
        )

    # ------------------------------------------------------------------
    # Tool-specific synthesizers
    # ------------------------------------------------------------------

    def _synthesize_read_file(self, arguments: dict, result: str) -> SynthesisResult:
        path = arguments.get("path", "file")
        lines = result.strip().splitlines()
        if len(lines) > 20:
            return SynthesisResult(
                summary=f"Read *{path}* ({len(lines)} lines). Here's the content:",
                detail=result,
                needs_follow_up=False,
            )
        return SynthesisResult(
            summary=f"Read *{path}*:",
            detail=result,
        )

    def _synthesize_write_file(self, arguments: dict, result: str) -> SynthesisResult:
        path = arguments.get("path", "file")
        return SynthesisResult(
            summary=f"Created/wrote *{path}*.",
            detail=result,
        )

    def _synthesize_edit_file(self, arguments: dict, result: str) -> SynthesisResult:
        path = arguments.get("path", "file")
        return SynthesisResult(
            summary=f"Edited *{path}*.",
            detail=result,
        )

    def _synthesize_list_files(self, arguments: dict, result: str) -> SynthesisResult:
        path = arguments.get("path", ".")
        lines = [line for line in result.strip().splitlines() if line.strip()]
        count = len(lines)
        if count == 0:
            return SynthesisResult(summary=f"Directory *{path}* is empty.")
        if count <= 10:
            files = ", ".join(f"`{line}`" for line in lines[:10])
            return SynthesisResult(summary=f"Found in *{path}*: {files}")
        return SynthesisResult(
            summary=f"Found {count} items in *{path}*.",
            detail=result,
        )

    def _synthesize_search_code(self, arguments: dict, result: str) -> SynthesisResult:
        query = arguments.get("query", "query")
        if not result.strip() or "no matches" in result.lower():
            return SynthesisResult(
                summary=f"No matches found for '{query}'.",
                needs_follow_up=True,
                suggested_next=["Try a broader search term", "Check a different directory"],
            )
        lines = result.strip().splitlines()
        return SynthesisResult(
            summary=f"Found {len(lines)} match(es) for '{query}':",
            detail=result,
        )

    def _synthesize_run_shell(self, arguments: dict, result: str) -> SynthesisResult:
        command = arguments.get("command", "command")
        # Truncate very long output
        lines = result.strip().splitlines()
        if len(lines) > 30 or len(result) > 2000:
            return SynthesisResult(
                summary=f"Ran `{command}` (output truncated, {len(lines)} lines):",
                detail=result,
            )
        if "error" in result.lower()[:100] or "ERROR" in result[:100]:
            return SynthesisResult(
                summary=f"`{command}` encountered an issue:",
                detail=result,
                needs_follow_up=True,
                suggested_next=["Check the error details", "Run with verbose flags"],
            )
        return SynthesisResult(
            summary=f"`{command}` completed successfully:",
            detail=result,
        )

    def _synthesize_remember(self, arguments: dict, result: str) -> SynthesisResult:
        return SynthesisResult(summary="💾 Saved to memory.")

    def _synthesize_recall(self, arguments: dict, result: str) -> SynthesisResult:
        query = arguments.get("query", "query")
        if not result.strip() or result == "[]":
            return SynthesisResult(summary=f"No memories found for '{query}'.")
        return SynthesisResult(
            summary=f"Recalled memories for '{query}':",
            detail=result,
        )

    def _synthesize_create_task(self, arguments: dict, result: str) -> SynthesisResult:
        desc = arguments.get("description", "task")
        return SynthesisResult(summary=f"✅ Task added: {desc}")

    def _synthesize_list_tasks(self, arguments: dict, result: str) -> SynthesisResult:
        if not result.strip():
            return SynthesisResult(summary="No active tasks.")
        return SynthesisResult(summary="Current tasks:", detail=result)

    def _synthesize_unknown_tool(self, arguments: dict, result: str) -> SynthesisResult:
        return SynthesisResult(
            summary="Unknown tool result.",
            detail=result,
        )

    def _synthesize_generic(self, arguments: dict, result: str) -> SynthesisResult:
        """Fallback synthesizer for any tool."""
        lines = result.strip().splitlines()
        if len(lines) > 20:
            return SynthesisResult(
                summary=f"Result ({len(lines)} lines):",
                detail=result,
            )
        return SynthesisResult(
            summary="Result:",
            detail=result,
        )
