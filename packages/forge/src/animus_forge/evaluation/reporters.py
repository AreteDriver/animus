"""Reporters for evaluation results."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .base import EvalStatus
from .runner import SuiteResult


class Reporter(ABC):
    """Abstract base class for evaluation reporters."""

    @abstractmethod
    def report(self, result: SuiteResult) -> str:
        """Generate a report from suite results.

        Args:
            result: Suite evaluation result

        Returns:
            Formatted report string
        """
        pass

    def save(self, result: SuiteResult, path: str | Path) -> None:
        """Save report to file.

        Args:
            result: Suite evaluation result
            path: Output file path
        """
        report = self.report(result)
        Path(path).write_text(report)


class ConsoleReporter(Reporter):
    """Console-friendly text reporter."""

    def __init__(self, verbose: bool = False, show_output: bool = False):
        """Initialize console reporter.

        Args:
            verbose: Show detailed metrics per case
            show_output: Show actual outputs
        """
        self.verbose = verbose
        self.show_output = show_output

    def report(self, result: SuiteResult) -> str:
        """Generate console report."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append(f"Evaluation Results: {result.suite.name}")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append("Summary:")
        lines.append(f"  Total Cases:  {result.total}")
        lines.append(f"  Passed:       {result.passed} ({self._pct(result.passed, result.total)})")
        lines.append(f"  Failed:       {result.failed} ({self._pct(result.failed, result.total)})")
        lines.append(f"  Errors:       {result.errors} ({self._pct(result.errors, result.total)})")
        lines.append(
            f"  Skipped:      {result.skipped} ({self._pct(result.skipped, result.total)})"
        )
        lines.append(f"  Avg Score:    {result.total_score:.2%}")
        lines.append(f"  Duration:     {result.duration_ms:.0f}ms")
        lines.append("")

        # Case details
        if self.verbose or result.failed > 0 or result.errors > 0:
            lines.append("-" * 60)
            lines.append("Case Details:")
            lines.append("-" * 60)

            for case_result in result.results:
                status_icon = {
                    EvalStatus.PASSED: "[PASS]",
                    EvalStatus.FAILED: "[FAIL]",
                    EvalStatus.ERROR: "[ERR!]",
                    EvalStatus.SKIPPED: "[SKIP]",
                }.get(case_result.status, "[????]")

                lines.append(f"\n{status_icon} {case_result.case.name}")
                lines.append(f"  Score: {case_result.score:.2%}")

                if case_result.metrics:
                    lines.append("  Metrics:")
                    for metric, score in case_result.metrics.items():
                        lines.append(f"    - {metric}: {score:.2%}")

                if case_result.error:
                    lines.append(f"  Error: {case_result.error}")

                if self.show_output and case_result.output:
                    output_preview = str(case_result.output)[:200]
                    if len(str(case_result.output)) > 200:
                        output_preview += "..."
                    lines.append(f"  Output: {output_preview}")

        lines.append("")
        lines.append("=" * 60)

        # Final verdict
        if result.pass_rate >= result.suite.threshold:
            lines.append(f"PASSED (threshold: {result.suite.threshold:.0%})")
        else:
            lines.append(f"FAILED (threshold: {result.suite.threshold:.0%})")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _pct(self, part: int, total: int) -> str:
        if total == 0:
            return "0%"
        return f"{part / total:.0%}"


class JSONReporter(Reporter):
    """JSON format reporter."""

    def __init__(self, indent: int = 2, include_outputs: bool = False):
        """Initialize JSON reporter.

        Args:
            indent: JSON indentation
            include_outputs: Include full outputs in report
        """
        self.indent = indent
        self.include_outputs = include_outputs

    def report(self, result: SuiteResult) -> str:
        """Generate JSON report."""
        data = result.to_dict()

        if not self.include_outputs:
            # Remove outputs for smaller files
            for case_result in data.get("results", []):
                case_result.pop("output", None)

        return json.dumps(data, indent=self.indent, default=str)


class HTMLReporter(Reporter):
    """HTML format reporter with styling."""

    def __init__(self, title: str | None = None):
        """Initialize HTML reporter.

        Args:
            title: Custom report title
        """
        self.title = title

    def report(self, result: SuiteResult) -> str:
        """Generate HTML report."""
        title = self.title or f"Evaluation: {result.suite.name}"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{ color: #666; }}
        .stat.passed .stat-value {{ color: #22c55e; }}
        .stat.failed .stat-value {{ color: #ef4444; }}
        .stat.error .stat-value {{ color: #f97316; }}
        .results {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .status.passed {{ background: #dcfce7; color: #166534; }}
        .status.failed {{ background: #fee2e2; color: #991b1b; }}
        .status.error {{ background: #ffedd5; color: #9a3412; }}
        .status.skipped {{ background: #f3f4f6; color: #374151; }}
        .score-bar {{
            width: 100px;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
        }}
        .score-bar-fill {{
            height: 100%;
            background: #22c55e;
            border-radius: 4px;
        }}
        .metrics {{ font-size: 0.85em; color: #666; }}
        .error-msg {{ color: #ef4444; font-size: 0.85em; }}
        .verdict {{
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .verdict.passed {{ background: #dcfce7; color: #166534; }}
        .verdict.failed {{ background: #fee2e2; color: #991b1b; }}
        footer {{ margin-top: 40px; text-align: center; color: #999; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="summary">
        <div class="stat">
            <div class="stat-value">{result.total}</div>
            <div class="stat-label">Total Cases</div>
        </div>
        <div class="stat passed">
            <div class="stat-value">{result.passed}</div>
            <div class="stat-label">Passed</div>
        </div>
        <div class="stat failed">
            <div class="stat-value">{result.failed}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat error">
            <div class="stat-value">{result.errors}</div>
            <div class="stat-label">Errors</div>
        </div>
        <div class="stat">
            <div class="stat-value">{result.total_score:.0%}</div>
            <div class="stat-label">Avg Score</div>
        </div>
        <div class="stat">
            <div class="stat-value">{result.duration_ms:.0f}ms</div>
            <div class="stat-label">Duration</div>
        </div>
    </div>

    <div class="results">
        <table>
            <thead>
                <tr>
                    <th>Case</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Metrics</th>
                    <th>Latency</th>
                </tr>
            </thead>
            <tbody>
                {self._render_rows(result)}
            </tbody>
        </table>
    </div>

    <div class="verdict {"passed" if result.pass_rate >= result.suite.threshold else "failed"}">
        {"PASSED" if result.pass_rate >= result.suite.threshold else "FAILED"}
        (Pass rate: {result.pass_rate:.0%}, Threshold: {result.suite.threshold:.0%})
    </div>

    <footer>
        Generated on {result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
        by Gorgon Evaluation Framework
    </footer>
</body>
</html>"""

        return html

    def _render_rows(self, result: SuiteResult) -> str:
        rows = []
        for case_result in result.results:
            status_class = case_result.status.value
            metrics_html = "<br>".join(f"{k}: {v:.0%}" for k, v in case_result.metrics.items())
            error_html = (
                f'<div class="error-msg">{case_result.error}</div>' if case_result.error else ""
            )

            row = f"""
            <tr>
                <td>
                    <strong>{case_result.case.name}</strong>
                    {error_html}
                </td>
                <td><span class="status {status_class}">{case_result.status.value.upper()}</span></td>
                <td>
                    <div class="score-bar">
                        <div class="score-bar-fill" style="width: {case_result.score * 100}%"></div>
                    </div>
                    {case_result.score:.0%}
                </td>
                <td class="metrics">{metrics_html or "-"}</td>
                <td>{case_result.latency_ms:.0f}ms</td>
            </tr>
            """
            rows.append(row)

        return "\n".join(rows)


class MarkdownReporter(Reporter):
    """Markdown format reporter."""

    def report(self, result: SuiteResult) -> str:
        """Generate Markdown report."""
        lines = [
            f"# Evaluation Results: {result.suite.name}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Cases | {result.total} |",
            f"| Passed | {result.passed} ({result.passed / result.total:.0%}) |"
            if result.total > 0
            else "| Passed | 0 |",
            f"| Failed | {result.failed} |",
            f"| Errors | {result.errors} |",
            f"| Avg Score | {result.total_score:.2%} |",
            f"| Duration | {result.duration_ms:.0f}ms |",
            "",
            "## Results",
            "",
            "| Case | Status | Score | Latency |",
            "|------|--------|-------|---------|",
        ]

        for case_result in result.results:
            status = case_result.status.value.upper()
            emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "⚠️", "SKIPPED": "⏭️"}.get(status, "❓")
            lines.append(
                f"| {case_result.case.name} | {emoji} {status} | {case_result.score:.0%} | {case_result.latency_ms:.0f}ms |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                f"**Verdict**: {'✅ PASSED' if result.pass_rate >= result.suite.threshold else '❌ FAILED'}",
                f"(Threshold: {result.suite.threshold:.0%})",
            ]
        )

        return "\n".join(lines)
