"""
Real Constraint Gates — subprocess-backed evidence producers.

This module bridges the constraint engine to actual tools:
  - pytest: Run tests and produce test_pass/test_fail evidence
  - mypy: Run type checking and produce type_check evidence
  - compile: Run compilation (cargo, gcc, etc.) and produce code_committed evidence
  - custom: Run arbitrary commands as constraint validators

"Constraint satisfied" means the code compiled and tests passed,
not "the signature has the right fields."

Each gate runs a subprocess, captures output, and produces Evidence
objects that feed back into the intent graph's stability scoring.

Usage:
    gate = PytestGate(test_path="tests/", timeout=30)
    result = gate.run(intent)
    if result.passed:
        for evidence in result.evidence:
            intent.add_evidence(evidence)

    # Or use the GateRunner for all gates at once:
    runner = GateRunner()
    runner.add(PytestGate("tests/"))
    runner.add(MypyGate("src/"))
    report = runner.run_all(intent)
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from convergent.intent import Evidence, EvidenceKind, Intent

# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class GateRunResult:
    """Result of running a single constraint gate."""

    gate_name: str
    passed: bool
    evidence: list[Evidence] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration_seconds: float = 0.0
    error_summary: str = ""

    @property
    def failed(self) -> bool:
        return not self.passed


@dataclass
class GateReport:
    """Aggregate report from running multiple gates."""

    results: list[GateRunResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_evidence(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for r in self.results:
            evidence.extend(r.evidence)
        return evidence

    def summary(self) -> str:
        lines = [
            f"Gate Report: {self.passed_count}/{len(self.results)} passed "
            f"({self.total_duration:.2f}s)",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.gate_name} ({r.duration_seconds:.2f}s)")
            if r.error_summary:
                lines.append(f"         {r.error_summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract gate
# ---------------------------------------------------------------------------


class ConstraintGate(ABC):
    """Abstract base for subprocess-backed constraint gates."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this gate."""
        ...

    @abstractmethod
    def run(self, intent: Intent) -> GateRunResult:
        """Run the gate and return evidence."""
        ...

    def _run_subprocess(
        self,
        cmd: list[str],
        cwd: str | Path | None = None,
        timeout: int = 120,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str, float]:
        """Run a subprocess and return (returncode, stdout, stderr, duration).

        Returns (-1, "", error_message, duration) if the process times out
        or fails to start.
        """
        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
            duration = time.monotonic() - start
            return result.returncode, result.stdout, result.stderr, duration
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return -1, "", f"Timed out after {timeout}s", duration
        except FileNotFoundError as e:
            duration = time.monotonic() - start
            return -1, "", f"Command not found: {e}", duration
        except OSError as e:
            duration = time.monotonic() - start
            return -1, "", f"OS error: {e}", duration


# ---------------------------------------------------------------------------
# Pytest gate
# ---------------------------------------------------------------------------


class PytestGate(ConstraintGate):
    """Run pytest and produce test evidence.

    Parses pytest output to extract pass/fail counts and produces
    Evidence.test_pass or Evidence.test_fail accordingly.
    """

    def __init__(
        self,
        test_path: str = "tests/",
        markers: str | None = None,
        timeout: int = 120,
        cwd: str | Path | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.test_path = test_path
        self.markers = markers
        self.timeout = timeout
        self.cwd = cwd
        self.extra_args = extra_args or []

    @property
    def name(self) -> str:
        return f"pytest({self.test_path})"

    def run(self, intent: Intent) -> GateRunResult:
        cmd = [sys.executable, "-m", "pytest", self.test_path, "-v", "--tb=short"]
        if self.markers:
            cmd.extend(["-m", self.markers])
        cmd.extend(self.extra_args)

        returncode, stdout, stderr, duration = self._run_subprocess(
            cmd, cwd=self.cwd, timeout=self.timeout
        )

        evidence: list[Evidence] = []
        error_summary = ""

        if returncode == 0:
            # All tests passed
            passed = True
            evidence.append(Evidence.test_pass(f"pytest {self.test_path}: all tests passed"))
        elif returncode == -1:
            # Timeout or command not found
            passed = False
            error_summary = stderr.strip()[:200]
            evidence.append(
                Evidence(
                    kind=EvidenceKind.TEST_FAIL,
                    description=f"pytest {self.test_path}: {error_summary}",
                )
            )
        else:
            # Test failures
            passed = False
            # Extract failure summary from output
            error_summary = _extract_pytest_summary(stdout) or stderr.strip()[:200]
            evidence.append(
                Evidence(
                    kind=EvidenceKind.TEST_FAIL,
                    description=f"pytest {self.test_path}: {error_summary}",
                )
            )

        return GateRunResult(
            gate_name=self.name,
            passed=passed,
            evidence=evidence,
            stdout=stdout,
            stderr=stderr,
            return_code=returncode,
            duration_seconds=duration,
            error_summary=error_summary,
        )


def _extract_pytest_summary(output: str) -> str:
    """Extract the summary line from pytest output."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if (line.startswith("FAILED") or line.startswith("=")) and (
            "passed" in line or "failed" in line or "error" in line
        ):
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            return clean.strip("= ").strip()
    return ""


# ---------------------------------------------------------------------------
# Mypy gate
# ---------------------------------------------------------------------------


class MypyGate(ConstraintGate):
    """Run mypy type checking and produce evidence.

    Treats any mypy error as a gate failure. Produces Evidence.test_pass
    on clean type check, Evidence.test_fail on type errors.
    """

    def __init__(
        self,
        target_path: str = ".",
        timeout: int = 120,
        cwd: str | Path | None = None,
        strict: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        self.target_path = target_path
        self.timeout = timeout
        self.cwd = cwd
        self.strict = strict
        self.extra_args = extra_args or []

    @property
    def name(self) -> str:
        return f"mypy({self.target_path})"

    def run(self, intent: Intent) -> GateRunResult:
        cmd = ["mypy", self.target_path]
        if self.strict:
            cmd.append("--strict")
        cmd.extend(self.extra_args)

        returncode, stdout, stderr, duration = self._run_subprocess(
            cmd, cwd=self.cwd, timeout=self.timeout
        )

        evidence: list[Evidence] = []
        error_summary = ""

        if returncode == 0:
            passed = True
            evidence.append(Evidence.test_pass(f"mypy {self.target_path}: no type errors"))
        elif returncode == -1:
            passed = False
            error_summary = stderr.strip()[:200]
            evidence.append(
                Evidence(
                    kind=EvidenceKind.TEST_FAIL,
                    description=f"mypy {self.target_path}: {error_summary}",
                )
            )
        else:
            passed = False
            # Count errors from mypy output
            error_lines = [ln for ln in stdout.splitlines() if ": error:" in ln]
            error_summary = f"{len(error_lines)} type error(s)"
            if error_lines:
                error_summary += f" — first: {error_lines[0].strip()[:100]}"
            evidence.append(
                Evidence(
                    kind=EvidenceKind.TEST_FAIL,
                    description=f"mypy {self.target_path}: {error_summary}",
                )
            )

        return GateRunResult(
            gate_name=self.name,
            passed=passed,
            evidence=evidence,
            stdout=stdout,
            stderr=stderr,
            return_code=returncode,
            duration_seconds=duration,
            error_summary=error_summary,
        )


# ---------------------------------------------------------------------------
# Compile gate (cargo, gcc, go build, etc.)
# ---------------------------------------------------------------------------


class CompileGate(ConstraintGate):
    """Run a compilation command and produce evidence.

    Works with any compiler: cargo build, gcc, go build, javac, etc.
    """

    def __init__(
        self,
        cmd: list[str],
        gate_name: str = "compile",
        timeout: int = 120,
        cwd: str | Path | None = None,
    ) -> None:
        self._cmd = cmd
        self._gate_name = gate_name
        self.timeout = timeout
        self.cwd = cwd

    @property
    def name(self) -> str:
        return self._gate_name

    def run(self, intent: Intent) -> GateRunResult:
        returncode, stdout, stderr, duration = self._run_subprocess(
            self._cmd, cwd=self.cwd, timeout=self.timeout
        )

        evidence: list[Evidence] = []
        error_summary = ""

        if returncode == 0:
            passed = True
            evidence.append(Evidence.code_committed(f"{self._gate_name}: compilation succeeded"))
        else:
            passed = False
            error_summary = (stderr or stdout).strip()[:200]
            evidence.append(
                Evidence(
                    kind=EvidenceKind.TEST_FAIL,
                    description=f"{self._gate_name}: compilation failed — {error_summary}",
                )
            )

        return GateRunResult(
            gate_name=self.name,
            passed=passed,
            evidence=evidence,
            stdout=stdout,
            stderr=stderr,
            return_code=returncode,
            duration_seconds=duration,
            error_summary=error_summary,
        )


# ---------------------------------------------------------------------------
# Custom command gate
# ---------------------------------------------------------------------------


class CommandGate(ConstraintGate):
    """Run an arbitrary command as a constraint gate.

    Exit code 0 = pass, anything else = fail.
    Useful for linters, formatters, security scanners, etc.
    """

    def __init__(
        self,
        cmd: list[str],
        gate_name: str = "custom",
        timeout: int = 60,
        cwd: str | Path | None = None,
        pass_description: str = "check passed",
        fail_description: str = "check failed",
        evidence_kind_on_pass: EvidenceKind = EvidenceKind.TEST_PASS,
        evidence_kind_on_fail: EvidenceKind = EvidenceKind.TEST_FAIL,
    ) -> None:
        self._cmd = cmd
        self._gate_name = gate_name
        self.timeout = timeout
        self.cwd = cwd
        self.pass_description = pass_description
        self.fail_description = fail_description
        self.evidence_kind_on_pass = evidence_kind_on_pass
        self.evidence_kind_on_fail = evidence_kind_on_fail

    @property
    def name(self) -> str:
        return self._gate_name

    def run(self, intent: Intent) -> GateRunResult:
        returncode, stdout, stderr, duration = self._run_subprocess(
            self._cmd, cwd=self.cwd, timeout=self.timeout
        )

        evidence: list[Evidence] = []
        error_summary = ""

        if returncode == 0:
            passed = True
            evidence.append(
                Evidence(
                    kind=self.evidence_kind_on_pass,
                    description=f"{self._gate_name}: {self.pass_description}",
                )
            )
        else:
            passed = False
            error_summary = (stderr or stdout).strip()[:200]
            evidence.append(
                Evidence(
                    kind=self.evidence_kind_on_fail,
                    description=f"{self._gate_name}: {self.fail_description} — {error_summary}",
                )
            )

        return GateRunResult(
            gate_name=self.name,
            passed=passed,
            evidence=evidence,
            stdout=stdout,
            stderr=stderr,
            return_code=returncode,
            duration_seconds=duration,
            error_summary=error_summary,
        )


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------


class GateRunner:
    """Runs multiple constraint gates and aggregates results.

    Usage:
        runner = GateRunner()
        runner.add(PytestGate("tests/"))
        runner.add(MypyGate("src/"))
        runner.add(CompileGate(["cargo", "build"], gate_name="cargo"))

        report = runner.run_all(intent)
        if report.all_passed:
            for evidence in report.all_evidence:
                intent.add_evidence(evidence)
        else:
            print(report.summary())
    """

    def __init__(self) -> None:
        self._gates: list[ConstraintGate] = []

    def add(self, gate: ConstraintGate) -> None:
        self._gates.append(gate)

    @property
    def gate_count(self) -> int:
        return len(self._gates)

    def run_all(
        self,
        intent: Intent,
        stop_on_failure: bool = False,
    ) -> GateReport:
        """Run all registered gates and return an aggregate report.

        Args:
            intent: The intent to validate.
            stop_on_failure: If True, stop after first failure.
        """
        report = GateReport()
        start = time.monotonic()

        for gate in self._gates:
            result = gate.run(intent)
            report.results.append(result)

            if stop_on_failure and result.failed:
                break

        report.total_duration = time.monotonic() - start
        return report

    def apply_evidence(self, intent: Intent, report: GateReport) -> int:
        """Apply all evidence from a gate report to an intent.

        Returns the number of evidence items applied.
        """
        count = 0
        for evidence in report.all_evidence:
            intent.add_evidence(evidence)
            count += 1
        return count
