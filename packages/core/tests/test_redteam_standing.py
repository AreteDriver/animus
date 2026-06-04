"""Tests for ``animus.redteam.standing``.

Three layers under test:

1. ``normalize_attack`` + ``attack_hash`` — collapsing of trivial
   reformatting variants.
2. ``FindingsLedger`` — JSONL persistence + lazy hash loading.
3. ``run_standing_sweep`` — orchestrator with a mocked driver (no live
   model dependency).
4. Dashboard rendering — markdown structure stays grep-friendly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from animus.redteam.driver import AttackProbe, RedTeamReport, Severity
from animus.redteam.standing import (
    FindingsLedger,
    LedgerEntry,
    StandingSweepResult,
    append_dashboard,
    attack_hash,
    gate_fires,
    normalize_attack,
    render_dashboard_section,
    run_standing_sweep,
)


class TestNormalizeAndHash:
    def test_case_and_whitespace_collapse_to_same_hash(self):
        # Identical attack content with reformatting must produce same hash.
        a = "Bearer Token: ABC-123"
        b = "  bearer\ntoken:abc-123  "
        c = "BEARER!TOKEN-ABC-123"
        assert attack_hash(a) == attack_hash(b) == attack_hash(c)

    def test_distinct_attacks_distinct_hashes(self):
        assert attack_hash("path/to/etc/passwd") != attack_hash("path/to/etc/shadow")

    def test_normalize_strips_to_alphanumerics(self):
        # Lowercased, non-alphanumerics removed; digits preserved
        # (``T0k3n`` stays as ``t0k3n`` — only the colon/space/special
        # chars get stripped).
        assert normalize_attack("Bearer: T0k3n!@#") == "bearert0k3n"

    def test_empty_input_safe(self):
        # Empty/whitespace-only inputs hash to a stable value, not a crash.
        assert attack_hash("") == attack_hash("   \n\t   ")


class TestFindingsLedger:
    def test_append_and_reload(self, tmp_path: Path):
        path = tmp_path / "ledger.jsonl"
        ledger = FindingsLedger(path)
        entry = LedgerEntry(
            ts="2026-05-27T00:00:00Z",
            sweep_id="abc123",
            category="dlp_bypass",
            severity="high",
            attack_input="Bearer: Tk-1234",
            observed="missed",
            normalized_hash=attack_hash("Bearer: Tk-1234"),
            is_novel=True,
            model="hauhaucs",
        )
        ledger.append(entry)

        # Fresh instance must reload the hash from disk.
        ledger2 = FindingsLedger(path)
        assert entry.normalized_hash in ledger2.known_hashes()

    def test_known_hashes_cached_after_first_load(self, tmp_path: Path):
        """Repeated calls don't re-read the file."""
        path = tmp_path / "ledger.jsonl"
        path.write_text('{"normalized_hash": "abcdef0123456789", "ts": "2026-05-27"}\n')
        ledger = FindingsLedger(path)
        first = ledger.known_hashes()
        assert "abcdef0123456789" in first
        # Mutating the file after first load should not affect the cached set.
        path.write_text("")
        assert ledger.known_hashes() is first

    def test_corrupt_line_does_not_crash(self, tmp_path: Path):
        """Resilient: garbage line in the middle of the ledger is skipped."""
        path = tmp_path / "ledger.jsonl"
        path.write_text(
            '{"normalized_hash": "good1"}\nthis is not json\n{"normalized_hash": "good2"}\n'
        )
        ledger = FindingsLedger(path)
        hashes = ledger.known_hashes()
        assert "good1" in hashes
        assert "good2" in hashes
        # Garbage line just dropped — no exception.

    def test_append_updates_known_hashes_in_memory(self, tmp_path: Path):
        path = tmp_path / "ledger.jsonl"
        ledger = FindingsLedger(path)
        _ = ledger.known_hashes()  # Force lazy load
        entry = LedgerEntry(
            ts="2026-05-27T00:00:00Z",
            sweep_id="x",
            category="pi",
            severity="ok",
            attack_input="probe",
            observed="wrapped",
            normalized_hash="fresh_hash_value",
            is_novel=True,
            model="qwen2.5:14b",
        )
        ledger.append(entry)
        assert "fresh_hash_value" in ledger.known_hashes()


def _fake_report(probes: list[AttackProbe], model: str = "qwen2.5:14b") -> RedTeamReport:
    """Build a RedTeamReport with the given probes for orchestrator tests."""
    by_sev: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for p in probes:
        by_sev[p.severity.value] = by_sev.get(p.severity.value, 0) + 1
        by_cat[p.category] = by_cat.get(p.category, 0) + 1
    return RedTeamReport(
        ts="2026-05-27T00:00:00Z",
        model=model,
        quick_mode=False,
        total_probes=len(probes),
        by_severity=by_sev,
        by_category=by_cat,
        probes=probes,
    )


class TestRunStandingSweep:
    @pytest.fixture
    def patched_driver(self, monkeypatch):
        """Replace RedTeamDriver with a controllable mock."""
        instances: list[MagicMock] = []

        def factory(*args, **kwargs):
            inst = MagicMock()
            inst.run = MagicMock(return_value=instances.pop(0)._return)
            return inst

        def queue_report(report: RedTeamReport):
            placeholder = MagicMock()
            placeholder._return = report
            instances.append(placeholder)

        monkeypatch.setattr("animus.redteam.standing.RedTeamDriver", factory)
        return queue_report

    def test_all_novel_on_fresh_ledger(self, tmp_path: Path, patched_driver):
        ledger_path = tmp_path / "ledger.jsonl"
        patched_driver(
            _fake_report(
                [
                    AttackProbe(
                        category="dlp_bypass",
                        attack_input="Bearer: Tk-1234567890",
                        expected_behavior="blocked",
                        observed="DLP missed",
                        severity=Severity.HIGH,
                        blocked=False,
                    )
                ]
            )
        )

        result = run_standing_sweep(ledger_path=ledger_path, audit_dir=tmp_path)
        assert result.total_probes == 1
        assert result.new_high == 1
        assert result.novel_findings == 1
        assert result.repeat_findings == 0
        # Ledger should now contain the entry.
        assert FindingsLedger(ledger_path).known_hashes() == {attack_hash("Bearer: Tk-1234567890")}

    def test_repeat_finding_marked_known(self, tmp_path: Path, patched_driver):
        ledger_path = tmp_path / "ledger.jsonl"
        # Pre-seed the ledger with a hash we expect to see again.
        attack = "Bearer: Tk-1234567890"
        known_hash = attack_hash(attack)
        ledger_path.write_text(
            f'{{"normalized_hash": "{known_hash}", "ts": "2026-05-20T00:00:00Z"}}\n'
        )

        patched_driver(
            _fake_report(
                [
                    AttackProbe(
                        category="dlp_bypass",
                        attack_input=attack,  # Same as seeded
                        expected_behavior="blocked",
                        observed="DLP missed",
                        severity=Severity.HIGH,
                        blocked=False,
                    )
                ]
            )
        )

        result = run_standing_sweep(ledger_path=ledger_path, audit_dir=tmp_path)
        assert result.novel_findings == 0
        assert result.repeat_findings == 1
        assert result.new_high == 0  # Not new — repeat

    def test_ok_probes_appended_but_not_counted_as_finding(self, tmp_path: Path, patched_driver):
        ledger_path = tmp_path / "ledger.jsonl"
        patched_driver(
            _fake_report(
                [
                    AttackProbe(
                        category="pi",
                        attack_input="some attack",
                        expected_behavior="wrapped",
                        observed="wrapped",
                        severity=Severity.OK,
                        blocked=True,
                    )
                ]
            )
        )

        result = run_standing_sweep(ledger_path=ledger_path, audit_dir=tmp_path)
        assert result.total_probes == 1
        assert result.novel_findings == 0  # OK doesn't count
        assert result.repeat_findings == 0
        # But the ledger still has it (for trend analysis).
        assert len(FindingsLedger(ledger_path).known_hashes()) == 1


class TestDashboardRendering:
    def test_section_contains_sweep_id_and_severity_counts(self):
        result = StandingSweepResult(
            sweep_id="abc12345",
            ts="2026-05-27T00:00:00Z",
            model="qwen2.5:14b",
            total_probes=10,
            novel_findings=2,
            repeat_findings=1,
            new_high=1,
            new_medium=1,
            by_severity={"ok": 7, "medium": 2, "high": 1},
        )
        md = render_dashboard_section(result)
        assert "## Sweep abc12345" in md
        assert "ok=7" in md
        assert "high=1" in md
        assert "**NEW**: 1 HIGH, 1 MEDIUM" in md

    def test_dashboard_header_written_on_first_append(self, tmp_path: Path):
        path = tmp_path / "dash.md"
        result = StandingSweepResult(
            sweep_id="abc",
            ts="2026-05-27T00:00:00Z",
            model="x",
            total_probes=0,
        )
        append_dashboard(result, path)
        content = path.read_text()
        assert content.startswith("# Standing Red-Team Dashboard")
        assert "## Sweep abc" in content

    def test_dashboard_header_only_written_once(self, tmp_path: Path):
        path = tmp_path / "dash.md"
        for sweep_id in ("a", "b", "c"):
            result = StandingSweepResult(
                sweep_id=sweep_id,
                ts="2026-05-27T00:00:00Z",
                model="x",
                total_probes=0,
            )
            append_dashboard(result, path)
        content = path.read_text()
        # The h1 header appears exactly once.
        assert content.count("# Standing Red-Team Dashboard") == 1
        # All three sweeps recorded.
        for sweep_id in ("a", "b", "c"):
            assert f"## Sweep {sweep_id}" in content

    def test_findings_use_novel_vs_repeat_markers(self):
        result = StandingSweepResult(
            sweep_id="z",
            ts="2026-05-27T00:00:00Z",
            model="x",
            total_probes=2,
            findings_detail=[
                {
                    "category": "dlp_bypass",
                    "severity": "high",
                    "is_novel": True,
                    "attack_input": "Bearer foo bar baz",
                    "observed": "missed",
                },
                {
                    "category": "dlp_bypass",
                    "severity": "medium",
                    "is_novel": False,
                    "attack_input": "Bearer qux quux",
                    "observed": "missed",
                },
            ],
        )
        md = render_dashboard_section(result)
        assert "🆕" in md
        assert "🔁" in md


class TestGate:
    """Exit-gate semantics: fire on ANY finding (novel OR repeat) at or
    above ``--alert-on`` severity. The key behavior is that a *repeat*
    HIGH still breaks the build — a known attack class reaching a gate is
    a regression, not noise.
    """

    @staticmethod
    def _result(*findings: tuple[str, bool]) -> StandingSweepResult:
        """Build a result whose findings_detail carries (severity, is_novel)."""
        return StandingSweepResult(
            sweep_id="gate",
            ts="2026-05-27T00:00:00Z",
            model="x",
            total_probes=len(findings),
            findings_detail=[
                {
                    "category": "dlp_bypass",
                    "severity": sev,
                    "is_novel": novel,
                    "attack_input": "x",
                    "observed": "missed",
                }
                for sev, novel in findings
            ],
        )

    def test_repeat_high_fires_gate_at_high(self):
        # The whole point of this change: a known (repeat) HIGH must fail.
        result = self._result(("high", False))
        assert result.novel_findings == 0  # not novel...
        assert gate_fires(result, "high") is True  # ...but still fires

    def test_novel_high_still_fires_gate_at_high(self):
        assert gate_fires(self._result(("high", True)), "high") is True

    def test_medium_does_not_fire_at_high_threshold(self):
        assert gate_fires(self._result(("medium", True)), "high") is False
        assert gate_fires(self._result(("medium", False)), "high") is False

    def test_repeat_medium_fires_at_medium_threshold(self):
        assert gate_fires(self._result(("medium", False)), "medium") is True

    def test_high_fires_at_medium_threshold(self):
        assert gate_fires(self._result(("high", False)), "medium") is True

    def test_no_findings_never_fires(self):
        empty = self._result()
        assert gate_fires(empty, "high") is False
        assert gate_fires(empty, "medium") is False
        assert gate_fires(empty, "low") is False

    def test_low_threshold_fires_on_any_finding(self):
        assert gate_fires(self._result(("medium", False)), "low") is True
        assert gate_fires(self._result(("high", True)), "low") is True
