"""Tests #14, #15 from the build spec §16.

Process-classification provenance rules:

- #14: Unknown processes must never be killable by name.
- #15: Recoverable and Orphaned classifications require the
  documented proofs.
"""

from __future__ import annotations

import pytest

from animus_bootstrap.lifecycle import (
    ClassificationInput,
    ProcessClassification,
    classify_process,
    default_provenance_threshold,
)
from animus_bootstrap.lifecycle.classification import (
    PROOF_EXECUTABLE,
    PROOF_CMDLINE,
    PROOF_UID,
    PROOF_STARTTIME,
)


# ---------------------------------------------------------------------------
# Test #14 — Unknown never killable, never auto-classified higher
# ---------------------------------------------------------------------------


def test_unknown_when_only_name_matches() -> None:
    res = classify_process(
        ClassificationInput(
            pid=9999,
            executable="/usr/bin/python3",
            command_line="animus_discord_bot.py",
        )
    )
    assert res.classification == ProcessClassification.UNKNOWN


def test_unknown_when_no_registry_identity() -> None:
    """No registry identity, even with multiple proofs, is still UNKNOWN.

    The classification requires *registry identity* first; the
    dashboard or cleanup CLI must never claim orphan status from
    bare ``/proc`` data alone.
    """
    res = classify_process(
        ClassificationInput(
            pid=9999,
            executable="/usr/bin/python3",
            command_line="animus daemon",
            start_time=1234,
            uid=1000,
            expected_uid=1000,
            registry_identity=False,
            unit_active=False,
        )
    )
    assert res.classification == ProcessClassification.UNKNOWN


def test_unknown_is_report_only_no_kill_authority() -> None:
    """The classification result carries no authority to kill.

    The classification function is the only consumer of these
    inputs. There is no ``allow_kill`` field; the dashboard must
    enforce the rule. This test asserts that the data shape
    contains no kill authority.
    """
    res = classify_process(
        ClassificationInput(pid=9999, executable="/usr/bin/python3")
    )
    assert res.classification == ProcessClassification.UNKNOWN
    assert not hasattr(res, "allow_kill")


# ---------------------------------------------------------------------------
# Test #15 — Recoverable and Orphaned require proofs
# ---------------------------------------------------------------------------


def test_managed_requires_unit_active() -> None:
    res = classify_process(
        ClassificationInput(
            pid=1,
            executable="/usr/bin/python3",
            command_line="animus daemon",
            start_time=100,
            uid=1000,
            expected_uid=1000,
            registry_identity=True,
            unit_active=True,
        )
    )
    assert res.classification == ProcessClassification.MANAGED


def test_recoverable_requires_unit_inactive_and_one_proof() -> None:
    """Recoverable fires when there is registry identity + unit
    inactive + exactly one reliable proof but not enough for Orphaned.
    """
    res = classify_process(
        ClassificationInput(
            pid=2,
            executable="/usr/bin/python3",
            # only one proof; not enough for Orphaned
            registry_identity=True,
            unit_active=False,
        )
    )
    assert res.classification == ProcessClassification.RECOVERABLE


def test_recoverable_falls_back_to_orphan_when_proofs_sufficient() -> None:
    """Recoverable path requires at least one of (executable, cmdline,
    start-time). If registry identity is True and there are two
    independent proofs, the classification can be Orphaned even when
    unit_active is False (which is the case for a service that
    crashed). The cgroup may itself be the thing that was lost.
    """
    res = classify_process(
        ClassificationInput(
            pid=3,
            executable="/usr/bin/python3",
            command_line="animus mcp",
            start_time=100,
            uid=1000,
            expected_uid=1000,
            registry_identity=True,
            unit_active=False,
        )
    )
    # Both classifications are defensible; the rule promotes to
    # Orphaned when 2+ proofs exist.
    assert res.classification in (
        ProcessClassification.RECOVERABLE,
        ProcessClassification.ORPHANED,
    )


def test_orphaned_requires_two_proofs() -> None:
    """Two independent proofs + registry identity => ORPHANED."""
    res = classify_process(
        ClassificationInput(
            pid=4,
            executable="/usr/bin/python3",
            command_line="animus mcp",
            start_time=100,
            uid=1000,
            expected_uid=1000,
            registry_identity=True,
            unit_active=False,
        )
    )
    # 4 reliable proofs (exe, cmdline, uid, starttime) plus registry
    # identity => orphaned (because threshold is met).
    assert res.classification == ProcessClassification.ORPHANED


def test_orphaned_blocked_by_uid_mismatch() -> None:
    """UID mismatch disqualifies Orphaned even with proofs."""
    res = classify_process(
        ClassificationInput(
            pid=5,
            executable="/usr/bin/python3",
            command_line="animus mcp",
            start_time=100,
            uid=0,  # running as root
            expected_uid=1000,
            registry_identity=True,
            unit_active=False,
        )
    )
    # UID mismatch prevents ORPHANED; falls through to UNKNOWN.
    assert res.classification == ProcessClassification.UNKNOWN


def test_orphaned_with_cgroup_alive_decisive() -> None:
    """Cgroup membership, when present and true, is decisive."""
    res = classify_process(
        ClassificationInput(
            pid=6,
            executable="/usr/bin/python3",
            registry_identity=True,
            unit_active=False,
            cgroup_alive=True,
        )
    )
    assert res.classification == ProcessClassification.ORPHANED


def test_default_provenance_threshold_is_two() -> None:
    """The ADR-mandated threshold is 2 proofs (one for registry, two
    independent)."""
    assert default_provenance_threshold() == 2


def test_unknown_with_one_proof_only() -> None:
    """One proof without registry identity stays UNKNOWN."""
    res = classify_process(
        ClassificationInput(
            pid=7,
            executable="/usr/bin/python3",
        )
    )
    assert res.classification == ProcessClassification.UNKNOWN


def test_proof_kinds_are_distinct() -> None:
    """The PROOF_* constants are the canonical proof identifiers."""
    assert PROOF_EXECUTABLE == "executable_path"
    assert PROOF_CMDLINE == "command_line_launch_token"
    assert PROOF_UID == "uid"
    assert PROOF_STARTTIME == "start_time_fingerprint"
