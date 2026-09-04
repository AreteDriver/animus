"""Tests for the exit-code → typed-exception mapping."""

from __future__ import annotations

import pytest

from animus_forge.governor.errors import (
    ContractIntegrityError,
    ContractRejectedError,
    GovernorError,
    PermissionDeniedError,
    VerifyDeniedError,
)
from animus_forge.governor.exit_codes import map_exit_code


def _expect_success() -> None:
    """Helper: returns ``None``; used as the ``return`` value."""
    return None


def test_exit_0_returns_none() -> None:
    """rc 0 is the success path — no exception."""
    assert map_exit_code(returncode=0, stderr="", subcommand="verify") is None
    assert map_exit_code(returncode=0, stderr="noise", subcommand="start") is None


def test_exit_1_permission_sniff() -> None:
    """rc 1 + ``PermissionDenied`` → :class:`PermissionDeniedError`."""
    with pytest.raises(PermissionDeniedError):
        map_exit_code(
            returncode=1,
            stderr="PermissionDenied: worker may not emit change_mapped",
            subcommand="record",
        )


def test_exit_1_integrity_sniff() -> None:
    """rc 1 + ``integrity`` → :class:`ContractIntegrityError`."""
    with pytest.raises(ContractIntegrityError):
        map_exit_code(
            returncode=1,
            stderr="contract integrity violation: contract.sha256 drift",
            subcommand="verify",
        )


def test_exit_1_generic() -> None:
    """rc 1 with no sniff match → :class:`GovernorError`."""
    with pytest.raises(GovernorError) as excinfo:
        map_exit_code(
            returncode=1,
            stderr="some other failure",
            subcommand="verify",
        )
    assert excinfo.value.subcommand == "verify"
    assert excinfo.value.exit_code == 1


def test_exit_2_compile() -> None:
    """rc 2 + ``compile`` → :class:`ContractRejectedError`."""
    with pytest.raises(ContractRejectedError):
        map_exit_code(
            returncode=2,
            stderr="requirement ids must be unique",
            subcommand="compile",
        )


def test_exit_2_other_subcommand_is_generic() -> None:
    """rc 2 + non-compile subcommand → :class:`GovernorError`."""
    with pytest.raises(GovernorError):
        map_exit_code(returncode=2, stderr="bad", subcommand="verify")


def test_exit_3_verify() -> None:
    """rc 3 + ``verify`` → :class:`VerifyDeniedError`."""
    with pytest.raises(VerifyDeniedError):
        map_exit_code(
            returncode=3,
            stderr="completion denied: missing evidence",
            subcommand="verify",
        )


def test_exit_3_other_subcommand_is_generic() -> None:
    """rc 3 outside ``verify`` → :class:`GovernorError`."""
    with pytest.raises(GovernorError):
        map_exit_code(returncode=3, stderr="bad", subcommand="start")


def test_exit_4_unknown_maps_to_generic() -> None:
    """Unexpected non-zero rc → :class:`GovernorError` with that rc."""
    with pytest.raises(GovernorError) as excinfo:
        map_exit_code(returncode=4, stderr="???", subcommand="verify")
    assert excinfo.value.exit_code == 4


def test_empty_stderr_does_not_crash() -> None:
    """Stderr may be empty; sniffers only fire on actual matches."""
    with pytest.raises(GovernorError) as excinfo:
        map_exit_code(returncode=1, stderr="", subcommand="verify")
    assert excinfo.value.stderr == ""


def test_case_insensitive_sniff() -> None:
    """``PermissionDenied`` sniff is case-insensitive."""
    with pytest.raises(PermissionDeniedError):
        map_exit_code(
            returncode=1,
            stderr="PERMISSIONDENIED example",
            subcommand="record",
        )
