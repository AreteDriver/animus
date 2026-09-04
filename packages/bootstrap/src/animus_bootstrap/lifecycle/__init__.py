"""Animus runtime lifecycle — primitives for the unified systemd lifecycle.

This package implements the architectural decision in
``adrs/ADR-007-runtime-lifecycle.md`` (Accepted) and the build contract in
``docs/specifications/animus-runtime-lifecycle-build-spec.md``.

Public surface:

- :class:`ProcessClassification` — the four-state classification with
  provenance rules.
- :class:`ProfileConfig` — desired-state schema for ``profile.json``.
- :class:`ProfileSwitcher` — atomic profile switch with rollback.
- :class:`HealthState` — the seven-state health enum and derivation logic.
- :class:`SystemdStateReader` — machine-readable state via ``systemctl show``.
- :class:`HealthContract` — versioned response contract for ``/healthz``.

The package intentionally does not start, stop, or signal processes. The
control app and dashboard consume these primitives; the ``animus-cleanup``
CLI is the only place that kills anything, and it uses the
:class:`ProcessClassification` provenance rules.
"""

from __future__ import annotations

from animus_bootstrap.lifecycle.classification import (
    ClassificationInput,
    ClassificationResult,
    ProcessClassification,
    ProcessEvidence,
    classify_process,
    default_provenance_threshold,
)
from animus_bootstrap.lifecycle.health import (
    HealthContract,
    HealthSnapshot,
    HealthState,
    ServiceHealth,
    derive_health_state,
)
from animus_bootstrap.lifecycle.profile import (
    PROFILE_TARGET_BINDINGS,
    ProfileConfig,
    ProfileMode,
    ProfileSwitcher,
    ProfileSwitchError,
    ProfileSwitchResult,
    SwitchBackend,
    load_profile,
    save_profile,
)
from animus_bootstrap.lifecycle.systemd import (
    SystemdInvoker,
    SystemdStateError,
    SystemdStateReader,
    UnitState,
    parse_show_output,
)

__all__ = [
    "PROFILE_TARGET_BINDINGS",
    "ClassificationInput",
    "ClassificationResult",
    "HealthContract",
    "HealthSnapshot",
    "HealthState",
    "ProcessClassification",
    "ProcessEvidence",
    "ProfileConfig",
    "ProfileMode",
    "ProfileSwitchError",
    "ProfileSwitchResult",
    "ProfileSwitcher",
    "ServiceHealth",
    "SwitchBackend",
    "SystemdInvoker",
    "SystemdStateError",
    "SystemdStateReader",
    "UnitState",
    "classify_process",
    "default_provenance_threshold",
    "derive_health_state",
    "load_profile",
    "parse_show_output",
    "save_profile",
]
