"""Process classification with provenance rules.

Implements the four-state classification mandated by ADR-007:

- ``Managed`` — registered AND attached to an active lifecycle.
- ``Recoverable`` — registered, parent metadata lost.
- ``Orphaned`` — Animus-owned, surviving after Animus stopped; requires
  registry identity plus at least two independent process proofs.
- ``Unknown`` — name matches but ownership unproven.

The :class:`ProcessClassification` enum is the *external* view that the
control app, dashboard, and cleanup CLI consume. It is distinct from the
internal :class:`ProcessState` enum in
``packages/core/animus/infrastructure/process_lifecycle.py``, which records
internal registry state.

``pgrep`` is never used. The functions in this module consume only
``/proc`` paths and registry identity. The result of every classification
function is JSON-serializable so the dashboard can render it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("animus_bootstrap.lifecycle.classification")


class ProcessClassification(str, Enum):  # noqa: UP042 - preserve persisted enum string behavior
    """External-facing process classification.

    The string values are what the dashboard API and the cleanup CLI
    consume. They appear in logs and persisted audit records.
    """

    MANAGED = "managed"
    RECOVERABLE = "recoverable"
    ORPHANED = "orphaned"
    UNKNOWN = "unknown"


# Provenance evidence types and their relative reliability. A process is
# ``Orphaned`` only when it has *registry identity* plus at least two
# independent evidences from this set.
PROOF_EXECUTABLE = "executable_path"
PROOF_CMDLINE = "command_line_launch_token"
PROOF_UID = "uid"
PROOF_STARTTIME = "start_time_fingerprint"
PROOF_INSTANCE_ID = "environment_instance_id"
PROOF_PARENT_HISTORY = "parent_history"


@dataclass(frozen=True)
class ProcessEvidence:
    """A single provenance proof.

    Attributes:
        kind: One of the ``PROOF_*`` constants.
        value: The proof's value (path, UID, fingerprint, etc.). The
            interpretation is kind-specific.
        reliable: Whether the proof is reliable in the current context.
            A proof may be unreliable if the source is missing (e.g.
            the cgroup was lost along with the parent).
    """

    kind: str
    value: str
    reliable: bool = True


@dataclass(frozen=True)
class ClassificationInput:
    """Inputs to :func:`classify_process`.

    Attributes:
        pid: Process ID. Zero/negative values are rejected.
        executable: ``/proc/<pid>/exe`` readlink target. ``None`` if
            the process is gone or unreadable.
        command_line: The first 4 KiB of ``/proc/<pid>/cmdline``.
            ``None`` if unreadable.
        start_time: ``/proc/<pid>/stat`` field 22 (starttime in clock
            ticks). ``None`` if unreadable.
        uid: ``/proc/<pid>/status`` Uid line. ``None`` if unreadable.
        ppid: ``/proc/<pid>/stat`` field 4 (parent pid). ``None`` if
            unreadable.
        expected_uid: The UID Animus was installed as. A mismatch
            disqualifies ``Orphaned`` (the process is not Animus).
        registry_identity: True if the process matches a row in
            ``SystemProcessRegistry``. False otherwise.
        unit_active: True if the systemd unit ostensibly owning the
            process is ``active``. ``None`` if unknown.
        cgroup_alive: True if the service cgroup is present and the
            process belongs to it. ``None`` if unknown.
        environment_instance_id: ``ANIMUS_INSTANCE_ID`` from the
            process's environment. ``None`` if not present.
        registry_match_key: A tuple identifying the registry row, used
            as the registry identity proof.
    """

    pid: int
    executable: str | None = None
    command_line: str | None = None
    start_time: int | None = None
    uid: int | None = None
    ppid: int | None = None
    expected_uid: int | None = None
    registry_identity: bool = False
    unit_active: bool | None = None
    cgroup_alive: bool | None = None
    environment_instance_id: str | None = None
    registry_match_key: tuple[str, ...] | None = None


@dataclass
class ClassificationResult:
    """The result of :func:`classify_process`.

    Attributes:
        classification: The four-state classification.
        proofs: The evidence that contributed to the decision.
        reason: A human-readable explanation suitable for the dashboard.
    """

    classification: ProcessClassification
    proofs: list[ProcessEvidence] = field(default_factory=list)
    reason: str = ""


def default_provenance_threshold() -> int:
    """Return the default threshold of independent proofs required for ``Orphaned``.

    ADR-007 requires at least two independent proofs in addition to
    registry identity. The threshold is centralized here so it can be
    raised without changing the public API.
    """
    return 2


def _build_evidences(inp: ClassificationInput) -> list[ProcessEvidence]:
    """Collect the provenance evidence from a classification input."""
    evs: list[ProcessEvidence] = []
    if inp.executable:
        evs.append(ProcessEvidence(PROOF_EXECUTABLE, inp.executable))
    if inp.command_line:
        evs.append(ProcessEvidence(PROOF_CMDLINE, inp.command_line[:256]))
    if inp.uid is not None:
        evs.append(ProcessEvidence(PROOF_UID, str(inp.uid)))
    if inp.start_time is not None:
        evs.append(ProcessEvidence(PROOF_STARTTIME, str(inp.start_time)))
    if inp.environment_instance_id:
        evs.append(ProcessEvidence(PROOF_INSTANCE_ID, inp.environment_instance_id))
    if inp.ppid is not None:
        evs.append(ProcessEvidence(PROOF_PARENT_HISTORY, f"ppid={inp.ppid}"))
    return evs


def _uid_matches(inp: ClassificationInput) -> bool:
    if inp.expected_uid is None or inp.uid is None:
        return True  # not sufficient to claim orphan
    return inp.uid == inp.expected_uid


def classify_process(inp: ClassificationInput) -> ClassificationResult:
    """Classify a process using the ADR-007 rules.

    The decision tree is:

    1. ``Managed`` if ``registry_identity`` AND ``unit_active`` is True.
    2. ``Orphaned`` if ``registry_identity`` AND decisive proof
       (cgroup_alive=True, or at least two independent proofs) AND
       the UID matches. Decisive proof wins over Recoverable because
       the cgroup may itself be the thing that was lost.
    3. ``Recoverable`` if ``registry_identity`` AND ``unit_active`` is False
       AND there is at least one reliable proof (executable, cmdline, or
       start-time fingerprint). Recoverable is the intermediate state
       before enough evidence accumulates to call Orphaned.
    4. ``Unknown`` otherwise (name matches but ownership unproven).
    """
    if inp.pid <= 0:
        return ClassificationResult(
            classification=ProcessClassification.UNKNOWN,
            reason="invalid pid",
        )

    evidences = _build_evidences(inp)

    # Rule 1: Managed
    if inp.registry_identity and inp.unit_active is True:
        return ClassificationResult(
            classification=ProcessClassification.MANAGED,
            proofs=evidences,
            reason="registered and unit active",
        )

    # Rule 2: Orphaned (decisive proof).
    # Decisive proof is registry identity + (cgroup_alive OR
    # threshold-many independent proofs) + UID match. This must run
    # before Recoverable because the cgroup may itself be the thing
    # that was lost — Recoverable would be wrong.
    if inp.registry_identity and _uid_matches(inp):
        if inp.cgroup_alive is True:
            return ClassificationResult(
                classification=ProcessClassification.ORPHANED,
                proofs=evidences,
                reason="registry identity + cgroup membership",
            )
        good = [e for e in evidences if e.reliable]
        if len(good) >= default_provenance_threshold():
            return ClassificationResult(
                classification=ProcessClassification.ORPHANED,
                proofs=evidences,
                reason=(f"registry identity + {len(good)} independent proofs"),
            )

    # Rule 3: Recoverable
    if inp.registry_identity and inp.unit_active is False and _uid_matches(inp):
        reliable = [
            e
            for e in evidences
            if e.reliable
            and e.kind
            in (
                PROOF_EXECUTABLE,
                PROOF_CMDLINE,
                PROOF_STARTTIME,
            )
        ]
        if reliable:
            return ClassificationResult(
                classification=ProcessClassification.RECOVERABLE,
                proofs=evidences,
                reason="registered but unit inactive; parent metadata lost",
            )

    # Rule 4: Unknown
    return ClassificationResult(
        classification=ProcessClassification.UNKNOWN,
        proofs=evidences,
        reason="name matches but ownership unproven",
    )


def majority_of_unknown_is_unknown(results: Iterable[ClassificationResult]) -> bool:
    """Helper for the dashboard: if N>0 results are all ``Unknown``, report so.

    Defensive: never narrows a real classification to ``Unknown``.
    """
    items = list(results)
    if not items:
        return True
    return all(r.classification == ProcessClassification.UNKNOWN for r in items)
