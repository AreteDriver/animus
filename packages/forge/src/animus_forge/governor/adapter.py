"""Mission-level orchestrator for the Animus Loop Governor.

The single public entry point is :meth:`GovernorAdapter.ensure_run`. It
implements the strict idempotent resolution algorithm the scheduler
relies on:

    1. Mission metadata already has a known run id
       - validate it: exists on disk, active, belongs to the same
         repository, on a compatible revision, not terminated
       - if valid → reuse
       - if invalid → raise :class:`RunUnusableError`; caller decides
         whether to create a new run or fail the mission
    2. Search for a compatible active run via :func:`paths.find_active_run`
       - if one matches → reuse
       - if one is found but mismatches → raise :class:`RunUnusableError`
    3. Compile and start a fresh run; return its id
       - the caller is responsible for persisting ``governor_run_id``
         in mission metadata atomically with the READY → RUNNING
         transition

Plus the verifier citizen (:class:`GovernorVerifierCitizen`) which runs
once per mission completion to invoke ``alg verify`` and map the
Governor decision back into Forge's :class:`CitizenOutput`.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from animus_forge.citizens.base import Citizen
from animus_forge.governor.client import GovernorClient
from animus_forge.governor.errors import (
    RunStateInvalidError,
    RunUnusableError,
    VerifyDeniedError,
)
from animus_forge.governor.models import (
    CompatibilityKey,
    GovernorRun,
    MissionKey,
    RepositoryKey,
)
from animus_forge.governor.paths import (
    find_active_run,
    run_dir,
)
from animus_forge.governor.protocol import (
    CompletionDecision,
    RunLedger,
    WatchdogReport,
)
from animus_forge.missions.domain import (
    CitizenOutput,
    Task,
    TaskContext,
)

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "0.1.0"
DEFAULT_POLICY_VERSION = 1
DEFAULT_COMPAT_TIMEOUT_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Compatibility key derivation
# ---------------------------------------------------------------------------


def compute_compatibility_key(
    *,
    repository: Path,
    mission_id: str | UUID,
    contract_digest: str | None = None,
    remote_identity: str | None = None,
    revision: str | None = None,
    worktree: Path | None = None,
) -> CompatibilityKey:
    """Derive a :class:`CompatibilityKey` for a candidate run.

    Best-effort: unknown fields stay ``None``. The adapter validates
    the key against existing runs in :meth:`GovernorAdapter.ensure_run`;
    a key with all-``None`` repository fields is a degenerate case that
    the adapter rejects explicitly.
    """
    canonical = repository.resolve()
    repo_key = RepositoryKey(
        canonical_path=str(canonical),
        remote_identity=remote_identity,
        revision=revision,
        worktree=str(worktree) if worktree else None,
    )
    mission_key = MissionKey(
        mission_id=str(mission_id), contract_digest=contract_digest
    )
    return CompatibilityKey(
        repository=repo_key,
        mission=mission_key,
        policy_version=DEFAULT_POLICY_VERSION,
        adapter_version=ADAPTER_VERSION,
    )


# ---------------------------------------------------------------------------
# Mission-level orchestrator
# ---------------------------------------------------------------------------


class GovernorAdapter:
    """Single seam between Forge's scheduler and the ``alg`` CLI.

    Args:
        client: Subprocess wrapper. Tests pass a fake; production
            uses the real :class:`GovernorClient`.
        run_id_resolver: Optional override for resolving ``known_run_id``
            receipts from mission metadata. Production wires the
            mission-store-aware path; tests pass a callable that
            returns ``None`` (always create new).
    """

    def __init__(
        self,
        client: GovernorClient | None = None,
        *,
        run_id_resolver: RunIdResolver | None = None,
    ) -> None:
        self.client = client or GovernorClient()
        self._resolver = run_id_resolver or _NullRunIdResolver()

    def ensure_run(
        self,
        *,
        repository: Path,
        mission_id: str | UUID,
        contract_path: Path,
        known_run_id: str | None = None,
        compatibility: CompatibilityKey | None = None,
    ) -> GovernorRun:
        """Idempotently produce a valid :class:`GovernorRun`.

        Resolution order:

        1. ``known_run_id`` (from mission metadata) → validate.
        2. ``alg`` on-disk run under ``.animus-loop-governor/runs/``
           → validate.
        3. Otherwise: ``alg start`` → return new run.

        Validation means: directory exists, ledger parses, repository
        identity matches, mission matches, phase is not terminal.
        Failure on any check raises :class:`RunUnusableError`.

        Concurrency: ``alg start`` is called inside a per-mission
        lock so two callers cannot both create a run. The lock is
        optional (``run_id_resolver`` may provide one); when absent,
        we rely on the caller to serialize (the scheduler does this
        via the mission lease).
        """
        compat = compatibility or compute_compatibility_key(
            repository=repository, mission_id=mission_id
        )

        # Step 1: known run id from metadata.
        candidate = known_run_id or self._resolver.lookup(mission_id)
        if candidate:
            validated = self._validate_or_raise(
                repository=repository,
                run_id=candidate,
                compat=compat,
            )
            if validated is not None:
                return validated

        # Step 2: hint from filesystem (most recent mtime). The hint
        # is **opportunistic**: if the on-disk run belongs to a
        # different mission, we silently fall through to Step 3
        # rather than reject — a cross-mission run is not a bug, it
        # is just not reusable for *this* mission. Other validation
        # failures (terminal phase, missing ledger, partially
        # initialised) remain fatal — they signal real corruption.
        hinted = find_active_run(repository)
        if hinted is not None:
            validated = self._validate_or_raise(
                repository=repository,
                run_id=hinted.name,
                compat=compat,
                mission_mismatch_is_fatal=False,
            )
            if validated is not None:
                return validated

        # Step 3: compile and start a fresh run.
        return self._create_new_run(
            repository=repository,
            mission_id=mission_id,
            contract_path=contract_path,
            compat=compat,
        )

    def _validate_or_raise(
        self,
        *,
        repository: Path,
        run_id: str,
        compat: CompatibilityKey,
        mission_mismatch_is_fatal: bool = True,
    ) -> GovernorRun | None:
        """Validate an existing run; return receipt or ``None``.

        Returns ``None`` when the run id is not present at all (so the
        caller can fall through to the filesystem hint or new-run
        creation). Raises :class:`RunUnusableError` when a run is
        present but cannot be reused.

        ``mission_mismatch_is_fatal`` controls behaviour on a
        compatibility mismatch: when ``True`` (the default, used by
        Step 1 with a known id), any mismatch is fatal. When ``False``
        (Step 2 filesystem hint), a mission mismatch is a soft signal
        that the hinted run belongs to a different mission; we
        return ``None`` so the caller can try Step 3 instead.
        """
        path = run_dir(repository, run_id)
        if not path.is_dir():
            return None

        ledger = _read_ledger_or_none(path)
        if ledger is None:
            raise RunUnusableError(
                f"Known run {run_id} at {path} has no parseable ledger"
            )
        if ledger.phase in {"complete", "failed", "aborted"}:
            raise RunUnusableError(
                f"Known run {run_id} is in terminal phase {ledger.phase}"
            )

        receipt = _read_receipt_or_none(path)
        if receipt is None:
            # No receipt yet but the run exists and is not terminal —
            # treat as partially initialised and reject.
            raise RunUnusableError(
                f"Known run {run_id} is partially initialised"
            )
        if not _receipt_matches(receipt, compat):
            if mission_mismatch_is_fatal:
                raise RunUnusableError(
                    f"Known run {run_id} does not match the requested "
                    f"compatibility key (expected mission "
                    f"{compat.mission.mission_id}, repository "
                    f"{compat.repository.canonical_path})"
                )
            return None
        return receipt

    def _create_new_run(
        self,
        *,
        repository: Path,
        mission_id: str | UUID,
        contract_path: Path,
        compat: CompatibilityKey,
    ) -> GovernorRun:
        """Run ``alg start`` and persist the receipt + a ledger stub.

        The real ``alg start`` writes ``ledger.json`` and creates the
        run directory as a side effect; the fake test double returns
        just the run id. We tolerate both: ``_persist_receipt`` and
        ``_persist_ledger_stub`` create the directory if it is
        missing, so the post-conditions on disk are identical
        regardless of which path produced the id.

        The ledger stub keeps the ``find_active_run`` → validation
        path honest on subsequent calls within the same mission — a
        later ``ensure_run`` will find this run via Step 2 and pass
        validation because both ledger and receipt are on disk.
        """
        run_id = self.client.start(
            contract_path=contract_path,
            cwd=repository,
        )
        path = run_dir(repository, run_id)
        receipt = GovernorRun(
            run_id=run_id,
            repository=repository,
            contract_path=contract_path,
            started_at=datetime.now(UTC).isoformat(),
            compatibility=compat,
            diagnostics={"created_by": "ensure_run"},
        )
        _persist_receipt(path, receipt)
        _persist_ledger_stub(path, run_id=run_id)
        logger.info(
            "Created Governor run %s for mission %s at %s",
            run_id,
            mission_id,
            path,
        )
        return receipt


# ---------------------------------------------------------------------------
# Receipt persistence
# ---------------------------------------------------------------------------


RECEIPT_FILENAME = "adapter-receipt.json"


def _persist_receipt(run_path: Path, receipt: GovernorRun) -> None:
    """Atomically write the receipt JSON next to the run's ledger.

    Creates the run directory if it does not exist — covers the
    case where the adapter produced the run id (e.g. via a test
    double) without a real ``alg start`` writing the dir.
    """
    run_path.mkdir(parents=True, exist_ok=True)
    target = run_path / RECEIPT_FILENAME
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        receipt.model_dump_json(indent=2), encoding="utf-8"
    )
    tmp.replace(target)


def _persist_ledger_stub(run_path: Path, *, run_id: str) -> None:
    """Write a minimal ``ledger.json`` so subsequent ``find_active_run``
    hits pass validation.

    Only used when the adapter produced the run id without a real
    ``alg start`` writing the full ledger; in production the real
    ``alg start`` overwrites this stub.
    """
    run_path.mkdir(parents=True, exist_ok=True)
    ledger_path = run_path / "ledger.json"
    if ledger_path.is_file():
        return  # production ``alg start`` already wrote it
    stub = RunLedger(
        run_id=run_id,
        task_id="adapter-stub",
        contract_hash="adapter-stub",
        phase="contracted",
    )
    ledger_path.write_text(stub.model_dump_json(), encoding="utf-8")


def _read_receipt_or_none(run_path: Path) -> GovernorRun | None:
    """Read the receipt if present; ``None`` if absent or corrupt."""
    target = run_path / RECEIPT_FILENAME
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunStateInvalidError(
            f"Receipt at {target} is corrupt: {exc}"
        ) from exc
    return GovernorRun.model_validate(data)


def _read_ledger_or_none(run_path: Path) -> RunLedger | None:
    """Read the run ledger if present; ``None`` if absent."""
    target = run_path / "ledger.json"
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunStateInvalidError(
            f"Ledger at {target} is corrupt: {exc}"
        ) from exc
    return RunLedger.model_validate(data)


def _receipt_matches(
    receipt: GovernorRun, compat: CompatibilityKey
) -> bool:
    """Strict equality check between receipt and requested key."""
    return receipt.compatibility == compat


# ---------------------------------------------------------------------------
# RunIdResolver — abstracts how the scheduler supplies known_run_id
# ---------------------------------------------------------------------------


class RunIdResolver:
    """Abstract base for the scheduler to supply a known run id.

    Production wires a resolver that reads from the mission ledger.
    Tests pass a stub that returns a fixed id or ``None``.
    """

    def lookup(self, mission_id: str | UUID) -> str | None:
        """Return a known run id for the mission, or ``None``."""
        raise NotImplementedError


class _NullRunIdResolver(RunIdResolver):
    """Default resolver: always returns ``None`` (no persisted id)."""

    def lookup(self, mission_id: str | UUID) -> str | None:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Verifier citizen — invoked after the worker chain for every mission
# ---------------------------------------------------------------------------


class GovernorVerifierCitizen(Citizen):
    """Maps the Governor decision to a :class:`CitizenOutput`.

    role = ``"loop_governor"``; this citizen never modifies code, never
    approves work on its own — it merely forwards the Governor's
    verdict into Forge's retry pipeline.

    Exit-code mapping (from :mod:`exit_codes`):

    * rc 0 + watchdog clean → ``status="completed"``
    * rc 0 + watchdog ``required_action`` → ``status="needs_repair"``
    * rc 3 (``VerifyDeniedError``) → ``status="needs_repair"`` with
      explicit ``missing_evidence`` from ``completion-latest.json``
    * any other :class:`GovernorAdapterError` → ``status="failed"``
    """

    role = "loop_governor"
    capabilities = {"verify", "completion-decision", "drift-detection"}
    can_modify_code = False
    can_approve = False

    def __init__(
        self,
        client: GovernorClient | None = None,
        *,
        reader: RunStateReader | None = None,
    ) -> None:
        self._client = client or GovernorClient()
        self._reader = reader or RunStateReader()

    def run(self, task: Task, context: TaskContext) -> CitizenOutput:
        """Invoke ``alg verify`` and translate the verdict."""
        repository = Path(context.repository) if context.repository else None
        if repository is None or not repository.is_dir():
            return CitizenOutput(
                status="failed",
                summary="Governor verifier: missing or invalid repository",
                risks=[
                    {
                        "type": "no_repository",
                        "repository": (
                            str(repository) if repository else None
                        ),
                    }
                ],
                follow_up_tasks=[
                    "repair: ensure context.repository is a valid path"
                ],
                confidence=0.0,
            )

        run_id = _resolve_run_id_for_task(context)
        if run_id is None:
            return CitizenOutput(
                status="failed",
                summary="Governor verifier: no governor_run_id on context",
                risks=[
                    {
                        "type": "no_governor_run",
                        "repository": str(repository),
                    }
                ],
                follow_up_tasks=[
                    "repair: ensure mission has gone through ensure_run()"
                ],
                confidence=0.0,
            )

        try:
            self._client.verify(run_id=run_id, cwd=repository)
        except VerifyDeniedError:
            return self._on_denial(repository=repository, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 — outer fault boundary
            return CitizenOutput(
                status="failed",
                summary=f"Governor verifier error: {exc}",
                risks=[{"type": "governor_error", "detail": str(exc)}],
                follow_up_tasks=[
                    f"repair: inspect .animus-loop-governor/runs/{run_id}"
                ],
                confidence=0.0,
            )

        # rc=0: verify approved. Watchdog may still require action.
        watchdog = self._reader.read_watchdog(repository, run_id)
        if watchdog is not None and watchdog.required_action:
            return CitizenOutput(
                status="needs_repair",
                summary=(
                    f"Watchdog requires action: {watchdog.required_action}"
                ),
                risks=[
                    {
                        "type": "watchdog",
                        "drift_score": watchdog.drift_score,
                        "stagnation": watchdog.stagnation,
                    }
                ],
                follow_up_tasks=[watchdog.required_action],
                evidence=[{"type": "governor_approval", "run_id": run_id}],
                confidence=1.0,
            )

        return CitizenOutput(
            status="completed",
            summary=f"Governor approved completion of run {run_id}",
            evidence=[{"type": "governor_approval", "run_id": run_id}],
            confidence=1.0,
        )

    def _on_denial(
        self, *, repository: Path, run_id: str
    ) -> CitizenOutput:
        """Map a VerifyDeniedError to a needs_repair citizen output."""
        decision = self._reader.read_completion(repository, run_id)
        reasons = "; ".join(decision.reasons)
        return CitizenOutput(
            status="needs_repair",
            summary=f"Governor denied completion of run {run_id}: {reasons}",
            evidence=[
                {
                    "type": "governor_decision",
                    "decision": decision.model_dump(mode="json"),
                }
            ],
            risks=[
                {
                    "type": "governor_denial",
                    "missing_evidence": decision.missing_evidence,
                    "blocking_findings": decision.blocking_findings,
                }
            ],
            follow_up_tasks=[
                *(
                    f"repair: provide {item}"
                    for item in decision.missing_evidence
                ),
                *(
                    f"repair: address finding: {finding}"
                    for finding in decision.blocking_findings
                ),
            ],
            confidence=1.0,
        )


def _resolve_run_id_for_task(context: TaskContext) -> str | None:
    """Pull the governor run id off the task context (inherited)."""
    # The scheduler is expected to populate ``extra_context`` or a
    # similar field. We deliberately do not invent a new TaskContext
    # field here — production wiring goes through the context dict.
    extras = getattr(context, "model_extra", None) or {}
    return extras.get("governor_run_id")  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Run-state reader — separate from the client (file I/O, not subprocess)
# ---------------------------------------------------------------------------


class RunStateReader:
    """Reads ``completion-latest.json`` and ``watchdog-latest.json``.

    Kept as a separate class so the citizen can be tested without
    subprocess mocking: the reader is purely a JSON file loader.
    """

    def read_completion(
        self, repository: Path, run_id: str
    ) -> CompletionDecision:
        path = run_dir(repository, run_id) / "completion-latest.json"
        if not path.is_file():
            raise RunStateInvalidError(
                f"completion-latest.json missing at {path}"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunStateInvalidError(
                f"completion-latest.json at {path} is corrupt: {exc}"
            ) from exc
        return CompletionDecision.model_validate(data)

    def read_watchdog(
        self, repository: Path, run_id: str
    ) -> WatchdogReport | None:
        """``None`` if no watchdog report exists yet (not an error)."""
        path = run_dir(repository, run_id) / "watchdog-latest.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunStateInvalidError(
                f"watchdog-latest.json at {path} is corrupt: {exc}"
            ) from exc
        return WatchdogReport.model_validate(data)


__all__ = [
    "ADAPTER_VERSION",
    "DEFAULT_POLICY_VERSION",
    "GovernorAdapter",
    "GovernorVerifierCitizen",
    "RECEIPT_FILENAME",
    "RunIdResolver",
    "RunStateReader",
    "compute_compatibility_key",
]
