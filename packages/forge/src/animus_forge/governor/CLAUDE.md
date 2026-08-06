# CLAUDE.md — animus_forge.governor

Adapter that wraps the [`animus-loop-governor`](https://github.com/AreteDriver/animus-loop-governor)
control plane as an Animus Forge verifier. Adopted via ADL-20260805-001.

## Headline property

> **The worker may claim completion. Only the Governor decides completion.**

This module enforces that property at the scheduler boundary: a mission
cannot enter `RUNNING` until its repository has a valid Governor run,
and a mission cannot exit `RUNNING` until the verifier citizen returns
the Governor's `alg verify` decision.

## Local engineering constraints

- Follow Forge's p95 line-length target of 77.
- Use `pathlib.Path` for filesystem paths.
- Catch specific exceptions; never use bare `except:`.
- Use `sys.executable` when invoking Python subprocesses.
- Never invoke the Governor CLI through `shell=True`.
- Keep this adapter independent of Quorum.
- Mission metadata owns `governor_run_id`.
- Governor preparation occurs once at mission start,
  never once per task.
- A mission must not enter RUNNING until workspace
  preparation succeeds.
- Maintain at least 97% test coverage for this adapter.

## Hard rules (inherited from ADL-20260805-001)

1. **Subprocess only — no in-process import.** Never
   `import animus_loop_governor`. Communication through the `alg` CLI.
2. **Never mutate a sealed task contract.** Once `alg start` runs,
   the contract is sealed with `contract.sha256`; further changes
   raise `ContractIntegrityError`.
3. **Verifier only.** This module never emits worker events. The
   citizen only calls `alg verify`. Worker event recording
   (`alg record`) and command execution (`alg exec`) are builder /
   reviewer concerns, not ours.
4. **Fail loud, never silent.** Missing `alg` or rejected contracts
   raise typed exceptions. The scheduler treats them as a `FAILED`
   mission transition (or stays in `READY` for retry).
5. **Strict compatibility.** A known run id is only reused when
   the `CompatibilityKey` matches exactly: same canonical
   repository path, same mission id, same policy version, same
   adapter version. Mismatch → `RunUnusableError`.

## Public API (one seam)

```python
from animus_forge.governor import GovernorAdapter

adapter = GovernorAdapter()
receipt = await adapter.ensure_run(
    repository=Path("/path/to/repo"),
    mission_id=mission.mission_id,
    contract_path=Path("contracts/mission-001.yaml"),
    known_run_id=mission.metadata.get("governor_run", {}).get("run_id"),
)
# Persist receipt to mission.metadata["governor_run"] atomically
# with the READY → RUNNING transition.
```

## Module map

| File | Purpose |
|---|---|
| `errors.py` | Typed exception hierarchy; one root, ten subclasses |
| `exit_codes.py` | `alg` exit-code → typed exception mapping |
| `paths.py` | `.animus-loop-governor/` run-dir resolution |
| `protocol.py` | Pydantic mirrors of the 5 consumer-side Governor JSON schemas |
| `models.py` | Adapter-side models: `CompatibilityKey`, `GovernorRun` |
| `client.py` | Subprocess wrapper (no `shell=True`, sanitized env) |
| `adapter.py` | `GovernorAdapter.ensure_run` + verifier citizen + state reader |

## Exit-code contract

| `alg` rc | Subcommand | Adapter exception | Scheduler impact |
|---|---|---|---|
| 0 | any | (none) | run is created / verified successfully |
| 1 | any | `GovernorError` (sniffed to `PermissionDeniedError` / `ContractIntegrityError`) | mission → `FAILED` |
| 2 | `compile` | `ContractRejectedError` | mission prep fails; stays in `READY` |
| 3 | `verify` | `VerifyDeniedError` | verifier citizen returns `needs_repair` |

## Test plan

See `tests/test_governor/`:

* `test_unit.py` — pure Python helpers (errors, paths, models, protocol)
* `test_exit_codes.py` — exit-code mapping
* `test_client.py` — command construction + output parsing
* `test_adapter.py` — run-resolution matrix
* `test_verifier_citizen.py` — verifier citizen output mapping
* `test_scheduler_integration.py` — mission-level lifecycle contract

Coverage target: **≥97%** (`coverage.report.fail_under = 97`).

## Dependencies

- **Production:** stdlib only (`pathlib`, `subprocess`, `shutil`, `json`,
  `re`, `logging`, `uuid`).
- **Existing Forge:** `animus_forge.citizens.base.Citizen`,
  `animus_forge.missions.domain.{CitizenOutput, Task, TaskContext, Mission}`.
- **Optional dev:** `pytest`, `pytest-asyncio` (the adapter is sync;
  tests use `asyncio_mode = "auto"` only for the scheduler-integration
  tests that follow Forge conventions).

The adapter does **not** depend on Quorum, on the Governor Python
package, or on any HTTP/CLI framework.