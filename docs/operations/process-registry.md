# Animus Process Registry & Provenance

This document describes how the dashboard and the cleanup CLI
distinguish **Animus processes** from processes that merely share a
name. It pairs with ADR-007 and the build specification
([`docs/specifications/animus-runtime-lifecycle-build-spec.md`](../specifications/animus-runtime-lifecycle-build-spec.md)).

## Scope

- **In:** the four-state classification, the proof thresholds, the
  process registry, the cleanup-CLI rules.
- **Out:** the runtime target's start/stop — see
  [`docs/systemd/animus-runtime.md`](../systemd/animus-runtime.md).
- **Out:** killing decisions — the dashboard never kills a process
  the registry cannot prove Animus owns.

## The four-state classification

| State        | What it means                                                    |
|--------------|------------------------------------------------------------------|
| `Managed`    | Registered AND the systemd unit is active                         |
| `Recoverable`| Registered, unit inactive, at least one reliable proof            |
| `Orphaned`   | Registered, plus ≥2 independent proofs OR cgroup_alive, UID matches |
| `Unknown`    | Name matches, ownership unproven                                 |

The classification is pure: it consumes only `/proc` paths and
registry identity, never `pgrep`. The decision tree is:

1. **Managed**: `registry_identity` AND `unit_active=True`.
2. **Orphaned**: `registry_identity` AND `cgroup_alive=True` (decisive)
   OR `registry_identity` AND ≥2 reliable proofs AND UID matches.
3. **Recoverable**: `registry_identity` AND `unit_active=False` AND
   at least one reliable proof (executable, cmdline, start-time
   fingerprint). This is the intermediate state before enough evidence
   accumulates to call Orphaned.
4. **Unknown**: anything else.

`Orphaned` deliberately runs **before** `Recoverable` in the decision
tree — the cgroup itself may be the thing that was lost.

## What counts as a proof

The :mod:`animus_bootstrap.lifecycle.classification` module defines
six proof kinds:

| Constant                      | Source                              |
|-------------------------------|-------------------------------------|
| `PROOF_EXECUTABLE`            | `/proc/<pid>/exe` readlink target   |
| `PROOF_CMDLINE`               | `/proc/<pid>/cmdline` (first 4 KiB) |
| `PROOF_UID`                   | `/proc/<pid>/status` Uid line       |
| `PROOF_STARTTIME`             | `/proc/<pid>/stat` field 22 (ticks) |
| `PROOF_INSTANCE_ID`           | `ANIMUS_INSTANCE_ID` env var        |
| `PROOF_PARENT_HISTORY`        | `/proc/<pid>/stat` field 4 (ppid)  |

`Recoverable` requires only **one** of executable, cmdline, or
start-time. `Orphaned` requires **two independent** proofs in
addition to registry identity, or cgroup membership (decisive). The
threshold is centralized in
:func:`default_provenance_threshold` so it can be raised without
changing the public API.

A UID mismatch disqualifies `Orphaned` (and `Recoverable`). A process
running as the wrong user is not Animus's, by definition.

## The process registry

The :class:`SystemProcessRegistry` records every Animus service
launched under the user manager, keyed by `(unit, pid, instance_id)`.
The dashboard reads the registry; the cleanup CLI deletes from it. The
registry is a SQLite database at
`${XDG_CONFIG_HOME}/animus/data/process_registry.db`.

```bash
# Inspect the registry (control app).
animus-ctl registry list

# Tail a specific unit.
animus-ctl registry tail animus.service
```

The registry is **append-heavy by design**. Old rows are not deleted
unless the cleanup CLI explicitly removes them after a successful
classification-driven end-of-life.

## UID mismatch

A UID mismatch is one of the strongest disqualifiers. If a process
claims to be `animus-discord-bot` but its UID is not the user Animus
was installed as, it is `Unknown` — even with multiple `/proc`
proofs. The dashboard refuses to operate on such processes; the
cleanup CLI refuses to kill them.

This matters on shared hosts and CI runners, where the same binary
path may exist under a different UID.

## The cleanup CLI

```bash
# Show all not-Managed processes whose name matches an Animus unit.
animus-ctl cleanup list

# Show only Orphaned ones (with reason and proofs).
animus-ctl cleanup list --state=orphaned

# Kill one Orphaned PID (requires --confirm).
animus-ctl cleanup kill 12345 --confirm

# Show the proofs that backed a specific Orphaned classification.
animus-ctl cleanup why 12345
```

`Unknown` processes are report-only. The CLI prints them with their
reason and refuses any kill action. This is enforced at the data
shape: the classification result carries no `allow_kill` field. The
dashboard and CLI both check the classification state before exposing
a destructive action.

## Why no `pgrep`?

`pgrep` is never used. A name is not a proof. Two Python processes
named `animus_discord_bot.py` may exist under different UIDs, on
different hosts, with different parent cgroups. Killing on a name
match has historically been a "rm -rf" of process hygiene. The
classification function deliberately consumes only `/proc` paths and
registry identity so that two processes with identical names collapse
only when the registry + proofs confirm it.

This rule is enforced by static AST analysis — see
`tests/test_runtime_lifecycle/test_no_pgrep_in_lifecycle.py`.

## When classification is not enough

`Unknown` is the answer when the registry has nothing and the proofs
are silent. The dashboard's posture is: do nothing, expose the data,
let the operator decide. There is no "best-guess kill" codepath. If a
user wants to override, they can use `kill -<signal> <pid>` directly
against a `Unknown` PID — but the CLI does not do it for them.
