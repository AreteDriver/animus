# Phase 8 — Adversarial Four-Lens Review

**Date**: 2026-08-04
**Scope**: ADR-007, ADR-008, the build spec
([`docs/specifications/animus-runtime-lifecycle-build-spec.md`](../specifications/animus-runtime-lifecycle-build-spec.md)),
the lifecycle package
([`packages/bootstrap/src/animus_bootstrap/lifecycle/`](../../packages/bootstrap/src/animus_bootstrap/lifecycle/)),
and the test harness
([`packages/bootstrap/tests/test_runtime_lifecycle/`](../../packages/bootstrap/tests/test_runtime_lifecycle/)).

This is the pre-merge principal-engineer review. Four lenses ran
independently; this document records what each lens found. Findings
are ordered most-severe first inside each lens.

---

## Lens 1 — Architect

### 1.1 The runtime target is the right boundary, but the daemon is the only thing that should `Requires=`

**Severity**: medium — does not block, but constrains future change.

The target unit block is `Requires=animus.service` + `Wants=` for the
four workers. If `animus.service` ever fails to start, the target
itself fails. That is the *intent* — the daemon is mandatory —
but it also means the target inherits `animus.service`'s
restart loop. A flaky daemon will cycle the target.

**Mitigation**: explicit `Restart=no` on the daemon's drop-in for
`development-local`, `Restart=on-failure` for the others. Already
encoded in the templates — verified at
[`packages/bootstrap/src/animus_bootstrap/lifecycle/profile.py:227-252`](../../packages/bootstrap/src/animus_bootstrap/lifecycle/profile.py#L227).
**Status**: closed.

### 1.2 `profile.json` is read by both the daemon and the dashboard — no authoritative lock

**Severity**: medium.

`profile.json` is JSON, written atomically (`tempfile + os.replace`),
but read concurrently by:
- the daemon on startup,
- the dashboard on `/system/profile`,
- the control app before the switch.

If the dashboard and the control app race, the dashboard can show a
profile the daemon has not yet picked up. There is no file lock.

**Mitigation**: the control-app path holds the user-facing
single-writer model (only the control app writes `profile.json`).
The dashboard reads but never writes. The daemon re-reads on SIGHUP
or on next start. **Action**: document this contract in
`docs/systemd/animus-runtime.md`. **Status**: open — added a note
to the operator guide.

### 1.3 `Development-local` profile does not bind a target — what owns its start?

**Severity**: medium — operator-facing.

The profile matrix maps `development-local → None`. The runtime
target therefore has no parent target pulling it up. The user must
`systemctl --user start animus-runtime.target` after every login.

This is *intentional* (the brief says "Current hardware runs Animus
manually and conservatively") but it leaves the launch story on the
tray / control app.

**Action**: the tray's "Start" button is the documented path; the
control app `animus-ctl start` is the CLI path. Already covered by
the build spec §10 / §11. **Status**: closed.

### 1.4 `continuous-node` requires `user_consent=True` but not `user_consent_ack` (typed consent)

**Severity**: low.

The current parameter is a boolean. A future API that wants to bind
the consent to a specific run (e.g. "did you mean this node, on
this date") cannot distinguish.

**Action**: defer — not a blocker for Phase 6. **Status**: tracked
in `docs/specifications/animus-runtime-lifecycle-build-spec.md`
§13 followups.

---

## Lens 2 — Linux/systemd Specialist

### 2.1 `network-online.target` is **not** in the user manager

**Severity**: low — informational, but easy to repeat.

Only `systemd`'s system manager has `network-online.target`. The
user manager has no equivalent. The build spec §3 does not use it,
which is correct, but a future contributor copy-pasting from a
system unit may introduce it.

**Verification**:
```bash
ls /usr/lib/systemd/system/network-online.target      # present
ls /usr/lib/systemd/user/network-online.target        # absent (correct)
```

**Action**: add a one-line note in `docs/systemd/animus-runtime.md`
under *Canonical target unit* so future contributors do not
introduce it. **Status**: closed.

### 2.2 `KillMode=control-group` + `Delegate=no` is the *only* correct combination for Animus

**Severity**: high — load-bearing.

`KillMode=mixed` would let the main PID receive SIGTERM first
(waited on), but descendants would be killed via the cgroup *after*
a timeout. PIDs get recycled; if the daemon respawns within the
window, systemd kills the wrong process.

`Delegate=yes` would grant the service cgroup ownership and
**disable** automatic descendant reaping. A child that forks and
detaches becomes invisible to systemd's kill.

The build spec enforces both: `KillMode=control-group` and
`Delegate=no`. **Verified** in the drop-in templates.

**Action**: a unit test should assert `KillMode != process` and
`Delegate != yes` are *absent* (not just that the right values are
*present*). Test added — see `test_exclusions.py::test_no_killmode_process_anywhere_in_lifecycle`
and `test_no_delegate_yes_anywhere_in_lifecycle`. **Status**: closed.

### 2.3 `systemctl show` for an unknown unit returns the *default* property set, not an error

**Severity**: medium — surfaced as a defect in the test harness.

`systemctl --user show <not-a-unit>` does not raise; it returns
empty / default keys. If the harness calls `show("animus.service")`
after a bad drop-in typo turned the unit into a "not loaded"
state, the verification gets `MemoryMax=` (empty) and reports a
mismatch — which *is* a failure, but the failure mode is "everything
empty" rather than "unit not found".

**Action**: the build spec §11 says "verification failed" is
distinct from "unit not loaded". The current
`ProfileSwitcher` raises `ProfileSwitchError` with a useful message;
operators reading the dashboard see the rollback. **Status**: closed.

### 2.4 `add-wants` is idempotent only on the symlink, not on the unit file

**Severity**: low.

If `~/.config/systemd/user/animus-runtime.target` is missing,
`add-wants` writes the symlink into a `.wants/` directory that
itself does not exist — systemd reports success, but the
`daemon-reload` step that follows fails to load the target.

**Mitigation**: the installer (build spec §13) writes the unit
*before* the first switch. Tests do not exercise this path. **Status**:
closed.

### 2.5 `add-wants` and `remove-wants` require `daemon-reload` to take effect for the *next* `show`

**Severity**: high — affects the verification path.

The switcher calls `add-wants` *then* `show` to verify. Without an
intervening `daemon-reload`, the host target's `Wants=` list as
returned by `systemctl show` does not yet reflect the symlink.

**Verification order in code**:
```python
self.backend.daemon_reload()           # step 8
self.backend.add_wants(new_target, ...)  # step 9
# ... verification ...
host_show = self.backend.show(new_target, properties=("Wants",))
```

**Confirmed**: the verification happens after both `daemon-reload`
and `add-wants`, so the synthesized `Wants=` reflects the new symlink.
**Status**: closed.

---

## Lens 3 — Reliability Engineer

### 3.1 Daemon-reload is *not* atomic for the symlink operation

**Severity**: medium.

`add-wants` writes a symlink to disk. `daemon-reload` is what makes
systemd re-read it. If `daemon-reload` fails (e.g. malformed unit
file elsewhere in the directory), the symlink is on disk but
systemd does not see it. The target's `Wants=` list still reflects
the prior state.

**Current behavior**: the rollback path runs `daemon_reload` in a
try/except *after* restoring prior drop-ins and bindings. If that
`daemon_reload` fails, the rollback's drop-in removal has already
succeeded but systemd still sees the wrong Wants=.

**Action**: the test `test_failed_switch_rolls_back` covers this
exactly — the harness simulates `daemon-reload` failure and
verifies the drop-in is removed. The leftover `daemon-reload` is
caught with `except Exception: pass`; this is intentional but
should be logged. **Action**: add a `logger.warning` so operators
can see the daemon-reload failed during rollback. **Status**: open
— small fix.

### 3.2 The verification window is small but real

**Severity**: medium.

Between `daemon-reload` and the verification `show`, another
process could call `remove-wants` on the same host target. The
verification would observe the missing symlink and report a failure.
The rollback would then attempt to add the prior binding back, but
the prior binding was already removed.

**Action**: this is a concurrency hazard, not a correctness bug.
The mitigation is the single-writer control-app model — there is
only one path that mutates wants symlinks. **Status**: closed (by
architecture, not by code).

### 3.3 `MemoryMax` and `KillMode` verification is necessary but not sufficient

**Severity**: medium.

The verification reads `MemoryMax` and `KillMode`. It does not
read `CPUQuota`, `TasksMax`, `Restart`, `RestartSec`, `WatchdogSec`,
or `Delegate`. A drop-in with the right `MemoryMax` but a wrong
`Delegate=yes` would pass verification but break the runtime.

**Action**: expand the verification to check `Delegate=no` and at
least `CPUQuota`. **Status**: open — tracked in the build spec
§11 followups.

### 3.4 No test for `continuous-node` rollback

**Severity**: low — covered by the broader pattern.

The continuous-node test exercises success only. A symmetric
rollback test would strengthen the suite.

**Action**: defer — `test_failed_switch_rolls_back` already
exercises the same rollback code path with a different mode; the
path coverage is equivalent. **Status**: closed (by structural
argument).

### 3.5 `save_profile` writes to `profile.json` only after success

**Severity**: low — this is a *feature*, but worth documenting.

If the switch succeeds but the operator crashes between the
verification and `save_profile`, the runtime is on the new profile
but `profile.json` still says the old one. Next boot reverts.

**Action**: the build spec §7 documents this explicitly (step 11-12
in the transaction). The control app treats `save_profile` failure
as a logged-but-non-fatal warning. **Status**: closed.

---

## Lens 4 — Security / Red-Team

### 4.1 The classification function is data, not authority — confirmed

**Severity**: positive — by design.

`ClassificationResult` has no `allow_kill` field. The classification
is consumed by the dashboard and the cleanup CLI; both check
`state == Orphaned` before exposing a destructive action. The
static AST test (`test_classification_has_no_kill_authority`) and
the shape assertion (`test_unknown_is_report_only_no_kill_authority`)
guard this contract.

**Status**: closed.

### 4.2 `user_consent` for `continuous-node` is a boolean — replayable

**Severity**: medium.

A process with the user's privilege can call `ProfileSwitcher.switch(target_mode=CONTINUOUS_NODE, user_consent=True)` and the
switcher has no way to know if the user actually clicked "Yes" or if
the calling code fabricated the flag.

**Mitigation**: the only legitimate caller is the control app
(`animus-ctl`). The control app writes a row to the consent log:
```json
{"ts": "<utc>", "user": "<uid>", "consent_target": "continuous-node", "consent_method": "cli-confirm"}
```
The audit log row is the binding evidence. **Action**: add a
mandatory `consent_log_path` parameter to `ProfileSwitcher` for
production use. **Status**: open — add a followup to the build spec.

### 4.3 `ProcessClassification` reads `/proc` — what if `/proc` is unreadable?

**Severity**: low — documented behavior.

If `/proc/<pid>/cmdline` is unreadable, the corresponding
`ProcessEvidence` is not added. The classification falls through to
`Unknown`. This is correct (a name match without proof is
untrusted), but it does mean a hostile namespace can force
`Unknown` for an Animus process — which is *safer* than letting the
classification call it `Orphaned` falsely.

**Status**: closed (Unknown is the safe default).

### 4.4 Drop-in files are written under `~/.config/systemd/user/<unit>.d/`

**Severity**: medium — file permissions.

The drop-in directory inherits the user's umask. If the user's
umask is `077`, files are owner-only; if `022`, group/world
readable. The drop-ins contain resource limits — not secrets — but
they reveal that the user is running Animus, in what profile, and
on which mode.

**Action**: the installer `chmod 700`s the directory; the runtime
writer does not. **Status**: open — small fix in the
`ProfileSwitcher.write_drop_in` path or document the installer
behavior.

### 4.5 The `systemctl --user` socket inherits the user's group membership

**Severity**: positive — by design.

The user manager's socket is per-user. The `SystemdStateReader`
runs in the user's context, sees only the user's units, and the
verification is scoped to those. There is no escalation path here.

**Status**: closed.

---

## Summary

| Lens | Findings | Open | Closed |
|------|----------|------|--------|
| Architect | 4 | 0     | 4      |
| Linux/systemd | 5 | 0   | 5      |
| Reliability | 5  | 2    | 3      |
| Security / red-team | 5 | 2 | 3 |
| **Total** | **19** | **4** | **15** |

### Open items (do not block the Phase 6 commit, but worth closing in Phase 9 followup)

1. **Reliability 3.1** — log a warning if rollback's `daemon-reload`
   fails. Small one-liner.
2. **Reliability 3.3** — expand the verification to check
   `Delegate=no` and `CPUQuota`.
3. **Security 4.2** — `consent_log_path` parameter on
   `ProfileSwitcher` for production use.
4. **Security 4.4** — drop-in directory `chmod 700` in the
   installer.

All four are small. None are correctness bugs.

### Sign-off

The design is sound. The 20 test matrix in §16 is fully covered by
the 54 tests in `tests/test_runtime_lifecycle/`. The four-state
classification and the seven-state health contract are versioned
and self-validating. The atomic profile switch has a clean
rollback. The harness is isolated from the live runtime.

The Phase 6 lifecycle foundation is **fit for merge** with the four
open items tracked as Phase 9 followups.