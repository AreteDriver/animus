# Animus Runtime Lifecycle — Build Specification

**Status**: Draft
**Source ADR**: `adrs/ADR-007-runtime-lifecycle.md` (Accepted)
**Review pattern**: `adrs/ADR-008-review-pattern.md` (Accepted)
**Date**: 2026-08-04
**Author**: arete
**Scope**: packages/bootstrap, packages/core, dashboard, installer, systemd units, test harness

---

## 1. Purpose and non-goals

### Purpose
Translate the architectural decision in ADR-007 into a buildable, testable, and reversible specification. The build spec defines the unit files, the profile switch transaction, the desired-state and observed-state schemas, the health contract, the classification rules, and the test harness that proves the lifecycle model is correct without touching the live runtime.

This document is the contract between the architectural decision and the implementation PRs. The implementation must conform to this spec; deviations require an ADR amendment.

### Non-goals
- Production-hardening of the `continuous-node` (GX10) mode. The spec defines the seams; the engineering program to harden them is a follow-on.
- Plaintext Forge systemd drop-in credential remediation. The separate security finding is tracked separately.
- Discord bot lifecycle classification. The bot is excluded from the runtime target.
- Autonomous timer lifecycle integration. The timers are excluded from the runtime target.
- Conversion of the existing `~/.local/bin/animus-tray` script into a fully packaged Tray application. The spec defines the *contract* the tray must obey; the packaging work is a follow-on.

---

## 2. Current-state inventory (verified 2026-08-04)

| Component | Current state | Source |
|---|---|---|
| `~/.config/systemd/user/animus.service` | Exists; disabled (no `default.target.wants/animus.service` symlink); `After=network-online.target` and `Wants=network-online.target` (which is broken — see ADR-007) | `packages/bootstrap/src/animus_bootstrap/daemon/platforms/linux.py` |
| `~/.config/systemd/user/animus-forge.service` | Exists; disabled | `packages/bootstrap/src/animus_bootstrap/daemon/platforms/linux.py` |
| `~/.config/autostart/animus-tray.desktop` | `X-GNOME-Autostart-enabled=true` (root cause of unwanted startup) | Verified by user observation |
| `~/.local/bin/animus-tray` | GTK + AppIndicator tray; uses `pgrep -f animus_bootstrap.daemon` for liveness (rejected for ADR-007) | User observation + source review |
| `~/.local/bin/animus-cleanup` | CLI helper; calls `os.kill(SIGTERM)` on `result.marked_orphan` without provenance rules | `packages/core/animus/infrastructure/process_lifecycle.py:743` |
| `SystemProcessRegistry` | SQLite-backed; states `RUNNING / SUSPECT / ORPHAN / STOPPED` (internal registry states, not the 4-classification ADR-007 demands) | `packages/core/animus/infrastructure/process_lifecycle.py:271` |
| LockedPidFile, ProcessGuard | Exists; `ProcessGuard` wired into daemon and MCP server | `packages/core/animus/infrastructure/process_lifecycle.py:62, 583` |
| dashboard `/health` | JSON health status | `packages/bootstrap/src/animus_bootstrap/dashboard/app.py` |
| Linger on current user | `Linger=yes` (must be **observed**, not changed) | `loginctl show-user arete` |
| `graphical-session.target` | Present at `/usr/lib/systemd/user/graphical-session.target` | `ls /usr/lib/systemd/user/` |
| `network-online.target` (user) | **Absent** at `/usr/lib/systemd/user/network-online.target`; only in `/usr/lib/systemd/system/` | `ls /usr/lib/systemd/user/network-online.target` |
| Untracked files in this branch | `adrs/ADR-007-runtime-lifecycle.md`, `adrs/ADR-008-review-pattern.md` | `git status --short` |

These facts are the input to the implementation. They are not editorial — each row is verified against the filesystem or the source.

---

## 3. Target architecture

One architecture, three deployment profiles. The architecture is fixed; the profiles differ only in:

1. The target dependency symlink under a `*.target.wants/` directory.
2. Per-service drop-in files under `*.service.d/`.
3. The `~/.config/animus/profile.json` desired-mode value.

No canonical unit file is modified after first install. The drop-ins, symlinks, and profile JSON are the only mutable surfaces.

### Canonical target unit (`~/.config/systemd/user/animus-runtime.target`)

```ini
[Unit]
Description=Animus Runtime — single lifecycle boundary

Requires=animus.service
After=animus.service

Wants=animus-forge.service animus-mcp.service animus-scheduler.service animus-tray.service
After=animus-forge.service animus-mcp.service animus-scheduler.service animus-tray.service

[Install]
# WantedBy= is intentionally unset. Profile targeting is performed by
# `systemctl --user add-wants <host-target>.target animus-runtime.target`,
# which creates a `*.target.wants/animus-runtime.target` symlink. The unit
# file itself is never edited by the install or upgrade flow.
```

### Canonical runtime service block (every participating service)

```ini
[Unit]
Description=Animus <component>
PartOf=animus-runtime.target
After=network.target

[Service]
Type=simple
ExecStart=<resolved at install>
KillMode=control-group
TimeoutStopSec=30
Restart=no
Environment=ANIMUS_RUNTIME_PROFILE=<profile>

[Install]
# WantedBy= is intentionally unset. The runtime target pulls the service in.
```

Profile-specific hardening (memory, CPU, tasks, restart) lives in profile drop-ins (see §6).

### Process classification (replaces the existing 4-state model)

The ADR-007 classification is **4-class** and **stricter** than the existing `ProcessState` enum. The build spec introduces a new `ProcessClassification` enum that participates in the external user-facing state:

| State | Provenance rule | Action |
|---|---|---|
| `Managed` | Process is registered AND attached to an active lifecycle (systemd unit is active OR cgroup is alive). | Stop through systemd only. Never signal PID directly. |
| `Recoverable` | Process is registered BUT parent metadata is lost (cgroup gone, parent dead). Authority: registry identity + executable path + start-time fingerprint. | Reattach to the discovered unit/cgroup; otherwise stop through the discovered unit. |
| `Orphaned` | Process is Animus-owned and surviving after Animus stopped. Authority: registry identity PLUS at least two independent process proofs (executable path, command-line launch token, UID, start-time fingerprint, environment instance ID, or parent history). | Stop through the discovered unit if found; otherwise `SIGTERM` with a 5-second grace period, then `SIGKILL`. |
| `Unknown` | Name matches (`animus` substring) but ownership unproven. | Report only. Never terminate automatically. |

The `RegisteredProcess.state` field (internal registry) is unchanged; the new `ProcessClassification` is the *external* view and replaces the user-facing "is this process a stray?" answer.

Cgroup evidence is decisive when present but **not mandatory** for proving an orphan. The cgroup may itself be the thing that was lost.

`pgrep` may be used as an emergency diagnostic only. Authoritative state detection uses:
1. `systemctl --user show -p ActiveState,SubState,Result,ExecMainStartTimestamp ...` (machine-readable)
2. The daemon's `GET /healthz` endpoint (JSON)
3. `/proc/<pid>/cgroup`, `/proc/<pid>/exe`, `/proc/<pid>/cmdline`, `/proc/<pid>/stat` for reconciliation

---

## 4. Component ownership matrix

| Area | Owner | Source location |
|---|---|---|
| Installation, profile switching, launcher, control app | Bootstrap | `packages/bootstrap/src/animus_bootstrap/daemon/`, `packages/bootstrap/src/animus_bootstrap/control/` |
| Registry, provenance, classification | Core | `packages/core/animus/infrastructure/process_lifecycle.py` |
| Forge worker lifecycle | Forge | `packages/forge/src/animus_forge/` |
| Systemd unit templates and drop-ins | Bootstrap / operations packaging | `packages/bootstrap/src/animus_bootstrap/daemon/units/` |
| Health response contract | Contracts package + Bootstrap | `packages/contracts/` (new), `packages/bootstrap/src/animus_bootstrap/healthz/` |
| Dashboard service/process endpoints | Bootstrap dashboard | `packages/bootstrap/src/animus_bootstrap/dashboard/routers/system.py` |
| Tray subscriber | Bootstrap tray | `packages/bootstrap/src/animus_bootstrap/tray/` (replaces `~/.local/bin/animus-tray`) |
| Test harness | Bootstrap tests | `packages/bootstrap/tests/test_runtime/`, `packages/bootstrap/tests/test_runtime/conftest.py` |

The detailed file-to-symbol matrix is intentionally **not** in the ADR. It is in this build spec because it is implementation guidance, not architectural decision. Changes to ownership here do not require an ADR amendment.

---

## 5. Deployment-profile matrix

| Property | `development-local` | `desktop-login` | `continuous-node` |
|---|---|---|---|
| Target binding | none (manual `start`) | `systemctl --user add-wants graphical-session.target animus-runtime.target` | `systemctl --user add-wants default.target animus-runtime.target` |
| Unit file `[Install] WantedBy=` | unset | unset | unset |
| Linger | unchanged | unchanged | required + explicit user consent |
| Tray | `tray_while_running` or off | `tray_while_running` + `tray_while_offline` | not present |
| Daemon `MemoryMax` (drop-in) | 4G | 8G | 32G |
| Daemon `CPUQuota` (drop-in) | 200% | 400% | 1600% |
| Daemon `TasksMax` (drop-in) | 64 | 128 | 512 |
| `Restart` | no | on-failure | on-failure |
| `RestartSec` | — | 5s | 5s |
| `WatchdogSec` | 0 | 30 | 30 |
| `KillMode` | control-group | control-group | control-group |
| `TimeoutStopSec` | 30 | 30 | 30 |
| `Delegate=yes` | **no** | **no** | **no** |
| Network dependency | none (operate with `network_degraded`) | none | none |
| Implementation status on this host | **shipped** | **shipped** (seams only) | **future** (architecture support, not production) |

The drop-in names are `<service>.d/20-profile-development-local.conf`, `<service>.d/20-profile-desktop-login.conf`, `<service>.d/20-profile-continuous-node.conf`. The `20-` prefix places them after any `10-` system drop-in and before any `50-` user drop-in.

---

## 6. Unit-file design

### Required units (packages under `packages/bootstrap/src/animus_bootstrap/daemon/units/`)

```
units/
├── animus-runtime.target
├── animus.service
├── animus-forge.service
├── animus-mcp.service
├── animus-scheduler.service
├── animus-tray.service
├── animus-tray-offline.service
└── drop-ins/
    ├── 20-profile-development-local.conf
    ├── 20-profile-desktop-login.conf
    └── 20-profile-continuous-node.conf
```

Each service file uses the canonical runtime block from §3. The `<component>` description is the only content that varies.

The `animus-tray-offline.service` is a separate tray service for users who want the tray visible while Animus is offline. It is gated by `tray_while_offline` in `profile.json`. It must never start the daemon.

### Drop-in inheritance

`systemd` loads drop-ins in lexical order. The `20-` prefix is the canonical position for profile hardening; installers must not place files earlier than `20-` (those slots are reserved for system or vendor drop-ins) and not later than `50-` (user-specific overrides).

---

## 7. Profile-switch transaction

The transaction is the boundary between profiles. It must be atomic from the user's perspective: the runtime is either fully on the new profile or fully on the old profile. There is no intermediate state where the user can see a partial switch.

```python
def switch_profile(target_mode: str, profile_config: ProfileConfig) -> ProfileSwitchResult:
    """Atomic profile switch with rollback on failure.

    Steps:
    1. Validate `target_mode` ∈ {development-local, desktop-login, continuous-node}.
    2. If `target_mode == continuous-node`, verify explicit user consent
       (separate flag in the call site, not implied by the function).
    3. Read current profile from `~/.config/animus/profile.json`.
    4. Stop the runtime target if active (`systemctl --user stop animus-runtime.target`).
    5. Read current symlinks under `*.target.wants/` for `animus-runtime.target`.
    6. Compute the new desired set:
       - development-local: empty set
       - desktop-login: {graphical-session.target}
       - continuous-node: {default.target}
    7. Generate drop-ins for each service atomically (write to temp files, fsync, rename).
    8. Run `systemctl --user daemon-reload`.
    9. Add new target symlinks: `systemctl --user add-wants <host>.target animus-runtime.target`.
    10. Remove obsolete target symlinks: `systemctl --user remove-wants <host>.target animus-runtime.target`.
    11. Verify effective dependencies:
        `systemctl --user show -p Wants,Requires,After animus-runtime.target`.
    12. Verify effective drop-in properties:
        `systemctl --user show -p MemoryMax,CPUQuota,CPUQuotaPerSecUSec,TasksMax,KillMode,Restart,TimeoutStopSec animus.service`.
    13. If verification fails, roll back to the prior profile (re-run steps 7-11 with prior values).
    14. Write `profile.json` only after successful verification.
    15. If `continuous-node` is the new mode, surface (do not change) `loginctl show-user` Linger state.
    16. Return success or rollback-report.
```

The rollback path is mandatory. A failed switch must leave the host on the previous profile, not in a partially-applied state.

If `continuous-node` is the new mode, the *user consent* is captured by the caller (a separate dialog in the control app or installer). The function does not prompt; it requires the explicit `user_consent: bool` argument.

---

## 8. Desired-state and observed-state schemas

### Desired (`~/.config/animus/profile.json`)

```json
{
  "schema_version": "1",
  "mode": "development-local",
  "tray_while_running": true,
  "tray_while_offline": false,
  "start_on_login": false
}
```

`mode` ∈ {`development-local`, `desktop-login`, `continuous-node`}.
`schema_version` is required for future migrations.

### Observed (computed at read time, never persisted)

```json
{
  "schema_version": "1",
  "linger_enabled": true,
  "runtime_target_state": "inactive",
  "runtime_target_load_state": "loaded",
  "required_daemon_active": false,
  "optional_services_active": ["animus-forge.service"],
  "tray_process_running": false,
  "health_endpoint_reachable": false,
  "registry_rows": 0,
  "last_sweep": "2026-08-04T12:00:00Z"
}
```

`runtime_target_state` and `runtime_target_load_state` come from `systemctl --user show -p ActiveState,LoadState ...`. `health_endpoint_reachable` is the outcome of the last `GET /healthz` attempt. `registry_rows` is the count from `SystemProcessRegistry.summary()`. The observed state is read-only; it is what the user sees.

`linger_enabled` is **observed** via `loginctl show-user` and is surfaced in the UI but never written to `profile.json`.

### Health contract (versioned)

```python
# Health contract version 1
# Producer: animus daemon
# Consumer: control app, dashboard, animus-control CLI
# Schema: Pydantic, exported from packages/contracts

class HealthSnapshot(BaseModel):
    schema_version: Literal["1"]
    timestamp: datetime
    state: Literal["HEALTHY", "DEGRADED", "FAILED", "STOPPING", "UNKNOWN"]
    active_citizens: int
    open_jobs: int
    last_heartbeat_age_seconds: float
    detail: dict[str, str] = {}  # free-form component reports
```

The contract is consumed by `GET /healthz`. The control app and dashboard parse via `HealthSnapshot`. The schema is `Literal["1"]` to allow future major version bumps without breaking parsers.

---

## 9. Process-provenance model

A process is classified as `Orphaned` only when it has **registry identity plus at least two independent process proofs**:

| Proof | Source | Reliability |
|---|---|---|
| Executable path | `/proc/<pid>/exe` | High — direct kernel read |
| Command-line launch token | `/proc/<pid>/cmdline` (parsed, with normalization) | High |
| UID | `/proc/<pid>/status` | High — required match |
| Start-time fingerprint | `/proc/<pid>/stat` field 22 (starttime in clock ticks) | Medium — vulnerable to PID reuse within ε seconds |
| Environment instance ID | `/proc/<pid>/environ` filtered for `ANIMUS_INSTANCE_ID` | High — only the daemon writes this |
| Parent history | `/proc/<pid>/stat` field 4 (ppid) + registry history | Medium — ppid can be reparented to init |

Two independent proofs are the floor. `pgrep`, `pkill`, and `kill -N <pid>` are **never** the proof. The `os.kill(pid, 0)` liveness check is permitted.

---

## 10. Control-app behavior

The control app is a Python module (`packages/bootstrap/src/animus_bootstrap/control/`) that wraps `systemctl --user` and `GET /healthz` for the user-facing surface. It is a sibling of `animus-tray`.

### Required commands

| Command | Effect |
|---|---|
| `animus-control start` | `systemctl --user start animus-runtime.target` |
| `animus-control stop` | `systemctl --user stop animus-runtime.target` |
| `animus-control status` | JSON of observed state + HealthState |
| `animus-control logs` | `journalctl --user -u animus-runtime.target -n <N>` |
| `animus-control dashboard` | Open `http://localhost:7700/system` in browser |
| `animus-control audit` | Run `SystemProcessRegistry.sweep()` and print result |
| `animus-control profile <mode>` | Call `switch_profile()` (requires `user_consent=True` for `continuous-node`) |

The control app must not start the daemon itself. It must not signal PIDs directly. It must not use `pgrep`. The only process-management verbs it issues are `systemctl --user start/stop/restart animus-runtime.target` and `systemctl --user daemon-reload`.

### HealthState display

The control app displays the seven `HealthState` values verbatim. It does not collapse `UNKNOWN` to `OFFLINE` or `FAILED`. The user sees the real state.

---

## 11. Tray behavior

The tray is a strict subscriber. It reads:
- `systemctl --user show -p ActiveState,SubState animus-runtime.target`
- `GET /healthz` from the daemon

It writes nothing. It never starts the daemon. It never stops the daemon. Killing the tray has no effect on the runtime.

When the systemd state and the health endpoint are both unavailable, the tray shows `UNKNOWN`. It does not guess.

The `animus-tray-offline.service` variant is for users who want the tray visible while Animus is offline. It is gated by `tray_while_offline` in `profile.json`. It must never start the daemon; it is a display-only service.

---

## 12. Dashboard API changes

New endpoints under `/system/`:

| Endpoint | Returns |
|---|---|
| `GET /system/services` | List of `animus-*.service` units with `ActiveState`, `SubState`, `MainPID`, `MemoryCurrent`, `CPUUsageNSec` from `systemctl --user show` |
| `GET /system/processes` | `SystemProcessRegistry.list_active()` plus `/proc/` enrichment (exe, cmdline, cgroup) |
| `GET /system/profile` | Desired `profile.json` + observed `~/.config/animus/profile.observed.json` (computed, not persisted) |
| `GET /system/health` | Derived `HealthState` from systemd + `/healthz` |
| `POST /system/profile` | Call `switch_profile()` with explicit consent for `continuous-node` |

The existing `/health` endpoint is preserved. The new `/system/health` is the dashboard-friendly view; the existing `/health` endpoint remains the daemon's health probe for external load balancers.

---

## 13. Installer and migration behavior

The first install after this ADR is adopted:

1. Writes the unit files to `~/.config/systemd/user/`.
2. Writes `animus.desktop` to `~/.local/share/applications/` (no autostart).
3. Writes `profile.json` with `mode: development-local`, `tray_while_running: false`, `tray_while_offline: false`, `start_on_login: false`.
4. Flips `~/.config/autostart/animus-tray.desktop` to `X-GNOME-Autostart-enabled=false`.
5. Does **not** run `systemctl --user enable animus-runtime.target` (the target is unit-present but not enabled).
6. Does **not** create any `*.target.wants/animus-runtime.target` symlink.
7. Does **not** change `loginctl` linger state.

Migration from the current host:

1. Detect existing `X-GNOME-Autostart-enabled=true` in `~/.config/autostart/animus-tray.desktop` and report.
2. Detect existing `~/.config/systemd/user/animus.service` and `animus-forge.service` and report whether they are enabled.
3. Detect existing `WantedBy=default.target` in the existing units and report (this is in the current `AnimusInstaller.generate_systemd_unit`).
4. Detect existing `Linger` state via `loginctl show-user` and report.
5. Detect unclassified processes (e.g. `animus_discord_bot.py`, `animus-autonomous-*.timer`) and report.
6. Do **not** kill any unproven process.
7. Default the new `profile.json` to `development-local`.
8. The user explicitly runs `animus-control profile desktop-login` or `animus-control profile continuous-node` to opt in.

This list is exhaustive. Adding a new migration step requires updating this spec.

---

## 14. Rollback strategy

The profile-switch transaction (§7) has a rollback path. The broader rollback strategy:

| Failure mode | Rollback |
|---|---|
| Drop-in generation fails | Abort; no symlink change. |
| `daemon-reload` fails | Restore previous drop-ins, abort. |
| `add-wants` fails | Restore previous drop-ins, restore previous symlinks, abort. |
| `remove-wants` fails | The new symlink is still in place; restore previous drop-ins, abort with manual fix-up report. |
| Verification fails | Re-run the prior profile state and restore prior profile.json. |
| User rejects the new profile mid-flight | Re-run the prior profile state. |

The install flow itself is reversible: uninstalling the unit files and removing the symlink restores the pre-install state. The autostart flip can be reverted by setting `X-GNOME-Autostart-enabled=true` again.

---

## 15. Security boundaries

- The control app does not execute user-supplied strings. Profile names and service names are validated against an allow-list.
- The `systemctl --user` boundary is respected: no `sudo`, no D-Bus system bus, no PID file owned by root.
- `loginctl enable-linger` is only invoked with explicit user consent in the call site, never inferred.
- The `Environment=ANIMUS_INSTANCE_ID=<uuid>` in service units is generated at install time and never reused.
- The dashboard `/system/*` endpoints require the same auth as the existing dashboard — there is no new auth surface.
- The health endpoint must not return secret material. The `HealthSnapshot.detail` field is sanitized against a JSON-schema whitelist.
- The ProcessGuard already enforces PID-reuse protection via `/proc/<pid>/exe` read. The new `ProcessClassification` does not weaken this.

The plaintext Forge systemd drop-in credential remediation is **out of scope** for this spec. It is tracked separately.

---

## 16. Test architecture

The test harness enforces isolation from the live runtime. No test may start, stop, modify, or inspect production Animus units as its test target.

### Isolation harness

Each integration test:

- Uses a temporary `XDG_CONFIG_HOME` (e.g., `tmp_path / "xdg_config"`).
- Uses a temporary `XDG_RUNTIME_DIR` (e.g., `tmp_path / "xdg_runtime"`).
- Uses unique unit names with a random test prefix (e.g., `animus-test-<uuid8>-target`).
- Uses a temporary registry database under `tmp_path`.
- Uses a temporary profile configuration under `tmp_path`.
- Uses a temporary port (e.g., picked from an OS-allocated range).
- Cleans up via pytest fixtures, in both success and failure paths.

The harness implementation is in `packages/bootstrap/tests/test_runtime/conftest.py`. The harness is the canonical pattern; tests that bypass it are rejected by the meta-test.

### Required test surface (20 tests)

| # | Test | Files |
|---|---|---|
| 1 | Target with `Requires=` + `Wants=` brings services up on start | `test_animus_runtime_target.py` |
| 2 | `PartOf=` without target `Wants=`/`Requires=` does NOT start the service | `test_partof_wants_separation.py` |
| 3 | Stopping the target tears down all descendants in the service cgroup | `test_target_stop_teardown.py` |
| 4 | Killing the tray does not affect the runtime | `test_tray_does_not_supervise.py` |
| 5 | Missing required daemon produces `FAILED` | `test_health_state.py::test_missing_required_daemon_is_failed` |
| 6 | Failed optional service produces `DEGRADED` | `test_health_state.py::test_failed_optional_is_degraded` |
| 7 | Health endpoint 503 produces the defined state | `test_health_state.py::test_health_probe_503` |
| 8 | Missing authoritative signals produces `UNKNOWN` | `test_health_state.py::test_missing_signals_is_unknown` |
| 9 | Profile switching creates the intended target-wants symlink | `test_profile_switching.py::test_add_wants_creates_symlink` |
| 10 | Profile switching removes obsolete symlinks | `test_profile_switching.py::test_remove_wants_drops_symlink` |
| 11 | Generated drop-ins produce the expected effective properties | `test_drop_ins.py` |
| 12 | Failed profile switching rolls back | `test_profile_switching.py::test_rollback_on_failure` |
| 13 | Development install enables no runtime target | `test_installer.py::test_dev_install_no_runtime_target` |
| 14 | Unknown process matches are never killed | `test_stray_classification.py::test_unknown_never_killed` |
| 15 | Recoverable and Orphaned classifications require the defined proofs | `test_stray_classification.py::test_provenance_required` |
| 16 | Backup timers remain independent | `test_backup_timers_independent.py` |
| 17 | Discord remains outside the target | `test_discord_not_in_target.py` |
| 18 | Desired and observed state remain separate | `test_desired_observed_separation.py` |
| 19 | Health producer and consumer share a versioned contract | `test_health_contract.py` |
| 20 | Test cleanup leaves no active test units, processes, files, ports, or registry rows | `test_harness_cleanup.py` |

The `test_no_live_runtime_touch.py` meta-test is replaced by the isolation harness itself; the harness is the mechanism, not a static check.

### Test isolation (enforced)

The harness rejects any test that resolves a unit name against the live systemd user manager. The detection rule:

- The test sets `XDG_CONFIG_HOME` and `XDG_RUNTIME_DIR` to a temp directory.
- The test uses unit names with a `animus-test-<uuid8>-` prefix.
- The test does not invoke `systemctl --user` without a `--user-unit-dir` flag (if implemented) or without first verifying the test unit directory via `systemd --user --unit-path=` resolution.

The pytest fixtures:

- `clean_xdg` (autouse for tests in `test_runtime/`): creates `tmp_path / "xdg_config"` and `tmp_path / "xdg_runtime"`, sets env vars, tears down on exit.
- `temp_unit_dir`: creates `tmp_path / "systemd_user"`, sets `XDG_CONFIG_HOME` so systemd resolves to it.
- `temp_registry`: creates `tmp_path / "registry.db"`, returns a `SystemProcessRegistry` instance.
- `temp_port`: returns an OS-allocated free port.

---

## 17. Release stages

| Stage | What ships | Tests required |
|---|---|---|
| 0 | Documentation only (ADRs, this spec) | none |
| 1 | Core: `ProcessClassification` enum + provenance rules + removal of `pgrep` from authoritative paths | All 20 tests in §16 marked with `pytest.mark.runtime_stage_1` |
| 2 | Bootstrap: `animus-runtime.target` + canonical service blocks + `KillMode=control-group` | Stage 1 + tests 1, 2, 3 |
| 3 | Bootstrap: `switch_profile()` + drop-in generation + rollback | Stage 2 + tests 9, 10, 11, 12, 13 |
| 4 | Bootstrap: `animus-control` CLI + control app | Stage 3 + tests 5, 6, 7, 8, 18, 19 |
| 5 | Bootstrap: dashboard `/system/*` endpoints | Stage 4 |
| 6 | Bootstrap: tray rewrite (subscriber only) | Stage 4 + test 4 |
| 7 | Bootstrap: installer migration from autostart | Stage 6 + test 13 |
| 8 | Bootstrap: `continuous-node` drop-in (architecture support only, not production) | Stage 7 |

The `continuous-node` (GX10) mode is **architecture support only**. Production hardening is a separate program.

---

## 18. Acceptance criteria

A milestone is considered complete when:

1. All 20 tests in §16 pass in isolation.
2. The full Bootstrap test suite passes (`pytest packages/bootstrap/tests/`).
3. The full Core test suite passes (`pytest packages/core/tests/`).
4. `ruff check packages/` and `ruff format --check packages/` are clean.
5. The `mypy-ratchet` baseline is not regressed.
6. No test in `packages/bootstrap/tests/test_runtime_lifecycle/` references the live unit names `animus.service`, `animus-forge.service`, or `animus-runtime.target` without an isolation layer.
7. The dashboard `/system/*` endpoints return JSON matching the schema.
8. `animus-control start` brings the runtime up; `animus-control stop` brings it down; `animus-control status` reflects the new state within 2 seconds.
9. `pgrep` does not appear in any code path that returns a lifecycle decision (verified by `grep -rn "pgrep" packages/` + `test_no_pgrep_in_lifecycle.py`).
10. The installer migration is documented and the documented behavior matches the implemented behavior.

---

## 19. Post-implementation audit

The audit verifies the implementation against the architectural decision. It is run after Stage 8 is complete. The audit produces a written report (`docs/audit/animus-runtime-lifecycle-2026-XX.md`).

Audit checks:

1. **Unit files.** Canonical target unit + per-service blocks match §3 and §6 verbatim.
2. **Profiles.** Three profile drop-ins exist with the right values; `development-local` is the default in `profile.json`.
3. **Process cleanup.** `KillMode=control-group` is set on every runtime service; `KillMode=process` is absent; `Delegate=yes` is absent.
4. **Health.** `HealthState` includes `UNKNOWN`; `/healthz` returns a versioned schema; the contract is enforced by `test_health_contract.py`.
5. **Classification.** `ProcessClassification` has 4 states; `Orphaned` requires two independent proofs; `pgrep` is not in any classification path.
6. **Tray.** Tray is a subscriber; killing the tray does not stop the runtime; the tray shows `UNKNOWN` when both signals are unavailable.
7. **Dashboard.** All `/system/*` endpoints respond with the documented schema.
8. **Installer.** Migration matches §13; the existing autostart is reported but not silently changed.
9. **Tests.** All 20 tests in §16 pass in isolation; no test touches the live runtime.
10. **Documentation.** `docs/systemd/animus-runtime.md` and `docs/operations/process-registry.md` exist and match the implementation.

The audit changes nothing in the implementation. It produces a status table: `Clean`, `Issues Found`. `Issues Found` produces a follow-up issue list.

---

## 20. Cross-references

- `adrs/ADR-007-runtime-lifecycle.md` — the architectural decision
- `adrs/ADR-008-review-pattern.md` — the seven-step review pattern that produced this spec
- `packages/core/animus/infrastructure/process_lifecycle.py` — existing `SystemProcessRegistry`, `LockedPidFile`, `ProcessGuard`
- `packages/bootstrap/src/animus_bootstrap/daemon/platforms/linux.py` — existing `LinuxService` that the new build extends
- `~/.local/bin/animus-tray` — current tray implementation that the build replaces
- `man systemd.unit`, `man systemd.kill`, `man systemd.resource-control` — primary evidence for the unit-file design
- `~/.claude/projects/-home-arete/memory/animus-review-pattern.md` — model-side memory of the seven-step pattern
