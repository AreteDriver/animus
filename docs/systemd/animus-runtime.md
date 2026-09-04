# Animus Runtime — systemd Operator Guide

This document is the operational reference for the Animus runtime under
systemd. It pairs with
[`docs/specifications/animus-runtime-lifecycle-build-spec.md`](../specifications/animus-runtime-lifecycle-build-spec.md)
and ADR-007.

## Scope

- **In:** the `animus-runtime.target` lifecycle, the unit-file design,
  the profile-switch transaction, the daemon-reload ordering.
- **Out:** message gateway details (see `docs/operators/configuration.md`),
  dashboard internals, intelligence layer, persona system.

## Unit summary

Five user units plus one target, all installed in
`${XDG_CONFIG_HOME}/systemd/user/` (default `~/.config/systemd/user/`):

| Unit                  | Type    | Required by target | Restart on stop of runtime target |
|-----------------------|---------|--------------------|-----------------------------------|
| `animus.service`      | daemon  | `Requires=`        | Yes (`PartOf=`)                   |
| `animus-forge.service`| worker  | `Wants=`           | Yes (`PartOf=`)                   |
| `animus-mcp.service`  | worker  | `Wants=`           | Yes (`PartOf=`)                   |
| `animus-scheduler.service` | worker | `Wants=`        | Yes (`PartOf=`)                   |
| `animus-tray.service` | optional| `Wants=`           | Yes (`PartOf=`)                   |
| `animus-runtime.target` | target | n/a              | n/a                               |

`PartOf=animus-runtime.target` on each service is **the only** way the
target's `systemctl --user stop animus-runtime.target` brings the
daemon and workers down. `Requires=` and `Wants=` on the target only
determine **start direction**, not stop. The tray is a `Wants=` — its
absence does not break the target.

## Canonical target unit

```ini
[Unit]
Description=Animus Runtime — single lifecycle boundary
Requires=animus.service
After=animus.service
Wants=animus-forge.service animus-mcp.service animus-scheduler.service animus-tray.service
After=animus-forge.service animus-mcp.service animus-scheduler.service animus-tray.service

[Install]
# WantedBy= is intentionally unset. The runtime target is bound to
# a host target (graphical-session.target or default.target) by
# `systemctl --user add-wants`. The target file itself is not
# auto-enabled.
```

The `Install` section is empty by design. The target file's
`WantedBy=` is the deployment-profile responsibility, set via
`add-wants`/`remove-wants`, not via `[Install]`.

## Canonical service unit

Every Animus service unit must carry:

```ini
[Unit]
PartOf=animus-runtime.target
After=animus-runtime.target

[Service]
Type=simple
ExecStart=/path/to/animus-entrypoint
KillMode=control-group
Delegate=no
TimeoutStopSec=30
```

- `KillMode=control-group` — the **only** safe choice. PIDs get
  recycled; cgroups do not. Without `control-group`, stopping the
  runtime target would orphan descendants and leave them running
  outside Animus's lifecycle.
- `Delegate=no` — Animus runs unprivileged under the user manager.
  `Delegate=yes` would grant cgroup ownership and disable automatic
  descendant reaping. The dashboard, tray, and bridge may own their
  cgroups; the runtime services must not.
- `TimeoutStopSec=30` — bounded shutdown. If a service does not
  honor SIGTERM in 30 s, systemd escalates to SIGKILL against the
  cgroup, not just the main PID.
- `PartOf=animus-runtime.target` — one-way stop/restart propagation.
  Stopping the target stops the service. Stopping the service does
  *not* stop the target.

## Profile switching

The three deployment profiles and their systemd targets:

| Profile            | `profile.json` value | Bound host target         | Auto-start? |
|--------------------|----------------------|---------------------------|-------------|
| `development-local`| `development-local`  | (no binding)              | Manual      |
| `desktop-login`    | `desktop-login`      | `graphical-session.target`| Yes        |
| `continuous-node`  | `continuous-node`    | `default.target`          | Yes        |

`continuous-node` requires explicit `user_consent=True` from the
control app. It is never inferred.

The :class:`ProfileSwitcher` performs the switch as a 16-step atomic
transaction. Operators running the switch by hand should call it via
the control app (`animus-ctl profile switch <mode>`), which holds the
right locks and persists `profile.json` only after verification.

### Manual switch (advanced)

Do this from a *single shell* so the steps are observed together:

```bash
# 1. Stop the runtime target if it is active.
systemctl --user stop animus-runtime.target

# 2. Back up the current profile.json.
cp ~/.config/animus/profile.json ~/.config/animus/profile.json.bak

# 3. Write the desired profile to disk (atomic; see the build spec §6).
# Use the control app if you can — it does the right ordering.

# 4. Write the per-profile drop-in for each Animus service under
# ~/.config/systemd/user/<unit>.d/20-profile-<mode>.conf.
# The drop-in must contain KillMode=control-group + Delegate=no at
# minimum; the per-profile limits (MemoryMax, CPUQuota, TasksMax,
# Restart, RestartSec, WatchdogSec) live in the build spec §6.

# 5. Reload systemd to read the new drop-ins.
systemctl --user daemon-reload

# 6. Move the wants symlink.
systemctl --user add-wants graphical-session.target animus-runtime.target    # desktop-login
systemctl --user remove-wants previous-host-target animus-runtime.target       # clean up

# 7. Verify the host target now Wants the runtime target.
systemctl --user show graphical-session.target --property=Wants | \
    grep -q animus-runtime.target

# 8. Verify the daemon's effective drop-in.
systemctl --user show animus.service --property=MemoryMax --property=KillMode
# MemoryMax=8G
# KillMode=control-group

# 9. Start the runtime target.
systemctl --user start animus-runtime.target
```

If step 7 or step 8 fails, **do not start the target.** Roll back by
re-writing the previous `profile.json` and drop-in, daemon-reload, and
re-running steps 6-8.

## Health

The runtime exposes a seven-state health contract at
`/health` (dashboard) and `/api/v1/health` (HTTP):

| State      | Meaning                                              |
|------------|------------------------------------------------------|
| `OFFLINE`  | Target inactive (stopped)                             |
| `STARTING` | Snapshot says STARTING; wait                         |
| `HEALTHY`  | All required services active, no probes failing      |
| `DEGRADED` | Optional service failing or HTTP probe returns 5xx  |
| `FAILED`   | Required daemon inactive                              |
| `STOPPING` | Snapshot says STOPPING; wait                         |
| `UNKNOWN`  | Both signals missing; cannot determine               |

The contract is versioned (`schema_version: "1"`). New states ship
under a new schema version; consumers should compare
`schema_version` against their known set and report `UNKNOWN` if
unrecognized.

## Lingering

Animus inherits `Linger=yes` from the user's login session — that is
the observation, not the requirement. For headless / dedicated
hardware, enabling lingering is a one-time manual step:

```bash
sudo loginctl enable-linger "$USER"
```

Animus does **not** enable lingering silently. The `continuous-node`
profile assumes lingering is already on; if it is off, the runtime
target stops when the user logs out and the state is reported as
`OFFLINE` on next boot.

## What this guide does NOT cover

- The tray icon (see the dashboard docs).
- The process registry and provenance rules — see
  [`docs/operations/process-registry.md`](../operations/process-registry.md).
- Migration from a pre-target install — see the *Migration* section
  in the build spec §13.
