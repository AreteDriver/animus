# Migrating a Current Animus Install to the Runtime Target

**Status**: Phase 6 — implementation in progress
**Last updated**: 2026-08-04

This is the operational migration from a "manually launched daemon"
install to a target-driven install under `systemd --user`. It pairs
with [`docs/specifications/animus-runtime-lifecycle-build-spec.md`](./animus-runtime-lifecycle-build-spec.md) §13 and ADR-007.

## When to run this

- The user has an existing Animus install running under tmux, a
  foreground process, or a hand-rolled systemd unit.
- The control app, dashboard, or installer offers
  `animus-ctl migrate runtime-target` (or the equivalent wizard step).
- The user has confirmed they want auto-start at login (or has
  reviewed the profile matrix and picked `development-local`).

## Pre-flight

Before the migration begins, the control app must read and capture
the **observed** state of the current install:

```bash
# Capture the live daemon's PID, the unit it would live under, and
# the command line that launched it. The migration writes this to
# `${XDG_DATA_HOME:-$HOME/.local/share}/animus/migration-baseline.json`.
animus-ctl migrate capture-baseline
```

Required fields:

- `pid` — the daemon's PID at capture time.
- `unit_path` — the path the migration would write a unit file to.
- `command_line` — first 4 KiB of `/proc/<pid>/cmdline`.
- `working_directory` — `pwdx <pid>` equivalent (`/proc/<pid>/cwd`).
- `environment_path_excerpt` — first 2 KiB of `PATH` from
  `/proc/<pid>/environ`.
- `open_fds[]` — `ls -la /proc/<pid>/fd` summary (path + target
  symlink).
- `listening_sockets[]` — `ss -ltnp` filtered by PID.
- `live_logs_tail` — last 100 lines of the daemon's log.

The migration refuses to proceed if `pid` no longer references a
process; capture is recorded as `stale` and the user is asked to
relaunch the daemon.

## Step 1: Stop the current daemon without losing context

The current daemon holds:
- a sqlite-on-FTS5 memory database under
  `${XDG_DATA_HOME}/animus/intelligence.db`,
- an open ChromaDB or Animus-Core connection,
- an HTTP listener on the user's chosen port.

Use the daemon's own SIGTERM handling (it persists its state on
SIGTERM). Do **not** SIGKILL — that skips the flush.

```bash
# Capture the PID from the baseline.
PID=$(jq -r .pid "$XDG_DATA_HOME/animus/migration-baseline.json")

# Send SIGTERM and wait up to 30 s.
kill -TERM "$PID"
for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
    echo "daemon did not exit within 30s; refusing to migrate"
    exit 1
fi
```

The 30-second budget matches `TimeoutStopSec=30` on the canonical
unit. The migration script treats "still alive after 30 s" as a
hard failure — the user must investigate.

## Step 2: Install the runtime target and service units

The installer writes:

- `${XDG_CONFIG_HOME}/systemd/user/animus.service`
- `${XDG_CONFIG_HOME}/systemd/user/animus-runtime.target`
- `${XDG_CONFIG_HOME}/systemd/user/animus-forge.service`
- `${XDG_CONFIG_HOME}/systemd/user/animus-mcp.service`
- `${XDG_CONFIG_HOME}/systemd/user/animus-scheduler.service`
- `${XDG_CONFIG_HOME}/systemd/user/animus-tray.service`

Each service unit carries `PartOf=animus-runtime.target` +
`KillMode=control-group` + `Delegate=no` +
`TimeoutStopSec=30`. The target carries `Requires=animus.service`
and `Wants=...` for the four workers, with an empty `Install`
section.

The installer is idempotent: re-running writes the same content. A
deviation (e.g. a hand-edited unit file) is reported but not
overwritten.

## Step 3: Render the per-profile drop-ins

```bash
# Initialize profile.json with the default profile.
mkdir -p "$XDG_CONFIG_HOME/animus/data"
cat > "$XDG_CONFIG_HOME/animus/profile.json" <<'EOF'
{
  "schema_version": "1",
  "mode": "development-local",
  "tray_while_running": false,
  "tray_while_offline": false,
  "start_on_login": false
}
EOF

# Render the default-profile drop-in for each service.
mkdir -p "$XDG_CONFIG_HOME/systemd/user/animus.service.d"
cat > "$XDG_CONFIG_HOME/systemd/user/animus.service.d/20-profile-development-local.conf" <<'EOF'
[Service]
KillMode=control-group
MemoryMax=4G
CPUQuota=200%
TasksMax=64
Restart=no
WatchdogSec=0
Delegate=no
EOF
# (same for the four worker services)
```

## Step 4: Daemon-reload and verify (no start yet)

```bash
systemctl --user daemon-reload

# The units should be loaded but not started.
systemctl --user show animus.service --property=ActiveState
# ActiveState=inactive

systemctl --user show animus.service --property=MemoryMax
# MemoryMax=4G

systemctl --user show animus.service --property=KillMode
# KillMode=control-group
```

If any of these reads wrong, stop and re-render the drop-in. Do not
start the target with a bad drop-in.

## Step 5: Start the runtime target

```bash
systemctl --user start animus-runtime.target

# The target brings up animus.service (Requires=) and the workers
# (Wants=, best-effort). Verify:
systemctl --user is-active animus-runtime.target
# active

systemctl --user is-active animus.service
# active

curl --silent --max-time 5 http://127.0.0.1:7700/health | jq .
# { "state": "HEALTHY", "schema_version": "1", ... }
```

The dashboard should be reachable and the registry should be
populated. The bridge between the old `logs/` file and the new
`journalctl --user -u animus.service` is one-way: the journalctl
side is the ground truth going forward; old log files are archived.

## Step 6: Switch profile (optional)

If the user wants `desktop-login`, they opt in via the control app:

```bash
animus-ctl profile switch desktop-login
```

That triggers the 16-step atomic switch transaction. The
transaction:
1. Stops the runtime target (already stopped on this path; no-op).
2. Writes the desktop-login drop-in for each service.
3. Daemon-reloads.
4. Adds `graphical-session.target.wants/animus-runtime.target`.
5. Removes any prior binding.
6. Verifies the host target now Wants= the runtime target.
7. Verifies the daemon's effective `MemoryMax` and `KillMode`.
8. Persists `profile.json` only after all checks pass.

Any failure rolls back. The migration is complete when:

- The unit is `active`.
- `profile.json` matches the desired mode.
- The dashboard `/health` is `HEALTHY` or `DEGRADED` (DEGRADED only
  if an optional worker is missing).

## Step 7: Mark migration complete

```bash
animus-ctl migrate mark-complete
```

The control app writes a one-line entry to its migration log:

```json
{"ts": "<utc>", "from": "manual-launch", "to": "runtime-target", "profile": "development-local"}
```

Once this is written, the migration wizard step is no longer
offered. Operators running `animus-ctl migrate capture-baseline`
again will see the migration log row and refuse to overwrite it.

## What is intentionally NOT migrated

- The user's hand-rolled systemd unit (if any) is left in place but
  has its `[Install]` section disabled. It does not auto-start.
- Old log files in `$XDG_DATA_HOME/animus/logs/` are *moved* to
  `archive/2026-08-04-runtime-target-migration/`, not deleted.
- PID files under `/run/user/<UID>/animus/` are deleted; the new
  unit's `RuntimeDirectory=` owns the namespace.
- The previous tmux session (if any) is killed after the daemon
  exits cleanly under SIGTERM.

## Rollback

If the user wants to revert within 24 hours:

```bash
animus-ctl migrate rollback
```

This:
1. Stops `animus-runtime.target` cleanly.
2. Disables all six units.
3. Restores the archived unit files and `profile.json`.
4. Re-launches the manual daemon from the archived command line.

The migration log is *added to*, not rewritten, so the rollback is
auditable.
