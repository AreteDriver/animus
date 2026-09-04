# ADR-007: Animus Runtime Lifecycle — One Architecture, Three Deployment Profiles

**Status**: Accepted
**Implementation**: Planned
**Validation**: Not started
**Date**: 2026-08-04
**Author**: arete
**Class**: ARCH (Architecture), OPS (Operations)

## Revision history

| Date | Revision | Notes |
|---|---|---|
| 2026-08-04 | r1 | Initial proposal. Defects identified in review: `Wants=`-only health signal is insufficient; `KillMode=process` is incorrect; profile switching mechanism was contradictory; user-vs-system `network-online.target` is invalid; provenance rules too rigid for true orphans; `pgrep` was both rejected and permitted; `linger_enabled` in config is observable state, not desired state. |
| 2026-08-04 | r2 | All seven review items corrected. Architectural decision is complete; status remains **Proposed** pending review. Implementation begins separately after the decision is **Accepted**. The regression tests listed in the **Consequences** section are acceptance criteria for the implementation, not prerequisites for accepting the architectural decision. |
| 2026-08-04 | r3 | Status / Implementation fields added per status-semantics correction. Duplicate Network handling cell in Profile 3 collapsed. Test-isolation guard added: tests do not touch the live Animus runtime. |
| 2026-08-04 | r4 | Principal-engineer review corrections integrated. C1 — `continuous-node` is selected manually only, never inferred from environment. C2 (refined) — `Delegate=yes` is not used by default; child processes remain in the service cgroup and are killed via `control-group`; any future delegated worker subtree is a separate design decision and must remain registry-discoverable. S2 — three concrete failure walkthroughs added (missing binary, mid-start crash, healthy process with 503 healthz). S5 — `UNKNOWN` added as a seventh `HealthState` for the case where both authoritative signals are unavailable. S6 (softened) — the health probe contract between control app and daemon is a versioned response contract with an integration test; the implementation form is left to the build spec. M2 — "do nothing" added to Alternatives Considered. M5 — credential reference removed; tracked as a separate security finding. C3, C4, and detailed ownership mapping moved to the build specification. C1 framing of profile-selection triggers added. |
| 2026-08-04 | r5 | **Accepted.** Canonical target unit block and per-service `PartOf=` block added to the Decision section so the unit-file contract is unambiguous. `KillMode=control-group` + `TimeoutStopSec=30` added as the canonical service block with `KillMode=process` explicitly forbidden and `KillMode=mixed` permitted only on a per-service basis. `Delegate=yes` no-delegate paragraph expanded. Status flipped to **Accepted**; Implementation to **Planned**; Validation to **Not started**. |

This ADR was reviewed against the seven-step adversarial review pattern in `adrs/ADR-008-review-pattern.md`. The review record is the **Revision history** table above; the specific errors and their corrections are also referenced in the **Consequences** section under each affected area.

## Context

The Animus project runs as multiple supervised processes on a Linux host:
- `animus_bootstrap.daemon` (dashboard + intelligence runtime, port 7700)
- `animus_forge.api` (orchestration API, port 8000)
- `animus.mcp_server` (model context protocol bridge)
- `animus_discord_bot.py` (channel adapter)
- `animus-tray` (GTK + AyatanaAppIndicator status/control icon)
- `animus-backup-*.service` and `animus-sync.timer` (cron-style housekeeping)

These processes are related but currently have **no unified lifecycle**. The user has reported:

1. **Animus starts when the user has not started it.** Root cause: `~/.config/autostart/animus-tray.desktop` carries `X-GNOME-Autostart-enabled=true` by default, so the tray launches at every graphical login. The systemd units `animus.service` and `animus-forge.service` are correctly disabled (no `default.target.wants` symlinks) and `WantedBy=default.target` does not auto-enable.
2. **Stray processes are visible but not bound to anything.** `pgrep -af animus` can show `animus.mcp_server` (twice historically — no singleton enforcement), orphaned `animus-tray` (no daemon parent, polls and spams notifications), and `animus_discord_bot.py` (currently a child of Plex, not Animus, on this host). Phase 1 of Process Herd Hardening (2026-07-25) added `LockedPidFile`, `SystemProcessRegistry`, and `ProcessGuard` to the core, and wired them into the daemon and MCP server, but the registry is not yet surfaced through the dashboard and the discord bot is not yet classified.
3. **No trustworthy "is Animus running?" signal exists.** The tray currently answers that with `pgrep -f animus_bootstrap.daemon`, which is unsafe as a kill primitive and only marginally useful as a liveness check. A correct answer requires two signals: `systemctl --user show animus-runtime.target` for **lifecycle intent** (the target's load state and its `Wants=/Requires=/BindsTo=` graph) and the daemon's `GET /healthz` endpoint for **runtime health** (active citizens, open jobs, last-heartbeat age). `pgrep` is not part of authoritative state detection.

The previous exploration (in chat, not in this repo) produced six candidate architectures and recommended tray-as-supervisor. That recommendation was incorrect on three load-bearing points: `PartOf=` alone does not start services, a tray icon requires a tray process, and "Run on login" should mean "start the runtime on login," not "show a launcher window on login." The other engineering lenses in the conversation corrected those errors. This ADR integrates the corrected position and adds the **deployment-profile axis** that reframes the question from "which architecture wins" to "which mode of one architecture runs on this host."

The current host is GPU-constrained (`nvidia-smi` absent, no discrete AMD GPU in `lsusb`) but CPU/RAM-rich (125 GiB RAM, 24 cores). It is a dev workstation, not an appliance. A future dedicated AI desktop ("GX10") is anticipated and will run Animus continuously and unattended. The architecture must support both, and the same code paths must run in both, with a profile flag controlling install/upgrade behavior.

## Decision

Adopt a **single Animus runtime architecture** that supports **three deployment profiles** by changing only install-time and unit-binding behavior, not code or unit-file content:

```
animus-runtime.target          (new — single lifecycle boundary)
├── Requires=animus.service         (required — runtime has no function without the daemon)
├── Wants=animus-forge.service       (optional)
├── Wants=animus-mcp.service         (optional)
├── Wants=animus-scheduler.service   (optional, profile-dependent)
└── Wants=animus-tray.service        (optional, profile-dependent)

Independent (never in the target):
├── animus-backup-hourly.timer
├── animus-backup-chroma.timer
├── animus-backup-forget.timer
├── animus-backup-check.timer
├── animus-sync.timer
├── animus-discord.service         (until explicitly classified)
└── animus-autonomous-*.timer      (until explicitly classified)
```

The canonical target unit is:

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

Every participating runtime service must declare:

```ini
[Unit]
PartOf=animus-runtime.target
```

The two-sided relationship is mandatory and the canonical block above is the only contract:

- `[Unit] Requires=animus.service` + `After=animus.service` in the **target** (daemon is required for the runtime to be meaningful)
- `[Unit] Wants=animus-forge.service animus-mcp.service animus-scheduler.service animus-tray.service` + `After=` for each in the **target** (optional services)
- `[Unit] PartOf=animus-runtime.target` in each **service** (so the target's stop/restart propagates to the service)

`Requires=` is used for the daemon because the runtime has no function without it. `Wants=` is used for the optional services because the runtime is meaningfully useful (dashboard, registry, control app) even when Forge, MCP, or the scheduler are unavailable. The man page is explicit that `Requires=` does not guarantee the required unit remains active — a service may exit on its own without propagation — so `Requires=` is a *startup* guarantee, not a *runtime health* guarantee. Runtime health is a separate signal (see below).

**Process cleanup is enforced via `KillMode=control-group`.** Every runtime service sets:

```ini
[Service]
KillMode=control-group
TimeoutStopSec=30
```

`KillMode=control-group` guarantees that when systemd stops the service, all remaining processes in the service's cgroup are terminated after `TimeoutStopSec=30` elapses. `KillMode=process` is **explicitly forbidden** — the man page states it is "not recommended" because it "allows processes to escape the service manager's lifecycle and resource management, and to remain running even while their service is considered stopped." `KillMode=control-group` is the only mode that gives the lifecycle model the cleanup guarantee it requires. `KillMode=mixed` is permitted only where a concrete service demonstrates that the SIGTERM-to-main / SIGKILL-to-cgroup split is required; the default is `control-group`.

**Cgroup delegation is not granted by default.** Runtime services do not receive `Delegate=yes` in their drop-ins. `Delegate=yes` grants the service authority to manage its own subhierarchy of control groups, makes the cgroup writable by the unit's user, and disables the kernel's automatic guarantee that all descendants die when the unit is reaped. Granting it casually invites the exact orphan scenario this ADR is designed to prevent: a delegated worker subtree can outlive the service and absent or stale registry data will hide it. Child processes remain in the service cgroup and are terminated through `KillMode=control-group`. Any future need for a delegated worker hierarchy (e.g., a citizen pool that owns its own cgroup subtree) requires a separate reviewed design decision and must remain visible through the `SystemProcessRegistry` so the lifecycle model continues to know about it.

`Requires=` is used for the daemon because the runtime has no function without it. `Wants=` is used for the optional services because the runtime is meaningfully useful (dashboard, registry, control app) even when Forge, MCP, or the scheduler are unavailable. The man page is explicit that `Requires=` does not guarantee the required unit remains active — a service may exit on its own without propagation — so `Requires=` is a *startup* guarantee, not a *runtime health* guarantee. Runtime health is a separate signal (see below).

The Animus daemon continues to own **logical work** (citizens, jobs, sessions, executions, Forge workers). The OS service manager owns **physical processes**. The `SystemProcessRegistry` reconciles the two. Processes are classified into four states, each with its own authority rule:

- **Managed** — registered and attached to an active lifecycle. Stop through systemd only. Never signal PID directly.
- **Recoverable** — registered, parent metadata lost. Authority: registry identity + executable + start-time fingerprint. Reattach or stop through the discovered unit/cgroup.
- **Orphaned** — Animus-owned process surviving after Animus stopped. Authority: registry identity plus at least two independent process proofs (executable path, command-line launch token, UID, start-time fingerprint, environment instance ID, or parent history). Cgroup membership is decisive when present, not mandatory for proving an orphan — the cgroup may itself be the thing that was lost.
- **Unknown** — name matches but ownership unproven. Report only. Never terminate automatically.

**`pgrep` is not part of authoritative state detection.** Runtime truth comes from (a) systemd D-Bus or `systemctl show` for unit state, (b) the daemon's health endpoint for application readiness, and (c) `/proc` enumeration for reconciliation and unknown-process discovery. `systemctl status` is human-oriented and is not a stable programmatic interface; `systemctl show` is machine-readable and is. `pgrep` may be used as an emergency diagnostic, not as a runtime signal.

**Health is a separate signal from lifecycle intent.** `is-active animus-runtime.target` represents *lifecycle intent* — did the user ask for Animus, and did the start transaction succeed? It does not represent *system health* — a required service may have exited on its own, or an optional service may have failed. The control app derives a six-state view:

| HealthState | Meaning |
|---|---|
| `OFFLINE` | Target inactive. No Animus processes. |
| `STARTING` | Activation transaction running. |
| `HEALTHY` | Target active, required daemon active, daemon health probe passes, all `Wants=` services either active or explicitly idle. |
| `DEGRADED` | Required daemon active; one or more optional services failed. Runtime is usable but not at full capacity. |
| `FAILED` | Required daemon failed to activate. |
| `STOPPING` | Deactivation transaction running. |
| `UNKNOWN` | Both authoritative signals (systemd `show` and the daemon's `/healthz`) are unavailable. Honest uncertainty; the control app displays the state without a guess. |

The health probe is a thin endpoint on the daemon (`GET /healthz` returning 200 with a JSON body listing active citizens, open jobs, and last-heartbeat age). The control app polls it after `is-active` returns active. The dashboard reconciles the systemd state, the health probe, and the `SystemProcessRegistry` into a single view per service.

**Failure mode walkthroughs.** Three concrete cases trace the model so future maintainers do not collapse lifecycle state into health:

| Failure case | Target state | HealthState | User-visible display |
|---|---|---|---|
| Daemon executable missing or `ExecStart=` fails | Target activation fails | `FAILED` | "Animus failed to start. Check `journalctl --user -u animus.service`." |
| Daemon starts, then crashes 5s later | Target active at activation; required service exits on its own and `Requires=` does not propagate to a re-stop; `is-active` may still report `active` for a brief window | `FAILED` (after the service exits and the health probe fails) or `UNKNOWN` (during the brief window) | "Animus stopped unexpectedly. View logs." |
| Daemon process active, but `/healthz` returns 503 | Target active; required service active; health probe fails | `DEGRADED` (if optional services are healthy) or `FAILED` (if the health probe is the required daemon itself reporting unhealthy) | "Animus is running with errors. Some capabilities may be unavailable." |

These walkthroughs also imply the test surface: the `HealthState` derivation function must be testable against each of these three inputs without involving the live runtime.

**Profile selection is manual only.** Each profile has exactly one selection trigger, and none of them is automatic:

- `development-local` — default on installation. No selection action required.
- `desktop-login` — selected manually through the control app's profile switcher, or through the installer. The user takes the action; nothing in Animus decides for them.
- `continuous-node` — selected manually through the control app or the installer. **Never inferred automatically** from hostname, hardware, GPU model, machine identity, or any other environmental signal. The GX10 will not silently activate `continuous-node` because Animus thinks it recognizes the machine. The user opts in.

**Cgroup delegation is not granted by default.** Runtime services do not receive `Delegate=yes` in their drop-ins. `Delegate=yes` grants the service authority to manage part of the cgroup hierarchy and can weaken systemd's direct control over descendants; granting it casually invites the exact orphan scenario this ADR is designed to prevent. Child processes remain in the service cgroup and are terminated through `KillMode=control-group`. Any future need for a delegated worker subtree (e.g., a citizen pool that owns its own cgroup subtree) requires a separate design decision and must remain discoverable through the `SystemProcessRegistry` so the lifecycle model continues to know about it.

The user-facing **launcher** is a `.desktop` file at `~/.local/share/applications/animus.desktop` with `Icon=animus` and `Exec=animus-control`. The launcher is always present in the application grid and dock; it costs no background process when not clicked. The launcher opens the `animus-control` window, which provides Start/Stop/Status/Logs/Dashboard/Audit and reads the active profile to surface the right controls. The launcher is not autostarted; the user clicks it to interact.

The **tray** (`animus-tray`) is a separate process that is *optional* and profile-dependent. It may run while Animus is running, or be configured to run while Animus is offline (advanced). It is never the supervisor; it observes systemd state via `systemctl --user show animus-runtime.target` and never owns the daemon. The existing `~/.config/autostart/animus-tray.desktop` entry is rewritten to `X-GNOME-Autostart-enabled=false` by default; the control app offers an opt-in to enable it.

The **discord bot** is excluded from the runtime target until its contract is explicitly classified by the user. Three legitimate classes exist: **core interface** (starts with Animus), **optional Animus adapter** (separate toggle and service, but visible in the dashboard), **independent application** (not controlled by Animus; only detected as external integration). Today, by systemd fact, the bot is the third class: the unit file does not exist in `~/.config/systemd/user/` and the running process is supervised by Plex, not Animus. Membership will not be inferred from the executable name.

The active **deployment profile** is stored at `~/.config/animus/profile.json`. The file holds **desired** state only:

```json
{
  "mode": "development-local",
  "tray_while_running": true,
  "tray_while_offline": false,
  "start_on_login": false
}
```

**`linger_enabled` is not stored as configuration.** It is observable system state owned by `loginctl`. Storing it as config invites drift between the file and reality. The control app exposes the actual state separately, computed at read time:

```json
{
  "observed": {
    "linger_enabled": true,
    "runtime_target_active": false,
    "tray_process_running": false
  }
}
```

The install/upgrade flow never changes the profile without explicit user consent. The default mode on first install is **`development-local`**.

### Profile 1 — `development-local` (default, this host today)

| Property | Value |
|---|---|
| `animus-runtime.target` `[Install] WantedBy=` | unset (unit file present, never enabled); the target becomes active only via manual `systemctl --user start animus-runtime.target` |
| Target binding mechanism | none — install writes the unit files, does not create any `*.target.wants/` symlinks |
| Tray autostart | off (`X-GNOME-Autostart-enabled=false`) |
| Process state on login | zero Animus processes |
| Start trigger | user clicks launcher → `animus-control` → Start button → `systemctl --user start animus-runtime.target` |
| Stop trigger | user clicks Stop → `systemctl --user stop animus-runtime.target` |
| Service unit hardening (drop-in) | strict: `MemoryMax=4G`, `CPUQuota=200%`, `TasksMax=64`, `Restart=no`, `WatchdogSec=0` |
| `KillMode` (drop-in) | `control-group` (cgroup teardown on stop, never leaves children behind) |
| Watchdog / self-heal | off (proves lifecycle works before proving recovery) |
| Network handling | start independently, retry remote with bounded backoff, report `NETWORK_DEGRADED` if Forge API or external integrations are unreachable |
| Primary control | launcher + control window + dashboard |

### Profile 2 — `desktop-login` (future, when user wants Animus at session start)

| Property | Value |
|---|---|
| Target binding mechanism | `systemctl --user add-wants graphical-session.target animus-runtime.target` (creates a symlink in `graphical-session.target.wants/`; the target unit file is not edited) |
| `animus-runtime.target` `[Install] WantedBy=` | still unset — binding is via the `.wants/` symlink, not the `[Install]` section |
| Tray autostart | optional, per `tray_while_running` / `tray_while_offline` |
| Process state | starts when graphical session begins, stops at session end |
| Service unit hardening (drop-in) | relaxed relative to dev profile, still bounded: `MemoryMax=8G`, `CPUQuota=400%`, `TasksMax=128`, `Restart=on-failure`, `RestartSec=5`, `WatchdogSec=30` |
| `KillMode` (drop-in) | `control-group` (same as dev; process-cleanup integrity does not vary by profile) |
| Watchdog | on, with `WatchdogSec=30` and `Restart=on-failure` |
| Network handling | same as dev |
| Primary control | launcher + tray + dashboard |

This mode requires the user to explicitly opt in. `graphical-session.target` exists at `/usr/lib/systemd/user/graphical-session.target` and is reachable on this host. Profile switching must call `systemctl --user daemon-reload` after changing the symlink.

### Profile 3 — `continuous-node` (GX10 appliance, future)

| Property | Value |
|---|---|
| Target binding mechanism | `systemctl --user add-wants default.target animus-runtime.target` (creates a symlink in `default.target.wants/`) |
| `animus-runtime.target` `[Install] WantedBy=` | still unset — binding is via the `.wants/` symlink |
| Linger | enabled (`loginctl enable-linger <user>`), but only with explicit user consent in the profile switch dialog |
| Tray | not present — headless |
| Service unit hardening (drop-in) | tuned for sustained load, still bounded: `MemoryMax=32G`, `CPUQuota=1600%`, `TasksMax=512`, `Restart=on-failure`, `RestartSec=5`, `TimeoutStopSec=30`, `WatchdogSec=30` |
| `KillMode` (drop-in) | `control-group` (same as other profiles; integrity over flexibility) |
| Watchdog + self-heal | on + remote telemetry |
| Network handling | start independently with bounded backoff; degraded mode for missing network; recovery when connectivity returns. **No `After=network-online.target`** — that target does not exist in the user manager's namespace on this host (`/usr/lib/systemd/user/network-online.target` is absent; only `/usr/lib/systemd/system/network-online.target` exists, and user units cannot depend on system units in the normal dependency model). |
| Required engineering | health checks, bounded-backoff restarts, child-process ownership, startup recovery after power loss, persistent job checkpoints, resource limits (GPU/CPU/memory/disk/temperature), maintenance and update windows, remote emergency stop, audit trail showing why every process exists, graceful degradation when models/storage/networking fail |
| Primary control | dashboard (remote), CLI, `systemctl --user status animus-runtime.target`, `animus-control` over SSH |

GX10 mode is **not** implemented on this host. The architecture supports it; the work to harden it for production is a follow-on program.

### Profile switching — explicit mechanism

All three profiles share identical unit files. The differences live in (a) the target dependency symlink under a `*.target.wants/` directory, and (b) per-service drop-in files under `*.service.d/`. The install/upgrade flow manipulates both, never the canonical units.

Switching from `development-local` to `desktop-login`:

```bash
# 1. Stop Animus if running
systemctl --user stop animus-runtime.target

# 2. Remove old drop-ins (dev profile hardening)
rm -f ~/.config/systemd/user/animus.service.d/20-profile.conf
rm -f ~/.config/systemd/user/animus-forge.service.d/20-profile.conf

# 3. Add target dependency symlink
systemctl --user add-wants graphical-session.target animus-runtime.target

# 4. Write new drop-ins (desktop-login profile hardening)
install -m 0644 animus-profile-desktop-login.conf \
    ~/.config/systemd/user/animus.service.d/20-profile.conf
# (repeat for forge, mcp, scheduler as appropriate)

# 5. Reload
systemctl --user daemon-reload

# 6. Verify the resulting dependency graph
systemctl --user show -p Wants,Requires,After animus-runtime.target
systemctl --user list-dependencies animus-runtime.target
```

The reverse (desktop-login → development-local) is the symmetric sequence with `add-wants` swapped for manual `rm` of the symlink. The continuous-node switch additionally runs `loginctl enable-linger <user>` (with explicit user consent shown in the control app's profile-switch dialog), and the dev switch additionally runs `loginctl disable-linger <user>` (only if linger was previously enabled *by this profile switch*; existing user-set linger is left alone).

**The canonical unit files are never modified by the install or upgrade flow.** Drops-ins, symlinks, and the profile JSON are the only mutable surfaces.

## Rationale

### Why a single architecture with three profiles, not three architectures

- **One codebase is cheaper than two.** Diverging dev-workstation and appliance code paths would create a parity tax. The same Animus daemon, registry, control app, and unit files run in all three modes; only the install/upgrade behavior changes.
- **Today's work compounds toward the future.** A lifecycle that is correct under `development-local` constraints is also correct under `desktop-login` and `continuous-node` constraints. The opposite is not true: a lifecycle designed for the GX10 would over-engineer the dev workstation.
- **Hardware limits the modes available, not the architecture.** A dev workstation cannot be made into an appliance by wishful design. The architecture supports the appliance path; the deployment decides when to walk it.

### Why `Requires=` for the daemon, `Wants=` for everything else

- `Requires=animus.service` is correct for the daemon because the runtime has no function without it. A failed start of the daemon should prevent the target from being reported as active, so the user is not misled into thinking Animus is up.
- `Wants=` is correct for `animus-forge.service`, `animus-mcp.service`, and `animus-scheduler.service` because the runtime is meaningfully useful without them (dashboard, registry, control app all work against the daemon). A failed Forge start should not block the user from seeing the dashboard or stopping the runtime.
- The systemd man page is explicit: `Requires=` does not guarantee the required unit remains active. A service may exit on its own without propagation, and `ConditionPathExists=` failures do not propagate either. This is why runtime **health** is modeled as a separate signal (the six-state `HealthState` above), not as the systemd unit state.
- This is a deliberate inversion of the original spec, which used `Wants=` for the daemon. The original spec was correct about the start direction (target pulls services in) but wrong about the dependency strength — `Wants=` is too weak to give the user a trustworthy "Animus is running" signal when the daemon has crashed.

### Why `PartOf=` on each service is mandatory

- `PartOf=` propagates stop and restart from the target to the service. Without it, `systemctl --user stop animus-runtime.target` leaves the daemon running.
- The systemd man page is explicit: `PartOf=` is a one-way back-reference; it does not start the service when the target is started. Starting is the responsibility of `Wants=` in the target. **Both sides are required.**

### Why the discord bot is not in the target

- The bot's contract has not been classified. Three legitimate classes exist (core / optional adapter / independent). The current systemd state (no unit file in the active user directory) is the third class by fact, not by intent.
- Inferring membership from the executable name (`animus_discord_bot.py`) is the same class of error as inferring ownership from `pgrep -f animus`. The ProcessRegistry's 4-state classification discipline must apply at design time, not only at runtime.
- The bot can be reclassified later by writing a new unit file and adding it to the target. This ADR is reversible on that point.

### Why the tray is a subscriber, not a supervisor

- A GTK + AppIndicator process is exactly the kind of long-lived UI component that gets reaped on desktop-shell restart, OOM, or DE reload. Putting process ownership on it is a single point of failure for the whole runtime.
- The tray already uses `LockedPidFile` for its own singleton (lines 100–130 of `~/.local/bin/animus-tray`); the design intent of Phase 1 Process Herd Hardening is preserved. What changes is the *authority* for state: `systemctl --user show animus-runtime.target` (machine-readable) for lifecycle intent, plus the daemon's `GET /healthz` endpoint for runtime health. The tray no longer participates in authoritative state detection. `pgrep` is removed entirely.

### Why the launcher is always present, the tray is not

- A `.desktop` file in `~/.local/share/applications/` is metadata, not a process. It costs nothing when the user does not click it. This is the only way to satisfy the user's "icon for the desktop/dock" requirement without spawning a background process.
- The tray is a process. It cannot exist without running. It is therefore opt-in, profile-dependent, and explicitly distinguished from the launcher in the UI.

## Consequences

### Required unit-file changes (proposed implementation)

1. **New** `~/.config/systemd/user/animus-runtime.target`:
   ```ini
   [Unit]
   Description=Animus Runtime — single lifecycle boundary
   Wants=animus.service animus-forge.service
   After=network.target

   [Install]
   # WantedBy= is set by the active profile (unset in development-local)
   ```

2. **Modified** existing `animus.service` and `animus-forge.service` to add `[Unit] PartOf=animus-runtime.target`.

3. **New** `~/.local/share/applications/animus.desktop` — launcher with `Icon=animus`, `Exec=animus-control`, no autostart.

4. **New** `animus-control` module (Python, sibling of `animus-tray`) — Start/Stop/Status/Logs/Dashboard/Audit. Thin wrapper over `systemctl --user start/stop animus-runtime.target` for lifecycle intent, plus the daemon's `GET /healthz` for runtime health. Reads `~/.config/animus/profile.json` and surfaces profile-appropriate controls. Displays the six-state `HealthState`, not a single "is-active" boolean.

5. **Modified** `~/.local/bin/animus-tray` — replace `pgrep -f animus_bootstrap.daemon` with `systemctl --user show animus-runtime.target` (machine-readable) and the daemon's `GET /healthz` endpoint. The tray observes lifecycle and health; it owns neither. Add opt-in "show tray offline" autostart unit `~/.config/systemd/user/animus-tray-offline.service` (gated by `tray_while_offline` in profile.json). The tray's `LockedPidFile` singleton enforcement is preserved.

6. **Modified** `~/.config/autostart/animus-tray.desktop` — flip `X-GNOME-Autostart-enabled` to `false` by default. The control app rewrites the line on user request.

7. **New** `~/.config/animus/profile.json` with default `{"mode": "development-local", "tray_while_running": true, "tray_while_offline": false, "start_on_login": false}`. Linger state is **not** in this file — it is read from `loginctl` at runtime and surfaced under a separate observed-state object.

8. **Modified** `animus-cleanup` (CLI) — 4-state classification with state-specific provenance rules per the Decision section. Hard rule: never `kill -9` an Unknown; Orphaned requires registry identity plus at least two independent process proofs; Managed services must be stopped through systemd, never signalled PID directly.

9. **New** dashboard endpoints `/system/services` and `/system/processes` returning `SystemProcessRegistry` rows reconciled against `systemctl --user list-units` and `pgrep` cross-check. Two sources of truth, not one.

### Required test changes

| Test | What it proves | Isolation strategy |
|---|---|---|
| `tests/test_animus_runtime_target.py` | Target with `Requires=animus.service` + `Wants=animus-forge.service` brings both up on start; target stop tears both down. | Uses **temporary uniquely named units** (e.g., `animus-test-target@<uuid>.target`) in an isolated test unit directory, not the live `animus-runtime.target`. Test cleanup removes the temp units in both success and failure paths. |
| `tests/test_partof_wants_separation.py` | Service with only `PartOf=animus-test-target@<uuid>.target` does NOT start when the target starts. Target without `Requires=animus-test-daemon@<uuid>.service` does NOT pull the daemon in. **This is the regression guard for the systemd error that motivated this ADR.** | Temp units; no live runtime. |
| `tests/test_profile_modes.py` | `profile.json` round-trips; mode change writes the right unit drop-in to the right path; `development-local` profile never creates a `default.target.wants/` symlink. | Operates on a temp `~/.config/animus-test-<uuid>/` directory tree; verifies the symlink is absent without touching the real `~/.config/animus/`. |
| `tests/test_stray_classification.py` | 4-state classification with state-specific provenance rules; provenance-deficient matches are reported as Unknown and never killed. | Uses fake process descriptors and a temp registry DB; no real PIDs. |
| `tests/test_tray_does_not_supervise.py` | Killing `animus-tray` does not stop the target. **This is the regression guard for the tray-as-supervisor error.** | Spawns a temp tray-shaped process in an isolated cgroup; verifies the target's state is unchanged. Does not kill the developer's actual tray. |
| `tests/test_discord_not_in_target.py` | The live `animus-discord.service` (or its absence) is not pulled in by `animus-runtime.target`; `systemctl --user stop animus-runtime.target` does not affect any discord-classified process. | Asserts the dependency graph of the real `animus-runtime.target` (a static `systemctl show` parse); does not modify the live runtime. |
| `tests/test_backup_timers_independent.py` | Backup timers continue to run when `animus-runtime.target` is stopped. | Uses temp `*.timer` units in the test unit directory; does not start or stop the live backup timers. |
| `tests/test_health_state.py` | The control app's `HealthState` derivation correctly maps `(is-active, health-probe, wants-service-states)` to the seven-state enum (including `UNKNOWN`), with the three failure-walkthrough cases from the Decision section as named test inputs. | Pure function tests on the state-derivation logic; no live processes. |
| `tests/test_health_contract.py` | The control app's `/healthz` parser and the daemon's `/healthz` endpoint conform to a versioned response contract. | Integration test using a temp daemon (or a recorded `/healthz` fixture); asserts both sides accept the same schema. The contract's *form* (OpenAPI, Pydantic, dataclasses, or the Contracts package) is an implementation decision in the build spec. |
| `tests/test_no_live_runtime_touch.py` | **Meta-test.** Walks the test directory and asserts that no test in `tests/test_animus_*.py` or `tests/test_runtime_*.py` references the live unit names `animus.service`, `animus-forge.service`, or `animus-runtime.target` without an isolation layer. | Static AST scan of test files. The build spec will likely replace this with a more robust sandboxed-harness check using `XDG_CONFIG_HOME`, `XDG_RUNTIME_DIR`, unique test unit names, and a temp registry database; that detail is out of scope for this ADR. |

**The regression tests must not start or stop the developer's actual Animus runtime.** Every test listed above either uses temporary units, an isolated cgroup, a temp config directory, or static parsing. The meta-test `tests/test_no_live_runtime_touch.py` enforces this property at the test-directory level.

### Required documentation changes

- Update `packages/bootstrap/CLAUDE.md` (or equivalent) with the lifecycle section above.
- Add `docs/systemd/animus-runtime.md` describing the target, the `Wants=`/`PartOf=` relationship, and the profile matrix.
- Add `docs/operations/process-registry.md` describing the 4-state classification and the dashboard reconciliation endpoint.
- Add the seven-step review pattern (see ADR-008) to the contributing guidelines.

### Operational consequences

- A separate security finding exists for plaintext credentials in a Forge systemd drop-in. **It is out of scope for this ADR** and is tracked as a separate security issue, ADR, or remediation PR. Mixing it into the lifecycle change set would increase review scope and complicate rollback.
- The current `Linger=yes` on this user must be surfaced (not changed) when the user is offered `desktop-login` or `continuous-node` profiles. Silent assumption is wrong in either direction.
- The first install after this ADR is adopted must not enable the runtime target. The installer writes the unit files, leaves them disabled, and writes the profile JSON with `mode: development-local`.

## Alternatives Considered

### A. Do nothing — accept the status quo (rejected)
Today's state is: Animus starts via the autostart-enabled `animus-tray.desktop` entry on every login, with the systemd units correctly disabled. Stray processes (`animus.mcp_server` duplicates, orphan trays, the discord bot supervised by Plex) exist with no unified registry or classification. The user is not given a Start/Stop control surface; "is Animus running?" is answered with `pgrep -f` and a desktop notification when the daemon flaps. Rejected because the user explicitly reported this state as the problem to be solved. The cost of doing nothing is operational noise (notifications, CPU/memory waste, stale locks on Chroma and SQLite) and erosion of trust in the daemon. The status quo is recorded here so a future maintainer does not relitigate the case for change.

### B. Tray-as-supervisor (rejected)
Make `animus-tray` own the daemon, MCP, forge, and discord bot as supervised children. Rejected because: (a) a GTK + AppIndicator process is a fragile supervisor — it dies on desktop-shell restart, OOM, DE reload; (b) the user explicitly named the orchestrator as the *cause* of the strays problem, and the orchestrator must be more robust than what it orchestrates; (c) systemd already provides the supervisor primitive; we should not reinvent it.

### B. Three separate architectures for the three profiles (rejected)
Build a dev profile code path, a desktop-login profile code path, and a continuous-node profile code path. Rejected because: (a) doubles the test surface; (b) creates a parity tax; (c) the differences are configuration, not code.

### C. Replace systemd and tray with a single Animus binary launcher (rejected)
Drop both systemd and the tray, use a Python launcher that supervises everything via `SystemProcessRegistry`. Rejected because: (a) the user explicitly asked for a desktop/dock icon, and the binary launcher does not provide one without a process; (b) it abandons the `loginctl` integration that the backup timers, sync, and discord bot already depend on; (c) it duplicates a primitive systemd already provides well.

### D. Tray stays as supervisor, only stop the strays (rejected partial)
Keep the tray as supervisor but kill the discord bot and orphan MCP processes. Rejected because: (a) the strays problem is a symptom of weak lifecycle ownership, not a separate problem; (b) the tray is still the wrong supervisor; (c) killing without classification discipline creates new failure modes (killing the wrong process).

## References

- `~/.local/bin/animus-tray` — current tray, lines 1–260 reviewed
- `~/.config/systemd/user/animus.service`, `animus-forge.service` — current user units
- `~/.config/autostart/animus-tray.desktop` — current autostart entry, `X-GNOME-Autostart-enabled=true`
- `man systemd.unit` — `PartOf=`, `Wants=`, `Requires=` semantics
- `packages/core/animus/infrastructure/process_lifecycle.py` — `LockedPidFile`, `SystemProcessRegistry`, `ProcessGuard`
- `adrs/ADR-005.md` — Kernel Extraction (precedent for unit-file-shaped work)
- `adrs/ADR-006.md` — Public/Private Repo Split (precedent for flat `adrs/ADR-NNN.md` format)

## Open Questions

1. **Should `animus-discord.service` be created and added to the target?** Depends on the user's classification decision. Tracked as A2 from the prior exploration; not resolved in this ADR.
2. **Should the `animus-autonomous-*.timer` units be in the runtime target or independent?** The autonomous timers (`autonomous`, `autonomous-all`, `autonomous-conversation`, `autonomous-knowledge`, `autonomous-test`) currently live in `~/projects/animus/systemd/` and are not in the active systemd user directory. Like the discord bot, they are not under Animus's lifecycle today. Reclassification is a separate decision.
3. **When does the GX10 mode ship?** Out of scope for this ADR. The architecture supports it; the engineering program to harden it is a follow-on.
