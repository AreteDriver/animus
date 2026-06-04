# deploy/ — Animus scheduled-job deployment

systemd `--user` units that schedule Animus background jobs, kept here as the
**source of truth** and copied into place by the install scripts. Copy, not
symlink: a symlinked unit pointing into this working tree would dangle whenever
the tree is checked out on a branch without `deploy/`, silently breaking the
job — the exact failure mode this setup exists to prevent.

## animus-sync timer

Replaces the old crontab line (`0 */4 * * * … animus_sync.py … >> ~/.animus/sync.log 2>&1`)
that **failed silently for 5 days** (2026-05-28 → 2026-06-02): the core venv lost
`animus_types` (local-only dep, not on PyPI), every run crashed on import, and the
traceback only appended to `~/.animus/sync.log` — which nobody reads. (Root cause
fixed separately by `scripts/setup-venv.sh`.)

Under systemd a failure **cannot hide**:
- `systemctl --user --failed` lists it (canonical "what's broken")
- `journalctl --user -u animus-sync` has the full traceback, queryable
- `OnFailure=animus-sync-fail.service` writes `~/.animus/sync.FAILED` (the Claude
  Code statusline surfaces it as `⚠sync:FAIL`) and fires a `notify-send` alert
- `animus-sync.service` clears the marker on the next successful run

### Install / redeploy

```bash
bash deploy/install-sync-timer.sh
```

Copies the units + notify script into `~/.config/systemd/user` and `~/.local/bin`,
runs `daemon-reload`, and enables the timer. Re-run after editing any unit here.

### Files

| File | Role |
|---|---|
| `systemd/user/animus-sync.service` | runs `animus_sync.py --quiet`; clears the fail marker on success |
| `systemd/user/animus-sync.timer` | every 4h (`OnCalendar=*-*-* 00/4:00:00`, `Persistent=true`) |
| `systemd/user/animus-sync-fail.service` | `OnFailure` handler — surfaces the failure |
| `bin/animus-sync-notify.sh` | writes `~/.animus/sync.FAILED` + desktop notification |

> The crontab job was removed when this landed; back up at
> `~/.animus/crontab.backup-*`. The statusline staleness check (`⚠sync:Nd`) is an
> independent backstop that catches the job not running for *any* reason.
