# Standing red-team — systemd units

Two templates that run `python -m animus.redteam.standing` on a weekly
cadence as the user (not root). Same hardening profile as the main
`animus.service` — read-only `$HOME`, write only to `~/.animus/audit/`,
loopback-only network.

## Install

```bash
# Copy to the user's systemd dir
mkdir -p ~/.config/systemd/user
cp packages/core/animus/redteam/systemd/animus-redteam.service ~/.config/systemd/user/
cp packages/core/animus/redteam/systemd/animus-redteam.timer ~/.config/systemd/user/

# Reload + enable
systemctl --user daemon-reload
systemctl --user enable --now animus-redteam.timer

# Check
systemctl --user list-timers animus-redteam
systemctl --user status animus-redteam.timer
```

## Run on demand

```bash
# Triggers the sweep immediately without waiting for the timer
systemctl --user start animus-redteam.service

# Watch the log
journalctl --user -fu animus-redteam.service
```

## Route through llama-server (HauhauCS Qwen3.6)

Override the env in a drop-in:

```bash
systemctl --user edit animus-redteam.service
```

Add:

```ini
[Service]
Environment="ANIMUS_REDTEAM_MODEL=hauhaucs"
Environment="ANIMUS_REDTEAM_BASE_URL=http://127.0.0.1:8081"
```

The OAI-compatible path means ollama and llama-server are
interchangeable backends from the sweep's perspective.

## Output locations

- `~/.animus/audit/standing-redteam-ledger.jsonl` — append-only history
  of every probe across every sweep. Persistent across reboots.
- `~/.animus/audit/standing-redteam-dashboard.md` — markdown summary
  of each sweep, tail-friendly. 🆕 marks novel findings, 🔁 marks
  repeats of probe shapes the ledger has seen before.

## Alerts

The sweep exits non-zero when a NEW finding lands at or above the
`--alert-on` severity (default `high`). Combine with a `OnFailure=`
unit (e.g. `animus-redteam-alert.service` that posts to a webhook or
sends mail) to get a push notification when something novel slips
through.
