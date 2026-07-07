# Animus Autonomous Loop

Headless, timer-driven improvement pipeline: Architect scans → generates proposals → stores to queue. No human input required.

## Components

| File | Purpose |
|---|---|
| `packages/core/animus/autonomous-loop.py` | Entry point — runs observation + proposal generation |
| `animus-autonomous.service` | systemd oneshot service |
| `animus-autonomous.timer` | systemd timer (every 6 hours) |

## Manual Run

```bash
PYTHONPATH="packages/core:packages/kernel/src:packages/types/src" \
  packages/core/.venv/bin/python3 -m animus.autonomous-loop --focus all
```

Exit codes:
- `0` — No actionable findings
- `1` — Error
- `2` — Proposal generated and queued

## systemd Install

```bash
sudo cp systemd/animus-autonomous.service /etc/systemd/system/
sudo cp systemd/animus-autonomous.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now animus-autonomous.timer
```

Check status:

```bash
systemctl status animus-autonomous.timer
journalctl -u animus-autonomous.service -f
```

## What It Does

1. **Architect observes codebase** — finds structural issues, debt, friction
2. **Generates proposal** — title, priority, confidence, affected files
3. **Stores to proposal queue** — persisted in Animus memory system
4. **Human reviews** via `animus proposal-queue list` and `animus proposal-queue approve <id>`

## Safety

- Proposals are **queued, not auto-executed**. Human approval gate remains.
- `--focus` controls scope: `codebase` (static analysis only), `conversation` (transcript patterns), `evaluation` (test / eval trends), `all` (default)
- `ProposalStatus.DRAFT` means pending review — never auto-committed
