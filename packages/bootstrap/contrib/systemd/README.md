# Systemd user units for Animus

These are reference unit files for running Animus Bootstrap (dashboard + intelligence layer) and Animus Forge (orchestration API) as systemd user services on Linux.

User services run without root, restart on logout/reboot, and integrate with `journalctl` for logs.

## Install

```bash
# Copy units to the user systemd dir
mkdir -p ~/.config/systemd/user/
cp animus.service animus-forge.service ~/.config/systemd/user/

# Edit the WorkingDirectory paths if your repo lives somewhere
# other than ~/projects/animus

# Reload systemd, enable, start
systemctl --user daemon-reload
systemctl --user enable --now animus.service animus-forge.service

# Verify
systemctl --user status animus.service animus-forge.service
curl http://127.0.0.1:7700/health
curl http://127.0.0.1:8000/health
```

## What each unit expects

| Path | Purpose | Required? |
|---|---|---|
| `~/projects/animus/.env` | Runtime config (Ollama host, ANIMUS_DATA_DIR, etc.) — already gitignored | Yes |
| `~/.local/share/animus/secrets.env` | API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY), `chmod 400` | Optional (Ollama-only installs work without) |
| `~/projects/animus/.venv/` | Bootstrap venv | Yes |
| `~/projects/animus/packages/forge/.venv/` | Forge venv | Yes |

The leading `-` on the secrets `EnvironmentFile=` line means systemd starts the service even when the file is missing.

## Logs

```bash
# Live tail
journalctl --user -u animus.service -f
journalctl --user -u animus-forge.service -f

# Last hour
journalctl --user -u animus.service --since "1 hour ago"
```

## Hardening notes

The units include:
- `PrivateTmp=true` — isolated `/tmp` per-service
- `ProtectSystem=strict` — read-only filesystem outside `ReadWritePaths`
- `NoNewPrivileges=true` — process can't gain privileges via setuid binaries

If a unit fails with permission errors after you change Animus internals, the most likely cause is a write to a path not listed in `ReadWritePaths=`. Add the path or relax the directive.

## Why user units, not system units

- No root needed for install
- Logs in your user journal, not `/var/log/`
- No conflict with system-wide service management
- Lifecycle naturally tied to your login session (start on login via `loginctl enable-linger $USER` if you want them to survive logout)

To run on boot without an active login session:

```bash
sudo loginctl enable-linger $USER
```
