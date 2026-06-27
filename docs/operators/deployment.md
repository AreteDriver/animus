# Deployment

> Run Animus as a persistent system service.

---

## Bootstrap Install

The recommended deployment path is via Bootstrap:

```bash
pip install animus-bootstrap
animus-bootstrap install
```

This:
1. Installs dependencies
2. Runs the onboarding wizard
3. Registers a systemd (Linux) or launchd (macOS) service
4. Opens the dashboard at `http://localhost:7700`

## Service Management

```bash
animus-bootstrap start      # Start the daemon
animus-bootstrap stop       # Stop the daemon
animus-bootstrap restart    # Restart
animus-bootstrap status     # Show system status
```

## Configuration

Config lives at `~/.config/animus/config.toml` (chmod 600).

See [Configuration](configuration.md) for the full config reference.

## Security

- Config file is permission-protected (chmod 600)
- No telemetry by default
- API keys stored locally, never transmitted
- See [Reference → Security](../reference/security.md) for threat model

---

## See Also

- [Configuration](configuration.md) — Config file reference
- [Monitoring](monitoring.md) — Health checks and logs
- [Troubleshooting](troubleshooting.md) — When things break
