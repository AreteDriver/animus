# Monitoring

> Health checks, logs, and observability for Animus.

---

## Dashboard

Bootstrap provides a local web dashboard at `http://localhost:7700`:

```bash
animus-bootstrap dashboard
```

Shows:
- Daemon status (running/stopped)
- Ollama connection status
- Memory backend status
- Forge connection status
- Pending identity proposals

## Logs

Logs are written to:
- `~/.local/share/animus/logs/animus.log`
- Console (if running interactively)

View recent logs:
```bash
tail -f ~/.local/share/animus/logs/animus.log
```

## Truth Baseline

The truth baseline script validates documented claims against reality:

```bash
python scripts/truth-baseline.py
```

Checks:
- Package versions match
- Schema counts match
- Test counts are in range
- ADR files exist

## Health Checks

Programmatic status:
```bash
animus-bootstrap status
```

Returns exit code 0 if all services healthy.

---

## See Also

- [Deployment](deployment.md) — Service setup
- [Configuration](configuration.md) — Config reference
- [Troubleshooting](troubleshooting.md) — When things break
