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

## Release Evidence Bundles

Before tagging a release, generate an evidence bundle that proves the codebase is tested, traceable, and schema-compliant:

```bash
python scripts/assemble_evidence_bundle.py
```

This produces a timestamped directory in `evidence/releases/` containing:

| File | Purpose |
|---|---|
| `manifest.json` | Git SHA, timestamp, version, builder identity |
| `test-results.json` | Aggregated pytest counts per package |
| `schema-validation.json` | JSON Schema parseability report |
| `git-info.txt` | Last 5 commits and dirty/clean status |
| `dependencies.lock` | `pip freeze`, `cargo tree`, `npm ls` |
| `report.md` | Human-readable summary with pass/fail badges |

Options:
- `--output-dir PATH` — write bundle to a custom directory
- `--allow-dirty` — allow dirty git working tree (default: fail if uncommitted changes exist)

See `evidence/releases/README.md` for the full bundle format specification.

---

## See Also

- [Configuration](configuration.md) — Config file reference
- [Monitoring](monitoring.md) — Health checks and logs
- [Troubleshooting](troubleshooting.md) — When things break
