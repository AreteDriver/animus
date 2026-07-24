# Security

> Security architecture, threat models, safety guidelines, and operational security practices for the Animus system.

---

## Policy Decision Point

Animus includes a deterministic Policy Decision Point (PDP) for capability-based authorization. It evaluates action requests against capability grants and enforces default-deny semantics.

### Components

- **PolicyDecisionPoint** — evaluates `(principal, action, resource, workspace)` tuples
- **CapabilityGrantStore** — in-memory store for grants (PostgreSQL-backed in production)

### Decisions

| Decision | Meaning |
|---|---|
| `ALLOW` | Grant permits the action |
| `DENY` | No grant, expired grant, or action not permitted |
| `ESCALATE` | High-risk action (delete, execute, delegate, export) requires approval |
| `ABSTAIN` | Reserved for future use |

### Denial Reasons

- `MISSING_SCOPE` — No grants found or action not in grant
- `CAPABILITY_REVOKED` — All grants expired or revoked
- `UNKNOWN_SCHEMA` — Schema not in grant's allowed list
- `ESCALATION_REQUIRED` — High-risk action

### Usage

```python
from animus.policy import PolicyDecisionPoint, CapabilityGrant, CapabilityGrantStore

store = CapabilityGrantStore()
store.create(CapabilityGrant(
    grant_id="grant-001",
    principal="user-alice",
    scope=["memory"],
    resource="ws-test",
    action=["read", "create"],
    granted_by="admin",
    granted_at=datetime.now(timezone.utc),
))

pdp = PolicyDecisionPoint(store)
result = pdp.evaluate(
    principal="user-alice",
    action="read",
    resource="mem-001",
    workspace_id="ws-test",
)
assert result.decision == "allow"
```

---
## Security Documents

| Document | Scope | Last Updated |
|---|---|---|
| [Safety Guidelines](safety.md) | Operational safety rules and constraints | 2026-02-27 |
| [Security Layer](security-layer.md) | Technical security architecture and controls | 2026-02-27 |
| [Threat Model](threat-model.md) | Identified threats, mitigations, and risk assessment | 2026-05-26 |

---

## Shell Execution Trust Boundary

The `ForgeToolRegistry.run_command` tool executes shell commands via `subprocess.run(command, shell=True, ...)`. It provides the following guardrails:

- **Opt-in only** — `enable_shell=False` by default
- **Command allowlist** — Only commands in `_allowed_commands` are permitted (`python`, `python3`, `pip`, `git`, `ls`, `cat`, `echo`, `mkdir`, `rm`, `mv`, `cp`, `touch`, `find`, `grep`, `sed`, `awk`, `curl`, `wget`, `tar`, `zip`, `unzip`, `chmod`, `chown`, `diff`, `head`, `tail`, `wc`, `sort`, `uniq`, `jq`, `node`, `npm`, `npx`, `make`, `pytest`, `ruff`, `black`, `mypy`, `cargo`, `rustc`, `go`, `gofmt`)
- **Timeout enforcement** — Default 30 seconds
- **Output size limits** — Large outputs are truncated
- **Working directory restriction** — Commands run within `project_root`

### What this means

This design is **appropriate for**:
- A trusted local operator executing trusted workflow files
- Development automation where the operator reviews workflow definitions before running them

This design is **not a sandbox** and should **not be used for**:
- Arbitrary agent-generated YAML without human review
- Remote or untrusted workflow submissions
- Multi-tenant environments where users submit workflows to a shared executor

### Why it is not a sandbox

- `shell=True` means shell metacharacters (`;`, `&&`, `|`, `$()`) are interpreted. A command like `python -c "..." && rm -rf /` bypasses the allowlist check on the first token.
- The allowlist checks only `cmd_parts[0]` (the first whitespace-separated token). Pipelines, command substitution, and chained commands are not analyzed.
- There is no filesystem namespace isolation (chroot, container, or Landlock). A workflow can read/write any file the Animus process has access to.
- There is no network policy enforcement. A workflow can open outbound connections.

### Hardening path for autonomous/remote use

If you need to run untrusted or agent-generated workflows:

1. **Replace `shell=True` with `shell=False` and parsed argv** — Use `shlex.split()` to build argv arrays, preventing shell injection entirely.
2. **Container isolation** — Run each workflow step in a Docker container or systemd-nspawn with minimal capabilities, read-only rootfs, and restricted network.
3. **Landlock or seccomp** — Use Linux Landlock LSM for per-workflow filesystem sandboxes, or seccomp-bpf to block dangerous syscalls.
4. **Per-command approval** — Require explicit human approval for any command that writes to disk, modifies git state, or executes non-allowlisted binaries.
5. **Network policy** — Block outbound connections by default; whitelist only required endpoints.

---

## Quick Reference

### Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x.x | Yes |
| < 2.0 | No |

### Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** open a public issue
2. Email **jamesyng79@gmail.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within 48 hours
4. A fix will be prioritized based on severity

### Security Measures

This project uses:
- **CodeQL** — static analysis on every push
- **gitleaks** — secret scanning on every push
- **pip-audit** — dependency vulnerability scanning
- **Dependabot** — automated dependency updates

### Scope

**In scope** for security reports:
- Code injection vulnerabilities
- Authentication/authorization bypasses
- Credential exposure
- Dependency vulnerabilities with known exploits

**Out of scope**:
- Denial of service (this is a local-first tool)
- Social engineering
- Issues in dependencies without a proof of concept

---

## Key Principles

- **No telemetry by default** — Your data stays on your machine
- **Cryptographic ownership** — Identity files are permission-protected
- **Local-first by default** — Works fully offline after install
- **Approval gates** — Significant changes require explicit human approval

---

## See Also

- [Architecture → Decisions](../architecture/decisions/README.md) — Security-related ADRs
- [Contributing → Guidelines](../contributing/guidelines.md) — Secure development practices
- [Operators → Deployment](../operators/deployment.md) — Secure deployment patterns
