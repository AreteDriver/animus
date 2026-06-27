# Security

> Security architecture, threat models, safety guidelines, and operational security practices for the Animus system.

---

## Security Documents

| Document | Scope | Last Updated |
|---|---|---|
| [Safety Guidelines](safety.md) | Operational safety rules and constraints | 2026-02-27 |
| [Security Layer](security-layer.md) | Technical security architecture and controls | 2026-02-27 |
| [Threat Model](threat-model.md) | Identified threats, mitigations, and risk assessment | 2026-05-26 |

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

- [Architecture → Decisions](../architecture/decisions/) — Security-related ADRs
- [Contributing → Guidelines](../contributing/guidelines.md) — Secure development practices
- [Operators → Deployment](../operators/deployment.md) — Secure deployment patterns
