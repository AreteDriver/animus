# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x.x | Yes |
| < 2.0 | No |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email **jamesyng79@gmail.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within 48 hours
4. A fix will be prioritized based on severity

## Security Measures

This project uses:
- **CodeQL** — static analysis on every push
- **gitleaks** — secret scanning on every push
- **pip-audit** — dependency vulnerability scanning
- **Dependabot** — automated dependency updates

## Scope

The following are in scope for security reports:
- Code injection vulnerabilities
- Authentication/authorization bypasses
- Credential exposure
- Dependency vulnerabilities with known exploits

Out of scope:
- Denial of service (this is a local-first tool)
- Social engineering
- Issues in dependencies without a proof of concept
