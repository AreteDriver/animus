# Animus Security Layer: Secret Manager + CodeQL

> Production-grade secret management and automated code scanning for the Animus ecosystem.

---

## 1. Secret Manager

### The Problem

Animus has secrets scattered across multiple components:

| Component | Secrets |
|-----------|---------|
| Marketing Engine | Twitter API keys, LinkedIn OAuth, Reddit credentials, YouTube OAuth, TikTok tokens |
| Forge | LLM provider API keys (Anthropic, OpenAI), CAPTCHA solver key (CapSolver) |
| Browser Adapter | Platform session cookies, CAPTCHA service credentials |
| BenchGoblins | Stripe keys, RevenueCat API key, Railway deploy tokens |
| Developer Tools | PyPI publish token, GitHub tokens |
| ESI Client | EVE SSO client credentials |

Right now these live in `.env` files, `secrets/` YAML (gitignored), and environment variables with no unified access pattern. Every component rolls its own credential loading. This is a security liability and an operational headache.

### Design: Unified Secret Interface

Animus gets a single secret access layer that abstracts the backend. Components never touch credentials directly — they call the secret manager.

```python
from animus.secrets import get_secret, set_secret, list_secrets

# Any component can request a secret by path
twitter_key = get_secret("marketing/twitter/api_key")
anthropic_key = get_secret("forge/providers/anthropic/api_key")
stripe_key = get_secret("benchgoblins/stripe/secret_key")

# Secrets are namespaced by component
list_secrets("marketing/")
# → marketing/twitter/api_key
# → marketing/twitter/api_secret
# → marketing/linkedin/client_id
# → marketing/reddit/client_id
# → ...
```

### Backend Options

The secret manager supports multiple backends via a common interface. You pick the backend based on your environment:

```yaml
# ~/.animus/secrets.yaml
backend: age  # Options: age, pass, 1password, env, plaintext

age:
  identity_file: ~/.animus/keys/animus.key
  secrets_dir: ~/.animus/secrets/

pass:
  store_dir: ~/.password-store/animus/

onepassword:
  vault: Animus
  account: my.1password.com

env:
  prefix: ANIMUS_  # Falls back to ANIMUS_MARKETING_TWITTER_API_KEY

plaintext:
  # NEVER use in production. Only for local dev/testing.
  file: ~/.animus/secrets.plaintext.yaml
```

### Recommended Backend: `age`

**Why age over the others:**

| Backend | Pros | Cons |
|---------|------|------|
| **age** | Simple, fast, no daemon, git-friendly (encrypted files), composable with sops | Manual key management |
| pass | GPG-based, mature, git-backed | GPG is complex, agent issues |
| 1Password CLI | Great UX, team sharing, audit log | $4/mo, external dependency, network required |
| env vars | Universal, works everywhere | No encryption at rest, leaks in logs, no namespacing |
| plaintext | Zero friction | Zero security |

**age** is the sweet spot for a solo developer: encrypted files that live alongside your code (gitignored), no daemon, no subscription, and compatible with `sops` for structured secret files.

### Implementation

```
shared/secrets/
├── __init__.py          ← Public API: get_secret, set_secret, list_secrets, delete_secret
├── manager.py           ← SecretManager class, backend dispatch
├── backends/
│   ├── base.py          ← Abstract backend interface
│   ├── age_backend.py   ← age encryption (recommended)
│   ├── pass_backend.py  ← pass (password-store) integration
│   ├── op_backend.py    ← 1Password CLI integration
│   ├── env_backend.py   ← Environment variable fallback
│   └── plaintext.py     ← Dev-only plaintext (with loud warnings)
├── config.py            ← Load ~/.animus/secrets.yaml
└── rotate.py            ← Key rotation utilities
```

### Core Interface

```python
# shared/secrets/backends/base.py
from abc import ABC, abstractmethod

class SecretBackend(ABC):
    @abstractmethod
    def get(self, path: str) -> str | None:
        """Retrieve a secret by path. Returns None if not found."""
        ...

    @abstractmethod
    def set(self, path: str, value: str) -> None:
        """Store a secret at the given path."""
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete a secret. Returns True if it existed."""
        ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List all secret paths matching the prefix."""
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a secret exists without retrieving it."""
        ...
```

### age Backend

```python
# shared/secrets/backends/age_backend.py
import subprocess
from pathlib import Path
from .base import SecretBackend

class AgeBackend(SecretBackend):
    def __init__(self, identity_file: str, secrets_dir: str):
        self.identity = Path(identity_file).expanduser()
        self.secrets_dir = Path(secrets_dir).expanduser()
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

        if not self.identity.exists():
            self._generate_key()

    def _secret_path(self, path: str) -> Path:
        """Convert secret path to filesystem path. marketing/twitter/api_key → secrets_dir/marketing/twitter/api_key.age"""
        clean = path.strip("/").replace("..", "")  # Sanitize
        return self.secrets_dir / f"{clean}.age"

    def get(self, path: str) -> str | None:
        fpath = self._secret_path(path)
        if not fpath.exists():
            return None
        result = subprocess.run(
            ["age", "--decrypt", "-i", str(self.identity), str(fpath)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to decrypt {path}: {result.stderr}")
        return result.stdout.strip()

    def set(self, path: str, value: str) -> None:
        fpath = self._secret_path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        # Get public key from identity file
        pubkey = self._get_pubkey()
        result = subprocess.run(
            ["age", "--encrypt", "-r", pubkey, "-o", str(fpath)],
            input=value, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to encrypt {path}: {result.stderr}")

    def delete(self, path: str) -> bool:
        fpath = self._secret_path(path)
        if fpath.exists():
            fpath.unlink()
            return True
        return False

    def list(self, prefix: str = "") -> list[str]:
        results = []
        search_dir = self.secrets_dir / prefix.strip("/")
        if not search_dir.exists():
            search_dir = self.secrets_dir
        for f in search_dir.rglob("*.age"):
            rel = f.relative_to(self.secrets_dir)
            results.append(str(rel).removesuffix(".age"))
        return sorted(results)

    def exists(self, path: str) -> bool:
        return self._secret_path(path).exists()

    def _generate_key(self):
        """Generate a new age keypair."""
        self.identity.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["age-keygen", "-o", str(self.identity)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate age key: {result.stderr}")
        self.identity.chmod(0o600)

    def _get_pubkey(self) -> str:
        """Extract public key from identity file."""
        with open(self.identity) as f:
            for line in f:
                if line.startswith("# public key:"):
                    return line.split(":")[1].strip()
        raise RuntimeError("Could not find public key in identity file")
```

### CLI

```bash
# Initialize secret manager
animus secrets init
# Generated age key at ~/.animus/keys/animus.key
# Secret store at ~/.animus/secrets/

# Store a secret
animus secrets set marketing/twitter/api_key
# Enter value: [hidden input]
# ✅ Stored marketing/twitter/api_key

# Retrieve a secret
animus secrets get marketing/twitter/api_key
# sk-abc123...

# List all secrets
animus secrets list
# marketing/twitter/api_key
# marketing/twitter/api_secret
# marketing/linkedin/client_id
# forge/providers/anthropic/api_key
# ...

# List by namespace
animus secrets list marketing/
# marketing/twitter/api_key
# marketing/twitter/api_secret
# marketing/linkedin/client_id

# Delete a secret
animus secrets delete marketing/twitter/api_key

# Rotate the age identity key
animus secrets rotate
# ⚠️  This will re-encrypt all secrets with a new key.
# Continue? [y/N]

# Export secrets for backup (encrypted archive)
animus secrets export > secrets-backup.age

# Import from backup
animus secrets import secrets-backup.age
```

### Integration with Forge

Forge workflows reference secrets by path. The secret manager resolves them at runtime:

```yaml
# configs/marketing_engine/weekly_content.yaml
agents:
  - name: publisher
    archetype: publisher
    credentials:
      twitter_api_key: secrets://marketing/twitter/api_key
      twitter_api_secret: secrets://marketing/twitter/api_secret
```

```python
# forge/tools/credentials.py
from animus.secrets import get_secret

def resolve_credentials(config: dict) -> dict:
    """Resolve secrets:// references in workflow configs."""
    resolved = {}
    for key, value in config.get("credentials", {}).items():
        if isinstance(value, str) and value.startswith("secrets://"):
            path = value.removeprefix("secrets://")
            secret = get_secret(path)
            if secret is None:
                raise ValueError(f"Secret not found: {path}")
            resolved[key] = secret
        else:
            resolved[key] = value
    return resolved
```

### Security Rules

- **age identity key** is never committed to git. It lives in `~/.animus/keys/` with `chmod 600`.
- **Encrypted .age files** CAN be committed to git (they're encrypted), but the default `.gitignore` excludes them anyway.
- **Secrets are never logged.** The secret manager masks values in any debug/error output.
- **Secrets are never passed as CLI arguments** (visible in `ps aux`). Always piped via stdin or environment.
- **Plaintext backend** prints a loud warning on every access: `⚠️  PLAINTEXT BACKEND — NOT FOR PRODUCTION`.

### Rotation Reality

**Most API keys cannot be rotated programmatically.** Providers require web console access to create new keys and revoke old ones. The secret manager doesn't pretend to automate what can't be automated. Instead, it provides a **guided rotation checklist** that tracks what needs updating and where.

**The actual rotation workflow for any key:**
1. Go to the provider's web console, create a new key
2. `animus secrets set <path>` to store the new key locally
3. For deployed services: update production secrets (e.g., `fly secrets set KEY=value`)
4. Revoke the old key on the provider's web console
5. `animus secrets verify <path>` to confirm the new key works

**What the secret manager automates:**
- Steps 2 and 5 (local storage and verification)
- Tracking which secrets are due for rotation (age-based reminders)
- Generating values that ARE locally generated (JWT secrets, encryption keys)

**What requires manual web console access:**
- Twitter/X API keys → developer.twitter.com
- LinkedIn OAuth → linkedin.com/developers
- Reddit credentials → reddit.com/prefs/apps
- YouTube OAuth → console.cloud.google.com
- TikTok tokens → developers.tiktok.com
- Anthropic API key → console.anthropic.com
- OpenAI API key → platform.openai.com
- Stripe keys → dashboard.stripe.com
- RevenueCat key → app.revenuecat.com
- CapSolver key → dashboard.capsolver.com
- CCP ESI app → developers.eveonline.com
- PyPI token → pypi.org/manage/account/token
- Fly.io → `fly secrets set` (CLI, but still manual)

```bash
# Guided rotation — NOT automated key creation
animus secrets rotate marketing/twitter/api_key
#
# ROTATION CHECKLIST: marketing/twitter/api_key
# ─────────────────────────────────────────────
# Current key stored: 2026-01-15 (36 days ago)
#
# Step 1: Go to https://developer.twitter.com/en/portal/dashboard
#         Create a new API key under your app's "Keys and Tokens" tab.
#
# Step 2: Enter the new key below.
#         New value: [hidden input]
#         ✅ Stored new key.
#
# Step 3: Update production deployments:
#         → BenchGoblins on Fly.io:
#           fly secrets set TWITTER_API_KEY=<new_key> --app benchgoblins
#         → Any other deployed service using this key
#
# Step 4: Revoke the old key at https://developer.twitter.com
#
# Step 5: Verify new key works:
#         Testing Twitter API connection... ✅ Authenticated.
#
# ✅ Rotation complete for marketing/twitter/api_key

# Rotate a locally-generated secret (CAN be fully automated)
animus secrets rotate benchgoblins/jwt_secret --generate
#
# ROTATION: benchgoblins/jwt_secret
# ─────────────────────────────────
# Generated new JWT secret (64 bytes, base64url)
# ✅ Stored locally.
#
# ⚠️  Update production:
#   fly secrets set JWT_SECRET_KEY=<new_value> --app benchgoblins
#
# Old sessions will be invalidated.

# Check which secrets are stale
animus secrets audit
#
# SECRET AUDIT
# ────────────
# ⚠️  marketing/twitter/api_key       — 36 days old (rotate at 90)
# ⚠️  forge/providers/anthropic/key   — 72 days old (rotate at 90)
# ❌ marketing/reddit/client_secret   — 104 days old (OVERDUE)
# ✅ benchgoblins/stripe/secret_key   — 12 days old
# ✅ benchgoblins/jwt_secret          — 3 days old
#
# 1 overdue, 2 approaching rotation window

# Set rotation policy
animus secrets policy set marketing/* --rotate-days 90
animus secrets policy set benchgoblins/* --rotate-days 180
```

### Secret Metadata

Each secret stores metadata alongside the encrypted value:

```python
class SecretMetadata(BaseModel):
    path: str
    created_at: datetime
    last_rotated: datetime | None = None
    rotate_days: int = 90  # Default rotation window
    console_url: str | None = None  # Where to go for manual rotation
    deploy_commands: list[str] = []  # Commands to update production
    notes: str | None = None
```

The metadata powers the rotation checklist and audit. When you first store a secret, you can attach the console URL and deploy commands:

```bash
animus secrets set marketing/twitter/api_key \
  --console-url "https://developer.twitter.com/en/portal/dashboard" \
  --deploy-cmd "fly secrets set TWITTER_API_KEY=\{value\} --app benchgoblins" \
  --rotate-days 90
```

After that, `animus secrets rotate marketing/twitter/api_key` knows exactly where to send you and what deploy commands to run.

### Dependencies

```
age          # System package: brew install age / apt install age
```

That's it. `age` is a single binary with zero dependencies. The Python wrapper uses `subprocess` — no pip packages needed.

---

## 2. CodeQL Integration

### The Problem

mcp-manager and agent-audit are Python tools published to PyPI. They handle user configs, file paths, and in agent-audit's case, parse YAML that could contain injection payloads. They need automated security scanning on every PR and push.

### What CodeQL Catches

| Vulnerability Class | Relevant To |
|-------------------|-------------|
| Command injection | mcp-manager (spawns MCP servers via subprocess) |
| Path traversal | Both (read config files from user-specified paths) |
| YAML deserialization | agent-audit (parses workflow YAML — `yaml.safe_load` required) |
| Information disclosure | Both (error messages could leak file paths or secrets) |
| Regex DoS (ReDoS) | agent-audit (if using regex for rule matching) |
| Hardcoded credentials | Both (should never happen, but catches mistakes) |

### GitHub Actions Workflow

Same workflow for both repos (and any future Animus Python project):

```yaml
# .github/workflows/codeql.yml
name: "CodeQL Security Scan"

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    # Run weekly on Monday at 6 AM UTC
    - cron: '0 6 * * 1'

jobs:
  analyze:
    name: Analyze Python
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      actions: read
      contents: read

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
          # Use extended query suite for deeper analysis
          queries: +security-extended,security-and-quality

      - name: Autobuild
        uses: github/codeql-action/autobuild@v3

      - name: Perform Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:python"
```

### Custom Query: Secret Leak Detection

CodeQL's built-in queries catch hardcoded secrets, but add a custom query for the `secrets://` pattern to ensure nobody accidentally logs resolved secrets:

```ql
// .github/codeql/queries/secret-logging.ql
/**
 * @name Potential secret value in log output
 * @description Detects when a variable that may contain a secret is passed to a logging function
 * @kind problem
 * @problem.severity warning
 * @id animus/secret-logging
 * @tags security
 */

import python
import semmle.python.security.dataflow.LogInjectionQuery

from DataFlow::Node source, DataFlow::Node sink
where
  source.asExpr().(Call).getFunc().(Attribute).getName() = "get_secret"
  and sink.asExpr().getEnclosingStmt() instanceof LogStatement
select sink, "Potential secret value passed to logging function"
```

### Per-Repo Configuration

Each repo gets a `codeql-config.yml`:

```yaml
# .github/codeql/codeql-config.yml
name: "Animus CodeQL Config"

queries:
  - uses: security-extended
  - uses: security-and-quality
  - uses: ./.github/codeql/queries  # Custom queries

paths:
  - src/

paths-ignore:
  - tests/
  - docs/
  - "**/*.md"
```

### Where to Add CodeQL

| Repo | Priority | Why |
|------|----------|-----|
| **mcp-manager** | HIGH | Spawns subprocesses (MCP servers), reads arbitrary configs |
| **agent-audit** | HIGH | Parses untrusted YAML workflow files |
| **memboot** | MEDIUM | Indexes arbitrary files, serves as MCP server |
| **Animus (monorepo)** | HIGH | Secret manager, browser automation, Forge agent execution |
| **claudemd-forge** | LOW | Reads codebases but doesn't execute anything |
| **BenchGoblins** | MEDIUM | Handles payment data (Stripe/RevenueCat) |
| **ai-spend** | MEDIUM | Stores API keys for multiple providers |

### Additional Security Tooling

CodeQL is the foundation. Layer these on top:

```yaml
# .github/workflows/security.yml
name: "Security Suite"

on:
  push:
    branches: [main]
  pull_request:

jobs:
  codeql:
    # ... (as above)

  bandit:
    name: Bandit (Python Security Linter)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install bandit
      - run: bandit -r src/ -f json -o bandit-report.json || true
      - uses: actions/upload-artifact@v4
        with:
          name: bandit-report
          path: bandit-report.json

  safety:
    name: Safety (Dependency Vulnerability Check)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install safety
      - run: safety check --file requirements.txt --json --output safety-report.json || true
      - uses: actions/upload-artifact@v4
        with:
          name: safety-report
          path: safety-report.json

  trivy:
    name: Trivy (Container/Filesystem Scan)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
          format: 'sarif'
          output: 'trivy-results.sarif'
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
```

This gives you four layers:
1. **CodeQL** — semantic code analysis (finds real vulnerabilities in logic)
2. **Bandit** — Python-specific security linter (catches common mistakes fast)
3. **Safety** — dependency vulnerability scanning (known CVEs in your pip packages)
4. **Trivy** — filesystem/container scanning (catches misconfigs, leaked secrets in files)

### Badge for READMEs

Add to each repo's README:

```markdown
[![CodeQL](https://github.com/AreteDriver/REPO_NAME/actions/workflows/codeql.yml/badge.svg)](https://github.com/AreteDriver/REPO_NAME/actions/workflows/codeql.yml)
```

Shows recruiters and users that you take security seriously. Especially relevant for Palantir/Scale AI applications.

---

## Claude Code Build Prompts

### Prompt 1 — Secret Manager Foundation

```
Create a secret management module for the Animus monorepo:

Location: shared/secrets/
Files:
- __init__.py: Public API (get_secret, set_secret, delete_secret, list_secrets)
- manager.py: SecretManager class that dispatches to the configured backend
- config.py: Loads backend config from ~/.animus/secrets.yaml
- backends/base.py: Abstract SecretBackend class
- backends/age_backend.py: age encryption backend (uses subprocess to call age CLI)
- backends/env_backend.py: Environment variable fallback (ANIMUS_ prefix)
- backends/plaintext.py: Dev-only plaintext with warnings on every access

The age backend should:
- Auto-generate a keypair if none exists (~/.animus/keys/animus.key)
- Store encrypted files at ~/.animus/secrets/{path}.age
- Sanitize paths (no .., no absolute paths)
- chmod 600 on the identity file

Tests in tests/secrets/ with mocked subprocess calls.
Include a conftest.py that sets up a temp directory for test secrets.
```

### Prompt 2 — CLI + Forge Integration

```
Add CLI and Forge integration to the secret manager:

- CLI: Add to the main animus CLI (Click group):
  - animus secrets init — generate age key, create directories
  - animus secrets set <path> — prompt for value (hidden input), encrypt and store
  - animus secrets get <path> — decrypt and print
  - animus secrets list [prefix] — list all secrets or filter by prefix
  - animus secrets delete <path> — delete with confirmation
  - animus secrets rotate — re-encrypt all secrets with a new key

- Forge integration:
  - forge/tools/credentials.py: resolve_credentials() function
  - Parses secrets:// URIs in workflow YAML configs
  - Resolves them via the secret manager at runtime
  - Never logs resolved values

Tests for CLI (use Click testing runner) and Forge credential resolution.
```

### Prompt 3 — CodeQL + Security CI

```
Add security CI to mcp-manager and agent-audit repos:

For each repo, create:
1. .github/workflows/codeql.yml — CodeQL with security-extended queries for Python
2. .github/workflows/security.yml — Bandit + Safety + Trivy
3. .github/codeql/codeql-config.yml — scan src/, ignore tests/

Also:
- Add CodeQL badge to each repo's README.md
- Ensure all YAML parsing uses yaml.safe_load (never yaml.load)
- Ensure all subprocess calls use list form (never shell=True)
- Ensure all file path operations sanitize against path traversal

Run bandit on the existing codebase and fix any HIGH/MEDIUM findings.
Commit as: "ci: add CodeQL and security scanning"
```

---

## Implementation Priority

| Phase | What | When |
|-------|------|------|
| Phase 1 | CodeQL workflow for mcp-manager + agent-audit | When those repos are created (Sprint Phase 2) |
| Phase 2 | Secret manager (age backend + CLI) | When Marketing Engine needs API keys (Sprint Phase 3) |
| Phase 3 | Forge credential resolution (secrets:// URIs) | When Forge runs real workflows (Sprint Phase 4) |
| Phase 4 | Full security suite (Bandit + Safety + Trivy) | When any repo hits v1.0 |
| Phase 5 | 1Password backend (if you switch to team-based secrets) | If/when needed |

CodeQL is zero-effort to add (just a workflow file) so it goes in immediately. The secret manager gets built when you actually have production secrets to manage — not before.
