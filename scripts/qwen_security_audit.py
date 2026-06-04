#!/usr/bin/env python3
"""Animus self-security-audit via its OWN provider stack + a local qwen model.

This drives animus_forge's ``OllamaProvider`` over the security-critical files
(the egress/DLP + integrity + provider-gate surface that the 2026-06-03 review
and Phase C0 touched). Everything runs against a LOCAL ollama — source code
never leaves the machine, so this is PUBLIC-tier safe.

Usage:
    python scripts/qwen_security_audit.py [--model qwen2.5-coder:latest]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The security-critical surface — the files an attacker would target to defeat
# the egress / DLP / integrity guarantees.
TARGETS = [
    "packages/types/src/animus_types/egress.py",
    "packages/types/src/animus_types/secrets.py",
    "packages/forge/src/animus_forge/network/egress.py",
    "packages/forge/src/animus_forge/providers/base.py",
    "packages/forge/src/animus_forge/providers/bedrock_provider.py",
    "packages/core/animus/integrity/checker.py",
]

PROMPT = """You are a rigorous security auditor reviewing a Python module from a \
personal AI exocortex ("Animus"). Security model: requests tagged CONFIDENTIAL/\
SECRET must NEVER reach a cloud provider; outbound payloads are scanned for \
credentials (content-aware DLP); an integrity checker hashes critical modules to \
detect tampering. The system must fail CLOSED.

Do NOT rubber-stamp. Work in two parts:

PART 1 — ATTACK SURFACE: list the 3 most security-relevant code paths in THIS file \
and, for each, state the specific way it could be bypassed or fail open, and \
whether the code actually prevents it. Be concrete (name functions / lines).

PART 2 — FINDINGS: for each real issue output one line:
[SEVERITY: CRITICAL|HIGH|MEDIUM|LOW] <file>:<line> — <issue> => <fix>
Consider: egress/DLP bypass, regex that misses real credentials or ReDoS-\
backtracks, TOCTOU, integrity evasion, unsafe deserialization, injection, \
fail-open logic, missing input validation. If genuinely none, write: NO FINDINGS

File: {path}

```python
{code}
```
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:latest")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "packages/forge/src"))
    sys.path.insert(0, str(REPO / "packages/types/src"))
    from animus_forge.providers.base import CompletionRequest
    from animus_forge.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(model=args.model, host=args.host)
    provider.initialize()
    print(f"# Animus self-security-audit — model={args.model}\n")

    for rel in TARGETS:
        path = REPO / rel
        code = path.read_text()
        # Keep within a sane context; truncate very large files.
        if len(code) > 16000:
            code = code[:16000] + "\n# ... (truncated)\n"
        req = CompletionRequest(
            prompt=PROMPT.format(path=rel, code=code),
            temperature=0.1,
            max_tokens=800,
        )
        t0 = time.time()
        resp = provider.complete(req)
        dt = time.time() - t0
        print(f"## {rel}  ({dt:.1f}s, {resp.output_tokens} out tok)")
        print(resp.content.strip())
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
