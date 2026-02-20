#!/usr/bin/env python3
"""Review a single file using local Ollama model."""
import json
import os
import sys
import urllib.request


def review_file(filepath: str) -> str:
    with open(filepath) as f:
        code = f.read()

    data = json.dumps({
        "model": os.getenv("OLLAMA_MODEL", "deepseek-coder-v2"),
        "prompt": f"""Senior Python engineer code review.
Project: Animus -- personal AI exocortex.
Three layers: Core (identity/memory), Forge (orchestration), Quorum (coordination).

Review this file. For each issue found, provide:
1. Line number
2. Issue category (correctness/security/performance/typing/style)
3. Current code
4. Suggested fix
5. Why it matters

Focus on: correctness bugs, error handling gaps, missing type hints, performance issues.
Skip: style preferences, docstring formatting, import ordering.

File: {filepath}

```python
{code}
```""",
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["response"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python review.py <filepath>")
        sys.exit(1)
    print(review_file(sys.argv[1]))
