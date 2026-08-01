# /loop Prompt: Animus Public Prep — Sprint 1

**Purpose:** Run the 3-phase public-prep checklist autonomously while you sleep/work.

**Trigger:** Type `/loop` in Claude Code at `~/projects/animus`, then paste this block.

---

## Phase 1 — Sanitize (1–2h)

**Task:** Run the pre-public sanitization scanner and resolve blockers.

```
1. READ scripts/pre_public_sanitize.py to understand scan rules
2. RUN: python3 scripts/pre_public_sanitize.py --repo . --json > /tmp/sanitize.json
3. PARSE /tmp/sanitize.json
4. FOR EACH finding with severity in ["critical", "high"]:
   a. READ the file at finding.file
   b. If category == "secret":
      - If file is .env or secrets.env: ADD to .gitignore, git rm --cached, VERIFY not in history
      - If snippet is hardcoded key: REPLACE with env var reference or placeholder
   c. If category == "owner":
      - REPLACE "your-org" with generic "your-org" or parameterize
   d. If category == "path":
      - REPLACE absolute path with Path.home() or env var
   e. WRITE the fix back to the file
5. RE-RUN scanner until critical + high count == 0
6. COMMIT: "chore: pre-public sanitization — remove secrets and owner-specific data"
```

**Acceptance:** `python3 scripts/pre_public_sanitize.py --repo .` exits 0.

---

## Phase 2 — Ollama-First Config (30m)

**Task:** Apply the Ollama-first patch.

```
1. READ scripts/apply_ollama_first_patch.py
2. RUN: python3 scripts/apply_ollama_first_patch.py
3. VERIFY: git diff shows changes to packages/core/animus/__main__.py and config.py
4. RUN: python3 -m pytest packages/core/tests/test_cli_commands.py -v -k "ollama" --tb=short
5. IF tests fail: READ failure, fix in-place, re-run
6. COMMIT: "feat(config): Ollama-first default, cloud providers opt-in only"
```

**Acceptance:** Core CLI tests pass with provider=ollama and no ANTHROPIC_API_KEY.

---

## Phase 3 — Public README + Install Script (1–2h)

**Task:** Write a public-facing README and single-command install.

```
1. READ existing README.md for current structure
2. WRITE README_PUBLIC.md with:
   - One-line pitch: "Personal AI exocortex — local-first, open-source"
   - Prerequisites: Ollama, Python 3.10+
   - Install: `pip install animus` or `git clone + pip install -e .`
   - Quick start: `animus init` → `animus brief`
   - Architecture diagram (ASCII or link to docs)
   - Contributing section (link to CONTRIBUTING.md)
   - License badge
3. WRITE scripts/install.sh:
   - Detect OS (macOS/Linux)
   - Check Python version
   - Check Ollama running on :11434
   - pip install from git
   - Run `animus init`
   - Print success message + next steps
4. MAKE scripts/install.sh executable
5. COMMIT: "docs: public README + one-command install script"
```

**Acceptance:** A friend can follow README_PUBLIC.md and get Animus running in under 10 minutes.

---

## Constraints

- **Budget:** 5000 effective tokens per phase (15000 total). If a phase exceeds, STOP and report.
- **Safety:** DO NOT commit .env files, secrets, or owner-specific data. The sanitization scanner gates this.
- **Cloud opt-in:** Never auto-enable Anthropic/OpenAI. If keys exist, warn user and require explicit `ANIMUS_CLOUD_PROVIDER`.
- **Scope:** No architecture changes. No new features. Just sanitize, patch docs, and make it runnable.

---

## Handoff Format (when loop completes or hits budget)

```
STATUS: [Phase 1/2/3 complete | budget exhausted | blocked]
BRANCH: [branch name]
COMMITS: [list of commit SHAs]
BLOCKERS: [any unresolved items]
NEXT: [what human needs to do]
```
