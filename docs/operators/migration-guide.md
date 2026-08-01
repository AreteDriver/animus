# Migration Guide: Monolith → Public/Private Split

**Status**: Draft — execute when ready to publicize
**Owner**: your-org
**Last updated**: 2026-06-29

---

## Overview

This guide walks through splitting the `animus` monorepo into:
- **Public**: `github.com/your-org/animus` — code, tests, docs, schemas
- **Private**: `github.com/your-org/animus-private` — owner data, secrets, PII

**Estimated time**: 2–4 hours (mostly waiting on GitHub/Git filter-repo)

---

## Prerequisites

- `git-filter-repo` installed (`pip install git-filter-repo`)
- GitHub CLI (`gh`) authenticated
- Admin access to the `your-org/animus` repo
- All pending work committed and pushed

---

## Phase A: Pre-Split Audit (30 min)

### Step 1: Verify no secrets in current working tree

```bash
# Run from repo root
grep -rn 'sk-ant-\|ghp_\|sk-\|api_key\|password\|secret' \
  --include='*.py' --include='*.md' --include='*.yaml' --include='*.json' \
  packages/ docs/ scripts/ | grep -v '.venv' | grep -v '__pycache__'
```

Expected result: **zero matches** in source files. Secrets should only exist in `.env` files (which are `.gitignore`d).

### Step 2: Verify `.gitignore` coverage

```bash
git status --ignored --short | grep -E '\.env|secrets|token|key|user_data|personal'
```

All private files should show as `!!` (ignored).

### Step 3: Check git history for secrets

```bash
# Scan all commits for common secret patterns
git log --all -p | grep -iE '(password|api_key|secret|token)\s*[:=]' | head -20
```

If any hits appear, note the commit SHA and file path. These will need purging.

### Step 4: Identify files to migrate to private repo

```bash
# Files that exist but are gitignored (potential private data)
git ls-files --others --ignored --exclude-standard
```

---

## Phase B: History Purge (30–60 min)

If secrets or private data exist in history, purge them before making the repo public.

### Option B1: git-filter-repo (Recommended)

```bash
# Backup first
cp -r /home/arete/projects/animus /tmp/animus-backup-$(date +%Y%m%d)

cd /home/arete/projects/animus

# Remove specific files from entire history
git filter-repo --path secrets.yaml --invert-paths
git filter-repo --path user_data/ --invert-paths
git filter-repo --path personal/ --invert-paths

# Remove files matching a pattern
git filter-repo --path-glob '*.env.local' --invert-paths

# Verify history is clean
git log --all --full-history -- secrets.yaml  # should return nothing
```

**Warning**: This rewrites history. All collaborators must re-clone or force-reset.

### Option B2: BFG Repo-Cleaner (Alternative)

```bash
java -jar bfg.jar --delete-files secrets.yaml
java -jar bfg.jar --delete-files user_data/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## Phase C: Create Private Repo (15 min)

```bash
# Create new private repo on GitHub
gh repo create your-org/animus-private --private --description "Animus private data and owner-specific configuration"

# Clone it
mkdir -p ~/projects/animus-private
cd ~/projects/animus-private
git init
git remote add origin git@github.com:your-org/animus-private.git

# Add a README
cat > README.md << 'EOF'
# Animus Private

Owner-specific data, secrets, and PII for the Animus exocortex.

This repo is **never** to be made public.

## Contents

- `config/local.yaml` — Local overrides
- `data/` — Memory dumps, eval results
- `secrets/` — API keys, tokens (encrypted)
- `backups/` — Evidence bundle archives

## Link to public repo

The public code lives at:
https://github.com/your-org/animus
EOF

git add README.md
git commit -m "init: private data repository"
git push -u origin main
```

---

## Phase D: Migrate Private Data (15 min)

```bash
cd ~/projects/animus-private

# Copy local configs (DO NOT commit to public repo)
mkdir -p config secrets data backups

# Example: copy your real .env (from local machine, not from git)
# cp ~/.config/animus/.env config/

# Example: copy memory dumps
# cp ~/animus-data/memory-store.sqlite data/

# Encrypt secrets before pushing
git add config/ data/
git commit -m "chore: add owner-specific data"
git push
```

---

## Phase E: Publicize Main Repo (10 min)

```bash
cd /home/arete/projects/animus

# Final verification — ensure no private data in tree
find . -name '.env' -o -name 'secrets.yaml' -o -name 'token.json' | grep -v '.gitignore'
# Should return nothing (all are gitignored)

# Ensure truth baseline passes
cd /home/arete/projects/animus
python3 scripts/truth-baseline.py truth-baseline.toml
```

Then on GitHub:
1. Settings → General → Danger Zone → Change visibility
2. Select "Make public"
3. Confirm repository name
4. Enable GitHub Pages: Settings → Pages → Source: Deploy from a branch → `main` → `/ (root)`

---

## Phase F: Post-Split Verification (15 min)

### Verify docs deploy

```bash
# Build docs locally first
mkdocs build

# Push should trigger Pages deployment (if using GitHub Actions)
git push origin main
```

Check `https://your-org.github.io/animus/` after 2–5 minutes.

### Verify no private data leaked

```bash
# Clone fresh copy to neutral location
cd /tmp
git clone --depth 1 https://github.com/your-org/animus.git animus-public-check
cd animus-public-check

# Run audit
grep -rn 'sk-ant-\|ghp_\|password' --include='*.py' --include='*.md' --include='*.yaml' packages/ docs/ scripts/
# Expected: zero matches
```

### Verify private repo is accessible

```bash
cd ~/projects/animus-private
git pull
# Should succeed (you have access)
```

---

## Rollback Plan

If publicization causes issues:

1. **Immediately** change visibility back to private on GitHub
2. If secrets were exposed: rotate ALL credentials (API keys, tokens, DB passwords)
3. Notify any external collaborators that the repo was temporarily public

---

## Ongoing Maintenance

| Task | Frequency | Responsible |
|---|---|---|
| Rotate API keys | Quarterly | your-org |
| Audit public repo for accidental private data | Monthly | Automated via CI (gitleaks) |
| Sync private data backups | Weekly | your-org |
| Review external PRs | As needed | your-org |

---

## See Also

- [Public/Private Split Specification](public-private-split.md)
- [ADR-006: Public/Private Repository Split](https://github.com/your-org/animus/blob/main/adrs/ADR-006.md)
