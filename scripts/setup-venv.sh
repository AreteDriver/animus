#!/usr/bin/env bash
# Rebuild the Animus core venv with EVERY workspace package editable-installed.
#
# Why this exists: packages/types (and other local-only siblings like quorum)
# are NOT published to PyPI. Core/Forge declare `animus-types` as a normal
# version constraint, so any install that omits `packages/types/` as an explicit
# `-e` target leaves the resolver unable to find it — imports then crash at
# runtime (e.g. tools/animus_sync.py -> ModuleNotFoundError: animus_types).
# That exact gap silently broke the 4h memory-sync cron for 5 days (2026-05-28
# -> 2026-06-02). This script makes a full, drift-proof rebuild a single command:
# it DISCOVERS every packages/*/pyproject.toml and installs them all in ONE pip
# invocation, so pip resolves all local workspace deps from source together.
#
# Usage:
#   scripts/setup-venv.sh            # create venv if missing, install everything
#   scripts/setup-venv.sh --dry-run  # print the discovered targets, install nothing
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/packages/core/.venv"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Discover every installable package (has a pyproject.toml). Dynamic discovery
# means new packages are picked up automatically — the list can't drift.
targets=()
for d in "$ROOT"/packages/*/; do
  [ -f "${d}pyproject.toml" ] || continue
  targets+=( -e "${d}[dev]" )
done

npkgs=$(( ${#targets[@]} / 2 ))
echo "Discovered ${npkgs} installable workspace packages:"
for ((i=1; i<${#targets[@]}; i+=2)); do echo "  - ${targets[$i]}"; done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "(--dry-run: nothing installed)"
  exit 0
fi

if [ ! -d "$VENV" ]; then
  echo "Creating venv at $VENV"
  python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"

"$PIP" install -q --upgrade pip
echo "Installing all ${npkgs} packages editable (single invocation)..."
"$PIP" install "${targets[@]}"

echo "Done. Animus/Convergent packages now in the venv:"
"$PIP" list | grep -iE 'animus|convergent' || true
