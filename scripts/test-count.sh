#!/usr/bin/env bash
# Emit per-package pytest collection counts as JSON.
#
# Usage:
#   scripts/test-count.sh                 # all packages
#   scripts/test-count.sh quorum          # one package
#
# Used by CI to publish accurate test counts to the README badge
# (.github/test-counts.json) and to flag silent test losses
# (collection regressions).
#
# Exit status is always 0 — measurement, not a gate.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

count_pkg() {
    local pkg="$1"
    local target=""
    local pypath=""
    local cwd=""

    case "$pkg" in
        core)
            target="tests"
            cwd="packages/core"
            ;;
        forge)
            target="tests"
            cwd="packages/forge"
            ;;
        quorum)
            target="tests"
            cwd="packages/quorum"
            pypath="$REPO_ROOT/packages/quorum/python"
            ;;
        bootstrap)
            target="tests"
            cwd="packages/bootstrap"
            pypath="$REPO_ROOT/packages/bootstrap/src"
            ;;
        *)
            echo "unknown package: $pkg" >&2
            return 1
            ;;
    esac

    local count
    count=$(cd "$cwd" && \
        PYTHONPATH="$pypath" "$PYTHON" -m pytest "$target" --collect-only -q 2>&1 \
        | grep -oP '\d+ tests? collected' \
        | head -1 \
        | grep -oP '\d+' \
        || echo 0)
    echo "${count:-0}"
}

if [ $# -eq 1 ]; then
    count_pkg "$1"
    exit 0
fi

core=$(count_pkg core)
forge=$(count_pkg forge)
quorum=$(count_pkg quorum)
bootstrap=$(count_pkg bootstrap)
total=$((core + forge + quorum + bootstrap))

cat <<JSON
{
  "schemaVersion": 1,
  "label": "tests",
  "message": "$total",
  "color": "brightgreen",
  "core": $core,
  "forge": $forge,
  "quorum": $quorum,
  "bootstrap": $bootstrap,
  "total": $total
}
JSON
