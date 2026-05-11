#!/usr/bin/env bash
# Run mypy per-package and emit error counts as a single JSON line.
#
# Usage:
#   scripts/mypy-count.sh                 # all packages
#   scripts/mypy-count.sh quorum          # one package
#
# CI compares the output against .github/mypy-baseline.json. Run
# locally before updating the baseline file when you've intentionally
# fixed errors:
#   scripts/mypy-count.sh > /tmp/mypy.json
#   # then write the new lower numbers into .github/mypy-baseline.json
#
# Exit status is always 0 — this is a measurement script, not a gate.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

count_pkg() {
    local pkg="$1"
    local target

    case "$pkg" in
        core)      target="packages/core/animus" ;;
        forge)     target="packages/forge/src" ;;
        quorum)    target="packages/quorum/python/convergent" ;;
        bootstrap) target="packages/bootstrap/src" ;;
        *)         echo "unknown package: $pkg" >&2; return 1 ;;
    esac

    if [ "$pkg" = "quorum" ]; then
        export PYTHONPATH="packages/quorum/python:${PYTHONPATH:-}"
    fi

    # mypy exits non-zero when errors are found — don't let that
    # propagate; we only want the count. Use python -m mypy so the
    # script works from any activated venv (CI, local, etc.) without
    # needing mypy on system PATH.
    local python_bin="${PYTHON:-python3}"
    local count
    count=$("$python_bin" -m mypy "$target" --ignore-missing-imports --no-error-summary 2>&1 \
            | grep -c "error:" || true)
    echo "$count"
}

if [ $# -eq 1 ]; then
    count_pkg "$1"
    exit 0
fi

# All packages — emit JSON
core=$(count_pkg core)
forge=$(count_pkg forge)
quorum=$(count_pkg quorum)
bootstrap=$(count_pkg bootstrap)

cat <<JSON
{"core": $core, "forge": $forge, "quorum": $quorum, "bootstrap": $bootstrap}
JSON
