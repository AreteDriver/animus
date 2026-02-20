#!/usr/bin/env bash
# Unified benchmark runner for Animus monorepo.
#
# Usage:
#   ./scripts/benchmark.sh              # Run all packages
#   ./scripts/benchmark.sh core         # Single package
#   ./scripts/benchmark.sh forge quorum # Multiple packages
#   ./scripts/benchmark.sh --json       # Save JSON output to .benchmarks/
#   ./scripts/benchmark.sh core --json  # Single package + JSON
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JSON_OUTPUT=false
PACKAGES=()

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --json) JSON_OUTPUT=true ;;
        core|forge|quorum) PACKAGES+=("$arg") ;;
        *)
            echo "Usage: $0 [core|forge|quorum ...] [--json]"
            exit 1
            ;;
    esac
done

# Default to all packages
if [ ${#PACKAGES[@]} -eq 0 ]; then
    PACKAGES=(core forge quorum)
fi

if $JSON_OUTPUT; then
    mkdir -p "$REPO_ROOT/.benchmarks"
fi

BENCH_ARGS="--benchmark-only --benchmark-disable-gc --benchmark-warmup=on -q"
FAILED=0

run_package() {
    local pkg="$1"
    local json_arg=""
    local json_file=""

    if $JSON_OUTPUT; then
        json_file="$REPO_ROOT/.benchmarks/${pkg}-${TIMESTAMP}.json"
        json_arg="--benchmark-json=$json_file"
    fi

    echo "════════════════════════════════════════════════════"
    echo "  Benchmarking: $pkg"
    echo "════════════════════════════════════════════════════"

    case "$pkg" in
        core)
            local venv="$REPO_ROOT/packages/core/.venv/bin/activate"
            if [ -f "$venv" ]; then
                # shellcheck disable=SC1090
                source "$venv"
            fi
            pytest "$REPO_ROOT/packages/core/tests/test_benchmarks.py" $BENCH_ARGS $json_arg || return 1
            ;;
        forge)
            local venv="$REPO_ROOT/packages/forge/.venv/bin/activate"
            if [ -f "$venv" ]; then
                # shellcheck disable=SC1090
                source "$venv"
            fi
            # Forge MUST run from its directory (relative paths for skills/workflows)
            (cd "$REPO_ROOT/packages/forge" && pytest tests/test_benchmarks.py $BENCH_ARGS $json_arg) || return 1
            ;;
        quorum)
            local venv="$REPO_ROOT/packages/quorum/.venv/bin/activate"
            if [ -f "$venv" ]; then
                # shellcheck disable=SC1090
                source "$venv"
            fi
            PYTHONPATH="$REPO_ROOT/packages/quorum/python" pytest "$REPO_ROOT/packages/quorum/tests/test_benchmarks.py" $BENCH_ARGS $json_arg || return 1
            ;;
    esac

    if $JSON_OUTPUT && [ -n "$json_file" ] && [ -f "$json_file" ]; then
        echo "  → JSON saved: $json_file"
    fi
    echo ""
}

for pkg in "${PACKAGES[@]}"; do
    if ! run_package "$pkg"; then
        echo "  ✗ $pkg benchmarks failed"
        FAILED=1
    fi
done

if $JSON_OUTPUT; then
    echo "Benchmark results in: $REPO_ROOT/.benchmarks/"
fi

exit $FAILED
