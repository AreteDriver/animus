#!/usr/bin/env bash
# Launch Animus CLI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install -e "${SCRIPT_DIR}[gorgon]" --quiet
else
    source "${VENV_DIR}/bin/activate"
fi

exec python -m animus "$@"
