#!/usr/bin/env bash
# Connect Animus to a running Gorgon instance.
# Usage: ./integrate-gorgon.sh [gorgon-url]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
GORGON_URL="${1:-http://localhost:8000}"

# Kill any existing Animus process
pkill -f "python -m animus" 2>/dev/null || true
pkill -f "python.*animus/__main__" 2>/dev/null || true
sleep 1

# Activate venv
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install -e "${SCRIPT_DIR}[gorgon]" --quiet
else
    source "${VENV_DIR}/bin/activate"
fi

# Verify httpx is installed
python -c "import httpx" 2>/dev/null || {
    echo "Installing httpx..."
    pip install httpx --quiet
}

# Check if Gorgon is reachable
echo "Checking Gorgon at ${GORGON_URL}..."
if python -c "
import httpx, sys
try:
    r = httpx.get('${GORGON_URL}/health', timeout=5.0)
    r.raise_for_status()
    print(f'  Gorgon healthy: {r.json()}')
except Exception as e:
    print(f'  Gorgon not reachable: {e}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "  Connection verified."
else
    echo ""
    echo "Gorgon is not running at ${GORGON_URL}."
    echo "Start it first:"
    echo "  cd /home/arete/projects/Gorgon && ALLOW_DEMO_AUTH=true ./run_api.sh"
    echo ""
    echo "Or specify a custom URL:"
    echo "  ./integrate-gorgon.sh http://host:port"
    exit 1
fi

# Authenticate with Gorgon
echo "Authenticating with Gorgon..."
GORGON_API_KEY=$(python -c "
import httpx, sys, os
url = '${GORGON_URL}'
user = os.environ.get('GORGON_USER', 'animus')
password = os.environ.get('GORGON_PASSWORD', 'demo')
try:
    r = httpx.post(f'{url}/v1/auth/login', json={'user_id': user, 'password': password}, timeout=5.0)
    r.raise_for_status()
    print(r.json()['access_token'])
except Exception as e:
    print(f'Auth failed: {e}', file=sys.stderr)
    sys.exit(1)
") || {
    echo ""
    echo "Authentication failed. Options:"
    echo "  1. Start Gorgon with ALLOW_DEMO_AUTH=true"
    echo "  2. Set GORGON_USER and GORGON_PASSWORD env vars"
    exit 1
}
echo "  Authenticated as ${GORGON_USER:-animus}"

# Set env vars and launch Animus
export GORGON_ENABLED=true
export GORGON_URL="${GORGON_URL}"
export GORGON_API_KEY="${GORGON_API_KEY}"

echo ""
echo "Launching Animus with Gorgon integration..."
echo "  GORGON_URL=${GORGON_URL}"
echo "  GORGON_API_KEY=${GORGON_API_KEY:0:20}..."
echo "  auto_delegate=true"
echo ""

exec python -m animus "$@"
