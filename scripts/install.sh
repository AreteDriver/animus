#!/usr/bin/env bash
# Animus one-command installer
# Detects OS, checks prerequisites, installs from git, and initializes.

set -euo pipefail

REPO_URL="https://github.com/your-org/animus.git"
INSTALL_DIR="${ANIMUS_INSTALL_DIR:-$HOME/projects/animus}"
OLLAMA_URL="http://localhost:11434"

red() { printf '\033[0;31m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
info() { printf '\033[0;34m%s\033[0m\n' "$*"; }

check_python() {
    if ! command -v python3 &>/dev/null; then
        red "Python 3 is required but not installed."
        exit 1
    fi

    local py_version
    py_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    local major minor
    major=$(echo "$py_version" | cut -d. -f1)
    minor=$(echo "$py_version" | cut -d. -f2)

    if [[ "$major" -lt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -lt 10 ]]; }; then
        red "Python 3.10+ required. Found: $py_version"
        exit 1
    fi

    green "✓ Python $py_version"
}

check_ollama() {
    if ! command -v ollama &>/dev/null; then
        yellow "Ollama not found. Install it first:"
        info "  curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi

    if ! curl -s "$OLLAMA_URL" &>/dev/null; then
        yellow "Ollama is installed but not running. Start it:"
        info "  ollama serve"
        exit 1
    fi

    green "✓ Ollama running on $OLLAMA_URL"
}

check_os() {
    local os
    os=$(uname -s)
    case "$os" in
        Linux)
            green "✓ Linux detected"
            ;;
        Darwin)
            yellow "macOS detected — support is on the roadmap. Proceed at your own risk."
            ;;
        *)
            red "Unsupported OS: $os. Animus requires Linux."
            exit 1
            ;;
    esac
}

install_animus() {
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        yellow "Animus already cloned at $INSTALL_DIR. Pulling latest..."
        git -C "$INSTALL_DIR" pull --ff-only
    else
        info "Cloning Animus into $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
    fi

    cd "$INSTALL_DIR"

    info "Installing core package..."
    pip install -e packages/core

    info "Installing additional packages (optional)..."
    pip install -e packages/types || true
    pip install -e packages/kernel || true
    pip install -e packages/forge || true
    pip install -e packages/quorum || true
    pip install -e packages/bootstrap || true
}

init_animus() {
    info "Initializing Animus..."
    animus init || true
}

main() {
    echo ""
    info "═══════════════════════════════════════════════════════"
    info "  Animus Installer — Local-First AI Exocortex"
    info "═══════════════════════════════════════════════════════"
    echo ""

    check_os
    check_python
    check_ollama
    install_animus
    init_animus

    echo ""
    green "═══════════════════════════════════════════════════════"
    green "  Animus installed successfully!"
    green "═══════════════════════════════════════════════════════"
    echo ""
    info "Next steps:"
    info "  animus brief      → Get a situation briefing"
    info "  animus chat       → Start a conversation"
    info "  animus status     → Check system health"
    info ""
    info "Docs: $INSTALL_DIR/README_PUBLIC.md"
    echo ""
}

main "$@"
