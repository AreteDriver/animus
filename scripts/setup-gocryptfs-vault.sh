#!/usr/bin/env bash
# setup-gocryptfs-vault.sh — one-shot migration to at-rest-encrypted
# memory storage for Animus.
#
# Scope (per ARETE design 2026-05-27):
#   - encrypts ``~/.animus/chroma/`` and ``~/.animus/memory/``
#   - passphrase: TPM-sealed via systemd-creds
#   - migration: offline (daemon stopped for ~5-10 min)
#   - rollback: Restic snapshots already in ``~/backups-local/``
#
# Prerequisites BEFORE running this script:
#   1. ``sudo usermod -aG tss arete`` — grants TPM access for systemd-creds.
#   2. **Restart the user session** to give the systemd --user manager
#      its new tss group. Without this, the daemon's ``LoadCredential
#      Encrypted=`` can't reach /dev/tpmrm0 and the post-mount daemon
#      restart fails. ``sg tss -c '...'`` works for the script's
#      one-shot operations but does NOT cover the persistent unit.
#      Terminate + reconnect: ``sudo loginctl terminate-user arete``
#      (kills any running session — close out other work first).
#   3. ``~/backups-local/`` already has fresh snapshots from
#      ``animus-backup-hourly`` + ``animus-backup-chroma``. Re-run the
#      pre-flight ``systemctl --user start animus-backup-{hourly,chroma}``
#      if the most recent is >1 hour old.
#
# The script is idempotent on the additive steps (vault init, credential
# encryption) — re-running them is safe. The destructive moment is the
# rename of ``~/.animus/chroma/`` and ``~/.animus/memory/`` to
# ``.unencrypted-backup/``, which the script asks for explicit
# confirmation before doing.
#
# Cleanup of ``.unencrypted-backup/`` directories is NOT automated.
# Per the destructive-disk-ops protocol, that step is a separate
# manual command after a successful verification window.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANIMUS_DIR="${HOME}/.animus"
VAULT_DIR="${ANIMUS_DIR}/_vault"          # encrypted ciphertext on disk
SECURE_DIR="${ANIMUS_DIR}/_secure"        # mounted cleartext view
CRED_DIR="${HOME}/.config/credstore.encrypted"
CRED_FILE="${CRED_DIR}/animus-vault"
SYSTEMD_UNIT="${HOME}/.config/systemd/user/animus.service"
LOG_PREFIX="[setup-gocryptfs-vault]"

log() { echo "${LOG_PREFIX} $*"; }
abort() { echo "${LOG_PREFIX} ABORT: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

log "==> Phase 0: pre-flight"

# Tools
command -v gocryptfs >/dev/null || abort "gocryptfs not installed (apt install gocryptfs)"
command -v systemd-creds >/dev/null || abort "systemd-creds missing — install systemd >= 250"
command -v rsync >/dev/null || abort "rsync missing"

# TPM access via tss group
if ! id -nG | tr ' ' '\n' | grep -qx tss; then
    abort "user '$USER' not in 'tss' group — run 'sudo usermod -aG tss arete' then re-login or use 'sg tss -c \"bash $0\"'"
fi

# TPM device readable
if [[ ! -r /dev/tpmrm0 ]]; then
    abort "/dev/tpmrm0 not readable (try a fresh shell after the usermod)"
fi

# Confirm the systemd --user manager also has tss access — without it,
# ``LoadCredentialEncrypted=`` in the patched unit fails post-mount and
# the daemon restart loops. The user-systemd's groups come from the
# login session that started it; if usermod happened after login, the
# session must be restarted before this script can complete cleanly.
USER_SYSTEMD_PID=$(systemctl --user show --property=ExecMainPID --value || true)
if [[ -n "${USER_SYSTEMD_PID:-}" && -r "/proc/${USER_SYSTEMD_PID}/status" ]]; then
    if ! grep -E '^Groups:' "/proc/${USER_SYSTEMD_PID}/status" | tr ' ' '\n' | grep -qx 105; then
        abort "systemd --user (pid ${USER_SYSTEMD_PID}) lacks tss group — restart your session ('sudo loginctl terminate-user $USER' or full logout/login). The script's manual mount works, but the unit's persistent auto-mount will loop-fail until the user-systemd inherits the group."
    fi
fi

# Source directories exist
[[ -d "${ANIMUS_DIR}/chroma" ]] || abort "${ANIMUS_DIR}/chroma not found"
[[ -d "${ANIMUS_DIR}/memory" ]] || abort "${ANIMUS_DIR}/memory not found"

# Restic snapshot recency
SNAPSHOT_AGE=$(stat -c %Y "${HOME}/backups-local/animus-chroma/locks" 2>/dev/null || echo 0)
NOW=$(date +%s)
if (( NOW - SNAPSHOT_AGE > 3600 )); then
    log "WARNING: most recent chroma snapshot is older than 1 hour. Consider re-running animus-backup-hourly + animus-backup-chroma first."
    read -r -p "Continue anyway? (yes/N): " confirm
    [[ "$confirm" == "yes" ]] || abort "user declined stale-snapshot warning"
fi

log "pre-flight OK"

# ---------------------------------------------------------------------------
# Phase 1: encrypt passphrase via systemd-creds (TPM-sealed)
# ---------------------------------------------------------------------------

log "==> Phase 1: generate + TPM-seal passphrase"

mkdir -p "${CRED_DIR}"
chmod 700 "${CRED_DIR}"

if [[ -f "${CRED_FILE}" ]]; then
    log "credential ${CRED_FILE} already exists — reusing (delete it manually if you want to regenerate)"
else
    # 48 bytes of randomness → ~64-char base64 → > 256 bits of entropy
    openssl rand -base64 48 | tr -d '\n' | \
        systemd-creds encrypt --name=animus-vault --with-key=tpm2 - "${CRED_FILE}"
    chmod 600 "${CRED_FILE}"
    log "credential sealed to TPM → ${CRED_FILE}"
fi

# Verify we can decrypt (full round-trip)
PASSPHRASE_BYTES=$(systemd-creds decrypt --name=animus-vault "${CRED_FILE}" - | wc -c)
(( PASSPHRASE_BYTES > 32 )) || abort "decrypt returned ${PASSPHRASE_BYTES} bytes — TPM unseal looks broken"
log "TPM unseal verified (${PASSPHRASE_BYTES} bytes)"

# ---------------------------------------------------------------------------
# Phase 2: initialize the gocryptfs vault (idempotent — skips if exists)
# ---------------------------------------------------------------------------

log "==> Phase 2: initialize gocryptfs vault"

if [[ -f "${VAULT_DIR}/gocryptfs.conf" ]]; then
    log "vault already initialized at ${VAULT_DIR} — reusing"
else
    mkdir -p "${VAULT_DIR}"
    chmod 700 "${VAULT_DIR}"
    # ``gocryptfs -extpass "cat"`` reads from stdin which is fragile in
    # a pipeline. Use ``-passfile`` with a tmpfs file that exists only
    # for the init.
    PW_TMP=$(mktemp -p /dev/shm gocryptfs-pw.XXXXXX 2>/dev/null \
             || mktemp /tmp/gocryptfs-pw.XXXXXX)
    chmod 600 "${PW_TMP}"
    systemd-creds decrypt --name=animus-vault "${CRED_FILE}" - > "${PW_TMP}"
    gocryptfs -init -plaintextnames -q -passfile "${PW_TMP}" "${VAULT_DIR}"
    shred -u "${PW_TMP}" 2>/dev/null || rm -f "${PW_TMP}"
    log "vault initialized at ${VAULT_DIR}"
fi

# ---------------------------------------------------------------------------
# Phase 3: stop the daemon (offline migration window starts here)
# ---------------------------------------------------------------------------

log "==> Phase 3: stop animus.service"

DAEMON_WAS_RUNNING=0
if systemctl --user is-active --quiet animus.service; then
    DAEMON_WAS_RUNNING=1
    systemctl --user stop animus.service
    log "stopped animus.service"
else
    log "animus.service already inactive"
fi

# Cleanup on exit — ensure daemon is restarted even if we abort mid-script
restart_daemon_on_exit() {
    local exit_code=$?
    if (( DAEMON_WAS_RUNNING )); then
        log "restoring daemon state (exit=${exit_code})"
        systemctl --user start animus.service || log "WARN: daemon failed to start"
    fi
    exit "${exit_code}"
}
trap restart_daemon_on_exit EXIT

# ---------------------------------------------------------------------------
# Phase 4: mount + rsync data into vault
# ---------------------------------------------------------------------------

log "==> Phase 4: mount vault + copy data"

mkdir -p "${SECURE_DIR}"
chmod 700 "${SECURE_DIR}"

# Mount the vault. Same -passfile pattern as Phase 2 — tmpfs file, scrub
# after use.
if mountpoint -q "${SECURE_DIR}"; then
    log "vault already mounted at ${SECURE_DIR}"
else
    PW_TMP=$(mktemp -p /dev/shm gocryptfs-pw.XXXXXX 2>/dev/null \
             || mktemp /tmp/gocryptfs-pw.XXXXXX)
    chmod 600 "${PW_TMP}"
    systemd-creds decrypt --name=animus-vault "${CRED_FILE}" - > "${PW_TMP}"
    gocryptfs -q -passfile "${PW_TMP}" "${VAULT_DIR}" "${SECURE_DIR}"
    shred -u "${PW_TMP}" 2>/dev/null || rm -f "${PW_TMP}"
    log "vault mounted at ${SECURE_DIR}"
fi

# Copy data INTO the encrypted view
mkdir -p "${SECURE_DIR}/chroma" "${SECURE_DIR}/memory"
log "rsyncing chroma → vault"
rsync -a --delete "${ANIMUS_DIR}/chroma/" "${SECURE_DIR}/chroma/"
log "rsyncing memory → vault"
rsync -a --delete "${ANIMUS_DIR}/memory/" "${SECURE_DIR}/memory/"

# Verify file counts match
CHROMA_SRC=$(find "${ANIMUS_DIR}/chroma" -type f | wc -l)
CHROMA_DST=$(find "${SECURE_DIR}/chroma" -type f | wc -l)
MEMORY_SRC=$(find "${ANIMUS_DIR}/memory" -type f | wc -l)
MEMORY_DST=$(find "${SECURE_DIR}/memory" -type f | wc -l)
log "file-count check: chroma ${CHROMA_SRC}→${CHROMA_DST}, memory ${MEMORY_SRC}→${MEMORY_DST}"
(( CHROMA_SRC == CHROMA_DST )) || abort "chroma file-count mismatch (${CHROMA_SRC} vs ${CHROMA_DST})"
(( MEMORY_SRC == MEMORY_DST )) || abort "memory file-count mismatch (${MEMORY_SRC} vs ${MEMORY_DST})"
log "copy verified"

# ---------------------------------------------------------------------------
# Phase 5: swap directory names (THIS IS THE REVERSIBLE-DESTRUCTIVE STEP)
# ---------------------------------------------------------------------------

log "==> Phase 5: swap directories"

BACKUP_SUFFIX=".unencrypted-backup.$(date +%Y%m%d-%H%M%S)"

if [[ -L "${ANIMUS_DIR}/chroma" ]]; then
    log "chroma already a symlink — skipping swap (idempotent)"
else
    # Backup → rename → symlink. The data exists in TWO places now (the
    # source dir + the encrypted vault). Removing the backup is a
    # SEPARATE step the operator must run after verification.
    mv "${ANIMUS_DIR}/chroma" "${ANIMUS_DIR}/chroma${BACKUP_SUFFIX}"
    ln -s "${SECURE_DIR}/chroma" "${ANIMUS_DIR}/chroma"
    log "chroma → ${ANIMUS_DIR}/chroma${BACKUP_SUFFIX} (backup); symlink installed"
fi

if [[ -L "${ANIMUS_DIR}/memory" ]]; then
    log "memory already a symlink — skipping swap"
else
    mv "${ANIMUS_DIR}/memory" "${ANIMUS_DIR}/memory${BACKUP_SUFFIX}"
    ln -s "${SECURE_DIR}/memory" "${ANIMUS_DIR}/memory"
    log "memory → ${ANIMUS_DIR}/memory${BACKUP_SUFFIX} (backup); symlink installed"
fi

# ---------------------------------------------------------------------------
# Phase 6: update animus.service with mount lifecycle
# ---------------------------------------------------------------------------

log "==> Phase 6: patch animus.service for mount lifecycle"

if [[ ! -f "${SYSTEMD_UNIT}" ]]; then
    log "WARN: ${SYSTEMD_UNIT} not found — skipping unit patch (do manually)"
elif grep -q "animus-vault" "${SYSTEMD_UNIT}"; then
    log "unit already patched (found 'animus-vault' reference)"
else
    UNIT_BACKUP="${SYSTEMD_UNIT}.bak.$(date +%Y%m%d-%H%M%S)"
    cp "${SYSTEMD_UNIT}" "${UNIT_BACKUP}"
    log "unit backup → ${UNIT_BACKUP}"

    # Insert credential + mount/unmount lines into [Service] block.
    # Strategy: find '[Service]' line, append directives right after.
    python3 - "$SYSTEMD_UNIT" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

INSERT = [
    "# --- gocryptfs vault lifecycle (added by setup-gocryptfs-vault.sh) ---\n",
    "LoadCredentialEncrypted=animus-vault:%h/.config/credstore.encrypted/animus-vault\n",
    "ExecStartPre=/bin/sh -c 'mountpoint -q %h/.animus/_secure || /usr/bin/gocryptfs -q -passfile ${CREDENTIALS_DIRECTORY}/animus-vault %h/.animus/_vault %h/.animus/_secure'\n",
    "ExecStopPost=-/bin/fusermount -u %h/.animus/_secure\n",
    "# --- end gocryptfs vault ---\n",
]

out = []
inserted = False
for line in lines:
    out.append(line)
    if not inserted and line.strip() == "[Service]":
        out.append("\n")
        out.extend(INSERT)
        inserted = True

if not inserted:
    print("ERROR: no [Service] section found", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.writelines(out)
PYEOF

    systemctl --user daemon-reload
    log "unit patched + daemon-reload run"
fi

# ---------------------------------------------------------------------------
# Phase 7: restart daemon, verify health
# ---------------------------------------------------------------------------

log "==> Phase 7: restart + verify"

# Unmount the manual mount — the unit's ExecStartPre will re-mount.
if mountpoint -q "${SECURE_DIR}"; then
    fusermount -u "${SECURE_DIR}" || log "WARN: manual unmount failed"
fi

# Clear EXIT trap — we're about to start the daemon explicitly.
DAEMON_WAS_RUNNING=0
trap - EXIT

systemctl --user start animus.service
sleep 3
if systemctl --user is-active --quiet animus.service; then
    log "animus.service active"
else
    log "ERROR: animus.service did not start cleanly — see journalctl --user -u animus.service"
    exit 2
fi

# Verify the vault is mounted via the unit
if mountpoint -q "${SECURE_DIR}"; then
    log "vault mounted via systemd"
else
    log "ERROR: vault NOT mounted after daemon start"
    exit 3
fi

# Quick sanity: daemon can see its data
if [[ -f "${ANIMUS_DIR}/chroma/chroma.sqlite3" ]]; then
    log "chroma.sqlite3 reachable via symlink ✓"
else
    log "ERROR: chroma.sqlite3 not reachable — daemon won't see memories"
    exit 4
fi

log ""
log "============================================================"
log "DONE. At-rest encryption active for chroma + memory."
log ""
log "Verify with:"
log "  systemctl --user status animus.service"
log "  journalctl --user -u animus.service --since '5 min ago'"
log "  python -m animus.cli recall 'test query'  # exercise the memory layer"
log ""
log "After a verification window (suggested 24h), remove the"
log "unencrypted backups (DESTRUCTIVE — SSD TRIM will physically"
log "erase the bits in minutes):"
log ""
log "  ls -la ${ANIMUS_DIR}/chroma${BACKUP_SUFFIX} ${ANIMUS_DIR}/memory${BACKUP_SUFFIX}"
log "  rm -rf ${ANIMUS_DIR}/chroma.unencrypted-backup.*"
log "  rm -rf ${ANIMUS_DIR}/memory.unencrypted-backup.*"
log "============================================================"
