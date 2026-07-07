#!/bin/bash
# Show Animus autonomous loop status — timers, last run, proposals queued

echo "=== systemd Timers ==="
systemctl list-timers --all | grep -E "animus|NEXT"

echo ""
echo "=== Last 20 lines from all Animus services ==="
journalctl -u 'animus-autonomous*' --no-pager -n 20

echo ""
echo "=== Pending Proposals ==="
PYTHONPATH="packages/core:packages/kernel/src:packages/types/src" \
  python3 -c "
from animus.citizens.architect import ArchitectCitizen
a = ArchitectCitizen()
proposals = a.list_pending_proposals()
print(f'Total pending: {len(proposals)}')
for p in proposals[:5]:
    print(f'  - {p.id}: {p.title[:60]} (priority={p.priority}, confidence={p.confidence})')
" 2>/dev/null || echo "Memory layer not available — proposals not persisted until daemon mode active."
