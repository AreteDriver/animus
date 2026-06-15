#!/usr/bin/env python3
"""Quick health check for animus-kernel imports."""

import sys

MODULES = [
    ("animus_kernel.budget", "BudgetManager"),
    ("animus_kernel.providers", "ProviderRouter"),
    ("animus_kernel.sandbox", "SelfImproveOrchestrator"),
    ("animus_kernel.agents", "SupervisorAgent"),
    ("animus_kernel.executor", "WorkflowExecutor"),
    ("animus_kernel.contracts", "ContractValidator"),
    ("animus_kernel.memory", "MemoryTier"),
    ("animus_kernel.coordination", "WorkflowEvolution"),
    ("animus_kernel.intelligence", "CostIntelligence"),
]

failures = 0
for mod, name in MODULES:
    try:
        obj = getattr(__import__(mod, fromlist=[name]), name)
        print(f"  ✅ {name}")
    except Exception as exc:
        print(f"  ❌ {name}: {exc}")
        failures += 1

if failures:
    print(f"\n{failures}/{len(MODULES)} imports failed")
    sys.exit(1)
else:
    print(f"\nAll {len(MODULES)} critical imports passed.")
    sys.exit(0)
