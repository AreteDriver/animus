# Kernel File Map

Maps Tier 1 (engine) source files to their new homes in `packages/kernel`.
Anything **not** listed here is considered Tier 2 (shell) or Tier 3 (waste) and stays behind.

## Legend
- ✅ Copied & adapted
- 🔄 Pending adaptation
- ⏳ Not yet copied
- ❌ Explicitly excluded (shell/waste)

---

## budget/ → kernel/budget/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/budget/manager.py` | `kernel/budget/manager.py` | ⏳ | BudgetManager, UsageRecord, atomic reservation |
| `packages/forge/src/animus_forge/budget/models.py` | `kernel/budget/models.py` | ⏳ | BudgetStatus, CostAudit, etc. |
| `packages/forge/src/animus_forge/budget/persistence.py` | `kernel/budget/persistence.py` | ⏳ | BudgetStore, SQLite/json |
| `packages/forge/src/animus_forge/budget/strategies.py` | `kernel/budget/strategies.py` | ⏳ | ProviderCostStrategy |
| `packages/forge/src/animus_forge/budget/preflight.py` | `kernel/budget/preflight.py` | ⏳ | Preflight check |
| `packages/forge/src/animus_forge/budget/cost_audit.py` | `kernel/budget/cost_audit.py` | ⏳ | Cost audit logic |
| `packages/forge/src/animus_forge/budget/__init__.py` | `kernel/budget/__init__.py` | ⏳ | Exports |
| `packages/core/animus/forge/budget.py` | `kernel/budget/core_bridge.py` | ⏳ | Core budget bridge |

---

## executor/ → kernel/executor/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/workflow/executor_core.py` | `kernel/executor/executor_core.py` | ⏳ | Main WorkflowExecutor class |
| `packages/forge/src/animus_forge/workflow/executor.py` | `kernel/executor/executor.py` | ⏳ | Legacy executor wrapper |
| `packages/forge/src/animus_forge/workflow/executor_step.py` | `kernel/executor/executor_step.py` | ⏳ | Step execution logic |
| `packages/forge/src/animus_forge/workflow/executor_results.py` | `kernel/executor/executor_results.py` | ⏳ | Result aggregation |
| `packages/forge/src/animus_forge/workflow/executor_error.py` | `kernel/executor/executor_error.py` | ⏳ | Error taxonomy, retries |
| `packages/forge/src/animus_forge/workflow/parallel.py` | `kernel/executor/parallel.py` | ⏳ | Parallel step runner |
| `packages/forge/src/animus_forge/workflow/scheduler.py` | `kernel/executor/scheduler.py` | ⏳ | Delayed/rescheduled jobs |
| `packages/forge/src/animus_forge/workflow/loader.py` | `kernel/executor/loader.py` | ⏳ | Workflow yaml/json loader |
| `packages/forge/src/animus_forge/workflow/versioning.py` | `kernel/executor/versioning.py` | ⏳ | Version pinning |
| `packages/forge/src/animus_forge/workflow/version_manager.py` | `kernel/executor/version_manager.py` | ⏳ | Semantic version ops |
| `packages/forge/src/animus_forge/workflow/graph_executor.py` | `kernel/executor/graph_executor.py` | ⏳ | DAG execution |
| `packages/forge/src/animus_forge/workflow/graph_models.py` | `kernel/executor/graph_models.py` | ⏳ | Graph dataclasses |
| `packages/forge/src/animus_forge/workflow/graph_walker.py` | `kernel/executor/graph_walker.py` | ⏳ | Topological walk |
| `packages/forge/src/animus_forge/workflow/rate_limited_executor.py` | `kernel/executor/rate_limited_executor.py` | ⏳ | Token-bucket rate limiting |
| `packages/core/animus/forge/engine.py` | `kernel/executor/core_engine.py` | ⏳ | Core engine bridge |
| `packages/core/animus/forge/checkpoint.py` | `kernel/executor/checkpoint.py` | ⏳ | Checkpoint/resume |

---

## sandbox/ → kernel/sandbox/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/self_improve/sandbox.py` | `kernel/sandbox/sandbox.py` | ⏳ | Clone → apply → test → lint |
| `packages/forge/src/animus_forge/self_improve/safety.py` | `kernel/sandbox/safety.py` | ⏳ | File/line limits, protected patterns |
| `packages/forge/src/animus_forge/self_improve/approval.py` | `kernel/sandbox/approval.py` | ⏳ | Approval gates |
| `packages/forge/src/animus_forge/self_improve/rollback.py` | `kernel/sandbox/rollback.py` | ⏳ | Git-based rollback |
| `packages/forge/src/animus_forge/self_improve/pr_manager.py` | `kernel/sandbox/pr_manager.py` | ⏳ | PR creation via gh cli |
| `packages/forge/src/animus_forge/self_improve/orchestrator.py` | `kernel/sandbox/orchestrator.py` | ⏳ | 10-stage self-improve loop |
| `packages/forge/src/animus_forge/self_improve/analyzer.py` | `kernel/sandbox/analyzer.py` | ⏳ | ImprovementCategory, CodebaseAnalyzer |

---

## providers/ → kernel/providers/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/providers/base.py` | `kernel/providers/base.py` | ⏳ | ProviderType, ModelTier, CompletionRequest |
| `packages/forge/src/animus_forge/providers/router.py` | `kernel/providers/router.py` | ⏳ | ProviderRouter, tier selection |
| `packages/forge/src/animus_forge/providers/manager.py` | `kernel/providers/manager.py` | ⏳ | Provider lifecycle |
| `packages/forge/src/animus_forge/providers/anthropic_provider.py` | `kernel/providers/anthropic_provider.py` | ⏳ | Anthropic API |
| `packages/forge/src/animus_forge/providers/openai_provider.py` | `kernel/providers/openai_provider.py` | ⏳ | OpenAI API |
| `packages/forge/src/animus_forge/providers/azure_openai_provider.py` | `kernel/providers/azure_openai_provider.py` | ⏳ | Azure OpenAI |
| `packages/forge/src/animus_forge/providers/bedrock_provider.py` | `kernel/providers/bedrock_provider.py` | ⏳ | AWS Bedrock |
| `packages/forge/src/animus_forge/providers/vertex_provider.py` | `kernel/providers/vertex_provider.py` | ⏳ | Google Vertex |
| `packages/forge/src/animus_forge/providers/ollama_provider.py` | `kernel/providers/ollama_provider.py` | ⏳ | Ollama (local) |
| `packages/forge/src/animus_forge/providers/llamacpp_provider.py` | `kernel/providers/llamacpp_provider.py` | ⏳ | llama.cpp |
| `packages/forge/src/animus_forge/providers/openrouter_provider.py` | `kernel/providers/openrouter_provider.py` | ⏳ | OpenRouter |
| `packages/forge/src/animus_forge/providers/hardware.py` | `kernel/providers/hardware.py` | ⏳ | GPU detection |
| `packages/forge/src/animus_forge/providers/model_pin.py` | `kernel/providers/model_pin.py` | ⏳ | Model pinning |
| `packages/forge/src/animus_forge/providers/mock_provider.py` | `kernel/providers/mock_provider.py` | ⏳ | Test double |

---

## agents/ → kernel/agents/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/agents/supervisor.py` | `kernel/agents/supervisor.py` | ⏳ | SupervisorAgent, 6-role delegation |
| `packages/forge/src/animus_forge/agents/task_runner.py` | `kernel/agents/task_runner.py` | ⏳ | Runs individual agent tasks |
| `packages/forge/src/animus_forge/agents/subagent_manager.py` | `kernel/agents/subagent_manager.py` | ⏳ | Spawns/manages subagents |
| `packages/forge/src/animus_forge/agents/provider_wrapper.py` | `kernel/agents/provider_wrapper.py` | ⏳ | Wrap provider for agents |
| `packages/forge/src/animus_forge/agents/agent_config.py` | `kernel/agents/agent_config.py` | ⏳ | Config dataclasses |
| `packages/forge/src/animus_forge/agents/config_loader.py` | `kernel/agents/config_loader.py` | ⏳ | Load agent_prompts.json |
| `packages/forge/src/animus_forge/agents/process_registry.py` | `kernel/agents/process_registry.py` | ⏳ | Track running processes |

---

## coordination/ → kernel/coordination/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/coordination/workflow_evolution.py` | `kernel/coordination/workflow_evolution.py` | ⏳ | IntentGraph mutation |
| `packages/forge/src/animus_forge/coordination/auto_promote.py` | `kernel/coordination/auto_promote.py` | ⏳ | Auto-promotion logic |
| `packages/forge/src/animus_forge/coordination/evolution_loop.py` | `kernel/coordination/evolution_loop.py` | ⏳ | Evolutionary pressure |
| `packages/core/animus/forge/models.py` | `kernel/coordination/models.py` | ⏳ | Core models bridge |
| `packages/core/animus/swarm/intent.py` | `kernel/coordination/intent.py` | ⏳ | InterfaceKind, InterfaceSpec |
| `packages/core/animus/swarm/models.py` | `kernel/coordination/swarm_models.py` | ⏳ | Swarm dataclasses |

**Excluded:**
- `consciousness_bridge.py`, `identity_anchor.py`, `identity_patch.py` → ❌ shell/personality

---

## memory/ → kernel/memory/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/core/animus/memory/tier.py` | `kernel/memory/tier.py` | ⏳ | HOT/WARM/COLD tiering |
| `packages/core/animus/memory/types.py` | `kernel/memory/types.py` | ⏳ | Memory types, constraints |
| `packages/core/animus/memory/stores/base.py` | `kernel/memory/stores/base.py` | ⏳ | Abstract store |
| `packages/core/animus/memory/stores/local.py` | `kernel/memory/stores/local.py` | ⏳ | Local JSON/SQLite store |
| `packages/core/animus/memory/fusion.py` | `kernel/memory/fusion.py` | ⏳ | Memory consolidation |
| `packages/core/animus/memory/evaluation.py` | `kernel/memory/evaluation.py` | ⏳ | Retrieval quality eval |
| `packages/forge/src/animus_forge/intelligence/cross_workflow_memory.py` | `kernel/memory/cross_workflow.py` | ⏳ | Cross-workflow recall |

---

## intelligence/ → kernel/intelligence/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/intelligence/cost_intelligence.py` | `kernel/intelligence/cost_intelligence.py` | ⏳ | Learn from cost outcomes |
| `packages/forge/src/animus_forge/intelligence/outcome_tracker.py` | `kernel/intelligence/outcome_tracker.py` | ⏳ | Success/failure tracking |
| `packages/forge/src/animus_forge/intelligence/feedback_engine.py` | `kernel/intelligence/feedback_engine.py` | ⏳ | Feedback loops |
| `packages/forge/src/animus_forge/intelligence/prompt_evolution.py` | `kernel/intelligence/prompt_evolution.py` | ⏳ | Prompt mutation |
| `packages/forge/src/animus_forge/intelligence/provider_router.py` | `kernel/intelligence/provider_router.py` | ⏳ | ML-based provider selection |

---

## contracts/ → kernel/contracts/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/contracts/base.py` | `kernel/contracts/base.py` | ⏳ | Base contract |
| `packages/forge/src/animus_forge/contracts/definitions.py` | `kernel/contracts/definitions.py` | ⏳ | Contract schemas |
| `packages/forge/src/animus_forge/contracts/enforcer.py` | `kernel/contracts/enforcer.py` | ⏳ | Runtime enforcement |
| `packages/forge/src/animus_forge/contracts/validator.py` | `kernel/contracts/validator.py` | ⏳ | Validation logic |

---

## resilience/ → kernel/resilience/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/resilience/bulkhead.py` | `kernel/resilience/bulkhead.py` | ⏳ | Bulkhead pattern |
| `packages/forge/src/animus_forge/resilience/fallback.py` | `kernel/resilience/fallback.py` | ⏳ | Fallback chains |
| `packages/forge/src/animus_forge/resilience/concurrency.py` | `kernel/resilience/concurrency.py` | ⏳ | Semaphore/limits |

---

## ratelimit/ → kernel/ratelimit/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/forge/src/animus_forge/ratelimit/limiter.py` | `kernel/ratelimit/limiter.py` | ⏳ | Token bucket |
| `packages/forge/src/animus_forge/ratelimit/provider.py` | `kernel/ratelimit/provider.py` | ⏳ | Rate-limit adapters |
| `packages/forge/src/animus_forge/ratelimit/quota.py` | `kernel/ratelimit/quota.py` | ⏳ | Quota models |

---

## safety/ → kernel/safety/

| Source | Destination | Status | Notes |
|---|---|---|---|
| `packages/core/animus/protocols/safety.py` | `kernel/safety/protocols.py` | ⏳ | Safety protocol ABC |
| `packages/forge/src/animus_forge/tools/safety.py` | `kernel/safety/tool_safety.py` | ⏳ | Tool call safety checks |
| `packages/forge/src/animus_forge/security/pi_wrap.py` | `kernel/safety/pii.py` | ⏳ | PII detection/redaction |

---

## Summary

| Category | Count | Approx LOC |
|---|---|---|
| Tier 1 (engine → kernel) | ~170 files | ~55,000 LOC |
| Tier 2 (shell, stays) | ~70 files | ~20,000 LOC |
| Tier 3 (waste, cut) | ~30 files | ~5,000 LOC |
| **Total** | **~270 files** | **~80,000 LOC** |

### Extraction Status (2026-06-14)

✅ **Kernel is importable and standalone**
- All 9 critical engine modules import successfully at runtime
- `pip install -e packages/kernel/` ready (needs venv)
- 172 Python source files, 55K LOC
- All absolute imports remapped from `animus_forge` / `animus` → `animus_kernel`
- Shell modules (dashboard, tui, cli, api routes, discord bot, etc.) explicitly excluded

### Known Cleanup Items
- Some secondary `__init__.py` files trimmed aggressively; may need partial restores if downstream consumers expect them
- `memory/stores/chroma.py` exists but ChromaDB is an optional dependency
- `config/settings.py` references env vars that may not be relevant in kernel-only mode
- No unit tests ported yet — next step is `tests/` extraction or new test suite

---

*Last updated: 2026-06-14*
