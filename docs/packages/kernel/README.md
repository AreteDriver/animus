# Animus Kernel

The autonomous builder engine. Extracted from the Animus monorepo for standalone use.

## What This Is

`animus-kernel` is the execution engine of the Animus system: the budget manager, workflow executor, sandbox validator, safety checker, multi-agent supervisor, and model provider abstraction. It is stripped of the personal-tool shell (dashboard, PWA, messaging channels, persona engine) and focused entirely on autonomous code building.

## Modules

| Module | Purpose | Origin |
|---|---|---|
| `budget` | Atomic token reservation, ET enforcement, cost tracking | `packages/forge/src/animus_forge/budget/` |
| `executor` | Workflow orchestration: sequential, parallel, loops, checkpoint/resume | `packages/forge/src/animus_forge/workflow/` |
| `sandbox` | Isolated execution: clone → apply → test → lint → rollback | `packages/forge/src/animus_forge/self_improve/sandbox.py` |
| `safety` | File/line limits, protected patterns, suspicious code detection | `packages/forge/src/animus_forge/self_improve/safety.py` |
| `providers` | Model-agnostic LLM calls: Anthropic, OpenAI, Ollama, OpenRouter, etc. | `packages/forge/src/animus_forge/providers/` |
| `agents` | Multi-agent delegation: Supervisor, Planner, Builder, Tester, Reviewer | `packages/forge/src/animus_forge/agents/` |
| `coordination` | Intent graph, workflow evolution, auto-promotion | `packages/forge/src/animus_forge/coordination/` |
| `intelligence` | Cross-workflow memory, cost intelligence, outcome tracking | `packages/forge/src/animus_forge/intelligence/` |
| `contracts` | Contract validation and enforcement | `packages/forge/src/animus_forge/contracts/` |
| `memory` | Project-scoped memory (HOT/WARM/COLD tiering) | `packages/core/animus/memory/` |

## Installation

```bash
pip install -e packages/kernel
```

## Usage

```python
from animus_kernel.executor import WorkflowExecutor
from animus_kernel.budget import BudgetManager
from animus_kernel.agents import SupervisorAgent

# Build a project
executor = WorkflowExecutor(
    budget_manager=BudgetManager(total_budget=10000),
    checkpoint_manager=...
)

supervisor = SupervisorAgent(
    executor=executor,
    provider_router=...
)

result = await supervisor.build_project(
    project_path="~/projects/my-app",
    plan="Add OAuth2 authentication",
    budget_limit=5000
)
```

## Roadmap

1. Extract all Tier 1 modules from `packages/forge/` and `packages/core/`
2. Verify standalone imports and tests pass
3. Build new builder modules: `plan_refiner`, `build_queue`, `repo_scanner`
4. Release v0.1.0 as installable package

---

*Part of the Animus system. See `docs/ENGINE_VS_SHELL_ASSESSMENT.md` for the strategic rationale.*
