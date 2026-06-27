# Documentation Migration Guide (2026-06-27)

> If you have bookmarks, external references, or muscle memory pointing to old documentation locations, use this map.

---

## What Changed

All documentation has been reorganized from a flat `docs/` directory and scattered root-level files into an **audience-based tree**.

---

## Quick Lookup Table

| Old Path | New Path | Category |
|---|---|---|
| `ARCHITECTURE.md` | `docs/architecture/overview.md` | Architecture |
| `CANON.md` | `docs/architecture/canonical-principles.md` | Architecture |
| `CONSTITUTIONAL_PRINCIPLES.md` | `docs/architecture/constitutional-principles.md` | Architecture |
| `CONSCIOUSNESS_QUORUM_BRIDGE.md` | `docs/architecture/consciousness-quorum-bridge.md` | Architecture |
| `EVOLUTION_LOOP.md` | `docs/architecture/evolution-loop.md` | Architecture |
| `WORKFLOW_EVOLUTION_CONSTRAINTS.md` | `docs/architecture/workflow-constraints.md` | Architecture |
| `OGMA.md` | `docs/architecture/ogma.md` | Architecture |
| `ENGINE_VS_SHELL_ASSESSMENT.md` | `docs/architecture/engine-vs-shell.md` | Architecture |
| `WORK_BOUNDARY.md` | `docs/architecture/work-boundary.md` | Architecture |
| `SOVEREIGNTY_STACK.md` | `docs/architecture/sovereignty-stack.md` | Architecture |
| `ARCHITECTURE_NAMING.md` | `docs/architecture/naming.md` | Architecture |
| `PROJECT_CHARTER.md` | `docs/architecture/charter.md` | Architecture |
| `CONTRIBUTING.md` | `docs/contributing/guidelines.md` | Contributing |
| `DEVELOPER_TOOLS.md` | `docs/contributing/developer-tools.md` | Contributing |
| `METHOD_AGENT_WORKFLOW.md` | `docs/contributing/method-agent-workflow.md` | Contributing |
| `PROJECT_ORGANIZATION_GUIDELINES.md` | `docs/contributing/organization.md` | Contributing |
| `ANIMUS_CONTEXT.md` | `docs/getting-started/animus-context.md` | Getting Started |
| `CASE_STUDY.md` | `docs/getting-started/case-study.md` | Getting Started |
| `INTERFACE_BOOTSTRAP_VISION.md` | `docs/getting-started/interface-vision.md` | Getting Started |
| `OLLAMA_SETUP.md` | `docs/getting-started/ollama-setup.md` | Getting Started |
| `USE_CASES.md` | `docs/getting-started/use-cases.md` | Getting Started |
| `BROWSER_AUTOMATION.md` | `docs/operators/browser-automation.md` | Operators |
| `CONNECTIVITY.md` | `docs/operators/connectivity.md` | Operators |
| `CUDA_AI_BOX_SETUP.md` | `docs/operators/cuda-setup.md` | Operators |
| `ISSUES.md` | `docs/operators/known-issues.md` | Operators |
| `OLLAMA_AGENT.md` | `docs/operators/ollama-setup.md` | Operators |
| `RECOVERY.md` | `docs/operators/recovery.md` | Operators |
| `REMOTE_ACCESS.md` | `docs/operators/remote-access.md` | Operators |
| `CHANGELOG.md` | `docs/reference/changelog.md` | Reference |
| `SECURITY.md` | `docs/reference/security.md` | Reference |
| `SAFETY.md` | `docs/reference/safety.md` | Reference |
| `SECURITY_LAYER.md` | `docs/reference/security-layer.md` | Reference |
| `THREAT_MODEL.md` | `docs/reference/threat-model.md` | Reference |
| `OPENCLAW_COMPARISON.md` | `docs/reference/openclaw-comparison.md` | Reference |
| `WHITEPAPER.md` | `docs/reference/whitepaper.md` | Reference |
| `WHITEPAPER_COMPARISON_2026-06.md` | `docs/reference/whitepaper-comparison-2026-06.md` | Reference |
| `PROJECT_CONTEXT.md` | `docs/reference/project-context.md` | Reference |
| `PROJECT_FOLDER_SETUP_EVALUATION_STANDARD.md` | `docs/reference/project-folder-evaluation-standard.md` | Reference |
| `ROADMAP.md` | `docs/roadmap/current.md` | Roadmap |
| `ROADMAP_TO_10.md` | `docs/roadmap/roadmap-to-10.md` | Roadmap |
| `ROADMAP_HERMES_2026-06.md` | `docs/roadmap/hermes-2026-06.md` | Roadmap |
| `ROADMAP_quorum_v2.md` | `docs/roadmap/quorum-v2.md` | Roadmap |
| `ROADMAP_research_assistant.md` | `docs/roadmap/research-assistant.md` | Roadmap |
| `PERSONAL_ROADMAP.md` | `docs/roadmap/personal.md` | Roadmap |
| `ANIMUS_MEMORY_GAPS.md` | `docs/reviews/animus-memory-gaps.md` | Reviews |
| `LEARNED_AUDIT_2026-05-15.md` | `docs/reviews/learned-audit-2026-05.md` | Reviews |
| `TOOL_AUDIT_2026-05-15.md` | `docs/reviews/tool-audit-2026-05.md` | Reviews |
| `WORK_BOUNDARY_AUDIT_2026-05-15.md` | `docs/reviews/work-boundary-audit-2026-05.md` | Reviews |
| `TPS_LEAN_AUDIT_2026-06.md` | `docs/reviews/tps-lean-audit-2026-06.md` | Reviews |
| `TARGETS_HIT_ANALYSIS_2026-06.md` | `docs/reviews/targets-hit-2026-06.md` | Reviews |

---

## Root-Level Redirects

These files still exist at root but now redirect to the canonical location:

- `CONTRIBUTING.md` → `docs/contributing/guidelines.md`
- `CHANGELOG.md` → `docs/reference/changelog.md`
- `ROADMAP.md` → `docs/roadmap/current.md`
- `PROJECT_CHARTER.md` → `docs/architecture/charter.md`
- `SECURITY.md` → `docs/reference/security.md`

---

## Deleted Files

These files were removed (not moved):

- `TODO_NEXT.md` — Ephemeral scratchpad
- `TODO_CHAT_AGENT.md` — Ephemeral scratchpad
- `animus-CLAUDE-addition.md` — Stale 37-line fragment
- `docs/ROADMAP.md` — Duplicate of root `ROADMAP.md`
- `docs/animus-build-spec.md` — Duplicate of `docs/specs/animus-build-spec.md`
- `docs/animus-landscape-and-additional-tools.md` — Duplicate of `docs/specs/animus-landscape-and-additional-tools.md`

---

## Package READMEs

Package READMEs were **not moved** — they stay at `packages/<name>/README.md`. However, new READMEs were added to packages that lacked them:

- `packages/quorum/README.md` — NEW
- `packages/pwa/README.md` — NEW
- `packages/contracts/README.md` — NEW

---

## Decision Logs

ADRs and ADL entries are now centralized:

- `adrs/ADR-001.md` → `docs/architecture/decisions/ADR-001.md`
- `decisions/2026-06.md` (ADL-20260618-001) → `docs/architecture/decisions/ADL-20260618-001.md`

New ADRs should be added to `docs/architecture/decisions/`.

---

## Need Help?

Start at `docs/README.md` — the new entry point for all Animus documentation.
