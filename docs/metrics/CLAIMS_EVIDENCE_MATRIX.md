# Claims-to-Evidence Matrix

This matrix maps external-facing project claims to verifiable evidence and reproducible commands.

## How to use

1. Every high-visibility claim in `README.md` should have a row.
2. Every row should point to a deterministic evidence source.
3. Prefer machine-generated evidence (CI artifacts, command output, coverage files) over narrative docs.

## Matrix

| Claim | Claim location | Evidence source | Reproduce command | Owner | Cadence |
|---|---|---|---|---|---|
| Multi-agent orchestration with budget controls, quality gates, checkpoint/resume | `README.md` intro + Forge section | Forge workflow schema + executor tests + workflow checkpoint state tables | `pytest packages/forge/tests -q` | Forge | Per release |
| 13,676+ tests across 4 packages | `README.md` status section | CI job summaries + pytest reports per package | `pytest packages/core/tests packages/quorum/tests -q && (cd packages/forge && pytest tests -q)` | Release | Per merge to main |
| Provider-agnostic model support (OpenAI/Anthropic/Ollama) | `README.md` design principles | Provider router and provider integration tests | `pytest packages/forge/tests -k provider -q` | Forge | Weekly |
| Self-improvement pipeline with safety + rollback | `README.md` self-improvement section + Forge README | self_improve orchestrator tests and safety checker tests | `pytest packages/forge/tests -k self_improve -q` | Forge | Per release |
| Quorum intent-graph coordination | `README.md` Quorum section | Quorum protocol tests and benchmark outputs | `pytest packages/quorum/tests -q` | Quorum | Per merge |

## Next steps

- Wire this file to CI as a required artifact in the docs/build workflow.
- Run `python tools/validate_claims_evidence.py` in CI to validate matrix structure and file references.
- Publish generated evidence snapshots under `docs/metrics/artifacts/`.
