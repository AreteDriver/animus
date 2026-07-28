# Animus Contracts

**Canonical JSON schemas for the Animus v2.3 Kernel-native architecture.**

These 20+ schemas define the data contracts across all Animus subsystems: memory, events, actions, assessments, and more. Every package that produces or consumes structured data validates against these schemas.

## Schemas

| Schema | Purpose |
|---|---|
| `action.schema.json` | Agent actions with parameters and outcomes |
| `action_receipt.schema.json` | Acknowledgment of executed actions |
| `approval_receipt.schema.json` | Human approval records for gated operations |
| `assessment.schema.json` | Evaluations, judgments, and scoring |
| `claim.schema.json` | Assertions with confidence and evidence |
| `common.schema.json` | Shared definitions referenced by other schemas |
| `context_envelope.schema.json` | Session context with metadata and provenance |
| `decision.schema.json` | Decision records with rationale and alternatives |
| `entity.schema.json` | Named entities (people, places, concepts) |
| `event.schema.json` | Time-stamped events with causality links |
| `forecast.schema.json` | Predictions with confidence intervals |
| `hypothesis.schema.json` | Testable hypotheses with evidence tracking |
| `lesson.schema.json` | Learned patterns and their applicability |
| `memory_candidate.schema.json` | Proposed memories awaiting validation |
| `observation.schema.json` | Raw observations from sensors or APIs |
| `outcome.schema.json` | Results of actions with metrics |
| `pattern.schema.json` | Recurring patterns with frequency data |
| `signal.schema.json` | Alerts and notifications with severity |
| `source.schema.json` | Provenance and citation tracking |
| `trace.schema.json` | Execution traces for debugging and audit |

### Registry Schemas (v2.3+)

| Schema | Purpose |
|---|---|
| `ledger_event.schema.json` | Immutable append-only event records |
| `object_version.schema.json` | Bitemporal canonical object snapshots |
| `outbox_entry.schema.json` | Transactional outbox for async delivery |
| `capability_grant.schema.json` | Scoped authorization grants |
| `policy_decision.schema.json` | Deterministic policy enforcement records |

## Usage

### Validate an object in Python

```python
import json
from jsonschema import validate

schema = json.load(open("packages/contracts/action.schema.json"))
instance = {"type": "deploy", "target": "staging"}
validate(instance, schema)
```

### Validate in CI

The `truth-baseline.yml` workflow validates all schema files are well-formed JSON Schema.

## Adding a New Schema

1. Copy `common.schema.json` as a template
2. Define `$id`, `title`, `description`, and `properties`
3. Add required fields to `required` array
4. Run `python scripts/truth-baseline.py` to validate
5. Submit PR with `docs:` prefix

## Part of the Animus Monorepo

- [Animus Core](https://github.com/AreteDriver/animus/tree/main/packages/core) — consumes these schemas for memory/events
- [Animus Forge](https://github.com/AreteDriver/animus/tree/main/packages/forge) — validates workflow outputs against schemas
- [Animus Kernel](https://github.com/AreteDriver/animus/tree/main/packages/kernel) — uses schemas for contract validation

## License

MIT — 2026, AreteDriver
