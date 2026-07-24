# Schema Usage Guide

> How Animus JSON Schema contracts become Pydantic models, and how to work with both.

---

## Overview

Animus uses a **schema-first** data model. Every core domain object is defined as JSON Schema Draft 2020-12 in `packages/contracts/`. These schemas are compiled into Pydantic v2 models in `packages/types/` and consumed by every package in the monorepo.

**Source of truth:** `packages/contracts/*.schema.json`
**Generated artifacts:** `packages/types/src/animus_types/*.py`
**Pipeline:** `scripts/compile_schemas.py` → `scripts/validate_schemas.py`

---

## Why Schema-First?

- **Validation at the boundary** — JSON Schema validates untrusted input before it reaches Python code
- **Language-neutral contracts** — Rust (Quorum), TypeScript (PWA), and Python share the same shape
- **CI-guarded drift** — The validation script fails if schemas and generated models diverge
- **Bitemporal integrity** — Every object carries `valid_time`, `transaction_time`, and `integrity` hashes

---

## The Compilation Pipeline

### Step 1: Write or update a JSON Schema

Place your schema in `packages/contracts/<name>.schema.json`. Requirements:

- Must declare `"$schema": "https://json-schema.org/draft/2020-12/schema"`
- Must declare a unique `"$id"` URL (e.g. `https://animus.local/schemas/<name>.schema.json`)
- Filename must match the `$id` basename
- Use `additionalProperties: false` for strict validation
- Reference `common.schema.json` for the canonical object envelope

Example: `packages/contracts/ledger_event.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://animus.local/schemas/ledger_event.schema.json",
  "title": "Ledger Event",
  "type": "object",
  "additionalProperties": false,
  "required": ["event_id", "event_type", "object_id", ...],
  "properties": {
    "event_id": {"type": "string", "pattern": "^evt-[a-z0-9_-]+$"},
    "event_type": {"enum": ["created", "updated", "superseded", ...]},
    ...
  }
}
```

### Step 2: Generate Pydantic models

```bash
python scripts/compile_schemas.py
```

This:
1. Copies all `.schema.json` files to a temp directory
2. Runs `datamodel-codegen` to generate Pydantic v2 models
3. Post-processes the output:
   - Strips `_schema` suffix from module names
   - Deduplicates classes that exist in `common.py`
   - Fixes import statements
   - Replaces `AnimusCanonicalObjectEnvelope` with `Common` as the base class
4. Writes generated modules to `packages/types/src/animus_types/`
5. Regenerates `__init__.py` with re-exports

### Step 3: Validate the gate

```bash
python scripts/validate_schemas.py
```

Checks:
1. Every `.schema.json` is valid Draft 2020-12
2. Every schema has a unique `$id`
3. Filename matches `$id` basename
4. No dangling `$ref` references
5. Every schema has a corresponding importable Pydantic model
6. Representative schemas pass round-trip validation (JSON → Pydantic → JSON)

This runs in CI (`.github/workflows/ci.yml`, job `schema-validate`).

---

## Using Generated Types

### Import from the package

```python
from animus_types import LedgerEvent, EventType, Common

# Construct a ledger event
event = LedgerEvent(
    event_id="evt-login-001",
    event_type=EventType.created,
    object_id="usr-alice",
    object_version=1,
    principal="usr-alice",
    workspace_id="ws-default",
    payload={"ip": "127.0.0.1"},
    integrity_hash="a" * 64,
    tx_time="2026-01-01T00:00:00Z",
)

# Validation happens automatically
event.model_validate({...})   # from dict
```

### Using the common envelope

Objects that extend `Common` inherit the canonical envelope:

```python
from animus_types import AnimusActionObject, Action

action = AnimusActionObject(
    object_id="act-deploy-001",
    object_version=1,
    schema_id="https://animus.local/schemas/action.schema.json",
    schema_version="1.0.0",
    owner_id="owner-alice",
    workspace_id="ws-prod",
    subject_domain=SubjectDomain.project,
    artifact_type=ArtifactType.action,
    cognitive_role=CognitiveRole.intelligence,
    workflow_status=WorkflowStatus.approved,
    epistemic_status=EpistemicStatus.supported,
    lifecycle_status=LifecycleStatus.active,
    storage_tier=StorageTier.hot,
    presentation=Presentation.canonical,
    security_class=SecurityClass.internal,
    valid_time=ValidTime(valid_from="2026-01-01T00:00:00Z", valid_to=None),
    transaction_time=TransactionTime(recorded_at="2026-01-01T00:00:00Z", superseded_at=None),
    provenance=Provenance(created_by="system", source_refs=[], derived_from=[], trace_id=None),
    integrity=Integrity(content_sha256="a" * 64),
    payload=Payload(
        action_kind="deploy",
        risk_class=RiskClass.R1,
        target="production",
        parameters={},
        approval_required=False,
        approval_id=None,
        idempotency_key="idemp-12345678",
        status=Status.proposed,
    ),
)
```

---

## Schema Catalog

The monorepo currently maintains **25 JSON Schema contracts**:

| Schema | Pydantic Model | Purpose |
|---|---|---|
| `common` | `Common` | Base envelope with bitemporal timestamps, provenance, integrity |
| `action` | `AnimusActionObject` | Deploy/execute actions with risk class and idempotency |
| `action_receipt` | `AnimusActionReceipt` | Outcome record for executed actions |
| `approval_receipt` | `AnimusApprovalReceipt` | Signed approval with grant reference |
| `assessment` | `DissentItem` | Challenge or review of a claim or decision |
| `capability_grant` | `CapabilityGrant` | Scoped authorization with budget limits |
| `claim` | `AnimusClaimObject` | Factual assertions with evidence |
| `context_envelope` | `Contradiction` | Cross-context contradiction detection |
| `decision` | `AnimusDecisionObject` | Recorded decisions with rationale |
| `entity` | `AnimusEntityObject` | Named entities (people, orgs, systems) |
| `event` | `AnimusEventObject` | Time-bounded occurrences |
| `forecast` | `AnimusForecastObject` | Predictions with confidence intervals |
| `hypothesis` | `AnimusHypothesisObject` | Testable statements with falsification criteria |
| `ledger_event` | `LedgerEvent` | Immutable append-only event store entries |
| `lesson` | `AnimusLessonObject` | Learned outcomes from actions |
| `memory_candidate` | `AnimusMemoryCandidate` | Proposed memories awaiting review |
| `object_version` | `ObjectVersion` | Versioned object snapshot with lineage |
| `observation` | `AnimusObservationObject` | Raw data points from sensors or APIs |
| `outbox_entry` | `OutboxEntry` | Pending outbound messages |
| `outcome` | `AnimusOutcomeObject` | Results of executed actions |
| `pattern` | `AnimusPatternObject` | Recurring structures or correlations |
| `policy_decision` | `Obligation` | Policy engine obligations |
| `signal` | `AnimusSignalObject` | Alerts and thresholds |
| `source` | `AnimusSourceObject` | Information origin tracking |
| `trace` | `AnimusTraceBundle` | Execution traces and telemetry |

**Legacy types** (not generated from schema):
- `Sensitivity` — Four-tier disclosure classification (`PUBLIC`, `PERSONAL`, `CONFIDENTIAL`, `SECRET`)
- `EgressDeniedError`, `is_egress_allowed` — Network egress helpers

---

## Adding a New Schema

1. **Create the schema** in `packages/contracts/<name>.schema.json`
2. **Add a minimal payload** to `scripts/validate_schemas.py` `_MINIMAL_PAYLOADS` (optional but recommended)
3. **Add a class mapping** to `_schema_name_to_class_name()` in `scripts/validate_schemas.py`
4. **Run the compiler:**
   ```bash
   python scripts/compile_schemas.py
   ```
5. **Run the validator:**
   ```bash
   python scripts/validate_schemas.py
   ```
6. **Commit both** the schema and the generated `.py` files

---

## Common Patterns

### Pattern: Validate untrusted input at the edge

```python
from animus_types import AnimusActionObject

raw = await request.json()
try:
    action = AnimusActionObject.model_validate(raw)
except ValidationError as e:
    raise HTTPException(status_code=422, detail=e.errors())
```

### Pattern: Serialize for storage

```python
json_str = action.model_dump_json()        # → str
json_dict = action.model_dump(mode="json") # → dict (JSON-serializable)
```

### Pattern: Immutable event hash

```python
import hashlib, json

payload_json = json.dumps(event.model_dump(mode="json")["payload"], sort_keys=True)
integrity_hash = hashlib.sha256(payload_json.encode()).hexdigest()
assert event.integrity_hash == integrity_hash
```

---

## CI Integration

The `schema-validate` job in `.github/workflows/ci.yml` runs on every push and PR:

```yaml
schema-validate:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.12" }
    - run: pip install jsonschema
    - run: pip install -e packages/types/
    - run: python scripts/validate_schemas.py
```

---

## See Also

- [Architecture → Packages](../architecture/packages.md) — Dependency map and package responsibilities
- [Packages → Types](../../packages/types/README.md) — Package README for `animus_types`
- [Contracts → README](../../packages/contracts/README.md) — JSON Schema design conventions
- `scripts/compile_schemas.py` — Compilation pipeline source
- `scripts/validate_schemas.py` — CI validation gate source
