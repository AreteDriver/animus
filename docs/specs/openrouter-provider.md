# Spec: OpenRouter Provider + Local-Default Egress Gate

- **Status:** DRAFT — parked for ARETE decision (not approved to build)
- **Author:** Claude Code session 2026-05-30
- **Mode:** Specification
- **Related:** `docs/THREAT_MODEL.md`, `docs/SECURITY_LAYER.md`, `docs/OPENCLAW_COMPARISON.md`, `packages/forge/src/animus_forge/providers/`
- **Prompted by:** Nous Research [Hermes Agent](https://github.com/nousresearch/hermes-agent) (#1 on OpenRouter global token rankings, May 2026). Hermes is cloud-default and model-agnostic via OpenRouter. This spec ports the *useful* piece — 200-model breadth — while **inverting** Hermes' posture to match Animus hardening.

---

## 1. Problem

Animus has six hand-wired providers (Anthropic, OpenAI, Azure OpenAI, Bedrock, Vertex, Ollama, llama.cpp) and three routers (`TierRouter` with CLOUD/LOCAL/air-gap modes, `ProviderRouter` with EMA cost/latency selection, `SefiroticRouter`). It does **not** have a single gateway to the broad model market. Adding each new model = a new provider + SDK.

[OpenRouter](https://openrouter.ai) solves this: one OpenAI-compatible endpoint, one key, 200+ models, automatic pricing and provider fallback. The routing machinery to consume it **already exists** in Animus — only the provider is missing.

## 2. Goal

Add OpenRouter as a CLOUD-tier provider, reachable through the existing routers, **without weakening the local-first / air-gapped posture locked in PRs #48–#67**.

The design principle is an inversion of Hermes:

> **Hermes = cloud-default. Animus = local-default, cloud-on-consent, DLP-gated.**

## 3. Non-Goals (out of scope for this spec)

- NVIDIA NIM provider (separate spec if wanted).
- Browser automation / vision tools (see `docs/BROWSER_AUTOMATION.md`; separate arc).
- Nous Portal Tool Gateway integration.
- Replacing Ollama as the default. Ollama/`qwen2.5:14b` stays the default and the only path for sensitive content.
- Any change to the self-improvement loop (`EVOLUTION_LOOP.md`) — already at parity with Hermes' headline feature.

## 4. Design

### 4.1 Provider

- New `ProviderType.OPENROUTER = "openrouter"` in `packages/forge/src/animus_forge/providers/base.py` (`ProviderType` enum, currently line ~38).
- New `OpenRouterProvider(Provider)` in `providers/openrouter_provider.py`, mirroring `AnthropicProvider` (cloud, `api_key`-based). OpenRouter is OpenAI-compatible → reuse the OpenAI request/response shaping where practical.
  - `base_url` default `https://openrouter.ai/api/v1`.
  - `is_available()` ⇒ `bool(api_key) AND egress_allowed()` (see 4.3). Must return `False` whenever egress is denied — the provider advertises itself as unavailable in local/offline mode so the router never selects it.
  - Sets OpenRouter ranking headers (`HTTP-Referer`, `X-Title`) to a static Animus identifier.
  - Implements `complete` / `complete_async` / `complete_stream(_async)` / `generate` per the `Provider` base contract.
- Register in `tui/providers.py::create_provider_manager()` **after** Ollama, and **only if** `settings.openrouter_api_key` is set. Default selection order is unchanged (anthropic > openai > ollama); OpenRouter is never auto-default.

### 4.2 Tier mapping — **open-weights only** (DECIDED)

Map `ModelTier` → OpenRouter model slugs via config (not hardcoded). **Only open-weight model slugs are permitted** — the provider rejects closed-vendor slugs (anthropic/*, openai/*, google/* etc.) at config-load and at request time, so cloud behaviour stays open-weight end-to-end, consistent with the local `qwen2.5:14b` default. This is a guarantee, not a default.

| Tier | Default slug intent (open-weight; exact slug confirmed vs live catalog at build) |
|---|---|
| `REASONING` | a strong open reasoning model (e.g. DeepSeek-R1-class / Qwen-large-class) |
| `STANDARD` | a mid open general model (Llama/Qwen-class) |
| `FAST` | a small/cheap open model |
| `EMBEDDING` | **not served via OpenRouter** — embeddings stay local (`nomic-embed-text`) |

Enforcement test: a closed-vendor slug in config or request raises and never reaches the network.

### 4.3 Egress gate (the load-bearing security component)

A single chokepoint — `egress_allowed(request) -> bool` — gates every cloud-bound call. OpenRouter calls pass through it; it is the spec's reason for existing.

Deny (force local Ollama) when **any** of:

1. `ANIMUS_OFFLINE` or `ANIMUS_LOCAL_ONLY` is set (also enforced at the network floor by `IPAddressDeny`; the gate is defence-in-depth, not the only barrier).
2. `TierRouter` mode is `LOCAL`.
3. The request's content classification is `PERSONAL`, `CONFIDENTIAL`, or `SECRET`. **PERSONAL is hard-deny, same tier as CONFIDENTIAL (DECIDED)** — no opt-in path. Memory is tagged 1408 PERSONAL / 4 CONFIDENTIAL / 0 SECRET (Stage 2.E backfill), so this means ~99% of classified content never egresses.

Allow **only** `PUBLIC` / unclassified-low-sensitivity. Default-deny: anything unclassified that can't be proven PUBLIC stays local.

**Redaction before egress:** when the gate allows a call, the request passes through the existing DLP redactor (`ANIMUS_REDACT_*`) first. No raw credential/PII markers leave the box. The CamelCase/leetspeak/Bearer-value detectors hardened in PRs #61/#62 apply.

### 4.4 Config / secrets

- `OPENROUTER_API_KEY` added to settings (env-sourced).
- **The env edit happens off-session.** Credentials have leaked in-session twice (B2 keys, Fly Postgres). Adding the key is an operator step run with Claude Code **not** running, per the credential-handling protocol.
- New env flags: `ANIMUS_OPENROUTER_MODEL_REASONING/STANDARD/FAST` (open-weight slug overrides). No `ALLOW_PERSONAL` flag — PERSONAL is hard-deny with no override.

## 5. Acceptance Criteria (testable)

1. `ProviderType.OPENROUTER` exists; `OpenRouterProvider` implements the full `Provider` contract; `pytest` provider-coverage suites pass.
2. With `ANIMUS_LOCAL_ONLY=1` (or `ANIMUS_OFFLINE=1`), `OpenRouterProvider.is_available()` returns `False` and `TierRouter`/`ProviderRouter` never select it — asserted by test.
3. With a `PERSONAL`/`CONFIDENTIAL`/`SECRET`-classified request, the egress gate denies OpenRouter and routes to Ollama — asserted by test, even when a key is configured and offline flags are unset. A closed-vendor model slug raises before any network call.
4. Allowed egress calls are passed through the DLP redactor; a request seeded with a credential marker reaches the (mocked) OpenRouter client with the marker redacted — asserted by test.
5. `EMBEDDING` tier never routes to OpenRouter.
6. Registration is conditional on `openrouter_api_key`; absent the key, provider count and default selection are unchanged.
7. No secret is ever written to the repo, logs, or test fixtures (gitleaks clean; mocked key in tests).
8. Live smoke (operator, off-session, key set): a `FAST`-tier `PUBLIC` prompt returns a completion via OpenRouter; the same prompt with offline mode set returns via Ollama.

## 6. Test Plan

- Unit: provider contract, tier→slug mapping, header injection, `is_available()` truth table.
- Integration: router selection under each `RoutingMode` × classification combination (the 4.3 truth table is the test matrix).
- DLP: redaction-before-egress with seeded markers (extend existing redactor tests).
- **Red-team gate (mandatory):** per `docs/THREAT_MODEL.md` §"Project policy: red-team-on-every-change" (line ~279) — *"Any new security-relevant feature must be run through the red-team driver before merge."* This is a security-relevant feature. Run `python -m animus.redteam.driver --category pi` (and the egress-relevant categories) before merge; merge blocked on failures. CI gate: `--quick`.
- `verify_hardening` must remain green (currently 24/24).

## 7. Risks

- **Prompt egress to a broker.** OpenRouter forwards prompts to downstream providers. Mitigated by classification gate + redactor + default-deny on PERSONAL. Residual: PUBLIC content still leaves the box — acceptable by design, documented for the user.
- **Key leakage.** Mitigated by off-session env edit + gitleaks + mocked tests.
- **Posture drift.** A future caller could bypass the gate by calling the provider directly. Mitigation: the gate lives in `is_available()` and the routers, and the provider's `complete*` methods re-check `egress_allowed()` and raise if denied (belt + suspenders).

## 8. Decisions — RESOLVED 2026-05-30

1. **Model intent: open-weights only.** Provider rejects closed-vendor slugs; exact open-weight slugs confirmed vs live catalog at build (4.2).
2. **PERSONAL egress: hard-deny**, same tier as CONFIDENTIAL, no opt-in flag (4.3 #3).
3. **Comparison doc: yes** — add the "local-default vs Hermes cloud-default" sovereignty line to `OPENCLAW_COMPARISON.md` as part of the build.

Spec is build-ready. Remaining gate is ARETE's explicit "go" + the off-session `OPENROUTER_API_KEY` env edit.

## 9. Estimate

Provider + enum + registration: ~1 day. Egress gate + DLP wiring + classification hookup: ~1–2 days. Tests + red-team gate: ~1 day. Total ~3–4 focused days, single arc, one PR (or PR-per-section if you want the gate reviewed independently — recommended, since the gate is the security-load-bearing half).
