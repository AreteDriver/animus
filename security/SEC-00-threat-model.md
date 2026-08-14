# SEC-00 — Execution-Plane Threat Model

**Baseline commit:** `6338784561efa6395fb83072318ff7771284842f`
**Scope:** Tool execution, shell dispatch, container isolation, memory/logging boundaries, and egress controls in the Animus execution plane.

> **Historical note:** This document was originally written against the pre-fix state at baseline commit `6338784`. Sections 1–4 below preserve that historical narrative so the original reasoning remains traceable. **Current canonical status is recorded in Section 5 (Reconciliation).** Do not treat the historical risk mapping as the current security posture.

---

## 1. Trust Boundaries

| Boundary | Inside | Outside | Enforcement Today |
|---|---|---|---|
| TB-1 Tool registry | `ToolRegistry.execute()` | Model-generated tool calls | Handler-level validation only; `requires_approval` is metadata and not enforced by the registry. |
| TB-2 Security config | `_security_config` singleton | Any module that imports `tools.py` | Global mutable variable; `None` means "unrestricted dev mode". |
| TB-3 Shell execution | `subprocess.run(shell=True)` | Host operating system | Token-based allowlist checks only the first whitespace-delimited word; remainder of the string is passed to the shell unreviewed. |
| TB-4 Container isolation | Docker/Podman runtime | Host workspace, secrets, network | Optional; worker pool silently falls back to `ProcessPoolExecutor` when no container manager is provided. |
| TB-5 Egress | `is_egress_allowed()` policy | External HTTP endpoints | Cognitive/providers use it; the generic `http_request` tool does not. |
| TB-6 Memory redaction | `redact()` at ingest | `logger.info()` previews | `remember()` stores the redacted copy but logs the raw original content at INFO level. |

---

## 2. Assets

| Asset | Sensitivity | Location | If Compromised |
|---|---|---|---|
| User file system | Confidential/Secret | Host paths reachable via `read_file`, `write_file`, `run_command` | Data exfiltration, destruction, or tampering. |
| API keys / tokens | Secret | `http_request` params, memory content, env vars | Credential leak to third-party services or logs. |
| Memory store | Public..Secret | `MemoryLayer` + Chroma/JSON backend | Disclosure of private memories via logging or MCP egress. |
| Model-provider keys | Secret | Environment variables | Container env passthrough and command logging can leak keys. |
| Mission workspace | Confidential | Git worktree / container mount | Read-write mounts let untrusted citizen code modify the host workspace. |
| Host network | Secret | Private IPs, cloud metadata endpoints | SSRF through `http_request` or shell-curl from `shell=True`. |

---

## 3. Risk Mapping

| ID | Defect | Threat | Likelihood | Impact | Risk | Target Fix |
|---|---|---|---|---|---|---|
| SEC-01 | `packages/core/animus/tools.py` uses module-level `_security_config = None`; missing config is interpreted as unrestricted. | Any caller that forgets to pass a config gets full filesystem + shell access. | High | Critical | **Critical** | Fail-closed default; reject operations when no config is set. |
| SEC-02 | `/build` in `packages/core/animus/__main__.py` calls `_set_security_config(None)` in a finally block, clearing policy globally. | A build sandbox temporarily widens `write_roots`; the finally block clears *all* policy, leaving later tool calls unprotected for the remainder of the process. | High | High | **High** | Restore the previous config, or set a safe default, instead of `None`. |
| SEC-03 | `packages/core/animus/mcp_server.py` calls `create_default_registry()` with no explicit security policy. | MCP-exposed tools run with `_security_config = None`, bypassing path and command gates. | High | Critical | **Critical** | Pass `config.tools_security` (or a hardened default) into `create_default_registry()`. |
| SEC-04 | `packages/kernel/src/animus_kernel/tools/registry.py` validates first command token and runs `shell=True`. | `python -c "<arbitrary code>"` passes the allowlist because `python` is allowed, then `shell=True` executes the injected body. | Medium | Critical | **High** | Disable `shell=True` and validate the full command argument vector. |
| SEC-05 | `packages/kernel/src/animus_kernel/head/tool_orchestrator.py` repeats the same `shell=True` pattern. | Same injection surface in the Head REPL / API. | Medium | Critical | **High** | Apply the same fix as SEC-04 and share one hardened runner. |
| SEC-06 | `Tool.requires_approval` is metadata; enforcement is not guaranteed inside `ToolRegistry.execute()`. | A malicious or confused model can call `run_command`, `write_file`, `edit_file`, or `http_request` without an approval gate. | Medium | High | **High** | Require an approval callback in `ToolRegistry.execute()` and fail if none is registered for approval-required tools. |
| SEC-07 | Generic HTTP tool does not enforce centralized egress policy and is exposed to SSRF. | `http_request` can reach `localhost`, `127.0.0.1`, `169.254.169.254`, internal RFC1918 hosts, and arbitrary cloud endpoints without consulting `is_egress_allowed()` or a sensitivity tier. | High | High | **High** | Route every outbound URL through `is_egress_allowed(url, tier)` with a default-deny for non-loopback private/internal targets. |
| SEC-08 | `packages/core/animus/memory/layer.py` logs raw unredacted memory content in an info-level preview. | Secrets ingested into memory are briefly stored in redacted form, but the original raw value is written to INFO logs before redaction is applied. | Medium | High | **High** | Log the redacted copy (or a length-only placeholder) and never the pre-redaction content. |
| SEC-09 | Container mode is optional, silently falls back to process mode, mounts workspace read-write, uses unpinned image, lacks runtime limits, and may log environment values. | A container-configured mission can execute on the host instead, the workspace is mutable from inside the container, the image is floating (`python:3.12-slim`), and container command construction logs env values. | Medium | High | **High** | Require explicit opt-in/availability check, mount workspace read-only, pin image digest, add CPU/memory caps, and redact env in logs. |

---

## 4. Safe Regression Strategy

All proofs in `test_security_execution_plane.py` use:

- `tempfile.TemporaryDirectory` / `tmp_path` for filesystem state.
- `unittest.mock` and `http.server` / `socketserver` for mock HTTP targets.
- Inert shell commands (`echo`, `python -c "print('safe')"`) — never destructive.
- Fake secrets (`sk-ant-api03-fakefakefakefakefakefake`, `ghp_fakefakefakefakefakefakefakefakefakefake`) that are not live credentials.
- Monkeypatching of `subprocess.run`, `shutil.which`, container runtimes, and log handlers to avoid touching real hosts or runtimes.

Each test asserts the **pre-fix vulnerable behavior** so that it fails today and passes once the corresponding SEC-01..SEC-07 fix is in place.

---

## 5. Security Reconciliation (current canonical state)

**Last reconciled:** 2026-08-13
**Reconciliation basis:** direct code inspection of current `main` + independent oracle-based verification + `test_security_execution_plane.py` + focused regression tests.

| ID | Historical claim | Current status | Evidence | Confidence |
|---|---|---|---|---|
| **SEC-01** | `_security_config = None` = unrestricted | **FIXED** | `WorkspaceToolPolicy.from_tools_security_config()` creates fail-closed policy; no `_security_config = None` path exists on current `main` | HIGH |
| **SEC-02** | `requires_approval` metadata-only | **FIXED** | `ToolRegistry.execute()` structurally enforces approval via `approval_store.lookup()` + `approval_store.verify()` (commit `248efea`, improved at `cffb0cf`) | HIGH |
| **SEC-03** | MCP server no security policy | **FIXED** | `create_default_registry()` receives explicit security policy | HIGH |
| **SEC-04** | `shell=True` in kernel registry | **FIXED** | `ForgeToolRegistry._handle_run_command` uses `subprocess.run(argv, shell=False)` with `shlex.split()` + injection char rejection + interpreter defense (commit `9b0ac6f`) | HIGH |
| **SEC-05** | `shell=True` in Head orchestrator | **FIXED** | `HeadToolOrchestrator._handle_run_shell` uses `subprocess.run(argv, shell=False)` with identical defense stack (commit `f0b210a`) | HIGH |
| **SEC-06** | Unredacted secrets in logs (17 files) | **PARTIALLY SUPERSEDED** | Memory-layer portion (layer.py, durable.py, local.py, chroma.py) superseded by SEC-08 fix (commit `7b24d6c`). Remaining non-memory paths (tools.py, mcp_server.py, kernel tools_core.py, head tool_orchestrator.py, Forge containers.py) remain **UNRECONCILED** — not independently verified on current `main`. | HIGH (memory); LOW (non-memory) |
| **SEC-07** | HTTP tool bypasses egress | **FIXED** | `_tool_http_request` enforces `policy.authorize_network()` before outbound request (part of SEC-05 integration) | HIGH |
| **SEC-08** | Memory layer logs raw content | **FIXED** (commit `7b24d6c`) | `remember()` no longer logs content previews; search() across all 3 stores no longer logs raw queries. 28 focused adversarial regression tests pass. | HIGH |
| **SEC-09** | Container silently falls back to process | **NOT_APPLICABLE** | No container execution exists in current architecture. `Sandbox` uses `tempfile.mkdtemp()` + `shutil.copytree()` + direct `subprocess.run()` with sanitized env. | HIGH |

### Remaining uncertainty

- **SEC-06 (non-memory):** Historical commit `65d82c8` claimed fixes across tools, MCP, kernel, and Forge logging. These paths were not independently revalidated in the SEC-08 session. Default disposition is **UNRECONCILED**.
- **Exception-logging paths:** Four `logger.debug(f"... failed: {e}")` calls in `layer.py` (entity linking/cleanup during remember/forget/import/consolidation) are **INVESTIGATION LEADS**. They are error paths with external-service exceptions, not the normal operational flow.

### Reachable Critical/High findings

**Critical: 0**
**High: 0** (within independently verified surface)

