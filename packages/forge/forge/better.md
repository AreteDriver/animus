# Definition of Better — Forge Evolution

## Goal
Improve Forge's YAML workflow definitions to produce higher-quality, more efficient agent orchestration.

## What "Better" Means

### Efficiency
- Fewer LLM calls per workflow (consolidate redundant steps)
- Lower total token usage for equivalent output quality
- Faster end-to-end execution time

### Quality
- Workflow outputs that are more structured and actionable
- Better error handling in workflow step definitions
- Clearer step descriptions that produce better LLM responses

### Safety
- All workflows have explicit token_budget fields
- No workflows with unbounded loops or missing halt conditions
- Every LLM step has a fallback or error handler

## Measurement
- Token count per workflow run (lower is better)
- Step success rate (higher is better)
- Output structure compliance (JSON parseable, required fields present)

## Constraints
- Do not modify Python source code — YAML workflows only
- Do not exceed 3000 tokens per evolution iteration
- Preserve existing workflow functionality — improve, don't break
