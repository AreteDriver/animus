# Performance & Observability

> Lightweight telemetry for measuring Animus runtime performance.

---

## Overview

Animus includes a built-in profiler with **zero external dependencies**. It measures latency for critical paths and writes structured JSON logs to a rotating file.

**Design principles:**
- No performance degradation — uses `time.perf_counter()` only
- No external services — logs to local file only
- No configuration required — works out of the box
- Privacy-first — no data leaves your machine

---

## Log File

| Property | Value |
|---|---|
| **Path** | `~/.animus/logs/performance.log` |
| **Format** | NDJSON (one JSON object per line) |
| **Rotation** | 10 MB max size (renames to `.log.1`) |
| **Retention** | 7 days |

### Log Schema

Each entry contains:

| Field | Type | Description |
|---|---|---|
| `timestamp` | ISO 8601 | Event time (UTC) |
| `phase` | string | What was measured: `model_generate`, `tool_execute`, `memory_recall`, `conversation_save` |
| `duration_ms` | float | Elapsed time in milliseconds |
| `tool_name` | string or null | Tool name (for `tool_execute` phase) |
| `model_provider` | string or null | Provider: `ollama`, `anthropic`, `openai` |
| `success` | boolean or null | Whether the operation succeeded |
| `context_tokens` | integer or null | Tokens in prompt context |
| `response_tokens` | integer or null | Tokens in response |

### Example Log Lines

```json
{"timestamp":"2026-07-01T14:30:00.123456+00:00","phase":"memory_recall","duration_ms":12.345,"success":true}
{"timestamp":"2026-07-01T14:30:00.234567+00:00","phase":"model_generate","duration_ms":2847.123,"model_provider":"ollama","success":true}
{"timestamp":"2026-07-01T14:30:03.456789+00:00","phase":"tool_execute","duration_ms":45.678,"tool_name":"read_file","success":true}
```

---

## Real-Time Summary

View performance statistics in the REPL:

```
>>> /stats --perf
```

**Output:**

```
Performance Summary

  model_generate       n=42  mean=2847.1ms  p95=4123.5ms  max=5234.2ms
  tool_execute         n=15  mean=45.7ms    p95=89.2ms    max=120.4ms
  memory_recall        n=42  mean=12.3ms    p95=28.1ms    max=34.5ms
  conversation_save    n=4   mean=156.2ms   p95=189.0ms   max=203.1ms

  Log file: ~/.animus/logs/performance.log
```

| Statistic | Meaning |
|---|---|
| **n** | Number of events in the window (last 100) |
| **mean** | Average duration |
| **p95** | 95th percentile — 95% of events are faster than this |
| **max** | Slowest event in the window |

---

## Measured Phases

| Phase | Trigger | Typical Range |
|---|---|---|
| `model_generate` | Every LLM response generation | 500 ms – 30 s (depends on model + hardware) |
| `tool_execute` | Every `/tool` invocation or agent loop tool call | 1 ms – 500 ms |
| `memory_recall` | Every user message (context building) | 5 ms – 50 ms |
| `conversation_save` | Every 10 messages (auto-save checkpoint) | 50 ms – 500 ms |

---

## Analysis Tips

### "Is Animus getting slower over time?"

```bash
# Extract model_generate durations over time
cat ~/.animus/logs/performance.log | \
  jq -r 'select(.phase=="model_generate") | [.timestamp, .duration_ms] | @tsv'
```

Look for a rising trend. Common causes:
- **Unbounded message growth** — `conversation.messages` grows without token-aware truncation
- **Cold-start latency** — First Ollama call after idle loads model from disk (slower)
- **Context bloat** — Long tool outputs bloat the context window

### "Which tool is slowest?"

```bash
cat ~/.animus/logs/performance.log | \
  jq -r 'select(.phase=="tool_execute") | [.tool_name, .duration_ms] | @tsv' | \
  sort | awk '{a[$1]+=$2; c[$1]++} END {for(i in a) print i, a[i]/c[i]}'
```

### "Is Ollama slower than Anthropic?"

Compare `model_generate` events filtered by `model_provider`:

```bash
cat ~/.animus/logs/performance.log | \
  jq -r 'select(.phase=="model_generate") | [.model_provider, .duration_ms] | @tsv' | \
  awk '{a[$1]+=$2; c[$1]++} END {for(i in a) print i, a[i]/c[i]}'
```

---

## Limitations

- **No token counting** — `context_tokens` and `response_tokens` are not currently populated (requires model-specific tokenizers)
- **No GPU monitoring** — GPU utilization is not tracked (would require `nvidia-smi` or `rocm-smi`)
- **No warmup detection** — First-call latency after idle is not flagged separately
- **In-memory only** — Real-time summary uses the last 1000 events in memory; older events require reading the log file

---

## Extending the Profiler

To add profiling to new code paths:

```python
from animus.profiler import perf_log

with perf_log("my_phase", model_provider="ollama") as ctx:
    result = do_something()
    ctx["success"] = result.success
    ctx["response_tokens"] = len(result.output.split())
```

The `ctx` dict accepts any extra fields. They will be included in the JSON log entry.

---

## See Also

- [Configuration](configuration.md) — Tuning model provider and backends
- [Working with Local Models](local-models.md) — Ollama performance characteristics
- [CLI Commands Reference](../reference/cli-commands.md) — `/stats --perf`
