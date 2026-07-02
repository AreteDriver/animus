# Working with Local Models

> Why natural language mode behaves differently with Ollama, and how to work around it.

---

## The Problem

When Animus uses **Ollama** (local models like Llama 3, Mistral, etc.), the natural language agent loop is **disabled by default**.

Why? The agent loop requires the model to follow a strict structured format:

```
TOOL: 3
path: /home/user/file.py
```

Local 7B–8B parameter models struggle with this level of instruction-following precision. Instead of selecting a tool, they tend to:

- Output prose summarizing the tool menu
- Repeat `TOOL: 12` without executing
- Hit the max-iteration limit and bail with garbled output

This is not a bug in Animus — it's a limitation of smaller local models.

---

## The Workaround: Direct Tool Invocation

When using Ollama, execute tools **directly** with the `/tool` command instead of describing what you want in natural language.

### Syntax

```
/tool <name> [param=value ...]
```

### Examples

```
>>> /tool get_datetime
# 2026-07-01 14:30:00

>>> /tool read_file path=/etc/hostname
# myhostname

>>> /tool list_files path=/home/user/projects
# [file list]

>>> /tool web_search query="Python context managers"
# [search results]

>>> /tool animus_watchlist_list
# [watchlist items]
```

**Positional arguments:** For tools with a single main parameter, you can omit `param=`:

```
>>> /tool web_search Python context managers
# Equivalent to query="Python context managers"
```

**See all tools:** Run `/tools` for a full list with parameter descriptions.

---

## What Works in Natural Language Mode

Even with Ollama, natural language still works for:

- **Basic conversation** — "Hello", "What time is it?" (Animus will respond without tools)
- **Memory commands** — `/remember`, `/recall`, `/tags` (these are REPL commands, not agent loop)
- **Reasoning modes** — `/deep`, `/research`, `/brief` (these use `think()`, not `think_with_tools()`)

Only **tool execution via natural language** is disabled.

---

## When the Agent Loop Works

The agent loop is fully functional when using:

- **Anthropic Claude** — Native `tool_use` content blocks (most reliable)
- **OpenAI GPT models** — Native function calling

If you have an `ANTHROPIC_API_KEY` set, Animus automatically enables dual-model routing: Claude handles planning and tool selection, Ollama handles cheap local execution.

---

## Tips for Local Model Users

1. **Learn the tool names** — Run `/tools` once at the start of each session
2. **Use `/tool` for file operations** — `read_file`, `write_file`, `edit_file`, `list_files`
3. **Use `/tool` for web access** — `web_search`, `http_request`
4. **Use memory commands directly** — `/remember`, `/recall` instead of "remember that I like pizza"
5. **Consider a larger model** — Models with 13B+ parameters (e.g., `llama3:70b`) handle structured formats better, though they require more VRAM

---

## Model Recommendations

| Model | Size | Tool Reliability | VRAM Required |
|---|---|---|---|
| `llama3:8b` | 8B | Low | ~6 GB |
| `llama3:70b` | 70B | Medium | ~40 GB |
| `mistral:7b` | 7B | Low | ~6 GB |
| `qwen2.5:14b` | 14B | Medium | ~10 GB |
| `deepseek-coder-v2` | 16B | Medium | ~12 GB |

**General rule:** Models under 13B parameters struggle with structured tool formats. Use `/tool` direct invocation.

---

## See Also

- [Ollama Setup](../getting-started/ollama-setup.md) — Installation and configuration
- [Tools Reference](../reference/tools.md) — All available tools and parameters
- [CLI Commands Reference](../reference/cli-commands.md) — Full REPL command list
