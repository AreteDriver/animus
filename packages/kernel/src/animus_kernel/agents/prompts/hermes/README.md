# Hermes Prompts

Hermes-format XML system prompts for Animus kernel agents.

## Usage

```python
from animus_kernel.agents.prompts.hermes import get_role_prompt

prompt = get_role_prompt("builder")
```

## Roles

| File | Role | Tool Access |
|---|---|---|
| `planner.xml` | Strategic planning, task decomposition | No |
| `builder.xml` | Code implementation, bug fixes | Yes |
| `tester.xml` | Test suite creation, QA automation | Yes |
| `reviewer.xml` | Code review, security audits | Yes |
| `architect.xml` | System design, technology decisions | No |
| `documenter.xml` | Documentation, guides, tutorials | No |

## Format

Each prompt is an XML document with a `<system>` block.
Tool-equipped roles declare available tools in a `<tools>` block,
and the model emits structured tool calls:

```xml
<tool_call>
<name>read_file</name>
<params>{"path": "src/main.py"}</params>
</tool_call>
```

Tool schemas match the `ForgeToolRegistry` definitions in
`kernel/tools/registry.py`.
