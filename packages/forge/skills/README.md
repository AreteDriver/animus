# Gorgon Skills Library

This directory contains the skill definitions for the Gorgon multi-agent orchestration system. Skills provide agents with structured capabilities, safety guardrails, and best practices for executing tasks.

## Directory Structure

```
gorgon-skills/
├── registry.yaml              # Master index of all skills
├── README.md                  # This file
│
├── system/                    # System Agent skills
│   ├── file_operations/
│   │   ├── SKILL.md          # Instructions and examples
│   │   └── schema.yaml       # Input/output contract
│   └── process_management/
│       ├── SKILL.md
│       └── schema.yaml
│
├── browser/                   # Browser Agent skills
│   ├── web_search/
│   │   ├── SKILL.md
│   │   └── schema.yaml
│   └── web_scrape/
│       ├── SKILL.md
│       └── schema.yaml
│
├── email/                     # Email Agent skills
│   └── compose/
│       ├── SKILL.md
│       └── schema.yaml
│
└── integrations/              # Third-party integrations
    └── github/
        ├── SKILL.md
        └── schema.yaml
```

## Skill Anatomy

Each skill consists of two files:

### SKILL.md
Human and LLM-readable instructions containing:
- **Purpose**: What the skill does
- **Safety Rules**: Critical guardrails and restrictions
- **Consensus Requirements**: What level of Triumvirate approval is needed
- **Capabilities**: Detailed documentation for each operation
- **Examples**: Step-by-step usage examples
- **Error Handling**: How to handle common failures

### schema.yaml
Machine-readable contract containing:
- **Capability definitions**: Input/output types and validation
- **Risk levels**: Classification for consensus routing
- **Dependencies**: Required and optional packages
- **Configuration**: Required settings

## Consensus Levels

The Triumvirate (Zorya Utrennyaya, Zorya Vechernyaya, Zorya Polunochnaya) uses these consensus levels:

| Level | Requirement | Use Case |
|-------|-------------|----------|
| `any` | 1 of 3 Zorya | Low-risk, read-only operations |
| `majority` | 2 of 3 Zorya | Medium-risk, recoverable operations |
| `unanimous` | 3 of 3 Zorya | High-risk, potentially irreversible |
| `unanimous + user` | 3 of 3 + explicit confirmation | Critical operations (email send, delete) |

## Adding New Skills

1. Create directory under appropriate category:
   ```
   mkdir -p skills/category/skill_name
   ```

2. Create SKILL.md following the template:
   ```markdown
   ---
   name: skill_name
   version: 1.0.0
   agent: system|browser|email
   risk_level: low|medium|high|critical
   description: "Brief description"
   ---
   
   # Skill Name
   
   ## Purpose
   ...
   
   ## Safety Rules
   ...
   
   ## Capabilities
   ...
   
   ## Examples
   ...
   ```

3. Create schema.yaml with capability definitions

4. Add entry to registry.yaml

5. Test skill with dry-run mode

## Using Skills in Agents

```python
from gorgon.skills import SkillLibrary

# Load all skills
library = SkillLibrary()

# Get skills for specific agent
system_skills = library.get_skills_for_agent("system")

# Get specific skill
file_ops = library.get_skill("file_operations")

# Build agent prompt with skill context
prompt = agent.build_system_prompt(task)  # Injects relevant skills
```

## Skill Development Guidelines

1. **Safety First**: Always define protected paths, dangerous patterns
2. **Explicit Consensus**: Every capability must declare its risk level
3. **Examples Matter**: Provide real, working examples
4. **Error Handling**: Document all failure modes
5. **Minimal Permissions**: Use least-privilege principle
6. **Audit Trail**: Operations should be loggable

## Testing Skills

```bash
# Dry run a skill operation
gorgon --dry-run "delete all .tmp files in /home/gorgon"

# Test specific skill
gorgon skills test file_operations

# Validate skill schema
gorgon skills validate system/file_operations
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add/modify skills following guidelines
4. Submit PR with examples demonstrating the skill

## License

MIT License - See LICENSE file
