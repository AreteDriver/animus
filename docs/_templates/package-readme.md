# Package README Template

> Use this template when creating or refreshing a package README.

---

# Package Name

**One-sentence summary of what this package does.**

## Features

- **Feature 1** — Brief description
- **Feature 2** — Brief description
- **Feature 3** — Brief description

## Install

```bash
pip install package-name
```

With optional extras:
```bash
pip install "package-name[extra1,extra2]"
```

## Quick Start

```bash
# Example CLI usage
package-name --help
```

```python
# Example Python usage
from package_name import SomeClass

obj = SomeClass()
result = obj.do_something()
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SOME_VAR` | `default` | What it controls |

## Part of the Animus Monorepo

- [Animus Core](https://github.com/your-org/animus/tree/main/packages/core) — exocortex engine
- [Animus Forge](https://github.com/your-org/animus/tree/main/packages/forge) — orchestration
- [Animus Quorum](https://pypi.org/project/convergentAI/) — coordination protocol
- [Animus Bootstrap](https://github.com/your-org/animus/tree/main/packages/bootstrap) — system daemon

## Development

```bash
git clone git@github.com:your-org/animus.git
cd animus/packages/<package-name>
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT — 2026, your-org
