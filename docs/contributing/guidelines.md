# Contributing to Animus

Animus is an open project. Contributions are welcome.

---

## Philosophy

This project exists because current AI assistants don't serve users - they serve platforms. If you share that frustration, you're in the right place.

We value:
- **Working code over perfect architecture**
- **User sovereignty over convenience features**
- **Privacy by default over opt-in protection**
- **Transparency over polish**

---

## How to Contribute

### 1. Start Using It

The best contributions come from actual use. Build Animus for yourself. Discover what's missing. Fix what's broken.

### 2. Report Issues

Found a bug? Something unclear? Missing feature? Open an issue.

Good issue reports include:
- What you expected
- What happened
- Steps to reproduce
- Your environment (OS, hardware, etc.)

### 3. Submit Code

Ready to contribute code? Great.

**Before you start:**
- Check existing issues and PRs
- For significant changes, open an issue first to discuss
- Keep changes focused - one PR per feature/fix

**Code standards:**
- Write clear, readable code
- Include tests for new functionality
- Update documentation as needed
- Follow existing patterns in the codebase

**PR process:**
1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit PR with clear description
6. Respond to feedback

### 4. Improve Documentation

Documentation is as important as code. If something was confusing to you, it's confusing to others. Fix it.

### 5. Share Ideas

Not ready to code? Ideas are valuable too. Open a discussion.

---

## Development Setup

### Prerequisites

- Python 3.10+ (Core), 3.11+ (Bootstrap), 3.12+ (Forge)
- Git
- A local LLM setup (Ollama recommended)

### Getting Started

```bash
# Clone the repo
git clone https://github.com/AreteDriver/animus.git
cd animus

# Install all packages in development mode
pip install -e "packages/quorum/[dev]" \
            -e "packages/forge/[dev]" \
            -e "packages/core/[dev]" \
            -e "packages/bootstrap/[dev]"

# Install Ollama and pull a model
# See: https://ollama.ai
ollama pull llama3.1:8b

# Run tests per package
pytest packages/core/tests/ -v
cd packages/forge && pytest tests/ -v && cd ../..
PYTHONPATH=packages/quorum/python pytest packages/quorum/tests/ -v
pytest packages/bootstrap/tests/ -v

# Lint
ruff check packages/ && ruff format --check packages/
```

### Project Structure

```
animus/
├── packages/
│   ├── core/                # Animus Core — exocortex, identity, memory, CLI
│   ├── forge/               # Animus Forge — multi-agent orchestration
│   ├── quorum/              # Animus Quorum — coordination protocol
│   └── bootstrap/           # Animus Bootstrap — install daemon, wizard, dashboard
├── docs/
├── scripts/
└── .github/workflows/
```

---

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Write docstrings for public functions
- Keep functions focused and small

### Commits

- Clear, descriptive commit messages
- Present tense ("Add feature" not "Added feature")
- Reference issues where relevant (#123)

### Testing

- Write tests for new functionality
- Maintain existing test coverage
- Integration tests for cross-module features

---

## Areas Needing Contribution

### Phase 0: Foundation
- [ ] CLI improvements
- [ ] Configuration management
- [ ] Logging infrastructure

### Phase 1: Memory
- [ ] Vector database optimization
- [ ] Memory export/import
- [ ] Search improvements

### Phase 2: Cognitive
- [ ] Tool framework
- [ ] Analysis modes
- [ ] Register translation

### Phase 3: Multi-Interface
- [ ] Mobile interface
- [ ] Voice integration
- [ ] Sync protocol

### Documentation
- [ ] Tutorial/getting started guide
- [ ] API documentation
- [ ] Architecture deep dives

### Testing
- [ ] Unit test coverage
- [ ] Integration tests
- [ ] Performance benchmarks

---

## Communication

### Discussions

Use GitHub Discussions for:
- Feature ideas
- Questions
- General conversation

### Issues

Use GitHub Issues for:
- Bug reports
- Specific feature requests
- Documentation problems

### Pull Requests

Use PRs for:
- Code contributions
- Documentation updates
- Configuration changes

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

---

## Code of Conduct

### Be Respectful

- Treat others as you'd want to be treated
- Assume good intent
- Disagree constructively

### Be Inclusive

- Welcome newcomers
- Help others learn
- Value diverse perspectives

### Be Professional

- Focus on the work
- Keep discussions productive
- Accept feedback gracefully

### Unacceptable Behavior

- Harassment or discrimination
- Personal attacks
- Trolling or inflammatory comments
- Violating others' privacy

Violations may result in removal from the project.

---

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

## Questions?

Open a discussion or reach out. We're building this together.
