# Gorgon Tests

This directory contains tests for Gorgon.

## 🧪 Test Structure

```
tests/
├── README.md              # This file
├── __init__.py            # Test package initialization
├── conftest.py            # Pytest fixtures and configuration
├── unit/                  # Unit tests
│   ├── test_workflow_engine.py
│   ├── test_prompt_manager.py
│   ├── test_api_clients.py
│   └── test_auth.py
├── integration/           # Integration tests
│   ├── test_workflow_execution.py
│   ├── test_api_endpoints.py
│   └── test_dashboard.py
└── e2e/                   # End-to-end tests
    └── test_user_workflows.py
```

## 🚀 Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only
pytest tests/e2e/
```

### Run Specific Test File

```bash
pytest tests/unit/test_workflow_engine.py
```

### Run with Coverage

```bash
# Run with coverage (adjust package path as needed)
pytest --cov=src/test_ai --cov-report=html

# View coverage report at htmlcov/index.html
```

### Run with Verbose Output

```bash
pytest -v
```

## 📝 Writing Tests

### Unit Test Example

```python
# tests/unit/test_workflow_engine.py
# Note: This is example code showing the testing pattern.
# Actual imports should match your implementation.

import pytest
# Example imports - adjust based on actual implementation
# from test_ai import WorkflowEngine, Workflow, WorkflowStep


def test_workflow_execution():
    """Test basic workflow execution.
    
    This example demonstrates the testing pattern.
    Adjust imports and assertions based on actual implementation.
    """
    # Example test code
    # engine = WorkflowEngine()
    # 
    # workflow = Workflow(
    #     workflow_id="test",
    #     name="Test Workflow",
    #     description="Test",
    #     steps=[
    #         WorkflowStep(
    #             step_id="step1",
    #             step_type="openai",
    #             action="generate_completion",
    #             parameters={"prompt": "test"}
    #         )
    #     ]
    # )
    # 
    # result = engine.execute_workflow(workflow)
    # assert result.success is True
    pass
```

### Integration Test Example

```python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from test_ai.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Fixture Example

```python
# tests/conftest.py
import pytest
from test_ai import Settings


@pytest.fixture
def test_settings():
    """Provide test settings."""
    return Settings(
        openai_api_key="test-key",
        secret_key="test-secret"
    )
```

## 🎯 Test Guidelines

### Unit Tests

- **Fast**: Should run in milliseconds
- **Isolated**: No external dependencies
- **Focused**: Test one thing at a time
- **Mocked**: Mock external API calls

### Integration Tests

- **Realistic**: Use real components
- **Controlled**: Use test credentials
- **Cleanup**: Reset state after tests
- **Documented**: Explain test scenarios

### End-to-End Tests

- **Complete**: Test full user workflows
- **Automated**: Run in CI/CD pipeline
- **Reliable**: Handle timing issues
- **Maintainable**: Keep tests simple

## 🔧 Test Configuration

### Environment Variables

Create `.env.test` for test-specific configuration:

```bash
# .env.test
OPENAI_API_KEY=test-key
GITHUB_TOKEN=test-token
NOTION_TOKEN=test-token
```

### Pytest Configuration

`pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

## 📊 Coverage Goals

- **Overall**: >80% coverage
- **Core Logic**: >90% coverage
- **API Clients**: >70% coverage
- **UI Code**: >60% coverage

## 🐛 Debugging Tests

### Run Single Test

```bash
pytest tests/unit/test_workflow_engine.py::test_workflow_execution
```

### Run with Print Statements

```bash
pytest -s
```

### Run with PDB Debugger

```bash
pytest --pdb
```

### Show Fixture Values

```bash
pytest --fixtures
```

## 🤝 Contributing Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Test happy path** and edge cases
3. **Add integration tests** for new endpoints
4. **Update test documentation**
5. **Ensure all tests pass** before submitting PR

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Streamlit Testing](https://docs.streamlit.io/library/advanced-features/testing)

---

**Happy Testing!**
