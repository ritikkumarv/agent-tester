# Agent Tester Package Structure

This package has been reorganized for better maintainability and ease of use.

## Package Structure

```
agent_tester/
├── __init__.py           # Main package exports
├── models.py             # Data models (Task, Trajectory, Memory, etc.)
├── cli.py                # Command-line interface
├── suite.py              # Test suite orchestration
├── validators/           # Validation modules
│   ├── __init__.py
│   ├── task_validator.py
│   ├── trajectory_validator.py
│   └── memory_validator.py
└── adapters/             # Platform adapters (Azure, OpenAI, etc.)
    └── __init__.py
```

## Quick Start

### Installation

```bash
pip install -e .
```

### CLI Usage

```bash
# Initialize a test configuration
agent-tester init

# Run tests
agent-tester run -c my_tests.yaml

# See examples
agent-tester examples
```

### Python API

```python
from agent_tester import TaskDefinition, TaskValidator

# Define a task
task = TaskDefinition(
    task_id="test1",
    goal="Do something useful",
    timeout_seconds=30
)

# Validate results
validator = TaskValidator()
result = validator.validate(agent_output, task, execution_time)

print(f"Passed: {result.passed}")
```

## Development

### Running Tests

```bash
# All tests
pytest test_agent_framework.py -v

# With coverage
pytest test_agent_framework.py --cov=agent_tester --cov-report=html

# Specific tests
pytest test_agent_framework.py -k "task" -v
```

### Building Package

```bash
pip install build
python -m build
```

## Migration from Old Structure

The old `test_agent_framework.py` file now acts as a backward compatibility wrapper. 
All core functionality has been moved into the `agent_tester` package.

### Old Code (still works)

```python
from test_agent_framework import TaskValidator, TaskDefinition
```

### New Code (recommended)

```python
from agent_tester import TaskValidator, TaskDefinition
```

Both imports will work, but the new package structure is recommended for maintainability.
