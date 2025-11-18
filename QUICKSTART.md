# 🚀 Quick Start Guide

Get started with Agent Tester in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/ritikkumarv/agent-tester.git
cd agent-tester

# Install the package
pip install -e .

# Verify installation
agent-tester --version
```

## Your First Test (3 Steps!)

### Step 1: Initialize Test Configuration

```bash
agent-tester init
```

This creates a sample `agent_tests.yaml` file with example tests.

### Step 2: Customize Your Tests

Edit `agent_tests.yaml`:

```yaml
name: My Agent Tests
description: Testing my AI agent

tests:
  - task_id: summarization_test
    goal: "Summarize a technical document"
    constraints:
      - name: word_count
        type: value_in_range
        min_value: 50
        max_value: 200
    expected_output_schema:
      required: ["summary"]
    timeout_seconds: 60
```

### Step 3: Run Your Tests

```bash
agent-tester run -c agent_tests.yaml
```

## Using the Python API

### Basic Example

```python
from agent_tester import TaskDefinition, TaskValidator

# 1. Define your test
task = TaskDefinition(
    task_id="test_sentiment",
    goal="Analyze customer sentiment",
    expected_output_schema={"required": ["sentiment", "confidence"]},
    timeout_seconds=30
)

# 2. Run your agent
result = my_agent.execute(task)

# 3. Validate the results
validator = TaskValidator()
validation = validator.validate(
    result["output"], 
    task, 
    result["execution_time"]
)

# 4. Check if it passed
if validation.passed:
    print("✅ Test passed!")
else:
    print("❌ Test failed:")
    print(f"  Goal achieved: {validation.goal_achieved}")
    print(f"  Constraints met: {validation.constraints_met}")
```

### Complete Example with All Validators

```python
from agent_tester import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator,
    AgentTestSuite
)

# Define multiple tests
tests = [
    TaskDefinition(
        task_id="test1",
        goal="Summarize text",
        expected_output_schema={"required": ["summary"]}
    ),
    TaskDefinition(
        task_id="test2",
        goal="Answer questions",
        expected_output_schema={"required": ["answer", "confidence"]}
    )
]

# Create test suite
suite = AgentTestSuite(
    task_validator=TaskValidator(strict_mode=False),
    trajectory_validator=TrajectoryValidator(max_actions=20),
    memory_validator=MemoryValidator(min_retention_score=0.7)
)

# Run all tests
results = suite.run_all_tests(my_agent, tests)

# Generate report
suite.generate_html_report(results, "report.html")

print(f"Pass rate: {results['summary']['pass_rate']:.1f}%")
```

## Testing Different Platforms

### Azure AI Foundry

```python
from agent_tester.adapters.azure import AzureAIFoundryAdapter
from agent_tester import TaskDefinition, TaskValidator

adapter = AzureAIFoundryAdapter(
    endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
    deployment=os.getenv("AZURE_AI_MODEL_DEPLOYMENT")
)

task = TaskDefinition(
    task_id="azure_test",
    goal="Process customer inquiry"
)

result = adapter.execute_task(task)
```

### OpenAI

```python
from agent_tester.adapters.openai import OpenAIAdapter
from agent_tester import TaskDefinition

adapter = OpenAIAdapter(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

task = TaskDefinition(
    task_id="openai_test",
    goal="Generate product description"
)

result = adapter.execute_task(task)
```

## CLI Commands Reference

```bash
# Initialize new test configuration
agent-tester init

# Run tests from configuration
agent-tester run -c my_tests.yaml

# Run with custom output
agent-tester run -c my_tests.yaml -o custom_report.html

# Quick validation (coming soon)
agent-tester validate task1 --goal "Summarize document"

# Show examples
agent-tester examples

# Show version
agent-tester version

# Help
agent-tester --help
```

## Configuration File Format

### YAML Format (Recommended)

```yaml
name: My Test Suite
description: Comprehensive agent tests

tests:
  - task_id: test_1
    goal: "Complete task successfully"
    constraints:
      - name: budget
        type: value_in_range
        min_value: 0
        max_value: 1000
    expected_output_schema:
      required: ["result", "cost"]
    timeout_seconds: 60

  - task_id: test_2
    goal: "Another test"
    expected_output_schema:
      required: ["output"]
    timeout_seconds: 30

validators:
  task:
    strict_mode: false
  trajectory:
    max_actions: 20
    allow_backtracking: true
  memory:
    min_retention_score: 0.7
```

### JSON Format

```json
{
  "name": "My Test Suite",
  "description": "Comprehensive agent tests",
  "tests": [
    {
      "task_id": "test_1",
      "goal": "Complete task successfully",
      "expected_output_schema": {
        "required": ["result"]
      },
      "timeout_seconds": 60
    }
  ],
  "validators": {
    "task": {"strict_mode": false},
    "trajectory": {"max_actions": 20},
    "memory": {"min_retention_score": 0.7}
  }
}
```

## Constraint Types

Agent Tester supports various constraint types for validation:

### key_exists
Check if a key exists in the output:
```yaml
constraints:
  - name: result
    type: key_exists
```

### value_equals
Check if a value equals expected:
```yaml
constraints:
  - name: status
    type: value_equals
    expected_value: "success"
```

### value_in_range
Check if a value is within range:
```yaml
constraints:
  - name: score
    type: value_in_range
    min_value: 0
    max_value: 100
```

### list_length
Check list length:
```yaml
constraints:
  - name: results
    type: list_length
    expected_value: 5
```

### not_empty
Check if value is not empty:
```yaml
constraints:
  - name: description
    type: not_empty
```

## Next Steps

1. 📚 Read the full [Documentation](DOCUMENTATION.md)
2. 🔍 Explore [Examples](examples/)
3. 🤝 See [Contributing Guidelines](CONTRIBUTING.md)
4. 🔒 Review [Security Policy](SECURITY.md)

## Get Help

- 📖 Documentation: [DOCUMENTATION.md](DOCUMENTATION.md)
- 🐛 Issues: [GitHub Issues](https://github.com/ritikkumarv/agent-tester/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/ritikkumarv/agent-tester/discussions)

Happy Testing! 🎉
