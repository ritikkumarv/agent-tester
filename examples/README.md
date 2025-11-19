# Agent Tester Examples

This directory contains example scripts demonstrating how to use the Agent Tester framework.

## Available Examples

### 1. Simple Example (`simple_example.py`)
Basic demonstration using a mock agent to show the complete testing workflow.

```bash
python examples/simple_example.py
```

**What it demonstrates:**
- Creating a mock agent
- Defining tasks
- Running validations (Task, Trajectory, Memory)
- Interpreting results

### 2. OpenAI Example (`openai_example.py`)
Shows how to test agents using the OpenAI adapter.

```bash
# Set your API key (optional - will use mock if not set)
export OPENAI_API_KEY="your-api-key"

# Run the example
python examples/openai_example.py
```

**What it demonstrates:**
- Using the OpenAI adapter
- Real API integration (if key is set)
- Fallback to mock adapter (if key not set)
- Task validation with structured output

### 3. Comprehensive Test Suite (`create_comprehensive_tests.py`)
Creates a sample test suite configuration file.

```bash
# Create the configuration file
python examples/create_comprehensive_tests.py

# Run the test suite using CLI
agent-tester run -c comprehensive_tests.yaml
```

**What it demonstrates:**
- YAML test configuration
- Multiple test scenarios
- Different output schemas
- Constraints and validations

## Using the CLI

### Initialize a new test configuration
```bash
agent-tester init
```

This creates an `agent_tests.yaml` file you can customize.

### Run tests from configuration
```bash
agent-tester run -c agent_tests.yaml
```

### Quick single task validation
```bash
agent-tester validate test1 --goal "Summarize this text"
```

### View all available commands
```bash
agent-tester --help
```

## Setting up API Keys

### OpenAI
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Azure AI Foundry
```bash
export AZURE_AI_PROJECT_ENDPOINT="https://your-resource.services.ai.azure.com/api/projects/your-project"
export AZURE_AI_MODEL_DEPLOYMENT="your-deployment-name"
```

## Example Test Configuration (YAML)

```yaml
name: My Agent Tests
description: Testing my AI agent

tests:
  - task_id: test_1
    goal: "Answer: What is 2+2?"
    expected_output_schema:
      required: ["answer"]
    timeout_seconds: 30
  
  - task_id: test_2
    goal: "Analyze sentiment: 'I love this!'"
    expected_output_schema:
      required: ["sentiment", "confidence"]
    timeout_seconds: 30

validators:
  task:
    strict_mode: false
  trajectory:
    max_actions: 20
  memory:
    min_retention_score: 0.7
```

## Next Steps

- Read the [QUICKSTART.md](../QUICKSTART.md) for a quick introduction
- Check [DOCUMENTATION.md](../DOCUMENTATION.md) for detailed API reference
- See [README.md](../README.md) for framework overview

## Need Help?

- Check existing test files in `tests/` directory
- Review the main test file `test_agent_framework.py`
- Read the comprehensive documentation
