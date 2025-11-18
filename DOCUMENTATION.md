# AI Agent Testing Framework - Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Testing Dimensions](#testing-dimensions)
8. [Supported Platforms](#supported-platforms)
9. [Enterprise Deployment](#enterprise-deployment)
10. [Troubleshooting](#troubleshooting)

## Overview

The AI Agent Testing Framework is a production-grade validation system for AI agents across three critical dimensions: Task Completion, Trajectory Efficiency, and Memory Consistency. It supports multiple AI platforms including Azure AI Foundry, OpenAI, GitHub Models, LangChain, and custom implementations.

### Key Capabilities

**Task Validation**
- Goal achievement verification
- Constraint satisfaction checking
- Output schema validation
- Execution timeout enforcement

**Trajectory Validation**
- Action sequence efficiency analysis
- Loop and redundancy detection
- Path optimization assessment
- Best practice compliance

**Memory Validation**
- Context retention measurement
- Consistency verification across interactions
- Relevance scoring of retained information
- Capacity management validation

## Architecture

### Core Components

```
test_agent_framework.py
├── TaskValidator        # Validates task completion
├── TrajectoryValidator  # Analyzes action sequences
├── MemoryValidator      # Tests context retention
└── AgentTestSuite       # Orchestrates all validators
```

### Platform Adapters

```
tests/
├── test_azure_ai_agents.py      # Azure AI Foundry adapter
├── test_real_agents.py          # OpenAI, LangChain, Custom adapters
└── test_azure_simple.py         # Simplified Azure testing
```

### Design Principles

- **Security-First**: No eval(), no hardcoded credentials, safe constraint validation
- **Extensible**: Adapter pattern for new platforms
- **Production-Ready**: Logging, error handling, type hints throughout
- **Framework-Agnostic**: Works with any AI agent implementation

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

### Basic Installation

```bash
git clone <repository-url>
cd agent-tester
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
```

### Platform-Specific Setup

**Azure AI Foundry**
```bash
pip install agent-framework-azure-ai --pre
az login
```

**OpenAI / GitHub Models**
```bash
# No additional setup required
# API key set via environment variable
```

**LangChain**
```bash
# Already included in requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Azure AI Foundry
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
AZURE_AI_MODEL_DEPLOYMENT=your-deployment-name

# OpenAI / GitHub Models
OPENAI_API_KEY=your-api-key

# Optional: Anthropic
ANTHROPIC_API_KEY=your-api-key
```

**Security Note**: Never commit `.env` files. Use `.env.example` as a template.

### Framework Configuration

The framework uses sensible defaults. Override via constructor parameters:

```python
from test_agent_framework import TaskValidator, TrajectoryValidator, MemoryValidator

task_validator = TaskValidator(strict_mode=True)
trajectory_validator = TrajectoryValidator(max_actions=20, efficiency_threshold=0.8)
memory_validator = MemoryValidator(max_memory_size=150)
```

## Usage

### Quick Start: Azure AI Foundry

```python
import os
import asyncio
from tests.test_azure_ai_agents import AzureAIFoundryAgentAdapter
from test_agent_framework import TaskDefinition, TaskValidator

# Configure
os.environ['AZURE_AI_PROJECT_ENDPOINT'] = 'your-endpoint'
os.environ['AZURE_AI_MODEL_DEPLOYMENT'] = 'your-deployment'

# Define task
task = TaskDefinition(
    task_id="example_task",
    goal="Analyze customer sentiment from feedback",
    expected_output_schema={"required": ["sentiment", "confidence"]}
)

# Execute
adapter = AzureAIFoundryAgentAdapter()
result = adapter.execute_task(task)

# Validate
validator = TaskValidator()
validation = validator.validate(
    result["output"],
    task,
    result["execution_time"]
)

print(f"Passed: {validation.passed}")
print(f"Goal Achieved: {validation.goal_achieved}")
```

### Quick Start: OpenAI

```python
from tests.test_real_agents import OpenAIAgentAdapter
from test_agent_framework import TaskDefinition

task = TaskDefinition(
    task_id="openai_task",
    goal="Summarize the following text",
    timeout_seconds=30
)

adapter = OpenAIAgentAdapter(model="gpt-4o-mini")
result = adapter.execute_task(task)
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific platform
pytest tests/test_azure_ai_agents.py -v
pytest tests/test_real_agents.py -v

# With HTML report
pytest tests/ --html=report.html --self-contained-html

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## API Reference

### TaskDefinition

Defines a task for agent execution.

```python
TaskDefinition(
    task_id: str,                           # Unique identifier
    goal: str,                              # Task objective
    constraints: List[Dict[str, Any]] = [], # Validation constraints
    expected_output_schema: Dict = {},      # Expected output structure
    timeout_seconds: int = 300,             # Max execution time
    max_retries: int = 3                    # Retry attempts on failure
)
```

### TaskConstraint Types

```python
# Check key existence
{"name": "field", "constraint_type": "key_exists"}

# Check exact value
{"name": "status", "constraint_type": "value_equals", "expected_value": "completed"}

# Check value range
{"name": "score", "constraint_type": "value_in_range", "min_value": 0, "max_value": 100}

# Check list length
{"name": "results", "constraint_type": "list_length", "expected_value": 5}

# Check non-empty
{"name": "description", "constraint_type": "not_empty"}
```

### TaskValidator

```python
validator = TaskValidator(strict_mode=False)
result = validator.validate(output: Dict, task: TaskDefinition, execution_time: float)

# Returns TaskValidationResult with:
# - passed: bool
# - goal_achieved: bool
# - constraints_met: Dict[str, bool]
# - output_valid: bool
# - execution_time: float
# - error_message: Optional[str]
```

### TrajectoryValidator

```python
validator = TrajectoryValidator(
    max_actions=15,              # Maximum allowed actions
    efficiency_threshold=0.7     # Minimum efficiency score (0-1)
)

result = validator.validate(trajectory: List[Dict])

# Returns TrajectoryValidationResult with:
# - is_efficient: bool
# - action_count: int
# - has_loops: bool
# - redundancy_score: float
# - efficiency_score: float
```

### MemoryValidator

```python
validator = MemoryValidator(max_memory_size=100)
result = validator.validate(
    memory: AgentMemory,
    conversation: List[Dict]
)

# Returns MemoryValidationResult with:
# - context_retention_score: float  # 0-100
# - consistency_score: float        # 0-100
# - relevance_score: float          # 0-100
# - within_capacity: bool
```

## Testing Dimensions

### 1. Task Validation

Ensures agents complete assigned tasks correctly.

**Validation Checks**:
- Goal achievement (fuzzy matching, keyword presence)
- Constraint satisfaction (all specified constraints met)
- Output schema compliance (required fields present)
- Timeout compliance (execution within limits)

**Use Cases**:
- Pre-deployment testing
- Regression testing
- Compliance verification

### 2. Trajectory Validation

Analyzes agent's action sequence for efficiency.

**Validation Checks**:
- Action count (within reasonable limits)
- Loop detection (no repeated action patterns)
- Redundancy analysis (minimal duplicate actions)
- Logical flow (coherent action sequence)

**Use Cases**:
- Performance optimization
- Cost reduction (fewer API calls)
- Debugging agent behavior

### 3. Memory Validation

Tests agent's context retention and consistency.

**Validation Checks**:
- Context retention (previous interactions remembered)
- Consistency (no contradictory statements)
- Relevance (important information prioritized)
- Capacity management (within memory limits)

**Use Cases**:
- Multi-turn conversation testing
- Long-running agent validation
- Context window optimization

## Supported Platforms

### Azure AI Foundry

**Features**:
- Agent creation and management
- Custom tool integration
- Streaming responses
- Azure AD authentication

**Adapter**: `tests.test_azure_ai_agents.AzureAIFoundryAgentAdapter`

**Example**:
```python
from tests.test_azure_ai_agents import AzureAIFoundryAgentWithToolsAdapter

adapter = AzureAIFoundryAgentWithToolsAdapter(tools=[get_weather, calculate_cost])
result = adapter.execute_task(task)
```

### OpenAI

**Features**:
- Chat completions API
- Function calling
- Multiple models support

**Adapter**: `tests.test_real_agents.OpenAIAgentAdapter`

**Example**:
```python
from tests.test_real_agents import OpenAIAgentAdapter

adapter = OpenAIAgentAdapter(model="gpt-4o-mini")
result = adapter.execute_task(task)
```

### LangChain

**Features**:
- Agent framework integration
- Tool usage
- Memory management

**Adapter**: `tests.test_real_agents.LangChainAgentAdapter`

**Example**:
```python
from tests.test_real_agents import LangChainAgentAdapter
from langchain_openai import ChatOpenAI

adapter = LangChainAgentAdapter(llm=ChatOpenAI(model="gpt-4o-mini"), tools=[])
result = adapter.execute_task(task)
```

### Custom Agents

**Features**:
- Bring your own agent implementation
- Minimal interface requirements

**Adapter**: `tests.test_real_agents.CustomAgentAdapter`

**Example**:
```python
from tests.test_real_agents import CustomAgentAdapter

class MyAgent:
    def run(self, prompt):
        return {"result": "agent response"}

adapter = CustomAgentAdapter(MyAgent())
result = adapter.execute_task(task)
```

## Enterprise Deployment

### CI/CD Integration

**GitHub Actions Example**:
```yaml
name: Agent Testing

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: |
          pip install -r requirements.txt
          pytest tests/ -v --html=report.html
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          AZURE_AI_PROJECT_ENDPOINT: ${{ secrets.AZURE_ENDPOINT }}
      - uses: actions/upload-artifact@v3
        with:
          name: test-report
          path: report.html
```

### Azure DevOps Example

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.10'
- script: |
    pip install -r requirements.txt
    pytest tests/ -v --junitxml=test-results.xml
  env:
    AZURE_AI_PROJECT_ENDPOINT: $(AZURE_ENDPOINT)
    AZURE_AI_MODEL_DEPLOYMENT: $(AZURE_DEPLOYMENT)
- task: PublishTestResults@2
  inputs:
    testResultsFiles: 'test-results.xml'
```

### Production Best Practices

1. **Secrets Management**
   - Use Azure Key Vault or AWS Secrets Manager
   - Never commit credentials
   - Rotate API keys regularly

2. **Monitoring**
   - Configure logging to central system (Azure Monitor, CloudWatch)
   - Set up alerts for test failures
   - Track performance metrics

3. **Scalability**
   - Run tests in parallel using pytest-xdist
   - Use containerization (Docker)
   - Implement test sharding for large suites

4. **Version Control**
   - Pin dependency versions
   - Tag releases
   - Maintain changelog

## Troubleshooting

### Common Issues

**ImportError: No module named 'agent_framework'**
```bash
pip install agent-framework-azure-ai --pre
```

**Azure Authentication Failed**
```bash
az login
az account show  # Verify correct subscription
```

**OpenAI API Key Not Found**
```bash
export OPENAI_API_KEY="your-key"  # Linux/Mac
$env:OPENAI_API_KEY="your-key"    # Windows PowerShell
```

**Tests Timeout**
- Increase `timeout_seconds` in TaskDefinition
- Check network connectivity
- Verify API rate limits

**Memory Validation Fails**
- Increase `max_memory_size` in MemoryValidator
- Check conversation history length
- Verify agent memory implementation

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Issues

- Use pytest-xdist for parallel execution: `pytest -n auto`
- Enable caching where appropriate
- Monitor API rate limits
- Consider using smaller models for development

## Comparison with Alternatives

| Feature | This Framework | LangSmith | PromptLayer | DeepEval |
|---------|---------------|-----------|-------------|----------|
| Task Validation | Yes | Limited | No | Yes |
| Trajectory Analysis | Yes | No | No | Limited |
| Memory Testing | Yes | No | No | No |
| Multi-Platform Support | Yes | Limited | Yes | Limited |
| Open Source | Yes | No | No | Yes |
| Self-Hosted | Yes | No | No | Yes |
| Enterprise Ready | Yes | Yes | Yes | Partial |
| Cost | Free | Paid | Paid | Free |

### Competitive Advantages

1. **Comprehensive**: Only framework testing all three dimensions (Task, Trajectory, Memory)
2. **Platform-Agnostic**: Works with Azure, OpenAI, LangChain, custom implementations
3. **Production-Ready**: Security, logging, error handling built-in
4. **Open Source**: No vendor lock-in, full customization
5. **Enterprise-Grade**: CI/CD integration, scalable, well-documented

### When to Use This Framework

**Best For**:
- Organizations with custom AI agent implementations
- Teams requiring comprehensive validation beyond basic metrics
- Enterprises needing self-hosted testing solutions
- Projects using multiple AI platforms
- Development teams prioritizing security and compliance

**Consider Alternatives If**:
- You only need basic prompt testing (use PromptLayer)
- You're exclusively on LangChain ecosystem (use LangSmith)
- You need managed SaaS solution with minimal setup

## Support and Contributing

See CONTRIBUTING.md for contribution guidelines.
See SECURITY.md for security vulnerability reporting.

## License

MIT License - See LICENSE file for details.
