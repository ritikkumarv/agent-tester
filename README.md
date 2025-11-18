# AI Agent Testing Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)

Production-grade testing framework for AI agents. Validates Task completion, Trajectory efficiency, and Memory consistency across Azure AI Foundry, OpenAI, LangChain, and custom implementations.

## Features

- **Task Validation**: Goal achievement, constraint satisfaction, output schema compliance
- **Trajectory Validation**: Action efficiency, loop detection, path optimization
- **Memory Validation**: Context retention, consistency checking, relevance scoring
- **Multi-Platform**: Azure AI Foundry, OpenAI, GitHub Models, LangChain, custom agents
- **Enterprise-Ready**: Security-first design, comprehensive logging, CI/CD integration
- **Extensible**: Adapter pattern for custom platforms

## Quick Start

### Installation

```bash
git clone <repository-url>
cd agent-tester
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
```

### Azure AI Foundry

```bash
# Install Azure SDK
pip install agent-framework-azure-ai --pre

# Configure
export AZURE_AI_PROJECT_ENDPOINT="https://your-resource.services.ai.azure.com/api/projects/your-project"
export AZURE_AI_MODEL_DEPLOYMENT="your-deployment-name"

# Test
python tests/test_azure_simple.py
```

### OpenAI

```bash
# Configure
export OPENAI_API_KEY="your-api-key"

# Test
python examples/example_test_openai_agent.py
```

## Usage

```python
from tests.test_azure_ai_agents import AzureAIFoundryAgentAdapter
from test_agent_framework import TaskDefinition, TaskValidator

# Define task
task = TaskDefinition(
    task_id="sentiment_analysis",
    goal="Analyze customer sentiment",
    expected_output_schema={"required": ["sentiment", "confidence"]}
)

# Execute
adapter = AzureAIFoundryAgentAdapter()
result = adapter.execute_task(task)

# Validate
validator = TaskValidator()
validation = validator.validate(result["output"], task, result["execution_time"])

print(f"Passed: {validation.passed}")
```

## Project Structure

```
agent-tester/
├── test_agent_framework.py    # Core framework
├── tests/                      # Platform adapters
│   ├── test_azure_ai_agents.py
│   ├── test_real_agents.py
│   └── test_azure_simple.py
├── examples/                   # Usage examples
├── DOCUMENTATION.md            # Complete documentation
├── SECURITY.md                 # Security policy
├── CONTRIBUTING.md             # Contribution guidelines
└── requirements.txt            # Dependencies
```

## Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Generate HTML report
pytest tests/ --html=report.html --self-contained-html
```

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for:
- Complete API reference
- Platform-specific guides
- Enterprise deployment patterns
- CI/CD integration examples
- Troubleshooting guide
- Competitive analysis

## Security

This framework follows security best practices:
- No code execution vulnerabilities
- Environment variable-based configuration
- Pinned dependencies
- Comprehensive input validation

Report security issues per [SECURITY.md](SECURITY.md).

## Enterprise Deployment

### CI/CD Integration

```yaml
# GitHub Actions
- run: |
    pip install -r requirements.txt
    pytest tests/ -v --html=report.html
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Supported Platforms

- Azure AI Foundry
- OpenAI
- GitHub Models
- LangChain
- Custom Agents

## Comparison with Alternatives

| Feature | This Framework | LangSmith | DeepEval |
|---------|---------------|-----------|----------|
| Task Validation | Yes | Limited | Yes |
| Trajectory Analysis | Yes | No | Limited |
| Memory Testing | Yes | No | No |
| Multi-Platform | Yes | Limited | Limited |
| Self-Hosted | Yes | No | Yes |
| Open Source | Yes | No | Yes |

**Key Differentiators**:
- Only framework testing all three dimensions
- Platform-agnostic architecture
- Production-ready security and logging
- No vendor lock-in

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).

## Support

- Documentation: [DOCUMENTATION.md](DOCUMENTATION.md)
- Issues: GitHub Issues
- Security: [SECURITY.md](SECURITY.md)
