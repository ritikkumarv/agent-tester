# 🤖 Agent Tester

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/agent-tester/)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)

**Like Postman for APIs, but for AI Agents** 🚀

Production-grade testing framework for AI agents. Validates Task completion, Trajectory efficiency, and Memory consistency across Azure AI Foundry, OpenAI, LangChain, and custom implementations.

---

## Features

- **Task Validation**: Goal achievement, constraint satisfaction, output schema compliance
- **Trajectory Validation**: Action efficiency, loop detection, path optimization
- **Memory Validation**: Context retention, consistency checking, relevance scoring
- **🔒 Security Analysis**: SAST scanning, dependency vulnerabilities, configuration security
- **🛡️ Security Scanning**: Built-in SAST, dependency scanning, secret detection, and configuration security checks
- **Multi-Platform**: Azure AI Foundry, OpenAI, GitHub Models, LangChain, custom agents
- **Enterprise-Ready**: Security-first design, comprehensive logging, CI/CD integration
- **Extensible**: Adapter pattern for custom platforms

## 🎯 Why Agent Tester?

Testing AI agents shouldn't be harder than testing APIs. Agent Tester brings the simplicity of Postman to AI agent testing:

- ✅ **Simple CLI** - Test agents with a single command
- 📝 **YAML/JSON Configuration** - Define tests like Postman collections
- 🎨 **Rich Output** - Beautiful, readable test results  
- 🔌 **Multi-Platform** - Works with Azure, OpenAI, LangChain, and more
- 🚀 **Production-Ready** - Enterprise-grade validation and reporting
- 🛡️ **Built-in Security** - Automated vulnerability scanning and security best practices

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (coming soon)
pip install agent-tester

# Or install from source
git clone https://github.com/ritikkumarv/agent-tester.git
cd agent-tester
pip install -e .
```

### Your First Test (30 seconds!)

```bash
# 1. Initialize a test configuration
agent-tester init

# 2. Edit agent_tests.yaml to define your tests
# 3. Run tests
agent-tester run -c agent_tests.yaml

# 4. View beautiful HTML report
open test_report.html
```

**👉 [Read the Complete Quick Start Guide](QUICKSTART.md)**

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

### Anthropic (Claude)

```bash
# Install Anthropic SDK
pip install anthropic

# Configure
export ANTHROPIC_API_KEY="your-api-key"

# Test
python examples/example_anthropic_agent.py
```

## 💻 Usage

### CLI (Recommended - Postman-like Experience)

```bash
# Initialize test configuration
agent-tester init

# Run all tests
agent-tester run -c my_tests.yaml

# Run with custom output
agent-tester run -c my_tests.yaml -o custom_report.html

# Quick validation
agent-tester validate task1 --goal "Summarize this document"

# Run security scan
agent-tester security

# Run security scan with custom output
agent-tester security --format json --output security_report.json
# Security Scanning
agent-tester security scan                    # Run full security scan
agent-tester security scan --type sast        # Run only SAST
agent-tester security scan --type dependency  # Check dependencies
agent-tester security report                  # Quick security summary
agent-tester security check-deps              # Check for vulnerable deps

# See all commands
agent-tester --help

# View examples
agent-tester examples
```

### Security Scanning

```bash
# Scan current directory
agent-tester security

# Scan specific path
agent-tester security --path /path/to/project

# Show only critical/high severity issues
agent-tester security --severity high

# Generate JSON report
agent-tester security --format json --output report.json
```

### Python API (For Programmatic Testing)

```python
from agent_tester import (
    TaskDefinition, 
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator
)

# Define your test
task = TaskDefinition(
    task_id="sentiment_analysis",
    goal="Analyze customer sentiment from reviews",
    expected_output_schema={"required": ["sentiment", "confidence"]},
    timeout_seconds=30
)

# Run your agent
result = my_agent.execute(task)

# Validate results
validator = TaskValidator()
validation = validator.validate(
    result["output"], 
    task, 
    result["execution_time"]
)

print(f"✅ Passed: {validation.passed}")
print(f"Goal Achieved: {validation.goal_achieved}")
print(f"Constraints Met: {all(validation.constraints_met.values())}")
```

### Security API

```python
from agent_tester.security import SecurityValidator

# Run comprehensive security scan
validator = SecurityValidator()
report = validator.validate_repository("./my-project")

# Get summary
summary = report.get_summary()
print(f"Total Issues: {summary['total_issues']}")
print(f"Critical: {summary['critical']}")
print(f"High: {summary['high']}")

# Get critical issues
for issue in report.critical_issues:
    print(f"🔴 {issue.title}")
    print(f"   File: {issue.file_path}:{issue.line_number}")
    print(f"   Fix: {issue.recommendation}")

# Export reports
validator.export_report(report, format="markdown", output_file="security_report.md")
validator.export_report(report, format="json", output_file="security_report.json")
```

### Test Configuration Format (YAML)

```yaml
name: My Agent Test Suite
description: Comprehensive tests for my AI agent

tests:
  - task_id: test_1
    goal: "Summarize a technical document"
    constraints:
      - name: word_count
        type: value_in_range
        min_value: 50
        max_value: 200
    expected_output_schema:
      required: ["summary", "key_points"]
    timeout_seconds: 60

  - task_id: test_2
    goal: "Answer customer questions accurately"
    expected_output_schema:
      required: ["answer", "confidence"]
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

## 🔒 Security Features

Agent Tester includes a comprehensive **Cybersecurity & Secure-Code Contributor** module that provides:

### Security Scanners

1. **Static Application Security Testing (SAST)**
   - Detects code injection vulnerabilities (eval, exec, compile)
   - Identifies SQL and command injection patterns
   - Finds hardcoded secrets and credentials
   - Detects insecure deserialization and weak cryptography

2. **Dependency Vulnerability Scanning**
   - Scans requirements.txt and pyproject.toml
   - Detects known CVEs in dependencies
   - Identifies unpinned dependencies

3. **Configuration Security Analysis**
   - Scans .env files for exposed secrets
   - Analyzes Dockerfile for security issues
   - Checks GitHub Actions workflows for injection vulnerabilities
   - Validates YAML configurations

### Security Knowledge Base

Integrated with industry standards:
- **OWASP Top 10 2021** - Web application security risks
- **SANS Top 25 CWE** - Most dangerous software weaknesses
- **MITRE ATT&CK** - Adversary tactics and techniques

### Security Reports

- **5-tier severity system**: Critical, High, Medium, Low, Info
- **Detailed remediation guidance** with code samples
- **Multiple export formats**: Markdown, JSON
- **CVE/CWE/OWASP references** for each issue

**👉 [Read the Complete Security Documentation](SECURITY_MODULE.md)**

## Project Structure

```
agent-tester/
├── agent_tester/                # Main package
│   ├── __init__.py              # Package exports
│   ├── models.py                # Core data models
│   ├── cli.py                   # Command-line interface
│   ├── suite.py                 # Test orchestration
│   ├── validators/              # Validation modules
│   │   ├── task_validator.py
│   │   ├── trajectory_validator.py
│   │   └── memory_validator.py
│   ├── adapters/                # Platform adapters
│   │   ├── azure_adapter.py     # Azure AI Foundry
│   │   └── openai_adapter.py    # OpenAI
│   └── security/                # 🔒 Security module
│       ├── sast_scanner.py      # Static code analysis
│       ├── dependency_scanner.py # CVE scanning
│       ├── config_scanner.py    # Configuration security
│       ├── security_validator.py # Orchestrator
│       ├── security_reporter.py # Report generation
│       └── knowledge_base.py    # OWASP/SANS/MITRE
│   └── security/                # Security scanning (NEW!)
│       ├── sast_scanner.py      # Static security analysis
│       ├── dependency_scanner.py # Dependency vulnerabilities
│       ├── secret_scanner.py    # Secret detection
│       ├── config_scanner.py    # Configuration security
│       └── security_orchestrator.py # Scan coordination
├── examples/                    # Usage examples
│   ├── simple_example.py
│   └── security_scan_example.py
├── tests/                       # Test files
│   └── security/                # Security module tests
├── .github/workflows/           # CI/CD workflows
│   └── security-scan.yml        # Automated security scanning
├── pyproject.toml               # Package configuration
├── QUICKSTART.md                # Getting started guide
├── SECURITY_MODULE.md           # Security documentation
└── README.md                    # This file
```

## 🛡️ Security Features

Agent Tester includes comprehensive security scanning capabilities:

### Built-in Security Scanners

- **SAST (Static Application Security Testing)**: Detects SQL injection, XSS, command injection, path traversal, eval usage, weak crypto, and more
- **Dependency Scanning**: Identifies known CVEs in your dependencies
- **Secret Detection**: Finds exposed API keys, passwords, tokens (AWS, GitHub, OpenAI, etc.)
- **Configuration Security**: Checks for insecure defaults and misconfigurations

### Quick Security Scan

```bash
# Run full security scan
agent-tester security scan

# Generate HTML report
agent-tester security scan --format html --output security_report

# Check only dependencies
agent-tester security check-deps
```

### Continuous Security Monitoring

The included GitHub Actions workflow automatically:
- Scans every push and PR
- Posts security summaries on PRs
- Runs daily security audits
- Blocks merges with critical vulnerabilities

See [Security Documentation](agent_tester/security/README.md) for detailed information.

## Testing

```bash
# All tests
pytest test_agent_framework.py -v

# With coverage
pytest test_agent_framework.py --cov=agent_tester --cov-report=html

# Run simple example
python examples/simple_example.py
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
- Anthropic (Claude)
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
