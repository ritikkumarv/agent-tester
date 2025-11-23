# 🔒 Security Module Documentation

## Cybersecurity & Secure-Code Contributor

The Agent Tester framework includes comprehensive security analysis capabilities that act as a **Cybersecurity & Secure-Code Contributor** for your AI agent projects.

## Overview

The security module provides:

1. **Static Application Security Testing (SAST)** - Source code vulnerability scanning
2. **Dependency Vulnerability Scanning** - CVE detection in third-party packages
3. **Configuration Security Analysis** - Misconfiguration and secret detection
4. **Security Knowledge Base** - Integration with OWASP, SANS, and MITRE standards
5. **Comprehensive Reporting** - Detailed security reports with remediation guidance

## Quick Start

### CLI Usage

```bash
# Run security scan on current directory
agent-tester security

# Scan specific directory
agent-tester security --path /path/to/project

# Generate JSON report
agent-tester security --format json --output security_report.json

# Show only critical and high severity issues
agent-tester security --severity high
```

### Python API Usage

```python
from agent_tester.security import SecurityValidator

# Initialize validator
validator = SecurityValidator()

# Run comprehensive security scan
report = validator.validate_repository("./my-project")

# Get summary
summary = report.get_summary()
print(f"Total Issues: {summary['total_issues']}")
print(f"Critical: {summary['critical']}")
print(f"High: {summary['high']}")

# Export report
validator.export_report(report, format="markdown", output_file="security_report.md")
```

## Features

### 1. Static Application Security Testing (SAST)

Detects common security vulnerabilities in Python code:

#### Detected Issues:
- **Code Injection** (`eval()`, `exec()`, `compile()`)
- **SQL Injection** (String-based query construction)
- **Command Injection** (`os.system()`, `subprocess` with `shell=True`)
- **Hardcoded Secrets** (API keys, passwords, tokens)
- **Insecure Deserialization** (`pickle.loads()`)
- **Weak Cryptography** (MD5, SHA1 usage)

#### Example:

```python
from agent_tester.security import SASTScanner

scanner = SASTScanner()
issues = scanner.scan_directory("./src")

for issue in issues:
    print(f"{issue.severity}: {issue.title}")
    print(f"  File: {issue.file_path}:{issue.line_number}")
    print(f"  Fix: {issue.recommendation}")
```

### 2. Dependency Vulnerability Scanning

Checks third-party dependencies for known vulnerabilities:

#### Supported Files:
- `requirements.txt` (Python pip)
- `pyproject.toml` (Python Poetry)

#### Detected Issues:
- Known CVEs in dependencies
- Unpinned dependency versions
- Outdated packages with security issues

#### Example:

```python
from agent_tester.security import DependencyScanner

scanner = DependencyScanner()
issues = scanner.scan_requirements_file("requirements.txt")

print(f"Dependencies checked: {scanner.dependencies_checked}")
print(f"Vulnerable packages: {len(issues)}")
```

### 3. Configuration Security Analysis

Scans configuration files for security misconfigurations:

#### Supported Files:
- `.env` files (secret detection)
- `Dockerfile` (container security)
- GitHub Actions workflows (CI/CD security)
- YAML configurations

#### Detected Issues:
- Exposed secrets in configuration files
- Committed `.env` files
- Dockerfile running as root
- Using `latest` tags in Docker
- GitHub Actions script injection vulnerabilities
- `pull_request_target` misuse

#### Example:

```python
from agent_tester.security import ConfigurationScanner

scanner = ConfigurationScanner()
issues = scanner.scan_directory(".")

for issue in issues:
    if issue.severity in ["critical", "high"]:
        print(f"⚠️  {issue.title}: {issue.file_path}")
```

### 4. Security Knowledge Base

Integration with industry-standard security frameworks:

#### Available Standards:
- **OWASP Top 10 2021** - Web application security risks
- **SANS Top 25 CWE** - Most dangerous software weaknesses
- **MITRE ATT&CK** - Adversary tactics and techniques

#### Example:

```python
from agent_tester.security import SecurityKnowledgeBase

kb = SecurityKnowledgeBase()

# Get OWASP guidance
injection_guidance = kb.get_owasp_guidance("A03:2021")
print(f"Category: {injection_guidance.title}")
print("Mitigations:")
for mitigation in injection_guidance.mitigations:
    print(f"  - {mitigation}")

# Search by keyword
results = kb.search_owasp_by_keyword("injection")
for result in results:
    print(f"  {result.category}: {result.title}")

# Get CWE information
cwe = kb.get_cwe_info("CWE-89")
print(f"CWE-89: {cwe['name']} (Rank: {cwe['rank']})")
```

## Security Report

The security scanner generates comprehensive reports with:

### Report Contents:
- Executive summary with issue counts by severity
- Detailed issue listings with:
  - File path and line number
  - Attack vector description
  - Impact assessment
  - Remediation recommendations
  - Code samples for fixes
  - CVE/CWE/OWASP references

### Severity Levels:

| Severity | Description | Examples |
|----------|-------------|----------|
| 🔴 **Critical** | Immediate action required | Code injection, hardcoded secrets, SQL injection |
| 🟠 **High** | Significant risk | Insecure deserialization, command injection |
| 🟡 **Medium** | Moderate risk | Weak cryptography, security misconfigurations |
| 🔵 **Low** | Minor risk | Unpinned dependencies, Docker best practices |
| ℹ️ **Info** | Best practice recommendations | Code quality improvements |

### Report Formats:

#### Markdown Report:
```python
validator.export_report(report, format="markdown", output_file="report.md")
```

#### JSON Report:
```python
validator.export_report(report, format="json", output_file="report.json")
```

## Integration with CI/CD

### GitHub Actions Example:

```yaml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install Agent Tester
        run: pip install agent-tester
      
      - name: Run Security Scan
        run: agent-tester security --severity high
      
      - name: Upload Security Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: security-report
          path: security_report_*.md
```

## Best Practices

### 1. Regular Scanning
Run security scans:
- On every commit (CI/CD)
- Before releases
- After dependency updates
- During code reviews

### 2. Prioritize Issues
Focus on:
1. Critical and High severity issues first
2. Issues in production code paths
3. Vulnerabilities with known exploits
4. Exposed secrets (rotate immediately)

### 3. Continuous Monitoring
Set up automated scanning:
```python
validator = SecurityValidator()
config = validator.continuous_monitor(
    repository_path=".",
    interval_hours=24
)
```

### 4. Developer Education
Use security reports to:
- Train developers on secure coding
- Share remediation examples
- Build security awareness
- Document security patterns

## Limitations

### Current Version:
- **Language Support**: Python only (more languages planned)
- **CVE Database**: Simplified (integrate with NVD API for production)
- **False Positives**: Some manual review required
- **Penetration Testing**: Defensive analysis only (no active exploitation)

### Future Enhancements:
- Integration with real-time CVE databases
- Support for more languages (JavaScript, Go, Java)
- Advanced DAST (Dynamic Application Security Testing)
- AI-powered vulnerability detection
- Automated fix suggestions

## Security Issue Examples

### Example 1: Code Injection

**Detected Code:**
```python
user_input = request.args.get('code')
result = eval(user_input)  # ❌ CRITICAL
```

**Recommendation:**
```python
# Use ast.literal_eval for safe evaluation
import ast
user_input = request.args.get('code')
result = ast.literal_eval(user_input)  # ✅ Safe
```

### Example 2: SQL Injection

**Detected Code:**
```python
query = f"SELECT * FROM users WHERE id = {user_id}"  # ❌ CRITICAL
cursor.execute(query)
```

**Recommendation:**
```python
# Use parameterized queries
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))  # ✅ Safe
```

### Example 3: Hardcoded Secrets

**Detected Code:**
```python
API_KEY = "sk_live_1234567890abcdef"  # ❌ CRITICAL
```

**Recommendation:**
```python
import os
API_KEY = os.getenv('API_KEY')  # ✅ Safe
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")
```

### Example 4: Weak Cryptography

**Detected Code:**
```python
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()  # ❌ MEDIUM
```

**Recommendation:**
```python
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())  # ✅ Safe
```

## API Reference

### SecurityValidator

Main orchestrator for security scanning.

```python
class SecurityValidator:
    def validate_repository(repository_path: str, branch: Optional[str] = None) -> SecurityReport
    def validate_file(file_path: str) -> List[SecurityIssue]
    def export_report(report: SecurityReport, format: str, output_file: Optional[str]) -> str
```

### SASTScanner

Static application security testing.

```python
class SASTScanner:
    def scan_file(file_path: str) -> List[SecurityIssue]
    def scan_directory(directory_path: str) -> List[SecurityIssue]
    def get_summary() -> Dict[str, Any]
```

### DependencyScanner

Dependency vulnerability scanning.

```python
class DependencyScanner:
    def scan_requirements_file(file_path: str) -> List[SecurityIssue]
    def scan_pyproject_toml(file_path: str) -> List[SecurityIssue]
    def scan_directory(directory_path: str) -> List[SecurityIssue]
```

### ConfigurationScanner

Configuration security analysis.

```python
class ConfigurationScanner:
    def scan_env_file(file_path: str) -> List[SecurityIssue]
    def scan_docker_file(file_path: str) -> List[SecurityIssue]
    def scan_yaml_config(file_path: str) -> List[SecurityIssue]
    def scan_github_workflow(file_path: str) -> List[SecurityIssue]
```

## Support

For security vulnerability reports in Agent Tester itself, see [SECURITY.md](../SECURITY.md).

For questions or issues:
- GitHub Issues: https://github.com/ritikkumarv/agent-tester/issues
- Documentation: https://github.com/ritikkumarv/agent-tester#readme
