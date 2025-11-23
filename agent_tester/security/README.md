# Security Module Documentation

## Overview

The Agent Tester Security Module provides comprehensive cybersecurity monitoring and vulnerability detection capabilities for your AI agent projects. It acts as an automated security reviewer, continuously scanning for vulnerabilities, insecure patterns, and security best practices violations.

## Features

### 1. Static Application Security Testing (SAST)
- Detects insecure function usage (eval, exec, pickle)
- Identifies SQL injection vulnerabilities
- Finds command injection risks
- Detects path traversal vulnerabilities
- Identifies weak cryptography usage
- Finds hardcoded credentials
- Detects unsafe deserialization

### 2. Dependency Security Analysis
- Scans for known CVEs in dependencies
- Detects outdated packages
- Identifies supply chain risks
- Monitors security-critical packages
- Supports requirements.txt and pyproject.toml

### 3. Secret Detection
- Scans for exposed API keys and tokens
- Detects hardcoded passwords
- Identifies AWS credentials
- Finds GitHub tokens
- Detects OpenAI API keys
- Identifies JWT tokens
- Detects database connection strings
- Supports 13+ secret patterns

### 4. Configuration Security
- Detects insecure default configurations
- Identifies debug mode in production
- Finds SSL verification disabled
- Detects permissive CORS settings
- Identifies missing security files

## Installation

The security module is included with agent-tester:

```bash
pip install agent-tester
```

## Quick Start

### Command Line Interface

#### Run a Full Security Scan

```bash
# Scan current directory
agent-tester security scan

# Scan specific directory
agent-tester security scan --path /path/to/project

# Generate HTML report
agent-tester security scan --format html --output my_report

# Run specific scan type
agent-tester security scan --type sast
agent-tester security scan --type dependency
agent-tester security scan --type secret
agent-tester security scan --type config
```

#### Quick Security Summary

```bash
agent-tester security report
```

#### Check Dependencies Only

```bash
agent-tester security check-deps
```

### Python API

#### Full Security Scan

```python
from agent_tester.security import SecurityOrchestrator

# Initialize orchestrator
orchestrator = SecurityOrchestrator(repository_path=".")

# Run full scan
report = orchestrator.run_full_scan()

# Generate HTML report
orchestrator.generate_report_file(report, format="html", output_path="security_report")

# Access results
print(f"Total issues: {len(report.issues)}")
print(f"Critical issues: {len(report.get_critical_issues())}")
```

#### Run Individual Scanners

```python
from agent_tester.security import (
    SASTScanner,
    DependencyScanner,
    SecretScanner,
    ConfigurationScanner
)

# SAST Scan
sast = SASTScanner()
sast_issues = sast.scan_directory("./src")

# Dependency Scan
dep_scanner = DependencyScanner()
dep_issues = dep_scanner.scan_requirements("requirements.txt")

# Secret Scan
secret_scanner = SecretScanner()
secret_issues = secret_scanner.scan_directory(".")

# Configuration Scan
config_scanner = ConfigurationScanner()
config_issues = config_scanner.scan_directory(".")
```

#### Custom Reporting

```python
from agent_tester.security import SecurityReporter, SecurityReport

reporter = SecurityReporter()

# Generate in different formats
reporter.generate_report(report, format="json", output_path="report")
reporter.generate_report(report, format="html", output_path="report")
reporter.generate_report(report, format="markdown", output_path="report")
reporter.generate_report(report, format="text", output_path="report")
```

## Security Issue Categories

### Critical Severity
- Eval/Exec usage (arbitrary code execution)
- Exposed API keys and secrets
- Command injection vulnerabilities
- Known CVEs in dependencies

### High Severity
- SQL injection vulnerabilities
- Unsafe deserialization (pickle)
- Path traversal vulnerabilities
- SSL verification disabled
- Vulnerable dependencies

### Medium Severity
- Weak cryptography (MD5, SHA1)
- Insecure random number generation
- Debug mode enabled
- Permissive CORS configuration
- Outdated security-critical packages

### Low Severity
- Code quality issues
- Unpinned dependencies
- Default admin URLs
- Assert statements in security checks

## Report Formats

### JSON
Structured data format for programmatic processing:
```json
{
  "report_id": "sec-abc123",
  "scan_type": "full",
  "issues": [...],
  "summary": {
    "total_issues": 15,
    "by_severity": {
      "critical": 2,
      "high": 5,
      "medium": 6,
      "low": 2
    }
  }
}
```

### HTML
Beautiful, interactive report with color-coded severity levels and detailed fix recommendations.

### Markdown
GitHub-friendly format perfect for PRs and documentation:
```markdown
# Security Scan Report

## Summary
- Total Issues: 15
- Critical: 2
- High: 5

### Critical Issues
1. **Exposed Secret: AWS Access Key**
   - Location: `config.py:45`
   - Fix: Remove from code and use environment variables
```

### Text
Plain text format for terminal output and logs.

## CI/CD Integration

### GitHub Actions

The module includes a pre-configured GitHub Actions workflow:

```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e .
      - run: agent-tester security scan --format html
      - uses: actions/upload-artifact@v4
        with:
          name: security-report
          path: security_report.html
```

### GitLab CI

```yaml
security-scan:
  image: python:3.10
  script:
    - pip install agent-tester
    - agent-tester security scan --format json
  artifacts:
    reports:
      security: security_report.json
```

## Best Practices

### 1. Run Scans Regularly
- On every commit (via CI/CD)
- Before merging PRs
- Scheduled daily/weekly scans
- Before releases

### 2. Address Critical Issues Immediately
- Fix critical issues within 24 hours
- High severity within 1 week
- Medium severity within 1 month

### 3. Use in Development
```bash
# Add to pre-commit hook
agent-tester security scan --type secret --type sast
```

### 4. Track Security Over Time
```python
from agent_tester.security import SecurityReporter

# Compare reports
current_report = orchestrator.run_full_scan()
changelog = reporter.create_changelog_entry(current_report, previous_report)

print(f"New issues: {changelog.issues_detected}")
print(f"Fixed issues: {changelog.issues_fixed}")
```

### 5. Customize Scanning
```python
# Skip certain directories
orchestrator = SecurityOrchestrator(repository_path=".")
orchestrator.sast_scanner.SKIP_PATTERNS.add("third_party")

# Add custom patterns
orchestrator.sast_scanner.INSECURE_PATTERNS["custom_pattern"] = {
    "patterns": [r"dangerous_function\("],
    "severity": SecuritySeverity.HIGH,
    "category": SecurityCategory.INSECURE_FUNCTION,
    "description": "Custom dangerous function detected",
    "fix": "Use safe alternative"
}
```

## OWASP Compliance

The security module helps ensure compliance with:

- **OWASP Top 10**: Detection of injection flaws, broken authentication, sensitive data exposure, XXE, broken access control, security misconfiguration, XSS, insecure deserialization, and more.

- **OWASP ASVS**: Application Security Verification Standard coverage for authentication, session management, access control, validation, cryptography, and error handling.

- **SANS Top 25**: Coverage of most dangerous software errors including improper input validation, improper neutralization of special elements, buffer overflow, and more.

## Exploitability Assessment

The scanner marks issues as exploitable when:
- Known CVE exists
- Attack vector is clear
- Proof-of-concept exists
- Secret is actively usable

```python
# Get only exploitable issues
exploitable = report.get_exploitable_issues()
for issue in exploitable:
    print(f"⚠️  {issue.title}")
    print(f"   Attack Vector: {issue.attack_vector}")
    print(f"   Impact: {issue.impact}")
```

## False Positive Handling

```python
# Mark false positives
for issue in report.issues:
    if issue.file_path.endswith("test_data.py"):
        issue.false_positive = True

# Filter out false positives
real_issues = [i for i in report.issues if not i.false_positive]
```

## Advanced Usage

### Penetration Testing Simulation

```python
# Identify exploitable vulnerabilities
exploitable = report.get_exploitable_issues()

for vuln in exploitable:
    if vuln.category == SecurityCategory.SQL_INJECTION:
        print(f"Potential SQL injection at {vuln.file_path}:{vuln.line_number}")
        print(f"Attack vector: {vuln.attack_vector}")
        print(f"Suggested payload: ' OR '1'='1")
```

### Security Metrics Dashboard

```python
import matplotlib.pyplot as plt

# Track security metrics
metrics = {
    "total_scans": 100,
    "avg_issues_per_scan": 12.5,
    "critical_resolved": 45,
    "mean_time_to_fix": "2.3 days"
}

# Visualize trends
severity_counts = report.summary["by_severity"]
plt.bar(severity_counts.keys(), severity_counts.values())
plt.title("Security Issues by Severity")
plt.savefig("security_metrics.png")
```

### Integration with Issue Trackers

```python
import requests

# Create GitHub issue for critical vulnerabilities
for issue in report.get_critical_issues():
    github_issue = {
        "title": f"[SECURITY] {issue.title}",
        "body": f"""
        **Severity**: {issue.severity.value.upper()}
        **Category**: {issue.category.value}
        **Location**: {issue.file_path}:{issue.line_number}
        
        **Description**: {issue.description}
        
        **Fix**: {issue.suggested_fix.description if issue.suggested_fix else 'N/A'}
        """,
        "labels": ["security", "critical"]
    }
    # Post to GitHub API
```

## Troubleshooting

### Scanner Not Finding Issues
- Check file extensions in `SCANNABLE_EXTENSIONS`
- Verify paths not in `SKIP_PATTERNS`
- Ensure files are readable

### False Positives
- Use `.env.example` for template files
- Add "example" or "test" to file names
- Mark issues as false positives programmatically

### Performance Issues
- Limit scan scope to specific directories
- Use specific scan types instead of full scan
- Exclude large binary or vendor directories

## Contributing

To add new security patterns:

```python
# Add to INSECURE_PATTERNS in sast_scanner.py
"new_pattern": {
    "patterns": [r"regex_pattern"],
    "severity": SecuritySeverity.HIGH,
    "category": SecurityCategory.INSECURE_FUNCTION,
    "description": "Description of the issue",
    "fix": "How to fix it"
}
```

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [MITRE ATT&CK](https://attack.mitre.org/)
- [NVD Database](https://nvd.nist.gov/)

## License

This security module is part of agent-tester and is released under the MIT License.
