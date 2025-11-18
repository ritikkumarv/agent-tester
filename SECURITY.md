# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of the AI Agent Testing Framework seriously. If you discover a security vulnerability, please follow these steps:

### 1. **DO NOT** Open a Public Issue

Please do not report security vulnerabilities through public GitHub issues.

### 2. Report via Private Channel

Instead, please report security vulnerabilities by email to:
- **Email**: [Will update later]

Include the following information:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability and how an attacker might exploit it

### 3. Response Timeline

- **Initial Response**: Within 48 hours of report
- **Status Update**: Within 7 days with validation of the vulnerability
- **Fix Timeline**: Varies based on severity (Critical: <7 days, High: <14 days, Medium: <30 days)

## Security Best Practices

### For Users

1. **Never commit credentials**
   - Never hardcode API keys, secrets, or passwords in code
   - Use environment variables for sensitive data
   - Add `.env` files to `.gitignore`
   - Use the provided `.env.example` as a template

2. **Keep dependencies updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Use virtual environments**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate      # Linux/Mac
   ```

4. **Validate external inputs**
   - Never pass untrusted data to `eval()` or `exec()`
   - Validate all user inputs before processing
   - Use the built-in safe constraint validators

5. **Secure API key storage**
   - Use Azure Key Vault for production deployments
   - Rotate API keys regularly
   - Use service principals with minimal permissions

### For Contributors

1. **Code Review**
   - All code changes require review before merging
   - Security-sensitive changes require additional review

2. **Dependency Management**
   - Pin dependency versions in `requirements.txt`
   - Review security advisories for dependencies
   - Use `pip-audit` to check for known vulnerabilities

3. **Testing**
   - Include security tests for new features
   - Test with invalid/malicious inputs
   - Never commit test data containing real credentials

## Known Security Considerations

### 1. API Key Management
- This framework requires API keys for AI services
- Keys must be stored securely using environment variables
- Never commit `.env` files to version control

### 2. Agent Output Validation
- Agent outputs should be validated before use
- Do not execute agent-generated code without review
- Be cautious with file system operations based on agent output

### 3. Dependency Security
- This framework depends on third-party packages
- Regularly update dependencies to patch security issues
- Review security advisories for:
  - `openai`
  - `anthropic`
  - `langchain`
  - `azure-ai-projects`
  - `pydantic`

## Security Updates

Security updates will be published as:
1. GitHub Security Advisories
2. Release notes with `[SECURITY]` tag
3. Updated documentation in this file

## Compliance

This framework follows security best practices including:
- OWASP Top 10 awareness
- Secure coding standards
- Least privilege principle
- Defense in depth

## Additional Resources

- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [Azure Security Best Practices](https://docs.microsoft.com/en-us/azure/security/fundamentals/best-practices-and-patterns)
- [OpenAI API Security](https://platform.openai.com/docs/guides/safety-best-practices)

## Disclosure Policy

We follow responsible disclosure:
- Security researchers are given credit for their findings
- Fixes are developed and tested privately
- Public disclosure occurs after patch release
- CVEs are assigned for significant vulnerabilities

## Contact

For security concerns, please contact:
- **Security Team**: [Will update the mail later]
- **Project Maintainer**: Ritik Kumar

---

**Last Updated**: November 2025
