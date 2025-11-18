# Contributing to AI Agent Testing Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- Clear, descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Code samples or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- Clear, descriptive title
- Detailed description of the proposed feature
- Rationale for why this enhancement would be useful
- Example code showing how it would work

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** if you've added code that should be tested
4. **Update documentation** to reflect any changes
5. **Ensure tests pass** by running `pytest`
6. **Create a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/agent-tester.git
cd agent-tester

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install agent-framework-azure-ai --pre

# Run tests
pytest tests/ -v
```

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use docstrings for all public modules, functions, classes, and methods

Example:
```python
def validate_task(
    output: Dict[str, Any],
    task: TaskDefinition,
    execution_time: float
) -> TaskValidationResult:
    """
    Validate task completion against requirements.
    
    Args:
        output: Agent's task output
        task: Task definition with requirements
        execution_time: Time taken to complete task
        
    Returns:
        TaskValidationResult with validation details
    """
    # Implementation here
    pass
```

### Documentation

- Use clear, concise language
- Include code examples for complex features
- Keep README.md updated with new features
- Document breaking changes in pull requests

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use descriptive test names
- Include both positive and negative test cases

Example:
```python
def test_task_validator_with_valid_output():
    """Test that valid output passes validation"""
    # Test implementation
    
def test_task_validator_with_invalid_output():
    """Test that invalid output fails validation"""
    # Test implementation
```

## Security

- Never commit API keys, secrets, or credentials
- Use environment variables for sensitive data
- Review [SECURITY.md](SECURITY.md) before contributing
- Report security vulnerabilities privately

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move file to..." not "Moves file to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

Examples:
```
Add support for custom constraint validators

Fix: Handle missing environment variables gracefully

Docs: Update Azure AI Foundry setup guide

Test: Add coverage for trajectory validation
```

## Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions or updates

## Testing Requirements

All pull requests must:
- Pass existing tests
- Include new tests for new functionality
- Maintain or improve code coverage
- Pass linting checks

Run tests before submitting:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_real_agents.py -v
```

## Documentation Requirements

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Update relevant documentation files in `docs/`
- Include usage examples for new features

## Review Process

1. **Automated checks** run on all pull requests
2. **Code review** by maintainers
3. **Changes requested** if needed
4. **Approval and merge** once all checks pass

## Questions?

Feel free to:
- Open an issue for questions
- Reach out to maintainers
- Check existing documentation and issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the AI Agent Testing Framework!
