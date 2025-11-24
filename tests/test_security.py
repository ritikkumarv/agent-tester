"""
Tests for Security Module

Tests the cybersecurity & secure-code contributor functionality.
"""

import os
import tempfile
import pytest
from pathlib import Path

from agent_tester.security import (
    SecurityValidator,
    SASTScanner,
    DependencyScanner,
    ConfigurationScanner,
    SecurityReporter,
    SecurityIssue,
    Severity,
    IssueCategory,
    SecurityKnowledgeBase,
)


class TestSASTScanner:
    """Test Static Application Security Testing scanner"""
    
    def test_detect_eval_usage(self, tmp_path):
        """Test detection of dangerous eval() usage"""
        test_file = tmp_path / "test_eval.py"
        test_file.write_text("result = eval(user_input)")
        
        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))
        
        assert len(issues) > 0
        assert any(issue.severity == Severity.CRITICAL for issue in issues)
        assert any("eval" in issue.title.lower() for issue in issues)
    
    def test_detect_exec_usage(self, tmp_path):
        """Test detection of dangerous exec() usage"""
        test_file = tmp_path / "test_exec.py"
        test_file.write_text("exec(user_code)")
        
        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))
        
        assert len(issues) > 0
        assert any(issue.severity == Severity.CRITICAL for issue in issues)
    
    def test_detect_hardcoded_password(self, tmp_path):
        """Test detection of hardcoded passwords"""
        test_file = tmp_path / "test_secret.py"
        test_file.write_text('password = "SuperSecret123!"')
        
        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))
        
        assert len(issues) > 0
        assert any(issue.category == IssueCategory.SENSITIVE_DATA for issue in issues)
    
    def test_detect_sql_injection(self, tmp_path):
        """Test detection of SQL injection vulnerabilities"""
        test_file = tmp_path / "test_sql.py"
        test_file.write_text('query = "SELECT * FROM users WHERE id = %s" % user_id')
        
        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))
        
        assert len(issues) > 0
        assert any(issue.category == IssueCategory.INJECTION for issue in issues)
    
    def test_detect_command_injection(self, tmp_path):
        """Test detection of command injection"""
        test_file = tmp_path / "test_cmd.py"
        test_file.write_text('os.system("ls " + user_input)')
        
        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))
        
        assert len(issues) > 0
        assert any("command" in issue.title.lower() for issue in issues)
    
    def test_detect_weak_crypto(self, tmp_path):
        """Test detection of weak cryptography"""
        test_file = tmp_path / "test_crypto.py"
        test_file.write_text('import hashlib\nhash = hashlib.md5(data)')
        
        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))
        
        assert len(issues) > 0
        assert any(issue.category == IssueCategory.CRYPTOGRAPHY for issue in issues)
    
    def test_scan_directory(self, tmp_path):
        """Test scanning entire directory"""
        # Create test files
        (tmp_path / "file1.py").write_text("result = eval(input())")
        (tmp_path / "file2.py").write_text("password = 'secret123'")
        (tmp_path / "safe.py").write_text("x = 1 + 1")
        
        scanner = SASTScanner()
        issues = scanner.scan_directory(str(tmp_path))
        
        assert len(issues) > 0
        assert scanner.files_scanned >= 3


class TestDependencyScanner:
    """Test dependency vulnerability scanner"""
    
    def test_scan_requirements_file(self, tmp_path):
        """Test scanning requirements.txt"""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("requests==2.25.0\nurllib3==1.26.0\npytest>=7.0.0")
        
        scanner = DependencyScanner()
        issues = scanner.scan_requirements_file(str(req_file))
        
        # Should detect some issues (based on known vulnerabilities list)
        assert scanner.dependencies_checked >= 3
    
    def test_detect_unpinned_dependency(self, tmp_path):
        """Test detection of unpinned dependencies"""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("flask\nrequests")
        
        scanner = DependencyScanner()
        issues = scanner.scan_requirements_file(str(req_file))
        
        # Should report unpinned dependencies
        assert len(issues) > 0
        assert any(issue.severity == Severity.LOW for issue in issues)
    
    def test_scan_pyproject_toml(self, tmp_path):
        """Test scanning pyproject.toml"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text("""
[project]
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0.0",
]
""")
        
        scanner = DependencyScanner()
        issues = scanner.scan_pyproject_toml(str(pyproject_file))
        
        assert scanner.dependencies_checked >= 2


class TestConfigurationScanner:
    """Test configuration security scanner"""
    
    def test_scan_env_file(self, tmp_path):
        """Test scanning .env files"""
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=secret123\nDATABASE_URL=postgres://localhost")
        
        scanner = ConfigurationScanner()
        issues = scanner.scan_env_file(str(env_file))
        
        # Should warn about .env file being committed
        assert len(issues) > 0
    
    def test_skip_env_example(self, tmp_path):
        """Test that .env.example files are skipped"""
        env_file = tmp_path / ".env.example"
        env_file.write_text("API_KEY=your-api-key-here")
        
        scanner = ConfigurationScanner()
        issues = scanner.scan_env_file(str(env_file))
        
        # Should not report issues for example files
        assert len(issues) == 0
    
    def test_scan_dockerfile(self, tmp_path):
        """Test scanning Dockerfile"""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""
FROM python:latest
USER root
ENV SECRET_KEY=hardcoded_secret
RUN pip install flask
""")
        
        scanner = ConfigurationScanner()
        issues = scanner.scan_docker_file(str(dockerfile))
        
        # Should detect multiple issues
        assert len(issues) > 0
        assert any("latest" in issue.title.lower() for issue in issues)
        assert any("root" in issue.title.lower() or "secret" in issue.title.lower() for issue in issues)
    
    def test_scan_github_workflow(self, tmp_path):
        """Test scanning GitHub Actions workflow"""
        workflow_file = tmp_path / "test.yml"
        workflow_file.write_text("""
name: Test
on: pull_request_target
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: echo ${{ github.event.pull_request.title }}
""")
        
        scanner = ConfigurationScanner()
        issues = scanner.scan_github_workflow(str(workflow_file))
        
        # Should detect security issues
        assert len(issues) > 0


class TestSecurityReporter:
    """Test security reporting functionality"""
    
    def test_create_report(self):
        """Test creating a security report"""
        reporter = SecurityReporter()
        report = reporter.create_report("test-001", "/path/to/repo", "main")
        
        assert report.report_id == "test-001"
        assert report.repository_path == "/path/to/repo"
        assert report.branch == "main"
        assert report.total_issues == 0
    
    def test_add_issues(self):
        """Test adding issues to report"""
        reporter = SecurityReporter()
        report = reporter.create_report("test-002", "/path/to/repo")
        
        # Add critical issue
        critical_issue = SecurityIssue(
            issue_id="TEST-001",
            title="Test Critical Issue",
            description="Test description",
            severity=Severity.CRITICAL,
            category=IssueCategory.INJECTION,
            recommendation="Fix it",
        )
        report.add_issue(critical_issue)
        
        assert report.total_issues == 1
        assert len(report.critical_issues) == 1
        assert len(report.high_issues) == 0
    
    def test_generate_markdown_report(self):
        """Test markdown report generation"""
        reporter = SecurityReporter()
        report = reporter.create_report("test-003", "/path/to/repo")
        
        issue = SecurityIssue(
            issue_id="TEST-001",
            title="Test Issue",
            description="Test description",
            severity=Severity.HIGH,
            category=IssueCategory.SENSITIVE_DATA,
            recommendation="Fix it",
            file_path="test.py",
            line_number=10,
        )
        report.add_issue(issue)
        
        markdown = reporter.generate_markdown_report(report)
        
        assert "Security Report" in markdown
        assert "Test Issue" in markdown
        assert "test.py" in markdown


class TestSecurityValidator:
    """Test security validator"""
    
    def test_validate_repository(self, tmp_path):
        """Test validating entire repository"""
        # Create test files
        (tmp_path / "test.py").write_text("result = eval(input())")
        (tmp_path / "requirements.txt").write_text("flask\nrequests")
        
        validator = SecurityValidator()
        report = validator.validate_repository(str(tmp_path))
        
        assert report.total_issues > 0
        assert report.total_files_scanned > 0
        assert "SAST" in report.scanners_used
        assert "Dependency" in report.scanners_used
    
    def test_export_markdown_report(self, tmp_path):
        """Test exporting markdown report"""
        (tmp_path / "test.py").write_text("x = 1")
        
        validator = SecurityValidator()
        report = validator.validate_repository(str(tmp_path))
        
        output_file = tmp_path / "report.md"
        content = validator.export_report(report, format="markdown", output_file=str(output_file))
        
        assert output_file.exists()
        assert "Security Report" in content
    
    def test_export_json_report(self, tmp_path):
        """Test exporting JSON report"""
        (tmp_path / "test.py").write_text("x = 1")
        
        validator = SecurityValidator()
        report = validator.validate_repository(str(tmp_path))
        
        output_file = tmp_path / "report.json"
        content = validator.export_report(report, format="json", output_file=str(output_file))
        
        assert output_file.exists()
        assert "{" in content  # Valid JSON


class TestSecurityKnowledgeBase:
    """Test security knowledge base"""
    
    def test_get_owasp_guidance(self):
        """Test retrieving OWASP guidance"""
        kb = SecurityKnowledgeBase()
        guidance = kb.get_owasp_guidance("A03:2021")
        
        assert guidance is not None
        assert "Injection" in guidance.title
        assert len(guidance.mitigations) > 0
    
    def test_search_owasp(self):
        """Test searching OWASP categories"""
        kb = SecurityKnowledgeBase()
        results = kb.search_owasp_by_keyword("injection")
        
        assert len(results) > 0
        assert any("Injection" in r.title for r in results)
    
    def test_get_cwe_info(self):
        """Test retrieving CWE information"""
        kb = SecurityKnowledgeBase()
        cwe = kb.get_cwe_info("CWE-89")
        
        assert cwe is not None
        assert "SQL" in cwe["name"]
    
    def test_get_all_owasp_categories(self):
        """Test getting all OWASP categories"""
        kb = SecurityKnowledgeBase()
        categories = kb.get_all_owasp_categories()
        
        assert len(categories) == 10  # OWASP Top 10
        assert "A01:2021" in categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
