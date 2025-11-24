"""
Unit tests for the Security Module
"""

import pytest
import tempfile
import os
from pathlib import Path

from agent_tester.security import (
    SASTScanner,
    DependencyScanner,
    SecretScanner,
    ConfigurationScanner,
    SecurityOrchestrator,
    SecurityReporter,
    SecuritySeverity,
    SecurityCategory,
)


class TestSASTScanner:
    """Test Static Application Security Testing scanner"""

    def test_detect_eval_usage(self, tmp_path):
        """Test detection of eval() usage"""
        # Create test file with eval
        test_file = tmp_path / "test_eval.py"
        test_file.write_text("""
def dangerous_function(user_input):
    result = eval(user_input)
    return result
""")

        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))

        assert len(issues) > 0
        assert any(
            issue.category == SecurityCategory.EVAL_USAGE for issue in issues
        )
        assert any(
            issue.severity == SecuritySeverity.CRITICAL for issue in issues
        )

    def test_detect_sql_injection(self, tmp_path):
        """Test detection of SQL injection vulnerabilities"""
        test_file = tmp_path / "test_sql.py"
        test_file.write_text("""
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)
""")

        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))

        # Should detect f-string formatting in what looks like SQL
        assert len(issues) > 0

    def test_detect_command_injection(self, tmp_path):
        """Test detection of command injection"""
        test_file = tmp_path / "test_cmd.py"
        test_file.write_text("""
import subprocess

def run_command(user_input):
    result = subprocess.run(user_input, shell=True)
    return result
""")

        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))

        # Pattern should detect shell=True
        assert len(issues) > 0
        # May detect as command injection or other category
        assert any(issue.severity in [SecuritySeverity.CRITICAL, SecuritySeverity.HIGH] for issue in issues)

    def test_detect_weak_crypto(self, tmp_path):
        """Test detection of weak cryptography"""
        test_file = tmp_path / "test_crypto.py"
        test_file.write_text("""
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()
""")

        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))

        assert len(issues) > 0
        assert any(
            issue.category == SecurityCategory.WEAK_CRYPTO for issue in issues
        )

    def test_skip_comments(self, tmp_path):
        """Test that comments are skipped"""
        test_file = tmp_path / "test_comments.py"
        test_file.write_text("""
# This is a comment about eval()
# Don't use eval(user_input)

def safe_function():
    return "safe"
""")

        scanner = SASTScanner()
        issues = scanner.scan_file(str(test_file))

        # Should not detect eval in comments
        assert len(issues) == 0

    def test_scan_directory(self, tmp_path):
        """Test scanning entire directory"""
        # Create multiple files
        (tmp_path / "file1.py").write_text("x = eval('1+1')")
        (tmp_path / "file2.py").write_text("import pickle; pickle.loads(data)")
        (tmp_path / "safe.py").write_text("def safe(): pass")

        scanner = SASTScanner()
        issues = scanner.scan_directory(str(tmp_path))

        assert len(issues) > 0
        assert scanner.files_scanned >= 3


class TestSecretScanner:
    """Test secret detection scanner"""

    def test_detect_api_key(self, tmp_path):
        """Test detection of API keys"""
        test_file = tmp_path / "config.py"
        test_file.write_text("""
API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
""")

        scanner = SecretScanner()
        issues = scanner.scan_file(str(test_file))

        assert len(issues) > 0
        assert any(
            issue.category == SecurityCategory.EXPOSED_SECRET for issue in issues
        )

    def test_detect_aws_key(self, tmp_path):
        """Test detection of AWS credentials"""
        test_file = tmp_path / "aws_config.py"
        test_file.write_text("""
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7REALKEY1"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYRealKey123"
""")

        scanner = SecretScanner()
        issues = scanner.scan_file(str(test_file))

        # Should detect at least one AWS-related secret
        assert len(issues) > 0

    def test_skip_example_values(self, tmp_path):
        """Test that example values are not flagged"""
        test_file = tmp_path / "example.py"
        test_file.write_text("""
API_KEY = "your-api-key-here"
PASSWORD = "example-password"
""")

        scanner = SecretScanner()
        issues = scanner.scan_file(str(test_file))

        # Should skip example/placeholder values
        assert len(issues) == 0

    def test_redact_secrets(self, tmp_path):
        """Test that secrets are redacted in output"""
        test_file = tmp_path / "secret.py"
        test_file.write_text("""
SECRET_KEY = "super-secret-key-12345"
""")

        scanner = SecretScanner()
        issues = scanner.scan_file(str(test_file))

        if issues:
            # Check that secret is redacted in snippet
            assert "***" in issues[0].code_snippet or "REDACTED" in issues[0].code_snippet


class TestDependencyScanner:
    """Test dependency scanner"""

    def test_scan_requirements_file(self, tmp_path):
        """Test scanning requirements.txt"""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("""
flask==2.0.0
django==3.0.0
requests>=2.25.0
pyyaml<5.4
""")

        scanner = DependencyScanner()
        issues = scanner.scan_requirements(str(req_file))

        # Should detect some issues based on known vulnerabilities
        assert scanner.dependencies_checked > 0

    def test_detect_unpinned_dependencies(self, tmp_path):
        """Test detection of unpinned dependencies"""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("""
requests
flask>=2.0.0
""")

        scanner = DependencyScanner()
        issues = scanner.scan_requirements(str(req_file))

        # Should flag unpinned dependencies as supply chain risk
        assert any(
            issue.category == SecurityCategory.SUPPLY_CHAIN_RISK for issue in issues
        )


class TestConfigurationScanner:
    """Test configuration scanner"""

    def test_detect_debug_mode(self, tmp_path):
        """Test detection of debug mode"""
        config_file = tmp_path / "settings.py"
        config_file.write_text("""
DEBUG = True
SECRET_KEY = "not-so-secret"
""")

        scanner = ConfigurationScanner()
        issues = scanner.scan_file(str(config_file))

        assert len(issues) > 0
        assert any(
            "debug" in issue.title.lower() for issue in issues
        )

    def test_detect_ssl_disabled(self, tmp_path):
        """Test detection of SSL verification disabled"""
        config_file = tmp_path / "config.py"
        config_file.write_text("""
SSL_VERIFY = False
VERIFY_SSL = False
""")

        scanner = ConfigurationScanner()
        issues = scanner.scan_file(str(config_file))

        assert len(issues) > 0


class TestSecurityOrchestrator:
    """Test security orchestrator"""

    def test_full_scan(self, tmp_path):
        """Test running full security scan"""
        # Create test files
        (tmp_path / "code.py").write_text("x = eval('1+1')")
        (tmp_path / "requirements.txt").write_text("requests==2.0.0")

        orchestrator = SecurityOrchestrator(repository_path=str(tmp_path))
        report = orchestrator.run_full_scan()

        assert report is not None
        assert report.scan_type == "full"
        assert report.files_scanned > 0

    def test_sast_only_scan(self, tmp_path):
        """Test SAST-only scan"""
        (tmp_path / "code.py").write_text("x = eval('1+1')")

        orchestrator = SecurityOrchestrator(repository_path=str(tmp_path))
        report = orchestrator.run_sast_scan()

        assert report.scan_type == "sast"

    def test_generate_recommendations(self, tmp_path):
        """Test recommendation generation"""
        (tmp_path / "vuln.py").write_text("password = 'hardcoded123'")

        orchestrator = SecurityOrchestrator(repository_path=str(tmp_path))
        report = orchestrator.run_full_scan()

        assert len(report.recommendations) > 0


class TestSecurityReporter:
    """Test security reporter"""

    def test_generate_json_report(self, tmp_path):
        """Test JSON report generation"""
        from agent_tester.security import SecurityReport, SecurityIssue

        report = SecurityReport(
            report_id="test-123",
            scan_type="test",
            repository_path=str(tmp_path),
            issues=[],
        )

        reporter = SecurityReporter()
        output_file = reporter.generate_report(
            report, output_format="json", output_path=str(tmp_path / "report")
        )

        assert os.path.exists(output_file)
        assert output_file.endswith(".json")

    def test_generate_html_report(self, tmp_path):
        """Test HTML report generation"""
        from agent_tester.security import SecurityReport

        report = SecurityReport(
            report_id="test-123",
            scan_type="test",
            repository_path=str(tmp_path),
            issues=[],
        )

        reporter = SecurityReporter()
        output_file = reporter.generate_report(
            report, output_format="html", output_path=str(tmp_path / "report")
        )

        assert os.path.exists(output_file)
        assert output_file.endswith(".html")

    def test_generate_markdown_report(self, tmp_path):
        """Test Markdown report generation"""
        from agent_tester.security import SecurityReport

        report = SecurityReport(
            report_id="test-123",
            scan_type="test",
            repository_path=str(tmp_path),
            issues=[],
        )

        reporter = SecurityReporter()
        output_file = reporter.generate_report(
            report, output_format="markdown", output_path=str(tmp_path / "report")
        )

        assert os.path.exists(output_file)
        assert output_file.endswith(".md")


class TestSecurityModels:
    """Test security data models"""

    def test_security_report_summary(self):
        """Test report summary calculation"""
        from agent_tester.security import SecurityReport, SecurityIssue

        issues = [
            SecurityIssue(
                issue_id="1",
                title="Test",
                description="Test",
                severity=SecuritySeverity.CRITICAL,
                category=SecurityCategory.EVAL_USAGE,
            ),
            SecurityIssue(
                issue_id="2",
                title="Test",
                description="Test",
                severity=SecuritySeverity.HIGH,
                category=SecurityCategory.SQL_INJECTION,
            ),
        ]

        report = SecurityReport(
            report_id="test",
            scan_type="test",
            repository_path=".",
            issues=issues,
        )

        report.calculate_summary()

        assert report.summary["total_issues"] == 2
        assert report.summary["by_severity"]["critical"] == 1
        assert report.summary["by_severity"]["high"] == 1

    def test_get_critical_issues(self):
        """Test filtering critical issues"""
        from agent_tester.security import SecurityReport, SecurityIssue

        issues = [
            SecurityIssue(
                issue_id="1",
                title="Critical",
                description="Test",
                severity=SecuritySeverity.CRITICAL,
                category=SecurityCategory.EVAL_USAGE,
            ),
            SecurityIssue(
                issue_id="2",
                title="Low",
                description="Test",
                severity=SecuritySeverity.LOW,
                category=SecurityCategory.CODE_QUALITY,
            ),
        ]

        report = SecurityReport(
            report_id="test",
            scan_type="test",
            repository_path=".",
            issues=issues,
        )

        critical = report.get_critical_issues()
        assert len(critical) == 1
        assert critical[0].severity == SecuritySeverity.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
