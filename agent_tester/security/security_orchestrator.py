"""
Security Orchestrator

Coordinates all security scanners and generates comprehensive reports
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import SecurityReport, SecurityIssue, SecuritySeverity
from .sast_scanner import SASTScanner
from .dependency_scanner import DependencyScanner
from .secret_scanner import SecretScanner
from .config_scanner import ConfigurationScanner
from .security_reporter import SecurityReporter


class SecurityOrchestrator:
    """
    Orchestrates comprehensive security scanning

    This is the main entry point for running security scans
    """

    def __init__(self, repository_path: str = "."):
        self.repository_path = repository_path
        self.sast_scanner = SASTScanner()
        self.dependency_scanner = DependencyScanner()
        self.secret_scanner = SecretScanner()
        self.config_scanner = ConfigurationScanner()
        self.reporter = SecurityReporter()

    def run_full_scan(self) -> SecurityReport:
        """
        Run all security scanners

        Returns:
            SecurityReport with all detected issues
        """
        start_time = time.time()
        all_issues: List[SecurityIssue] = []

        # Run SAST scan
        print("🔍 Running Static Application Security Testing (SAST)...")
        sast_issues = self.sast_scanner.scan_directory(self.repository_path)
        all_issues.extend(sast_issues)
        print(f"   Found {len(sast_issues)} SAST issues")

        # Run dependency scan
        print("📦 Scanning Dependencies...")
        dep_issues = []
        # Check requirements.txt
        req_path = Path(self.repository_path) / "requirements.txt"
        if req_path.exists():
            dep_issues.extend(self.dependency_scanner.scan_requirements(str(req_path)))

        # Check pyproject.toml
        pyproject_path = Path(self.repository_path) / "pyproject.toml"
        if pyproject_path.exists():
            dep_issues.extend(
                self.dependency_scanner.scan_pyproject_toml(str(pyproject_path))
            )

        all_issues.extend(dep_issues)
        print(f"   Found {len(dep_issues)} dependency issues")

        # Run secret scan
        print("🔐 Scanning for Exposed Secrets...")
        secret_issues = self.secret_scanner.scan_directory(self.repository_path)
        all_issues.extend(secret_issues)
        print(f"   Found {len(secret_issues)} exposed secrets")

        # Run configuration scan
        print("⚙️  Scanning Configurations...")
        config_issues = self.config_scanner.scan_directory(self.repository_path)
        all_issues.extend(config_issues)
        print(f"   Found {len(config_issues)} configuration issues")

        # Create report
        scan_duration = time.time() - start_time

        report = SecurityReport(
            report_id=self._generate_report_id(),
            scan_type="full",
            repository_path=self.repository_path,
            issues=all_issues,
            scan_duration_seconds=scan_duration,
            files_scanned=(
                self.sast_scanner.files_scanned + self.secret_scanner.files_scanned
            ),
            dependencies_checked=self.dependency_scanner.dependencies_checked,
        )

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Calculate summary
        report.calculate_summary()

        print(f"\n✅ Scan complete! Found {len(all_issues)} total issues")
        print(f"   Duration: {scan_duration:.2f}s")

        return report

    def run_sast_scan(self) -> SecurityReport:
        """Run only SAST scan"""
        start_time = time.time()

        print("🔍 Running SAST scan...")
        issues = self.sast_scanner.scan_directory(self.repository_path)

        report = SecurityReport(
            report_id=self._generate_report_id(),
            scan_type="sast",
            repository_path=self.repository_path,
            issues=issues,
            scan_duration_seconds=time.time() - start_time,
            files_scanned=self.sast_scanner.files_scanned,
        )

        report.calculate_summary()
        return report

    def run_dependency_scan(self) -> SecurityReport:
        """Run only dependency scan"""
        start_time = time.time()

        print("📦 Running dependency scan...")
        issues = []

        req_path = Path(self.repository_path) / "requirements.txt"
        if req_path.exists():
            issues.extend(self.dependency_scanner.scan_requirements(str(req_path)))

        pyproject_path = Path(self.repository_path) / "pyproject.toml"
        if pyproject_path.exists():
            issues.extend(
                self.dependency_scanner.scan_pyproject_toml(str(pyproject_path))
            )

        report = SecurityReport(
            report_id=self._generate_report_id(),
            scan_type="dependency",
            repository_path=self.repository_path,
            issues=issues,
            scan_duration_seconds=time.time() - start_time,
            dependencies_checked=self.dependency_scanner.dependencies_checked,
        )

        report.calculate_summary()
        return report

    def run_secret_scan(self) -> SecurityReport:
        """Run only secret scan"""
        start_time = time.time()

        print("🔐 Running secret scan...")
        issues = self.secret_scanner.scan_directory(self.repository_path)

        report = SecurityReport(
            report_id=self._generate_report_id(),
            scan_type="secret",
            repository_path=self.repository_path,
            issues=issues,
            scan_duration_seconds=time.time() - start_time,
            files_scanned=self.secret_scanner.files_scanned,
        )

        report.calculate_summary()
        return report

    def run_config_scan(self) -> SecurityReport:
        """Run only configuration scan"""
        start_time = time.time()

        print("⚙️  Running configuration scan...")
        issues = self.config_scanner.scan_directory(self.repository_path)

        report = SecurityReport(
            report_id=self._generate_report_id(),
            scan_type="config",
            repository_path=self.repository_path,
            issues=issues,
            scan_duration_seconds=time.time() - start_time,
            files_scanned=self.config_scanner.files_scanned,
        )

        report.calculate_summary()
        return report

    def _generate_recommendations(self, report: SecurityReport) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []

        critical_count = len(report.get_by_severity(SecuritySeverity.CRITICAL))
        high_count = len(report.get_by_severity(SecuritySeverity.HIGH))

        if critical_count > 0:
            recommendations.append(
                f"🔴 URGENT: Address {critical_count} critical security issues immediately"
            )

        if high_count > 0:
            recommendations.append(
                f"🟠 Address {high_count} high severity issues as soon as possible"
            )

        # Check for specific issue types
        has_secrets = any(
            issue.category.value == "exposed_secret" for issue in report.issues
        )
        if has_secrets:
            recommendations.append(
                "Immediately revoke and rotate all exposed credentials"
            )
            recommendations.append("Review git history to remove committed secrets")

        has_eval = any(issue.category.value == "eval_usage" for issue in report.issues)
        if has_eval:
            recommendations.append(
                "Replace eval()/exec() usage with safer alternatives like ast.literal_eval()"
            )

        has_sql_injection = any(
            issue.category.value == "sql_injection" for issue in report.issues
        )
        if has_sql_injection:
            recommendations.append("Use parameterized queries to prevent SQL injection")

        has_vuln_deps = any(
            issue.category.value == "vulnerable_dependency" for issue in report.issues
        )
        if has_vuln_deps:
            recommendations.append("Update vulnerable dependencies to secure versions")

        # General recommendations
        if len(report.issues) > 0:
            recommendations.append(
                "Integrate security scanning into CI/CD pipeline for continuous monitoring"
            )
            recommendations.append(
                "Schedule regular security audits and penetration testing"
            )
            recommendations.append(
                "Implement security training for development team on OWASP Top 10"
            )

        return recommendations

    def _generate_report_id(self) -> str:
        """Generate unique report ID"""
        return f"sec-{uuid.uuid4().hex[:8]}"

    def generate_report_file(
        self, report: SecurityReport, format: str = "html", output_path: str = "security_report"
    ) -> str:
        """
        Generate and save security report

        Args:
            report: SecurityReport object
            format: Output format ('json', 'html', 'markdown', 'text')
            output_path: Base path for output file

        Returns:
            Path to generated report file
        """
        return self.reporter.generate_report(report, format, output_path)
