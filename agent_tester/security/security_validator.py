"""
Security Validator - Continuous monitoring and validation of repository security
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .sast_scanner import SASTScanner
from .dependency_scanner import DependencyScanner
from .config_scanner import ConfigurationScanner
from .security_reporter import SecurityReporter, SecurityReport, SecurityIssue


class SecurityValidator:
    """
    Main security validation orchestrator.
    
    Coordinates different security scanners and produces comprehensive reports.
    """
    
    def __init__(self):
        self.sast_scanner = SASTScanner()
        self.dependency_scanner = DependencyScanner()
        self.config_scanner = ConfigurationScanner()
        self.reporter = SecurityReporter()
    
    def validate_repository(
        self,
        repository_path: str,
        branch: Optional[str] = None,
        report_id: Optional[str] = None,
    ) -> SecurityReport:
        """
        Perform comprehensive security validation of a repository.
        
        Args:
            repository_path: Path to the repository to scan
            branch: Git branch name (optional)
            report_id: Unique identifier for the report (auto-generated if not provided)
        
        Returns:
            SecurityReport with all findings
        """
        start_time = time.time()
        
        # Generate report ID if not provided
        if not report_id:
            report_id = f"security-scan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create new report
        report = self.reporter.create_report(report_id, repository_path, branch)
        report.scanners_used = ["SAST", "Dependency", "Configuration"]
        
        # Run SAST scan
        sast_issues = self.sast_scanner.scan_directory(repository_path)
        for issue in sast_issues:
            report.add_issue(issue)
        report.total_files_scanned = self.sast_scanner.files_scanned
        
        # Run dependency scan
        dep_issues = self.dependency_scanner.scan_directory(repository_path)
        for issue in dep_issues:
            report.add_issue(issue)
        report.total_dependencies_checked = self.dependency_scanner.dependencies_checked
        
        # Run configuration scan
        config_issues = self.config_scanner.scan_directory(repository_path)
        for issue in config_issues:
            report.add_issue(issue)
        
        # Calculate scan duration
        report.scan_duration_seconds = time.time() - start_time
        
        return report
    
    def validate_file(self, file_path: str) -> List[SecurityIssue]:
        """
        Validate a single file for security issues.
        
        Args:
            file_path: Path to the file to validate
        
        Returns:
            List of security issues found
        """
        issues = []
        
        # Determine file type and run appropriate scanner
        if file_path.endswith(".py"):
            issues.extend(self.sast_scanner.scan_file(file_path))
        
        elif file_path == "requirements.txt":
            issues.extend(self.dependency_scanner.scan_requirements_file(file_path))
        
        elif file_path.endswith(".env"):
            issues.extend(self.config_scanner.scan_env_file(file_path))
        
        elif file_path == "Dockerfile":
            issues.extend(self.config_scanner.scan_docker_file(file_path))
        
        return issues
    
    def get_critical_issues(self, report: SecurityReport) -> List[SecurityIssue]:
        """Get all critical severity issues from a report"""
        return report.critical_issues
    
    def get_high_priority_issues(self, report: SecurityReport) -> List[SecurityIssue]:
        """Get critical and high severity issues from a report"""
        return report.critical_issues + report.high_issues
    
    def export_report(
        self, report: SecurityReport, format: str = "markdown", output_file: Optional[str] = None
    ) -> str:
        """
        Export security report in specified format.
        
        Args:
            report: SecurityReport to export
            format: Output format ('markdown' or 'json')
            output_file: Optional file path to write report to
        
        Returns:
            Formatted report as string
        """
        if format == "markdown":
            content = self.reporter.generate_markdown_report(report)
        elif format == "json":
            import json
            content = json.dumps(self.reporter.generate_json_report(report), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
        
        return content
    
    def continuous_monitor(
        self, repository_path: str, interval_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Setup for continuous monitoring (returns configuration).
        
        In a real implementation, this would set up a scheduled task.
        
        Args:
            repository_path: Path to monitor
            interval_hours: How often to run scans
        
        Returns:
            Monitoring configuration
        """
        return {
            "repository": repository_path,
            "interval_hours": interval_hours,
            "enabled": True,
            "scanners": ["SAST", "Dependency", "Configuration"],
            "notification_on": ["critical", "high"],
        }
