"""
Security Reporter - Provides structured security issue reporting and management
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Security issue severity levels aligned with industry standards"""

    CRITICAL = "critical"  # Immediate action required, exploitable vulnerabilities
    HIGH = "high"  # Significant risk, should be addressed urgently
    MEDIUM = "medium"  # Moderate risk, should be addressed in near term
    LOW = "low"  # Minor risk, can be addressed in regular updates
    INFO = "info"  # Informational, best practice recommendations


class IssueCategory(str, Enum):
    """Categories of security issues"""

    INJECTION = "injection"  # SQL, Command, Code injection
    AUTHENTICATION = "authentication"  # Auth/AuthZ issues
    SENSITIVE_DATA = "sensitive_data"  # Exposed secrets, passwords, keys
    XXE = "xxe"  # XML External Entities
    BROKEN_ACCESS = "broken_access_control"  # Access control issues
    SECURITY_MISCONFIG = "security_misconfiguration"  # Configuration issues
    XSS = "xss"  # Cross-Site Scripting
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    COMPONENTS_VULNERABILITIES = "known_vulnerabilities"  # CVEs in dependencies
    LOGGING_MONITORING = "insufficient_logging"  # Logging/monitoring gaps
    SSRF = "ssrf"  # Server-Side Request Forgery
    CRYPTOGRAPHY = "cryptographic_failure"  # Crypto weaknesses
    CODE_QUALITY = "code_quality"  # Insecure coding patterns


class SecurityIssue(BaseModel):
    """Represents a single security issue found during analysis"""

    issue_id: str = Field(description="Unique identifier for the issue")
    title: str = Field(description="Brief description of the issue")
    description: str = Field(description="Detailed explanation of the vulnerability")
    severity: Severity = Field(description="Issue severity level")
    category: IssueCategory = Field(description="OWASP-aligned category")
    
    # Location information
    file_path: Optional[str] = Field(None, description="File where issue was found")
    line_number: Optional[int] = Field(None, description="Line number of the issue")
    commit_hash: Optional[str] = Field(None, description="Git commit where issue exists")
    function_name: Optional[str] = Field(None, description="Function/method name")
    
    # Attack vector information
    attack_vector: Optional[str] = Field(
        None, description="How the vulnerability could be exploited"
    )
    impact: Optional[str] = Field(None, description="Potential impact if exploited")
    
    # Remediation
    recommendation: str = Field(description="How to fix the issue")
    code_sample: Optional[str] = Field(None, description="Example of secure code")
    
    # References
    cve_ids: List[str] = Field(default_factory=list, description="Related CVE IDs")
    cwe_ids: List[str] = Field(default_factory=list, description="Related CWE IDs")
    owasp_references: List[str] = Field(
        default_factory=list, description="OWASP reference links"
    )
    
    # Metadata
    detected_at: datetime = Field(default_factory=datetime.now)
    false_positive: bool = Field(False, description="Marked as false positive")
    status: str = Field("open", description="open, fixed, accepted_risk, false_positive")
    
    model_config = {"use_enum_values": True}


class SecurityReport(BaseModel):
    """Comprehensive security report"""

    report_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    repository_path: str
    branch: Optional[str] = None
    
    # Issues by severity
    critical_issues: List[SecurityIssue] = Field(default_factory=list)
    high_issues: List[SecurityIssue] = Field(default_factory=list)
    medium_issues: List[SecurityIssue] = Field(default_factory=list)
    low_issues: List[SecurityIssue] = Field(default_factory=list)
    info_issues: List[SecurityIssue] = Field(default_factory=list)
    
    # Summary statistics
    total_issues: int = 0
    total_files_scanned: int = 0
    total_dependencies_checked: int = 0
    
    # Scan metadata
    scan_duration_seconds: float = 0.0
    scanners_used: List[str] = Field(default_factory=list)
    
    def add_issue(self, issue: SecurityIssue):
        """Add an issue to the appropriate severity list"""
        if issue.severity == Severity.CRITICAL:
            self.critical_issues.append(issue)
        elif issue.severity == Severity.HIGH:
            self.high_issues.append(issue)
        elif issue.severity == Severity.MEDIUM:
            self.medium_issues.append(issue)
        elif issue.severity == Severity.LOW:
            self.low_issues.append(issue)
        else:
            self.info_issues.append(issue)
        self.total_issues += 1
    
    def get_all_issues(self) -> List[SecurityIssue]:
        """Get all issues sorted by severity"""
        return (
            self.critical_issues
            + self.high_issues
            + self.medium_issues
            + self.low_issues
            + self.info_issues
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_issues": self.total_issues,
            "critical": len(self.critical_issues),
            "high": len(self.high_issues),
            "medium": len(self.medium_issues),
            "low": len(self.low_issues),
            "info": len(self.info_issues),
            "files_scanned": self.total_files_scanned,
            "dependencies_checked": self.total_dependencies_checked,
            "scan_duration": self.scan_duration_seconds,
        }


class SecurityReporter:
    """Manages security reporting and issue tracking"""

    def __init__(self):
        self.reports: List[SecurityReport] = []
    
    def create_report(
        self, report_id: str, repository_path: str, branch: Optional[str] = None
    ) -> SecurityReport:
        """Create a new security report"""
        report = SecurityReport(
            report_id=report_id, repository_path=repository_path, branch=branch
        )
        self.reports.append(report)
        return report
    
    def generate_markdown_report(self, report: SecurityReport) -> str:
        """Generate a markdown-formatted security report"""
        lines = [
            f"# Security Report: {report.report_id}",
            f"",
            f"**Generated**: {report.timestamp.isoformat()}",
            f"**Repository**: {report.repository_path}",
            f"**Branch**: {report.branch or 'N/A'}",
            f"",
            f"## Executive Summary",
            f"",
            f"- 🔴 Critical Issues: {len(report.critical_issues)}",
            f"- 🟠 High Issues: {len(report.high_issues)}",
            f"- 🟡 Medium Issues: {len(report.medium_issues)}",
            f"- 🔵 Low Issues: {len(report.low_issues)}",
            f"- ℹ️ Info: {len(report.info_issues)}",
            f"",
            f"**Total Issues**: {report.total_issues}",
            f"**Files Scanned**: {report.total_files_scanned}",
            f"**Dependencies Checked**: {report.total_dependencies_checked}",
            f"**Scan Duration**: {report.scan_duration_seconds:.2f}s",
            f"",
        ]
        
        # Add issues by severity
        for severity_name, issues in [
            ("Critical", report.critical_issues),
            ("High", report.high_issues),
            ("Medium", report.medium_issues),
            ("Low", report.low_issues),
            ("Informational", report.info_issues),
        ]:
            if issues:
                lines.append(f"## {severity_name} Severity Issues")
                lines.append("")
                for issue in issues:
                    lines.append(f"### {issue.title}")
                    lines.append(f"")
                    lines.append(f"- **Category**: {issue.category}")
                    lines.append(f"- **File**: {issue.file_path or 'N/A'}")
                    if issue.line_number:
                        lines.append(f"- **Line**: {issue.line_number}")
                    lines.append(f"")
                    lines.append(f"**Description**: {issue.description}")
                    lines.append(f"")
                    if issue.attack_vector:
                        lines.append(f"**Attack Vector**: {issue.attack_vector}")
                        lines.append(f"")
                    if issue.impact:
                        lines.append(f"**Impact**: {issue.impact}")
                        lines.append(f"")
                    lines.append(f"**Recommendation**: {issue.recommendation}")
                    lines.append(f"")
                    if issue.code_sample:
                        lines.append(f"**Secure Code Example**:")
                        lines.append(f"```python")
                        lines.append(issue.code_sample)
                        lines.append(f"```")
                        lines.append(f"")
                    if issue.cve_ids:
                        lines.append(f"**CVEs**: {', '.join(issue.cve_ids)}")
                    if issue.cwe_ids:
                        lines.append(f"**CWEs**: {', '.join(issue.cwe_ids)}")
                    if issue.owasp_references:
                        lines.append(f"**OWASP References**: {', '.join(issue.owasp_references)}")
                    lines.append(f"")
                    lines.append(f"---")
                    lines.append(f"")
        
        return "\n".join(lines)
    
    def generate_json_report(self, report: SecurityReport) -> Dict[str, Any]:
        """Generate a JSON-formatted security report"""
        # Use model_dump with mode='json' to handle datetime serialization
        return report.model_dump(mode='json')
    
    def get_latest_report(self) -> Optional[SecurityReport]:
        """Get the most recent security report"""
        if self.reports:
            return sorted(self.reports, key=lambda r: r.timestamp, reverse=True)[0]
        return None
Security Reporter

Generates comprehensive security reports in various formats
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .models import SecurityReport, SecurityIssue, SecuritySeverity, SecurityChangeLog


class SecurityReporter:
    """Generates security reports in various formats"""

    def __init__(self):
        self.reports: List[SecurityReport] = []

    def generate_report(
        self,
        report: SecurityReport,
        output_format: str = "json",
        output_path: str = "security_report",
    ) -> str:
        """
        Generate security report in specified format

        Args:
            report: SecurityReport object
            output_format: 'json', 'html', 'markdown', or 'text'
            output_path: Base path for output file (extension added automatically)

        Returns:
            Path to generated report file
        """
        # Calculate summary before generating report
        report.calculate_summary()

        if output_format == "json":
            return self._generate_json_report(report, output_path)
        elif output_format == "html":
            return self._generate_html_report(report, output_path)
        elif output_format == "markdown":
            return self._generate_markdown_report(report, output_path)
        elif output_format == "text":
            return self._generate_text_report(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_json_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate JSON report"""
        file_path = f"{output_path}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

        return file_path

    def _generate_text_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate plain text report"""
        file_path = f"{output_path}.txt"

        lines = []
        lines.append("=" * 80)
        lines.append("SECURITY SCAN REPORT")
        lines.append("=" * 80)
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Scan Type: {report.scan_type}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Repository: {report.repository_path}")
        lines.append(f"Files Scanned: {report.files_scanned}")
        lines.append(f"Dependencies Checked: {report.dependencies_checked}")
        lines.append(f"Scan Duration: {report.scan_duration_seconds:.2f}s")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Issues: {report.summary.get('total_issues', 0)}")
        lines.append("")
        lines.append("By Severity:")
        for severity, count in report.summary.get("by_severity", {}).items():
            lines.append(f"  {severity.upper()}: {count}")
        lines.append(f"Exploitable Issues: {report.summary.get('exploitable_count', 0)}")
        lines.append(f"Files with Issues: {report.summary.get('files_with_issues', 0)}")
        lines.append("")

        # Issues by severity
        for severity in [
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
        ]:
            issues = report.get_by_severity(severity)
            if issues:
                lines.append(f"\n{severity.value.upper()} SEVERITY ISSUES ({len(issues)})")
                lines.append("-" * 80)
                for i, issue in enumerate(issues, 1):
                    lines.append(f"\n{i}. {issue.title}")
                    lines.append(f"   ID: {issue.issue_id}")
                    lines.append(f"   Category: {issue.category.value}")
                    if issue.file_path:
                        location = f"{issue.file_path}"
                        if issue.line_number:
                            location += f":{issue.line_number}"
                        lines.append(f"   Location: {location}")
                    lines.append(f"   Description: {issue.description}")
                    if issue.code_snippet:
                        lines.append(f"   Code: {issue.code_snippet}")
                    if issue.suggested_fix:
                        lines.append(f"   Fix: {issue.suggested_fix.description}")
                    if issue.exploitable:
                        lines.append("   ⚠️  EXPLOITABLE")

        # Recommendations
        if report.recommendations:
            lines.append("\n\nRECOMMENDATIONS")
            lines.append("-" * 80)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")

        lines.append("\n" + "=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return file_path

    def _generate_markdown_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate Markdown report"""
        file_path = f"{output_path}.md"

        lines = []
        lines.append("# Security Scan Report")
        lines.append("")
        lines.append(f"**Report ID:** {report.report_id}")
        lines.append(f"**Scan Type:** {report.scan_type}")
        lines.append(f"**Timestamp:** {report.timestamp}")
        lines.append(f"**Repository:** {report.repository_path}")
        lines.append(f"**Files Scanned:** {report.files_scanned}")
        lines.append(f"**Dependencies Checked:** {report.dependencies_checked}")
        lines.append(f"**Scan Duration:** {report.scan_duration_seconds:.2f}s")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Issues:** {report.summary.get('total_issues', 0)}")
        lines.append(f"- **Exploitable Issues:** {report.summary.get('exploitable_count', 0)}")
        lines.append(f"- **Files with Issues:** {report.summary.get('files_with_issues', 0)}")
        lines.append("")

        # Severity breakdown
        lines.append("### Issues by Severity")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for severity, count in report.summary.get("by_severity", {}).items():
            emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
                "info": "⚪",
            }.get(severity, "")
            lines.append(f"| {emoji} {severity.upper()} | {count} |")
        lines.append("")

        # Detailed issues
        for severity in [
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
        ]:
            issues = report.get_by_severity(severity)
            if issues:
                emoji = {
                    SecuritySeverity.CRITICAL: "🔴",
                    SecuritySeverity.HIGH: "🟠",
                    SecuritySeverity.MEDIUM: "🟡",
                    SecuritySeverity.LOW: "🔵",
                }.get(severity, "")

                lines.append(f"## {emoji} {severity.value.upper()} Severity Issues")
                lines.append("")
                for issue in issues:
                    lines.append(f"### {issue.title}")
                    if issue.exploitable:
                        lines.append("⚠️ **EXPLOITABLE**")
                    lines.append("")
                    lines.append(f"- **ID:** `{issue.issue_id}`")
                    lines.append(f"- **Category:** {issue.category.value}")
                    if issue.file_path:
                        location = f"`{issue.file_path}`"
                        if issue.line_number:
                            location += f" (line {issue.line_number})"
                        lines.append(f"- **Location:** {location}")
                    if issue.cve_id:
                        lines.append(f"- **CVE:** [{issue.cve_id}](https://nvd.nist.gov/vuln/detail/{issue.cve_id})")
                    lines.append("")
                    lines.append(f"**Description:** {issue.description}")
                    lines.append("")
                    if issue.code_snippet:
                        lines.append("**Code:**")
                        lines.append("```")
                        lines.append(issue.code_snippet)
                        lines.append("```")
                        lines.append("")
                    if issue.suggested_fix:
                        lines.append("**Suggested Fix:**")
                        lines.append(issue.suggested_fix.description)
                        lines.append("")
                    lines.append("---")
                    lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return file_path

    def _generate_html_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate HTML report"""
        file_path = f"{output_path}.html"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Scan Report - {report.report_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .severity-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .critical {{ background: #dc3545; color: white; }}
        .high {{ background: #fd7e14; color: white; }}
        .medium {{ background: #ffc107; color: black; }}
        .low {{ background: #17a2b8; color: white; }}
        .info {{ background: #6c757d; color: white; }}
        .issue {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .issue.critical {{ border-left-color: #dc3545; }}
        .issue.high {{ border-left-color: #fd7e14; }}
        .issue.medium {{ border-left-color: #ffc107; }}
        .issue.low {{ border-left-color: #17a2b8; }}
        .code {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
        .fix {{
            background: #d4edda;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #28a745;
            margin-top: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Security Scan Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Scan Type:</strong> {report.scan_type}</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
        <p><strong>Repository:</strong> {report.repository_path}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{report.summary.get('total_issues', 0)}</div>
            <div>Total Issues</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report.files_scanned}</div>
            <div>Files Scanned</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report.dependencies_checked}</div>
            <div>Dependencies Checked</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report.scan_duration_seconds:.1f}s</div>
            <div>Scan Duration</div>
        </div>
    </div>

    <div class="summary">
        <h2>Severity Breakdown</h2>
        <p>
            <span class="severity-badge critical">CRITICAL: {report.summary.get('by_severity', {}).get('critical', 0)}</span>
            <span class="severity-badge high">HIGH: {report.summary.get('by_severity', {}).get('high', 0)}</span>
            <span class="severity-badge medium">MEDIUM: {report.summary.get('by_severity', {}).get('medium', 0)}</span>
            <span class="severity-badge low">LOW: {report.summary.get('by_severity', {}).get('low', 0)}</span>
            <span class="severity-badge info">INFO: {report.summary.get('by_severity', {}).get('info', 0)}</span>
        </p>
    </div>
"""

        # Add issues
        for severity in [
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
        ]:
            issues = report.get_by_severity(severity)
            if issues:
                html += f'<h2>{severity.value.upper()} Severity Issues ({len(issues)})</h2>\n'
                for issue in issues:
                    html += f'<div class="issue {severity.value}">\n'
                    html += f'<h3>{issue.title}</h3>\n'
                    if issue.exploitable:
                        html += '<span class="severity-badge critical">⚠️ EXPLOITABLE</span>\n'
                    html += f'<p><strong>ID:</strong> {issue.issue_id}</p>\n'
                    html += f'<p><strong>Category:</strong> {issue.category.value}</p>\n'
                    if issue.file_path:
                        location = issue.file_path
                        if issue.line_number:
                            location += f":{issue.line_number}"
                        html += f'<p><strong>Location:</strong> {location}</p>\n'
                    html += f'<p>{issue.description}</p>\n'
                    if issue.code_snippet:
                        html += f'<div class="code">{issue.code_snippet}</div>\n'
                    if issue.suggested_fix:
                        html += f'<div class="fix"><strong>Fix:</strong> {issue.suggested_fix.description}</div>\n'
                    html += '</div>\n'

        html += """
</body>
</html>
"""

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)

        return file_path

    def create_changelog_entry(
        self, current_report: SecurityReport, previous_report: Optional[SecurityReport] = None
    ) -> SecurityChangeLog:
        """Create a changelog entry comparing two reports"""
        changelog = SecurityChangeLog()

        if previous_report:
            current_issues = {issue.issue_id for issue in current_report.issues}
            previous_issues = {issue.issue_id for issue in previous_report.issues}

            new_issues = current_issues - previous_issues
            fixed_issues = previous_issues - current_issues

            changelog.issues_detected = len(new_issues)
            changelog.issues_fixed = len(fixed_issues)
            changelog.new_vulnerabilities = list(new_issues)
            changelog.fixed_vulnerabilities = list(fixed_issues)

        changelog.remaining_critical = len(current_report.get_critical_issues())

        return changelog
