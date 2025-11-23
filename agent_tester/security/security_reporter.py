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
