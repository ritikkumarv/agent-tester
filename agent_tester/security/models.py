"""
Security data models for vulnerability tracking and reporting
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class SecuritySeverity(str, Enum):
    """Severity levels for security issues"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityCategory(str, Enum):
    """Categories of security issues"""

    # SAST Categories
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    INSECURE_RANDOM = "insecure_random"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    WEAK_CRYPTO = "weak_crypto"
    EVAL_USAGE = "eval_usage"
    INSECURE_FUNCTION = "insecure_function"

    # Dependency Categories
    VULNERABLE_DEPENDENCY = "vulnerable_dependency"
    OUTDATED_DEPENDENCY = "outdated_dependency"
    SUPPLY_CHAIN_RISK = "supply_chain_risk"

    # Configuration Categories
    EXPOSED_SECRET = "exposed_secret"
    INSECURE_DEFAULT = "insecure_default"
    MISSING_SECURITY_HEADER = "missing_security_header"
    INSECURE_PERMISSION = "insecure_permission"

    # General
    CODE_QUALITY = "code_quality"
    INFORMATION_DISCLOSURE = "information_disclosure"
    OTHER = "other"


class VulnerabilityFix(BaseModel):
    """Suggested fix for a vulnerability"""

    description: str
    code_snippet: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class SecurityIssue(BaseModel):
    """Represents a detected security issue"""

    issue_id: str
    title: str
    description: str
    severity: SecuritySeverity
    category: SecurityCategory
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    attack_vector: Optional[str] = None
    impact: Optional[str] = None
    cve_id: Optional[str] = None
    cwe_id: Optional[str] = None
    suggested_fix: Optional[VulnerabilityFix] = None
    detected_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    exploitable: bool = False
    false_positive: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SecurityReport(BaseModel):
    """Comprehensive security report"""

    report_id: str
    scan_type: str  # "full", "sast", "dependency", "config"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    repository_path: str
    issues: List[SecurityIssue] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    scan_duration_seconds: float = 0.0
    files_scanned: int = 0
    dependencies_checked: int = 0

    def get_by_severity(self, severity: SecuritySeverity) -> List[SecurityIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_by_category(self, category: SecurityCategory) -> List[SecurityIssue]:
        """Get issues filtered by category"""
        return [issue for issue in self.issues if issue.category == category]

    def get_critical_issues(self) -> List[SecurityIssue]:
        """Get all critical severity issues"""
        return self.get_by_severity(SecuritySeverity.CRITICAL)

    def get_exploitable_issues(self) -> List[SecurityIssue]:
        """Get all exploitable issues"""
        return [issue for issue in self.issues if issue.exploitable]

    def calculate_summary(self):
        """Calculate summary statistics"""
        self.summary = {
            "total_issues": len(self.issues),
            "by_severity": {
                "critical": len(self.get_by_severity(SecuritySeverity.CRITICAL)),
                "high": len(self.get_by_severity(SecuritySeverity.HIGH)),
                "medium": len(self.get_by_severity(SecuritySeverity.MEDIUM)),
                "low": len(self.get_by_severity(SecuritySeverity.LOW)),
                "info": len(self.get_by_severity(SecuritySeverity.INFO)),
            },
            "exploitable_count": len(self.get_exploitable_issues()),
            "files_with_issues": len(
                set(issue.file_path for issue in self.issues if issue.file_path)
            ),
        }


class SecurityChangeLog(BaseModel):
    """Track security changes over time"""

    date: str = Field(default_factory=lambda: datetime.now().isoformat())
    issues_detected: int = 0
    issues_fixed: int = 0
    new_vulnerabilities: List[str] = Field(default_factory=list)
    fixed_vulnerabilities: List[str] = Field(default_factory=list)
    remaining_critical: int = 0
    notes: str = ""
