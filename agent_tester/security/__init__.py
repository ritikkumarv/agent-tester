"""
Security Module for AI Agent Testing Framework

This module provides comprehensive cybersecurity monitoring and analysis capabilities:
- Static Application Security Testing (SAST)
- Dependency vulnerability scanning
- Configuration security checks
- Secret detection
- Security reporting and recommendations
"""

from .models import (
    SecurityIssue,
    SecuritySeverity,
    SecurityCategory,
    SecurityReport,
    VulnerabilityFix,
)
from .sast_scanner import SASTScanner
from .dependency_scanner import DependencyScanner
from .secret_scanner import SecretScanner
from .config_scanner import ConfigurationScanner
from .security_reporter import SecurityReporter
from .security_orchestrator import SecurityOrchestrator

__all__ = [
    "SecurityIssue",
    "SecuritySeverity",
    "SecurityCategory",
    "SecurityReport",
    "VulnerabilityFix",
    "SASTScanner",
    "DependencyScanner",
    "SecretScanner",
    "ConfigurationScanner",
    "SecurityReporter",
    "SecurityOrchestrator",
]
