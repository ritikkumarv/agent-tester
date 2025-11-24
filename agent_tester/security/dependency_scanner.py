"""
Dependency Security Scanner

Scans project dependencies for known vulnerabilities:
- CVE detection in Python packages
- Outdated package detection
- Supply chain risk assessment
"""

import re
import json
import subprocess
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from .models import SecurityIssue, SecuritySeverity, SecurityCategory, VulnerabilityFix

logger = logging.getLogger(__name__)


class DependencyScanner:
    """Scans dependencies for security vulnerabilities"""

    # Known vulnerable packages (this would normally come from a vulnerability database)
    KNOWN_VULNERABILITIES = {
        # Example entries - in production, this would be fetched from a CVE database
        "pyyaml": {
            "versions": ["<5.4"],
            "cve": "CVE-2020-14343",
            "severity": SecuritySeverity.HIGH,
            "description": "PyYAML 5.3.1 allows remote code execution",
            "fix": "Upgrade to PyYAML >= 5.4",
        },
        "pillow": {
            "versions": ["<8.3.2"],
            "cve": "CVE-2021-34552",
            "severity": SecuritySeverity.HIGH,
            "description": "Pillow buffer overflow vulnerability",
            "fix": "Upgrade to Pillow >= 8.3.2",
        },
        "requests": {
            "versions": ["<2.31.0"],
            "cve": "CVE-2023-32681",
            "severity": SecuritySeverity.MEDIUM,
            "description": "Requests proxy-authorization header disclosure",
            "fix": "Upgrade to requests >= 2.31.0",
        },
    }

    # Packages that should use latest versions for security
    SECURITY_CRITICAL_PACKAGES = {
        "cryptography",
        "paramiko",
        "pycryptodome",
        "pyopenssl",
        "pysaml2",
        "pyjwt",
        "django",
        "flask",
        "fastapi",
    }

    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.dependencies_checked = 0

    def scan_requirements(self, requirements_path: str = "requirements.txt") -> List[SecurityIssue]:
        """Scan requirements.txt for vulnerable dependencies"""
        self.issues = []
        self.dependencies_checked = 0

        if not Path(requirements_path).exists():
            return self.issues

        try:
            with open(requirements_path, "r") as f:
                requirements = f.readlines()

            for line in requirements:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                self._check_requirement(line, requirements_path)
                self.dependencies_checked += 1

        except Exception as e:
            logger.warning(f"Error scanning requirements file {requirements_path}: {e}")

        return self.issues

    def scan_pyproject_toml(self, pyproject_path: str = "pyproject.toml") -> List[SecurityIssue]:
        """Scan pyproject.toml for vulnerable dependencies"""
        if not Path(pyproject_path).exists():
            return []

        try:
            with open(pyproject_path, "r") as f:
                content = f.read()

            # Simple regex-based parsing for dependencies
            # In production, use a proper TOML parser
            dep_pattern = r'["\']([a-zA-Z0-9_-]+)([><=!]+)([0-9.]+)["\']'
            matches = re.finditer(dep_pattern, content)

            for match in matches:
                package_name = match.group(1)
                operator = match.group(2)
                version = match.group(3)
                requirement = f"{package_name}{operator}{version}"
                self._check_requirement(requirement, pyproject_path)
                self.dependencies_checked += 1

        except Exception:
            pass

        return self.issues

    def scan_installed_packages(self) -> List[SecurityIssue]:
        """Scan currently installed packages using pip-audit if available"""
        try:
            # Try to run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                self._process_pip_audit_results(audit_data)

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # pip-audit not available or failed
            pass

        return self.issues

    def _check_requirement(self, requirement: str, source_file: str):
        """Check a single requirement for vulnerabilities"""
        # Parse requirement
        match = re.match(r"^([a-zA-Z0-9_-]+)([><=!]+)?([0-9.]+)?", requirement)
        if not match:
            return

        package_name = match.group(1).lower()
        operator = match.group(2) or ""
        version = match.group(3) or ""

        # Check against known vulnerabilities
        if package_name in self.KNOWN_VULNERABILITIES:
            vuln = self.KNOWN_VULNERABILITIES[package_name]
            issue_id = self._generate_issue_id(package_name, source_file)

            issue = SecurityIssue(
                issue_id=issue_id,
                title=f"Vulnerable Dependency: {package_name}",
                description=vuln["description"],
                severity=vuln["severity"],
                category=SecurityCategory.VULNERABLE_DEPENDENCY,
                file_path=source_file,
                cve_id=vuln["cve"],
                attack_vector=f"Exploit via {package_name} dependency",
                impact="Potential for remote code execution or data breach",
                suggested_fix=VulnerabilityFix(
                    description=vuln["fix"],
                    references=[
                        f"https://nvd.nist.gov/vuln/detail/{vuln['cve']}",
                        "https://pypi.org/",
                    ],
                ),
                exploitable=True,
            )
            self.issues.append(issue)

        # Check if security-critical package is pinned to old version
        if package_name in self.SECURITY_CRITICAL_PACKAGES:
            if operator in ["==", "<", "<="]:
                issue_id = self._generate_issue_id(
                    f"{package_name}_outdated", source_file
                )

                issue = SecurityIssue(
                    issue_id=issue_id,
                    title=f"Security-Critical Package May Be Outdated: {package_name}",
                    description=f"{package_name} is a security-critical package and should be kept up to date",
                    severity=SecuritySeverity.MEDIUM,
                    category=SecurityCategory.OUTDATED_DEPENDENCY,
                    file_path=source_file,
                    suggested_fix=VulnerabilityFix(
                        description=f"Review and update {package_name} to the latest secure version",
                        references=[
                            f"https://pypi.org/project/{package_name}/",
                        ],
                    ),
                )
                self.issues.append(issue)

        # Check for unpinned dependencies (supply chain risk)
        if not operator or operator in [">", ">="]:
            issue_id = self._generate_issue_id(f"{package_name}_unpinned", source_file)

            issue = SecurityIssue(
                issue_id=issue_id,
                title=f"Unpinned Dependency: {package_name}",
                description="Unpinned dependencies can introduce unexpected breaking changes or vulnerabilities",
                severity=SecuritySeverity.LOW,
                category=SecurityCategory.SUPPLY_CHAIN_RISK,
                file_path=source_file,
                suggested_fix=VulnerabilityFix(
                    description=f"Pin {package_name} to a specific version range for reproducible builds",
                ),
            )
            self.issues.append(issue)

    def _process_pip_audit_results(self, audit_data: Dict[str, Any]):
        """Process results from pip-audit"""
        for vuln in audit_data.get("vulnerabilities", []):
            package_name = vuln.get("name", "unknown")
            version = vuln.get("version", "unknown")
            cve_id = vuln.get("id", "")

            issue_id = self._generate_issue_id(f"{package_name}_{cve_id}", "pip-audit")

            issue = SecurityIssue(
                issue_id=issue_id,
                title=f"CVE Detected in {package_name}",
                description=vuln.get("description", "Vulnerability detected by pip-audit"),
                severity=self._map_pip_audit_severity(vuln.get("severity", "medium")),
                category=SecurityCategory.VULNERABLE_DEPENDENCY,
                cve_id=cve_id,
                metadata={"version": version, "fix_versions": vuln.get("fix_versions", [])},
                suggested_fix=VulnerabilityFix(
                    description=f"Upgrade {package_name} to version {vuln.get('fix_versions', ['latest'])[0]}",
                    references=[f"https://nvd.nist.gov/vuln/detail/{cve_id}"],
                ),
                exploitable=True,
            )
            self.issues.append(issue)

    def _map_pip_audit_severity(self, severity: str) -> SecuritySeverity:
        """Map pip-audit severity to SecuritySeverity"""
        severity_map = {
            "critical": SecuritySeverity.CRITICAL,
            "high": SecuritySeverity.HIGH,
            "medium": SecuritySeverity.MEDIUM,
            "low": SecuritySeverity.LOW,
        }
        return severity_map.get(severity.lower(), SecuritySeverity.MEDIUM)

    def _generate_issue_id(self, identifier: str, source: str) -> str:
        """Generate unique ID for an issue"""
        unique_str = f"dep:{identifier}:{source}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
