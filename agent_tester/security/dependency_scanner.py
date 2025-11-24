"""
Dependency Security Scanner

Analyzes third-party dependencies for known vulnerabilities (CVEs).
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .security_reporter import SecurityIssue, Severity, IssueCategory

# Set up logger
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
    """
    Scans project dependencies for known vulnerabilities.
    
    Supports:
    - requirements.txt (Python pip)
    - pyproject.toml (Python Poetry)
    - package.json (Node.js npm)
    - Gemfile (Ruby)
    """
    
    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.dependencies_checked = 0
        
        # Known vulnerable packages (simplified - in production, use a CVE database)
        # Format: (package_name, vulnerable_version_pattern, cve_id, severity, description)
        self.known_vulnerabilities = [
            # Example vulnerabilities - in production, fetch from NVD or GitHub Advisory Database
            {
                "package": "requests",
                "vulnerable_versions": ["<2.31.0"],
                "cve": "CVE-2023-32681",
                "severity": Severity.MEDIUM,
                "description": "Requests Proxy-Authorization header leak",
                "recommendation": "Upgrade to requests>=2.31.0",
            },
            {
                "package": "urllib3",
                "vulnerable_versions": ["<1.26.17"],
                "cve": "CVE-2023-43804",
                "severity": Severity.HIGH,
                "description": "Cookie request header isn't stripped on cross-origin redirects",
                "recommendation": "Upgrade to urllib3>=1.26.17",
            },
        ]
    
    def scan_requirements_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a requirements.txt file for vulnerable dependencies"""
        issues = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Parse package specification
                package_info = self._parse_requirement(line)
                if not package_info:
                    continue
                
                package_name, version = package_info
                self.dependencies_checked += 1
                
                # Check for vulnerabilities
                vuln_issues = self._check_vulnerability(
                    package_name, version, file_path, line_num
                )
                issues.extend(vuln_issues)
                
                # Check for unpinned versions
                if not version or version == "*":
                    issue = SecurityIssue(
                        issue_id=f"DEP-UNPIN-{package_name}-{file_path}",
                        title=f"Unpinned dependency: {package_name}",
                        description=f"Package '{package_name}' does not have a pinned version.",
                        severity=Severity.LOW,
                        category=IssueCategory.COMPONENTS_VULNERABILITIES,
                        file_path=file_path,
                        line_number=line_num,
                        attack_vector="Unpinned dependencies can introduce breaking changes or vulnerabilities in future updates.",
                        impact="Unexpected behavior, security vulnerabilities from automatic updates.",
                        recommendation=f"Pin the version: {package_name}==<specific_version>",
                        cwe_ids=["CWE-1104"],  # Use of Unmaintained Third Party Components
                    )
                    issues.append(issue)
        
        except Exception as e:
            pass
        
        return issues
    
    def scan_pyproject_toml(self, file_path: str) -> List[SecurityIssue]:
        """Scan a pyproject.toml file for vulnerable dependencies"""
        issues = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple TOML parsing for dependencies section
            in_dependencies = False
            for line_num, line in enumerate(content.split("\n"), 1):
                line = line.strip()
                
                if "[project.dependencies]" in line or "dependencies = [" in line:
                    in_dependencies = True
                    continue
                
                if in_dependencies:
                    if line.startswith("["):
                        in_dependencies = False
                        continue
                    
                    # Match dependency declarations
                    match = re.search(r'["\']([a-zA-Z0-9_-]+)(?:([>=<]+)([0-9.]+))?["\']', line)
                    if match:
                        package_name = match.group(1)
                        version = match.group(3) if match.group(3) else None
                        self.dependencies_checked += 1
                        
                        vuln_issues = self._check_vulnerability(
                            package_name, version, file_path, line_num
                        )
                        issues.extend(vuln_issues)
        
        except Exception as e:
            # Log error but continue with other files
            logger.warning(f"Error scanning requirements file {file_path}: {str(e)}")
        
        return issues
    
    def scan_directory(self, directory_path: str) -> List[SecurityIssue]:
        """Scan a directory for dependency files and check for vulnerabilities"""
        all_issues = []
        
        # Check for requirements.txt
        req_file = os.path.join(directory_path, "requirements.txt")
        if os.path.exists(req_file):
            issues = self.scan_requirements_file(req_file)
            all_issues.extend(issues)
        
        # Check for pyproject.toml
        pyproject_file = os.path.join(directory_path, "pyproject.toml")
        if os.path.exists(pyproject_file):
            issues = self.scan_pyproject_toml(pyproject_file)
            all_issues.extend(issues)
        
        self.issues = all_issues
        return all_issues
    
    def _parse_requirement(self, line: str) -> Optional[Tuple[str, Optional[str]]]:
        """Parse a requirement line to extract package name and version"""
        # Handle various formats:
        # package==1.0.0
        # package>=1.0.0
        # package~=1.0.0
        # package
        
        # Remove extras and options
        line = re.sub(r'\[.*?\]', '', line)
        line = line.split(";")[0].strip()
        
        # Match package with optional version specifier
        match = re.match(r'^([a-zA-Z0-9_-]+)(?:([><=!~]+)([0-9.]+(?:[a-zA-Z0-9]*)?))?', line)
        if match:
            package_name = match.group(1).lower()
            version = match.group(3) if match.group(3) else None
            return (package_name, version)
        
        return None
    
    def _check_vulnerability(
        self, package_name: str, version: Optional[str], file_path: str, line_num: int
    ) -> List[SecurityIssue]:
        """Check if a package version has known vulnerabilities"""
        issues = []
        
        for vuln in self.known_vulnerabilities:
            if vuln["package"].lower() == package_name.lower():
                # Simple version checking (in production, use proper version comparison)
                is_vulnerable = False
                
                if version:
                    # Extract numeric version for comparison
                    try:
                        current_version = self._parse_version(version)
                        for vuln_pattern in vuln["vulnerable_versions"]:
                            if self._matches_vulnerable_pattern(current_version, vuln_pattern):
                                is_vulnerable = True
                                break
                    except Exception:
                        # If we can't parse version, flag as potentially vulnerable
                        is_vulnerable = True
                else:
                    # No version specified, might be vulnerable
                    is_vulnerable = True
                
                if is_vulnerable:
                    issue = SecurityIssue(
                        issue_id=f"DEP-{vuln['cve']}-{package_name}",
                        title=f"Vulnerable dependency: {package_name}",
                        description=vuln["description"],
                        severity=vuln["severity"],
                        category=IssueCategory.COMPONENTS_VULNERABILITIES,
                        file_path=file_path,
                        line_number=line_num,
                        attack_vector=f"Exploiting {vuln['cve']} in {package_name}",
                        impact="Depends on the specific vulnerability - could range from information disclosure to remote code execution.",
                        recommendation=vuln["recommendation"],
                        cve_ids=[vuln["cve"]],
                        cwe_ids=["CWE-1035"],  # Using Components with Known Vulnerabilities
                        owasp_references=["https://owasp.org/www-project-top-ten/2017/A9_2017-Using_Components_with_Known_Vulnerabilities"],
                    )
                    issues.append(issue)
        
        return issues
    
    def _parse_version(self, version_string: str) -> Tuple[int, ...]:
        """Parse version string to tuple of integers for comparison"""
        # Remove any non-numeric suffixes (like 'b1', 'rc1')
        version_string = re.sub(r'[a-zA-Z].*$', '', version_string)
        parts = version_string.split(".")
        return tuple(int(p) for p in parts if p.isdigit())
    
    def _matches_vulnerable_pattern(self, version: Tuple[int, ...], pattern: str) -> bool:
        """Check if version matches vulnerability pattern"""
        # Simple pattern matching: <2.31.0 means version < 2.31.0
        if pattern.startswith("<"):
            threshold_str = pattern[1:].strip()
            threshold = self._parse_version(threshold_str)
            return version < threshold
        elif pattern.startswith("<="):
            threshold_str = pattern[2:].strip()
            threshold = self._parse_version(threshold_str)
            return version <= threshold
        elif pattern.startswith("=="):
            threshold_str = pattern[2:].strip()
            threshold = self._parse_version(threshold_str)
            return version == threshold
        
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of dependency scan results"""
        return {
            "dependencies_checked": self.dependencies_checked,
            "total_issues": len(self.issues),
            "vulnerable_packages": len(set(i.title for i in self.issues)),
        }
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
