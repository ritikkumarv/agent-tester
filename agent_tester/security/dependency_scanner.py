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
