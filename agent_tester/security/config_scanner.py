"""
Configuration Security Scanner

Analyzes configuration files for security misconfigurations and exposed secrets.
Checks for insecure configurations and defaults
"""

import os
import re
import logging
from typing import List, Dict, Any

from .security_reporter import SecurityIssue, Severity, IssueCategory

# Set up logger
from typing import List, Dict
from pathlib import Path
import hashlib

from .models import SecurityIssue, SecuritySeverity, SecurityCategory, VulnerabilityFix

logger = logging.getLogger(__name__)


class ConfigurationScanner:
    """
    Scans configuration files for security issues.
    
    Checks:
    - Environment files for exposed secrets
    - Docker configurations for security issues
    - CI/CD configurations for security weaknesses
    - General configuration files for insecure settings
    """
    
    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.files_scanned = 0
    
    def scan_env_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan .env files for exposed secrets"""
        issues = []
        
        # Don't scan .env.example files
        if ".example" in file_path or "sample" in file_path.lower():
            return issues
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            self.files_scanned += 1
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Check if .env file is committed to git
                if os.path.basename(file_path) == ".env":
                    issue = SecurityIssue(
                        issue_id=f"CONFIG-ENV-COMMITTED-{file_path}",
                        title=".env file should not be committed to version control",
                        description="The .env file contains sensitive configuration and should be excluded from version control.",
                        severity=Severity.HIGH,
                        category=IssueCategory.SENSITIVE_DATA,
                        file_path=file_path,
                        line_number=1,
                        attack_vector="Committed .env files expose secrets to anyone with repository access.",
                        impact="Unauthorized access to services, data breaches.",
                        recommendation="Add .env to .gitignore and remove from repository history. Rotate all exposed credentials.",
                        cwe_ids=["CWE-540"],  # Information Exposure Through Source Code
                        owasp_references=["https://owasp.org/www-project-top-ten/2017/A3_2017-Sensitive_Data_Exposure"],
                    )
                    issues.append(issue)
                    break  # Only report once per file
        
        except Exception as e:
            logger.warning(f"Error scanning env file {file_path}: {str(e)}")
        
        return issues
    
    def scan_docker_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan Dockerfile for security issues"""
        issues = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            self.files_scanned += 1
            
            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for running as root
                if line_stripped.startswith("USER root"):
                    issue = SecurityIssue(
                        issue_id=f"CONFIG-DOCKER-ROOT-{file_path}-{line_num}",
                        title="Docker container running as root",
                        description="Container is configured to run as root user, which is a security risk.",
                        severity=Severity.MEDIUM,
                        category=IssueCategory.SECURITY_MISCONFIG,
                        file_path=file_path,
                        line_number=line_num,
                        attack_vector="If container is compromised, attacker has root privileges.",
                        impact="Container escape, host system compromise.",
                        recommendation="Create and use a non-root user in the Dockerfile.",
                        code_sample="RUN adduser -D appuser\nUSER appuser",
                        cwe_ids=["CWE-250"],  # Execution with Unnecessary Privileges
                    )
                    issues.append(issue)
                
                # Check for latest tag
                if re.search(r'FROM\s+.*:latest', line_stripped, re.IGNORECASE):
                    issue = SecurityIssue(
                        issue_id=f"CONFIG-DOCKER-LATEST-{file_path}-{line_num}",
                        title="Docker image using 'latest' tag",
                        description="Using 'latest' tag makes builds non-reproducible and can introduce unexpected changes.",
                        severity=Severity.LOW,
                        category=IssueCategory.SECURITY_MISCONFIG,
                        file_path=file_path,
                        line_number=line_num,
                        attack_vector="Unpredictable image versions can introduce vulnerabilities.",
                        impact="Non-reproducible builds, potential security vulnerabilities.",
                        recommendation="Pin to specific image version tags.",
                        code_sample="FROM python:3.11-slim",
                        cwe_ids=["CWE-1104"],
                    )
                    issues.append(issue)
                
                # Check for secrets in ENV
                if line_stripped.startswith("ENV") and any(
                    keyword in line_stripped.upper()
                    for keyword in ["PASSWORD", "SECRET", "KEY", "TOKEN"]
                ):
                    issue = SecurityIssue(
                        issue_id=f"CONFIG-DOCKER-SECRET-{file_path}-{line_num}",
                        title="Potential secret in ENV variable",
                        description="Hardcoding secrets in Dockerfile ENV variables is insecure.",
                        severity=Severity.HIGH,
                        category=IssueCategory.SENSITIVE_DATA,
                        file_path=file_path,
                        line_number=line_num,
                        attack_vector="Secrets in Docker images are visible to anyone with image access.",
                        impact="Credential exposure, unauthorized access.",
                        recommendation="Use Docker secrets or runtime environment variables instead.",
                        cwe_ids=["CWE-798"],
                    )
                    issues.append(issue)
        
        except Exception as e:
            logger.warning(f"Error scanning Dockerfile {file_path}: {str(e)}")
        
        return issues
    
    def scan_yaml_config(self, file_path: str) -> List[SecurityIssue]:
        """Scan YAML configuration files for security issues"""
        issues = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
            
            self.files_scanned += 1
            
            # Check for hardcoded credentials
            secret_patterns = [
                (r'password:\s*["\']?[^"\'\s]+["\']?', "password"),
                (r'api[_-]?key:\s*["\']?[^"\'\s]+["\']?', "API key"),
                (r'secret:\s*["\']?[^"\'\s]+["\']?', "secret"),
                (r'token:\s*["\']?[^"\'\s]+["\']?', "token"),
            ]
            
            for pattern, credential_type in secret_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count("\n") + 1
                    
                    # Skip if it's clearly a placeholder
                    matched_text = match.group(0)
                    if any(
                        placeholder in matched_text.lower()
                        for placeholder in ["your-", "example", "placeholder", "xxx", "$"]
                    ):
                        continue
                    
                    issue = SecurityIssue(
                        issue_id=f"CONFIG-YAML-SECRET-{file_path}-{line_num}",
                        title=f"Hardcoded {credential_type} in configuration",
                        description=f"A {credential_type} appears to be hardcoded in the YAML configuration.",
                        severity=Severity.HIGH,
                        category=IssueCategory.SENSITIVE_DATA,
                        file_path=file_path,
                        line_number=line_num,
                        attack_vector="Hardcoded credentials can be extracted from configuration files.",
                        impact="Unauthorized access to systems or services.",
                        recommendation="Use environment variables or a secret management system.",
                        cwe_ids=["CWE-798"],
                    )
                    issues.append(issue)
        
        except Exception as e:
            logger.warning(f"Error scanning YAML config {file_path}: {str(e)}")
        
        return issues
    
    def scan_github_workflow(self, file_path: str) -> List[SecurityIssue]:
        """Scan GitHub Actions workflow files for security issues"""
        issues = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
            
            self.files_scanned += 1
            
            # Check for script injection in workflow files
            # Pattern: ${{ github.event.... }} used in run: commands
            injection_pattern = r'run:.*\$\{\{\s*github\.event\.'
            for match in re.finditer(injection_pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                
                issue = SecurityIssue(
                    issue_id=f"CONFIG-GHA-INJECTION-{file_path}-{line_num}",
                    title="Potential script injection in GitHub Actions",
                    description="Using github.event data directly in run commands can lead to script injection.",
                    severity=Severity.HIGH,
                    category=IssueCategory.INJECTION,
                    file_path=file_path,
                    line_number=line_num,
                    attack_vector="An attacker can craft malicious input through PR titles, issue comments, etc.",
                    impact="Arbitrary code execution in CI/CD pipeline, secret exfiltration.",
                    recommendation="Use intermediate environment variables to safely handle user input.",
                    code_sample="env:\n  TITLE: ${{ github.event.pull_request.title }}\nrun: echo \"$TITLE\"",
                    cwe_ids=["CWE-78"],
                    owasp_references=["https://owasp.org/www-community/attacks/Command_Injection"],
                )
                issues.append(issue)
            
            # Check for use of pull_request_target with checkout
            if "pull_request_target" in content and "actions/checkout" in content:
                issue = SecurityIssue(
                    issue_id=f"CONFIG-GHA-PWNTARGET-{file_path}",
                    title="Dangerous use of pull_request_target with checkout",
                    description="Using pull_request_target with actions/checkout can allow attackers to execute code in workflow context.",
                    severity=Severity.CRITICAL,
                    category=IssueCategory.SECURITY_MISCONFIG,
                    file_path=file_path,
                    line_number=1,
                    attack_vector="Attacker can modify workflow files in their fork and execute arbitrary code with secrets access.",
                    impact="Complete repository compromise, secret exfiltration, supply chain attack.",
                    recommendation="Use pull_request trigger instead, or carefully validate checkout ref.",
                    cwe_ids=["CWE-346"],  # Origin Validation Error
                )
                issues.append(issue)
        
        except Exception as e:
            logger.warning(f"Error scanning GitHub workflow {file_path}: {str(e)}")
        
        return issues
    
    def scan_directory(self, directory_path: str) -> List[SecurityIssue]:
        """Scan directory for configuration files and security issues"""
        all_issues = []
        
        for root, dirs, files in os.walk(directory_path):
            # Skip certain directories
            if any(skip in root for skip in [".git", "node_modules", "__pycache__"]):
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Scan different file types
                if file == ".env" or file.endswith(".env"):
                    issues = self.scan_env_file(file_path)
                    all_issues.extend(issues)
                
                elif file == "Dockerfile" or file.endswith(".dockerfile"):
                    issues = self.scan_docker_file(file_path)
                    all_issues.extend(issues)
                
                elif file.endswith(".yaml") or file.endswith(".yml"):
                    # Check if it's a GitHub workflow
                    if ".github/workflows" in file_path:
                        issues = self.scan_github_workflow(file_path)
                    else:
                        issues = self.scan_yaml_config(file_path)
                    all_issues.extend(issues)
        
        self.issues = all_issues
        return all_issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of configuration scan results"""
        return {
            "files_scanned": self.files_scanned,
            "total_issues": len(self.issues),
        }
    """Scans for insecure configurations"""

    # Insecure configuration patterns
    INSECURE_CONFIGS = {
        "debug_enabled": {
            "patterns": [
                r"DEBUG\s*=\s*True",
                r"debug\s*[:=]\s*true",
                r"--debug",
            ],
            "severity": SecuritySeverity.MEDIUM,
            "description": "Debug mode enabled in configuration",
            "fix": "Disable debug mode in production environments",
        },
        "insecure_ssl": {
            "patterns": [
                r"SSL_VERIFY\s*=\s*False",
                r"verify\s*=\s*False",
                r"VERIFY_SSL\s*=\s*False",
            ],
            "severity": SecuritySeverity.HIGH,
            "description": "SSL verification disabled",
            "fix": "Enable SSL verification to prevent man-in-the-middle attacks",
        },
        "permissive_cors": {
            "patterns": [
                r"Access-Control-Allow-Origin:\s*\*",
                r"cors\s*=\s*\*",
                r"CORS_ORIGIN_ALLOW_ALL\s*=\s*True",
            ],
            "severity": SecuritySeverity.MEDIUM,
            "description": "Permissive CORS configuration",
            "fix": "Restrict CORS to specific trusted origins",
        },
        "weak_session": {
            "patterns": [
                r"SESSION_COOKIE_SECURE\s*=\s*False",
                r"SESSION_COOKIE_HTTPONLY\s*=\s*False",
            ],
            "severity": SecuritySeverity.MEDIUM,
            "description": "Insecure session cookie configuration",
            "fix": "Set SESSION_COOKIE_SECURE and SESSION_COOKIE_HTTPONLY to True",
        },
        "exposed_admin": {
            "patterns": [
                r"/admin\s*['\"]",
                r"ADMIN_URL\s*=\s*['\"]admin['\"]",
            ],
            "severity": SecuritySeverity.LOW,
            "description": "Default admin URL detected",
            "fix": "Change admin URL to a non-standard path",
        },
        "insecure_port": {
            "patterns": [
                r"PORT\s*=\s*80\b",
                r"listen\s+80\b",
            ],
            "severity": SecuritySeverity.LOW,
            "description": "Insecure HTTP port configuration",
            "fix": "Use HTTPS (port 443) instead of HTTP (port 80)",
        },
    }

    SKIP_PATTERNS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        "test",
    }

    CONFIG_EXTENSIONS = {
        ".env",
        ".ini",
        ".conf",
        ".config",
        ".yaml",
        ".yml",
        ".json",
        ".py",
    }

    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.files_scanned = 0

    def scan_directory(self, directory_path: str) -> List[SecurityIssue]:
        """Scan directory for configuration issues"""
        self.issues = []
        self.files_scanned = 0

        path = Path(directory_path)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        for file_path in self._get_config_files(path):
            self.scan_file(str(file_path))

        # Check for missing security files
        self._check_missing_security_files(path)

        return self.issues

    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a single configuration file"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            self.files_scanned += 1

            for config_type, config_info in self.INSECURE_CONFIGS.items():
                for pattern in config_info["patterns"]:
                    for line_num, line in enumerate(lines, start=1):
                        if line.strip().startswith("#"):
                            continue

                        if re.search(pattern, line, re.IGNORECASE):
                            issue_id = self._generate_issue_id(
                                file_path, line_num, config_type
                            )

                            # Check for duplicates
                            if any(issue.issue_id == issue_id for issue in self.issues):
                                continue

                            issue = SecurityIssue(
                                issue_id=issue_id,
                                title=f"Insecure Configuration: {config_type.replace('_', ' ').title()}",
                                description=config_info["description"],
                                severity=config_info["severity"],
                                category=SecurityCategory.INSECURE_DEFAULT,
                                file_path=file_path,
                                line_number=line_num,
                                code_snippet=line.strip(),
                                suggested_fix=VulnerabilityFix(
                                    description=config_info["fix"],
                                ),
                            )
                            self.issues.append(issue)

        except Exception as e:
            logger.warning(f"Error scanning configuration file {file_path}: {e}")

        return self.issues

    def _get_config_files(self, path: Path) -> List[Path]:
        """Get list of configuration files to scan"""
        config_files = []

        for item in path.rglob("*"):
            if any(skip in item.parts for skip in self.SKIP_PATTERNS):
                continue

            if item.is_file() and item.suffix in self.CONFIG_EXTENSIONS:
                config_files.append(item)

        return config_files

    def _check_missing_security_files(self, path: Path):
        """Check for missing security-related files"""
        security_files = {
            "SECURITY.md": "Security policy file",
            ".gitignore": "Git ignore file to prevent committing sensitive files",
        }

        for filename, description in security_files.items():
            file_path = path / filename
            if not file_path.exists():
                issue_id = self._generate_issue_id(str(path), 0, f"missing_{filename}")

                issue = SecurityIssue(
                    issue_id=issue_id,
                    title=f"Missing Security File: {filename}",
                    description=f"{description} is missing from the repository",
                    severity=SecuritySeverity.INFO,
                    category=SecurityCategory.INSECURE_DEFAULT,
                    file_path=str(path),
                    suggested_fix=VulnerabilityFix(
                        description=f"Create {filename} file in the repository root",
                        references=[
                            "https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository",
                        ],
                    ),
                )
                self.issues.append(issue)

    def _generate_issue_id(self, file_path: str, line_num: int, config_type: str) -> str:
        """Generate unique ID for an issue"""
        unique_str = f"config:{file_path}:{line_num}:{config_type}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
