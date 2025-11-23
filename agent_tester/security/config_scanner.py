"""
Configuration Security Scanner

Checks for insecure configurations and defaults
"""

import os
import re
import logging
from typing import List, Dict
from pathlib import Path
import hashlib

from .models import SecurityIssue, SecuritySeverity, SecurityCategory, VulnerabilityFix

logger = logging.getLogger(__name__)


class ConfigurationScanner:
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
