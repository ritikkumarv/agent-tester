"""
Secret Scanner

Detects exposed secrets, API keys, tokens, and credentials in code and configuration files
"""

import re
import os
from typing import List, Dict, Pattern
from pathlib import Path
import hashlib

from .models import SecurityIssue, SecuritySeverity, SecurityCategory, VulnerabilityFix


class SecretScanner:
    """Scans for exposed secrets and credentials"""

    # Secret detection patterns
    SECRET_PATTERNS: Dict[str, Dict] = {
        "aws_access_key": {
            "pattern": r"AKIA[0-9A-Z]{16}",
            "description": "AWS Access Key ID detected",
        },
        "aws_secret_key": {
            "pattern": r"aws_secret_access_key\s*=\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
            "description": "AWS Secret Access Key detected",
        },
        "github_token": {
            "pattern": r"ghp_[A-Za-z0-9]{36}",
            "description": "GitHub Personal Access Token detected",
        },
        "github_oauth": {
            "pattern": r"gho_[A-Za-z0-9]{36}",
            "description": "GitHub OAuth Token detected",
        },
        "slack_token": {
            "pattern": r"xox[baprs]-[0-9]{10,12}-[0-9]{10,12}-[A-Za-z0-9]{24,32}",
            "description": "Slack Token detected",
        },
        "slack_webhook": {
            "pattern": r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
            "description": "Slack Webhook URL detected",
        },
        "openai_api_key": {
            "pattern": r"sk-[A-Za-z0-9]{48}",
            "description": "OpenAI API Key detected",
        },
        "azure_subscription_key": {
            "pattern": r"[0-9a-f]{32}",
            "description": "Potential Azure Subscription Key",
        },
        "generic_api_key": {
            "pattern": r"['\"]?api[_-]?key['\"]?\s*[:=]\s*['\"]([A-Za-z0-9_\-]{20,})['\"]",
            "description": "Generic API key detected",
        },
        "generic_secret": {
            "pattern": r"['\"]?secret['\"]?\s*[:=]\s*['\"]([A-Za-z0-9_\-]{16,})['\"]",
            "description": "Generic secret detected",
        },
        "password_assignment": {
            "pattern": r"['\"]?password['\"]?\s*[:=]\s*['\"]([^'\"]{8,})['\"]",
            "description": "Hardcoded password detected",
        },
        "private_key": {
            "pattern": r"-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----",
            "description": "Private key detected",
        },
        "jwt_token": {
            "pattern": r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
            "description": "JWT token detected",
        },
        "connection_string": {
            "pattern": r"(mongodb|mysql|postgresql|redis)://[^\s'\"]+",
            "description": "Database connection string detected",
        },
    }

    # Files to skip
    SKIP_PATTERNS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".pytest_cache",
        ".env.example",  # Template files are OK
        "example",
        "test",
        "mock",
    }

    # File extensions to scan
    SCANNABLE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".env",
        ".config",
        ".conf",
        ".ini",
        ".xml",
        ".properties",
        ".sh",
        ".bash",
        ".ps1",
    }

    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.files_scanned = 0

    def scan_directory(self, directory_path: str) -> List[SecurityIssue]:
        """Scan directory for exposed secrets"""
        self.issues = []
        self.files_scanned = 0

        path = Path(directory_path)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        for file_path in self._get_scannable_files(path):
            self.scan_file(str(file_path))

        return self.issues

    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a single file for secrets"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            self.files_scanned += 1

            for secret_type, secret_info in self.SECRET_PATTERNS.items():
                pattern = secret_info["pattern"]
                for line_num, line in enumerate(lines, start=1):
                    # Skip comments
                    if line.strip().startswith("#") or line.strip().startswith("//"):
                        continue

                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Avoid false positives from example values
                        if self._is_likely_example(match.group(0)):
                            continue

                        issue_id = self._generate_issue_id(
                            file_path, line_num, secret_type
                        )

                        # Check for duplicates
                        if any(issue.issue_id == issue_id for issue in self.issues):
                            continue

                        # Redact the actual secret in the snippet
                        redacted_line = self._redact_secret(line, match)

                        issue = SecurityIssue(
                            issue_id=issue_id,
                            title=f"Exposed Secret: {secret_type.replace('_', ' ').title()}",
                            description=secret_info["description"],
                            severity=SecuritySeverity.CRITICAL,
                            category=SecurityCategory.EXPOSED_SECRET,
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=redacted_line,
                            attack_vector="Credential theft from source code repository",
                            impact="Unauthorized access to services and data",
                            suggested_fix=VulnerabilityFix(
                                description=(
                                    "1. Immediately revoke this credential\n"
                                    "2. Remove from source code and git history\n"
                                    "3. Use environment variables or secret management services\n"
                                    "4. Add to .gitignore if in config file"
                                ),
                                references=[
                                    "https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password",
                                    "https://12factor.net/config",
                                ],
                            ),
                            exploitable=True,
                        )
                        self.issues.append(issue)

        except Exception:
            pass

        return self.issues

    def _get_scannable_files(self, path: Path) -> List[Path]:
        """Get list of files to scan"""
        scannable_files = []

        for item in path.rglob("*"):
            # Skip directories and files in skip patterns
            if any(skip.lower() in str(item).lower() for skip in self.SKIP_PATTERNS):
                continue

            if item.is_file() and item.suffix in self.SCANNABLE_EXTENSIONS:
                scannable_files.append(item)

        return scannable_files

    def _is_likely_example(self, value: str) -> bool:
        """Check if value is likely an example/placeholder"""
        examples = [
            "example",
            "placeholder",
            "your-key-here",
            "xxx",
            "yyy",
            "zzz",
            "abc123",
            "test",
            "dummy",
            "fake",
            "sample",
            "changeme",
        ]
        value_lower = value.lower()
        return any(ex in value_lower for ex in examples)

    def _redact_secret(self, line: str, match: re.Match) -> str:
        """Redact the secret from the line"""
        secret = match.group(0)
        if len(secret) > 8:
            redacted = secret[:4] + "***" + secret[-4:]
        else:
            redacted = "***REDACTED***"
        return line.replace(secret, redacted)

    def _generate_issue_id(self, file_path: str, line_num: int, secret_type: str) -> str:
        """Generate unique ID for an issue"""
        unique_str = f"secret:{file_path}:{line_num}:{secret_type}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
