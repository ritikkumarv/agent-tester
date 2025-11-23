"""
Static Application Security Testing (SAST) Scanner

Detects common security vulnerabilities in source code:
- Insecure function usage
- SQL injection vulnerabilities
- XSS vulnerabilities
- Command injection
- Path traversal
- Unsafe deserialization
- Weak cryptography
- Hardcoded secrets
"""

import re
import os
import ast
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import hashlib

from .models import SecurityIssue, SecuritySeverity, SecurityCategory, VulnerabilityFix


class SASTScanner:
    """Static Application Security Testing Scanner"""

    # Insecure functions and patterns
    INSECURE_PATTERNS = {
        # Python dangerous functions
        "eval_usage": {
            "patterns": [r"\beval\s*\(", r"\bexec\s*\("],
            "severity": SecuritySeverity.CRITICAL,
            "category": SecurityCategory.EVAL_USAGE,
            "description": "Use of eval() or exec() can lead to arbitrary code execution",
            "fix": "Avoid eval/exec. Use ast.literal_eval() for safe evaluation or json.loads() for JSON data",
        },
        "pickle_usage": {
            "patterns": [r"\bpickle\.loads\s*\(", r"\bpickle\.load\s*\("],
            "severity": SecuritySeverity.HIGH,
            "category": SecurityCategory.UNSAFE_DESERIALIZATION,
            "description": "Pickle deserialization can execute arbitrary code",
            "fix": "Use JSON or other safe serialization formats. If pickle is necessary, validate and sanitize input",
        },
        "sql_format": {
            "patterns": [
                r'\.format\s*\(',
                r'%\s*\(',
                r'f["\'].*{.*}.*["\']',
            ],
            "severity": SecuritySeverity.HIGH,
            "category": SecurityCategory.SQL_INJECTION,
            "description": "String formatting in SQL queries may lead to SQL injection",
            "fix": "Use parameterized queries with placeholders (?, %s) instead of string formatting",
        },
        "shell_injection": {
            "patterns": [
                r"os\.system\s*\(",
                r"shell\s*=\s*True",
            ],
            "severity": SecuritySeverity.CRITICAL,
            "category": SecurityCategory.COMMAND_INJECTION,
            "description": "Command execution with shell=True can lead to command injection",
            "fix": "Use shell=False and pass arguments as a list. Validate and sanitize all user inputs",
        },
        "path_traversal": {
            "patterns": [
                r"open\s*\([^,)]*\+",
                r"os\.path\.join\s*\([^)]*\.\.",
            ],
            "severity": SecuritySeverity.HIGH,
            "category": SecurityCategory.PATH_TRAVERSAL,
            "description": "Potential path traversal vulnerability",
            "fix": "Validate file paths, use os.path.realpath(), and check against allowed directories",
        },
        "weak_hash": {
            "patterns": [
                r"hashlib\.md5\s*\(",
                r"hashlib\.sha1\s*\(",
            ],
            "severity": SecuritySeverity.MEDIUM,
            "category": SecurityCategory.WEAK_CRYPTO,
            "description": "MD5 and SHA1 are cryptographically broken",
            "fix": "Use SHA-256 or stronger: hashlib.sha256(), hashlib.sha512()",
        },
        "hardcoded_secret": {
            "patterns": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            "severity": SecuritySeverity.CRITICAL,
            "category": SecurityCategory.HARDCODED_CREDENTIALS,
            "description": "Hardcoded credentials found in source code",
            "fix": "Store secrets in environment variables or secure vaults (e.g., Azure Key Vault, AWS Secrets Manager)",
        },
        "insecure_random": {
            "patterns": [r"random\.random\s*\(", r"random\.randint\s*\("],
            "severity": SecuritySeverity.MEDIUM,
            "category": SecurityCategory.INSECURE_RANDOM,
            "description": "random module is not cryptographically secure",
            "fix": "Use secrets module for security-sensitive randomness: secrets.token_bytes(), secrets.token_hex()",
        },
        "assert_usage": {
            "patterns": [r"\bassert\s+"],
            "severity": SecuritySeverity.LOW,
            "category": SecurityCategory.CODE_QUALITY,
            "description": "Assert statements are removed in optimized Python (-O flag)",
            "fix": "Use proper exception handling and validation instead of assert for security checks",
        },
    }

    # File extensions to scan
    SCANNABLE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rb", ".php"}

    # Files/directories to skip
    SKIP_PATTERNS = {
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
    }

    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.files_scanned = 0

    def scan_directory(self, directory_path: str) -> List[SecurityIssue]:
        """Scan entire directory for security issues"""
        self.issues = []
        self.files_scanned = 0

        path = Path(directory_path)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        for file_path in self._get_scannable_files(path):
            self.scan_file(str(file_path))

        return self.issues

    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a single file for security issues"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            self.files_scanned += 1

            # Pattern-based scanning
            self._scan_patterns(file_path, content, lines)

            # AST-based scanning for Python files
            if file_path.endswith(".py"):
                self._scan_python_ast(file_path, content)

            return self.issues

        except Exception as e:
            # Log error but continue scanning
            return self.issues

    def _get_scannable_files(self, path: Path) -> List[Path]:
        """Get list of files to scan"""
        scannable_files = []

        for item in path.rglob("*"):
            # Skip directories and files in skip patterns
            if any(skip in item.parts for skip in self.SKIP_PATTERNS):
                continue

            if item.is_file() and item.suffix in self.SCANNABLE_EXTENSIONS:
                scannable_files.append(item)

        return scannable_files

    def _scan_patterns(self, file_path: str, content: str, lines: List[str]):
        """Scan file using regex patterns"""
        for pattern_name, pattern_info in self.INSECURE_PATTERNS.items():
            for pattern in pattern_info["patterns"]:
                for line_num, line in enumerate(lines, start=1):
                    # Skip comments
                    if line.strip().startswith("#") or line.strip().startswith("//"):
                        continue

                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        issue_id = self._generate_issue_id(
                            file_path, line_num, pattern_name
                        )

                        # Check for duplicates
                        if any(issue.issue_id == issue_id for issue in self.issues):
                            continue

                        issue = SecurityIssue(
                            issue_id=issue_id,
                            title=f"{pattern_name.replace('_', ' ').title()} Detected",
                            description=pattern_info["description"],
                            severity=pattern_info["severity"],
                            category=pattern_info["category"],
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip(),
                            suggested_fix=VulnerabilityFix(
                                description=pattern_info["fix"],
                                references=[
                                    "https://owasp.org/www-project-top-ten/",
                                    "https://cwe.mitre.org/",
                                ],
                            ),
                        )
                        self.issues.append(issue)

    def _scan_python_ast(self, file_path: str, content: str):
        """Scan Python files using AST analysis"""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["pickle", "marshal", "shelve"]:
                            issue_id = self._generate_issue_id(
                                file_path, node.lineno, f"import_{alias.name}"
                            )
                            issue = SecurityIssue(
                                issue_id=issue_id,
                                title=f"Potentially Unsafe Import: {alias.name}",
                                description=f"Module {alias.name} can be unsafe with untrusted data",
                                severity=SecuritySeverity.MEDIUM,
                                category=SecurityCategory.UNSAFE_DESERIALIZATION,
                                file_path=file_path,
                                line_number=node.lineno,
                                suggested_fix=VulnerabilityFix(
                                    description="Review usage of this module and ensure proper input validation",
                                ),
                            )
                            self.issues.append(issue)

        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception:
            # Skip files that can't be parsed
            pass

    def _generate_issue_id(self, file_path: str, line_num: int, pattern: str) -> str:
        """Generate unique ID for an issue"""
        unique_str = f"{file_path}:{line_num}:{pattern}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]

    def get_issues_by_severity(
        self, severity: SecuritySeverity
    ) -> List[SecurityIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_critical_issues(self) -> List[SecurityIssue]:
        """Get all critical issues"""
        return self.get_issues_by_severity(SecuritySeverity.CRITICAL)
