"""
Static Application Security Testing (SAST) Scanner

Performs source code analysis to detect security vulnerabilities and insecure patterns.
"""

import os
import re
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
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .security_reporter import SecurityIssue, Severity, IssueCategory

# Set up logger
import hashlib

from .models import SecurityIssue, SecuritySeverity, SecurityCategory, VulnerabilityFix

logger = logging.getLogger(__name__)


class SASTScanner:
    """
    Static Application Security Testing scanner for Python code.
    
    Detects:
    - Insecure function usage (eval, exec, etc.)
    - SQL injection patterns
    - Command injection vulnerabilities
    - Hardcoded secrets and credentials
    - Unsafe deserialization
    - Cryptographic weaknesses
    - Path traversal vulnerabilities
    """
    
    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.files_scanned = 0
        
        # Dangerous functions to detect
        self.dangerous_functions = {
            "eval": {
                "severity": Severity.CRITICAL,
                "category": IssueCategory.INJECTION,
                "message": "Use of eval() allows arbitrary code execution",
                "recommendation": "Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.",
            },
            "exec": {
                "severity": Severity.CRITICAL,
                "category": IssueCategory.INJECTION,
                "message": "Use of exec() allows arbitrary code execution",
                "recommendation": "Avoid exec(). Refactor code to avoid dynamic code execution.",
            },
            "compile": {
                "severity": Severity.HIGH,
                "category": IssueCategory.INJECTION,
                "message": "Use of compile() can lead to code injection",
                "recommendation": "Avoid compile() with untrusted input. Use safer alternatives.",
            },
            "__import__": {
                "severity": Severity.HIGH,
                "category": IssueCategory.INJECTION,
                "message": "Dynamic imports can be exploited",
                "recommendation": "Use static imports or validate module names against a whitelist.",
            },
        }
        
        # Secret patterns to detect
        self.secret_patterns = [
            (r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]([a-zA-Z0-9_\-]{20,})['\"]", "API Key"),
            (r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]([^'\"]{3,})['\"]", "Password"),
            (r"(?i)(secret[_-]?key|secretkey)\s*[=:]\s*['\"]([a-zA-Z0-9_\-]{20,})['\"]", "Secret Key"),
            (r"(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[=:]\s*['\"]([A-Z0-9]{20})['\"]", "AWS Access Key"),
            (r"(?i)(private[_-]?key)\s*[=:]\s*['\"](.+?)['\"]", "Private Key"),
            (r"(?i)token\s*[=:]\s*['\"]([a-zA-Z0-9_\-\.]{20,})['\"]", "Token"),
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r"execute\s*\(\s*['\"].*?%s.*?['\"]\s*%\s*",
            r"\.format\s*\(.*?\)\s*.*?execute",
            r"f['\"].*?SELECT.*?\{.*?\}.*?['\"]",
            r'["\'].*?SELECT.*?WHERE.*?["\'].*?%\s+',  # String formatting with %
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r"os\.system\s*\(",
            r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
        ]
    
    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a single Python file for security issues"""
        issues = []
        
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
            
            # Check for dangerous functions
            issues.extend(self._check_dangerous_functions(file_path, content, lines))
            
            # Check for hardcoded secrets
            issues.extend(self._check_secrets(file_path, content, lines))
            
            # Check for SQL injection
            issues.extend(self._check_sql_injection(file_path, content, lines))
            
            # Check for command injection
            issues.extend(self._check_command_injection(file_path, content, lines))
            
            # Check for insecure deserialization
            issues.extend(self._check_deserialization(file_path, content, lines))
            
            # Check for weak cryptography
            issues.extend(self._check_cryptography(file_path, content, lines))
            
        except Exception as e:
            # Log error but continue scanning other files
            logger.warning(f"Error scanning file {file_path}: {str(e)}")
        
        return issues
    
    def scan_directory(self, directory_path: str, extensions: List[str] = None) -> List[SecurityIssue]:
        """Recursively scan a directory for security issues"""
        if extensions is None:
            extensions = [".py"]
        
        all_issues = []
        
        for root, _, files in os.walk(directory_path):
            # Skip common directories that don't need scanning
            if any(skip in root for skip in [".git", "__pycache__", "venv", ".venv", "node_modules"]):
                continue
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    issues = self.scan_file(file_path)
                    all_issues.extend(issues)
        
        self.issues = all_issues
        return all_issues
    
    def _check_dangerous_functions(self, file_path: str, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Check for dangerous function usage"""
        issues = []
        
        for func_name, details in self.dangerous_functions.items():
            pattern = rf"\b{func_name}\s*\("
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                
                issue = SecurityIssue(
                    issue_id=f"SAST-{func_name.upper()}-{file_path}-{line_num}",
                    title=f"Dangerous function: {func_name}()",
                    description=details["message"],
                    severity=details["severity"],
                    category=details["category"],
                    file_path=file_path,
                    line_number=line_num,
                    attack_vector=f"An attacker could provide malicious input to {func_name}() leading to arbitrary code execution.",
                    impact="Complete system compromise, data exfiltration, or denial of service.",
                    recommendation=details["recommendation"],
                    cwe_ids=["CWE-94"],  # Code Injection
                    owasp_references=["https://owasp.org/www-community/attacks/Code_Injection"],
                )
                issues.append(issue)
        
        return issues
    
    def _check_secrets(self, file_path: str, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Check for hardcoded secrets and credentials"""
        issues = []
        
        # Skip common example files (but not test files - we want to detect issues in tests)
        if any(x in file_path.lower() for x in [".env.example", "sample", "example_"]):
            return issues
        
        for pattern, secret_type in self.secret_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                
                # Skip if it's clearly a placeholder
                matched_value = match.group(2) if len(match.groups()) > 1 else match.group(1)
                if any(placeholder in matched_value.lower() for placeholder in [
                    "your-", "xxx", "example", "placeholder", "changeme", "***", "..."
                ]):
                    continue
                
                issue = SecurityIssue(
                    issue_id=f"SAST-SECRET-{file_path}-{line_num}",
                    title=f"Hardcoded {secret_type} detected",
                    description=f"A {secret_type.lower()} appears to be hardcoded in the source code.",
                    severity=Severity.CRITICAL,
                    category=IssueCategory.SENSITIVE_DATA,
                    file_path=file_path,
                    line_number=line_num,
                    attack_vector="Hardcoded credentials in source code can be extracted by anyone with repository access.",
                    impact="Unauthorized access to systems, data breaches, or service compromise.",
                    recommendation=(
                        "Remove hardcoded credentials. Use environment variables or a secure secret management system "
                        "(e.g., Azure Key Vault, AWS Secrets Manager, HashiCorp Vault). "
                        "Rotate the compromised credential immediately."
                    ),
                    code_sample="import os\n\n# Use environment variables\napi_key = os.getenv('API_KEY')\nif not api_key:\n    raise ValueError('API_KEY environment variable not set')",
                    cwe_ids=["CWE-798"],  # Use of Hard-coded Credentials
                    owasp_references=["https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password"],
                )
                issues.append(issue)
        
        return issues
    
    def _check_sql_injection(self, file_path: str, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Check for SQL injection vulnerabilities"""
        issues = []
        
        for pattern in self.sql_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                
                issue = SecurityIssue(
                    issue_id=f"SAST-SQLI-{file_path}-{line_num}",
                    title="Potential SQL Injection vulnerability",
                    description="SQL query appears to be constructed using string formatting, which is vulnerable to SQL injection.",
                    severity=Severity.CRITICAL,
                    category=IssueCategory.INJECTION,
                    file_path=file_path,
                    line_number=line_num,
                    attack_vector="An attacker could inject malicious SQL code through user input.",
                    impact="Unauthorized data access, data modification, or complete database compromise.",
                    recommendation="Use parameterized queries or an ORM. Never construct SQL queries with string formatting.",
                    code_sample="# Use parameterized queries\ncursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n\n# Or use an ORM\nuser = User.objects.get(id=user_id)",
                    cwe_ids=["CWE-89"],  # SQL Injection
                    owasp_references=["https://owasp.org/www-community/attacks/SQL_Injection"],
                )
                issues.append(issue)
        
        return issues
    
    def _check_command_injection(self, file_path: str, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Check for command injection vulnerabilities"""
        issues = []
        
        for pattern in self.command_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                
                issue = SecurityIssue(
                    issue_id=f"SAST-CMDI-{file_path}-{line_num}",
                    title="Potential Command Injection vulnerability",
                    description="Command execution with shell=True or os.system() is vulnerable to command injection.",
                    severity=Severity.CRITICAL,
                    category=IssueCategory.INJECTION,
                    file_path=file_path,
                    line_number=line_num,
                    attack_vector="An attacker could inject malicious commands through user input.",
                    impact="Arbitrary command execution, system compromise, or data exfiltration.",
                    recommendation="Use subprocess with shell=False and pass arguments as a list. Validate and sanitize all inputs.",
                    code_sample="# Safe command execution\nimport subprocess\nresult = subprocess.run(['ls', '-l', directory], capture_output=True, check=True)",
                    cwe_ids=["CWE-78"],  # OS Command Injection
                    owasp_references=["https://owasp.org/www-community/attacks/Command_Injection"],
                )
                issues.append(issue)
        
        return issues
    
    def _check_deserialization(self, file_path: str, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Check for insecure deserialization"""
        issues = []
        
        # Check for pickle usage
        if re.search(r"pickle\.(loads|load)\s*\(", content):
            matches = re.finditer(r"pickle\.(loads|load)\s*\(", content)
            for match in matches:
                line_num = content[:match.start()].count("\n") + 1
                
                issue = SecurityIssue(
                    issue_id=f"SAST-DESER-{file_path}-{line_num}",
                    title="Insecure deserialization using pickle",
                    description="Using pickle to deserialize untrusted data can lead to arbitrary code execution.",
                    severity=Severity.HIGH,
                    category=IssueCategory.INSECURE_DESERIALIZATION,
                    file_path=file_path,
                    line_number=line_num,
                    attack_vector="An attacker can craft malicious pickle data to execute arbitrary code during deserialization.",
                    impact="Remote code execution, complete system compromise.",
                    recommendation="Use safe serialization formats like JSON. If pickle is necessary, only deserialize from trusted sources and implement integrity checks.",
                    code_sample="import json\n\n# Use JSON instead of pickle\ndata = json.loads(json_string)",
                    cwe_ids=["CWE-502"],  # Deserialization of Untrusted Data
                    owasp_references=["https://owasp.org/www-community/vulnerabilities/Deserialization_of_untrusted_data"],
                )
                issues.append(issue)
        
        return issues
    
    def _check_cryptography(self, file_path: str, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Check for weak cryptographic practices"""
        issues = []
        
        # Check for MD5/SHA1 usage
        weak_hashes = ["md5", "sha1"]
        for hash_type in weak_hashes:
            pattern = rf"hashlib\.{hash_type}\s*\("
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                
                issue = SecurityIssue(
                    issue_id=f"SAST-CRYPTO-{hash_type.upper()}-{file_path}-{line_num}",
                    title=f"Weak cryptographic hash: {hash_type.upper()}",
                    description=f"{hash_type.upper()} is cryptographically broken and should not be used for security purposes.",
                    severity=Severity.MEDIUM,
                    category=IssueCategory.CRYPTOGRAPHY,
                    file_path=file_path,
                    line_number=line_num,
                    attack_vector="Weak hashing algorithms can be attacked with collision or pre-image attacks.",
                    impact="Password cracking, data integrity compromise, or authentication bypass.",
                    recommendation="Use SHA-256 or SHA-3 for hashing. For password hashing, use bcrypt, scrypt, or Argon2.",
                    code_sample="import hashlib\n\n# Use stronger hashing\nhash_value = hashlib.sha256(data).hexdigest()\n\n# For passwords, use bcrypt\nimport bcrypt\nhashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())",
                    cwe_ids=["CWE-327"],  # Use of Broken Cryptographic Algorithm
                    owasp_references=["https://owasp.org/www-project-top-ten/2017/A3_2017-Sensitive_Data_Exposure"],
                )
                issues.append(issue)
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of scan results"""
        return {
            "files_scanned": self.files_scanned,
            "total_issues": len(self.issues),
            "by_severity": {
                "critical": len([i for i in self.issues if i.severity == Severity.CRITICAL]),
                "high": len([i for i in self.issues if i.severity == Severity.HIGH]),
                "medium": len([i for i in self.issues if i.severity == Severity.MEDIUM]),
                "low": len([i for i in self.issues if i.severity == Severity.LOW]),
            },
        }

            self.files_scanned += 1

            # Pattern-based scanning
            self._scan_patterns(file_path, content, lines)

            # AST-based scanning for Python files
            if file_path.endswith(".py"):
                self._scan_python_ast(file_path, content)

            return self.issues

        except Exception as e:
            # Log error but continue scanning other files
            logger.warning(f"Error scanning file {file_path}: {e}")
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
