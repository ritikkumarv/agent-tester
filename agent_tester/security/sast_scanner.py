"""
Static Application Security Testing (SAST) Scanner

Performs source code analysis to detect security vulnerabilities and insecure patterns.
"""

import os
import re
import ast
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .security_reporter import SecurityIssue, Severity, IssueCategory

# Set up logger
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
