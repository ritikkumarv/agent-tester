# Security Report: security-scan-20251123-040015

**Generated**: 2025-11-23T04:00:15.144669
**Repository**: .
**Branch**: N/A

## Executive Summary

- 🔴 Critical Issues: 17
- 🟠 High Issues: 2
- 🟡 Medium Issues: 1
- 🔵 Low Issues: 0
- ℹ️ Info: 0

**Total Issues**: 20
**Files Scanned**: 31
**Dependencies Checked**: 27
**Scan Duration**: 0.05s

## Critical Severity Issues

### Dangerous function: eval()

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 29

**Description**: Use of eval() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to eval() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: eval()

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 31

**Description**: Use of eval() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to eval() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: eval()

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 98

**Description**: Use of eval() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to eval() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: eval()

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 279

**Description**: Use of eval() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to eval() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: exec()

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 41

**Description**: Use of exec() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to exec() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid exec(). Refactor code to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: exec()

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 43

**Description**: Use of exec() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to exec() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid exec(). Refactor code to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Hardcoded Password detected

- **Category**: sensitive_data
- **File**: ./tests/test_security.py
- **Line**: 54

**Description**: A password appears to be hardcoded in the source code.

**Attack Vector**: Hardcoded credentials in source code can be extracted by anyone with repository access.

**Impact**: Unauthorized access to systems, data breaches, or service compromise.

**Recommendation**: Remove hardcoded credentials. Use environment variables or a secure secret management system (e.g., Azure Key Vault, AWS Secrets Manager, HashiCorp Vault). Rotate the compromised credential immediately.

**Secure Code Example**:
```python
import os

# Use environment variables
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError('API_KEY environment variable not set')
```

**CWEs**: CWE-798
**OWASP References**: https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password

---

### Hardcoded Password detected

- **Category**: sensitive_data
- **File**: ./tests/test_security.py
- **Line**: 99

**Description**: A password appears to be hardcoded in the source code.

**Attack Vector**: Hardcoded credentials in source code can be extracted by anyone with repository access.

**Impact**: Unauthorized access to systems, data breaches, or service compromise.

**Recommendation**: Remove hardcoded credentials. Use environment variables or a secure secret management system (e.g., Azure Key Vault, AWS Secrets Manager, HashiCorp Vault). Rotate the compromised credential immediately.

**Secure Code Example**:
```python
import os

# Use environment variables
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError('API_KEY environment variable not set')
```

**CWEs**: CWE-798
**OWASP References**: https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password

---

### Potential SQL Injection vulnerability

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 65

**Description**: SQL query appears to be constructed using string formatting, which is vulnerable to SQL injection.

**Attack Vector**: An attacker could inject malicious SQL code through user input.

**Impact**: Unauthorized data access, data modification, or complete database compromise.

**Recommendation**: Use parameterized queries or an ORM. Never construct SQL queries with string formatting.

**Secure Code Example**:
```python
# Use parameterized queries
cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))

# Or use an ORM
user = User.objects.get(id=user_id)
```

**CWEs**: CWE-89
**OWASP References**: https://owasp.org/www-community/attacks/SQL_Injection

---

### Potential Command Injection vulnerability

- **Category**: injection
- **File**: ./tests/test_security.py
- **Line**: 76

**Description**: Command execution with shell=True or os.system() is vulnerable to command injection.

**Attack Vector**: An attacker could inject malicious commands through user input.

**Impact**: Arbitrary command execution, system compromise, or data exfiltration.

**Recommendation**: Use subprocess with shell=False and pass arguments as a list. Validate and sanitize all inputs.

**Secure Code Example**:
```python
# Safe command execution
import subprocess
result = subprocess.run(['ls', '-l', directory], capture_output=True, check=True)
```

**CWEs**: CWE-78
**OWASP References**: https://owasp.org/www-community/attacks/Command_Injection

---

### Dangerous function: eval()

- **Category**: injection
- **File**: ./agent_tester/models.py
- **Line**: 39

**Description**: Use of eval() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to eval() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: eval()

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 39

**Description**: Use of eval() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to eval() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: eval()

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 40

**Description**: Use of eval() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to eval() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or refactor to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: exec()

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 45

**Description**: Use of exec() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to exec() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid exec(). Refactor code to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: exec()

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 46

**Description**: Use of exec() allows arbitrary code execution

**Attack Vector**: An attacker could provide malicious input to exec() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid exec(). Refactor code to avoid dynamic code execution.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Potential SQL Injection vulnerability

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 77

**Description**: SQL query appears to be constructed using string formatting, which is vulnerable to SQL injection.

**Attack Vector**: An attacker could inject malicious SQL code through user input.

**Impact**: Unauthorized data access, data modification, or complete database compromise.

**Recommendation**: Use parameterized queries or an ORM. Never construct SQL queries with string formatting.

**Secure Code Example**:
```python
# Use parameterized queries
cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))

# Or use an ORM
user = User.objects.get(id=user_id)
```

**CWEs**: CWE-89
**OWASP References**: https://owasp.org/www-community/attacks/SQL_Injection

---

### Potential Command Injection vulnerability

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 249

**Description**: Command execution with shell=True or os.system() is vulnerable to command injection.

**Attack Vector**: An attacker could inject malicious commands through user input.

**Impact**: Arbitrary command execution, system compromise, or data exfiltration.

**Recommendation**: Use subprocess with shell=False and pass arguments as a list. Validate and sanitize all inputs.

**Secure Code Example**:
```python
# Safe command execution
import subprocess
result = subprocess.run(['ls', '-l', directory], capture_output=True, check=True)
```

**CWEs**: CWE-78
**OWASP References**: https://owasp.org/www-community/attacks/Command_Injection

---

## High Severity Issues

### Dangerous function: compile()

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 51

**Description**: Use of compile() can lead to code injection

**Attack Vector**: An attacker could provide malicious input to compile() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid compile() with untrusted input. Use safer alternatives.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

### Dangerous function: compile()

- **Category**: injection
- **File**: ./agent_tester/security/sast_scanner.py
- **Line**: 52

**Description**: Use of compile() can lead to code injection

**Attack Vector**: An attacker could provide malicious input to compile() leading to arbitrary code execution.

**Impact**: Complete system compromise, data exfiltration, or denial of service.

**Recommendation**: Avoid compile() with untrusted input. Use safer alternatives.

**CWEs**: CWE-94
**OWASP References**: https://owasp.org/www-community/attacks/Code_Injection

---

## Medium Severity Issues

### Weak cryptographic hash: MD5

- **Category**: cryptographic_failure
- **File**: ./tests/test_security.py
- **Line**: 87

**Description**: MD5 is cryptographically broken and should not be used for security purposes.

**Attack Vector**: Weak hashing algorithms can be attacked with collision or pre-image attacks.

**Impact**: Password cracking, data integrity compromise, or authentication bypass.

**Recommendation**: Use SHA-256 or SHA-3 for hashing. For password hashing, use bcrypt, scrypt, or Argon2.

**Secure Code Example**:
```python
import hashlib

# Use stronger hashing
hash_value = hashlib.sha256(data).hexdigest()

# For passwords, use bcrypt
import bcrypt
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
```

**CWEs**: CWE-327
**OWASP References**: https://owasp.org/www-project-top-ten/2017/A3_2017-Sensitive_Data_Exposure

---
