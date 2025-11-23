"""
Security Knowledge Base - Integration with security databases and best practices
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CVEInfo:
    """Information about a CVE (Common Vulnerabilities and Exposures)"""

    cve_id: str
    description: str
    severity: str
    cvss_score: float
    published_date: str
    affected_packages: List[str]
    references: List[str]


@dataclass
class OWASPMapping:
    """OWASP Top 10 mapping"""

    category: str
    title: str
    description: str
    examples: List[str]
    mitigations: List[str]


class SecurityKnowledgeBase:
    """
    Knowledge base for security best practices and vulnerability information.
    
    Provides:
    - OWASP Top 10 guidelines
    - SANS Top 25 CWE mappings
    - MITRE ATT&CK knowledge
    - CVE lookups (simplified - in production would integrate with NVD API)
    """
    
    def __init__(self):
        self.owasp_top10_2021 = self._initialize_owasp_top10()
        self.sans_top25 = self._initialize_sans_top25()
        self.mitre_attack_patterns = self._initialize_mitre_patterns()
    
    def _initialize_owasp_top10(self) -> Dict[str, OWASPMapping]:
        """Initialize OWASP Top 10 2021 mappings"""
        return {
            "A01:2021": OWASPMapping(
                category="A01:2021",
                title="Broken Access Control",
                description="Restrictions on what authenticated users can do are not properly enforced.",
                examples=[
                    "Accessing API with another user's identifier",
                    "Bypassing access control checks by modifying URL or application state",
                    "Elevation of privilege",
                ],
                mitigations=[
                    "Implement least privilege access control",
                    "Deny by default",
                    "Log access control failures and alert admins",
                ],
            ),
            "A02:2021": OWASPMapping(
                category="A02:2021",
                title="Cryptographic Failures",
                description="Failures related to cryptography leading to exposure of sensitive data.",
                examples=[
                    "Using weak or broken cryptographic algorithms",
                    "Transmitting sensitive data in clear text",
                    "Using insecure random number generators",
                ],
                mitigations=[
                    "Use strong, up-to-date cryptographic algorithms",
                    "Encrypt data in transit and at rest",
                    "Don't cache sensitive data",
                ],
            ),
            "A03:2021": OWASPMapping(
                category="A03:2021",
                title="Injection",
                description="User-supplied data is not validated, filtered, or sanitized.",
                examples=[
                    "SQL, NoSQL, OS command injection",
                    "LDAP injection",
                    "Expression Language (EL) injection",
                ],
                mitigations=[
                    "Use parameterized queries",
                    "Validate and sanitize all inputs",
                    "Use ORM/framework escaping",
                ],
            ),
            "A04:2021": OWASPMapping(
                category="A04:2021",
                title="Insecure Design",
                description="Missing or ineffective control design.",
                examples=[
                    "Lack of security requirements",
                    "Missing threat modeling",
                    "Insecure design patterns",
                ],
                mitigations=[
                    "Establish secure development lifecycle",
                    "Use threat modeling",
                    "Write unit and integration tests for security flows",
                ],
            ),
            "A05:2021": OWASPMapping(
                category="A05:2021",
                title="Security Misconfiguration",
                description="Missing security hardening or improperly configured permissions.",
                examples=[
                    "Default accounts with default passwords",
                    "Unnecessary features enabled",
                    "Error messages revealing sensitive information",
                ],
                mitigations=[
                    "Implement hardening procedures",
                    "Remove unnecessary features",
                    "Review configurations regularly",
                ],
            ),
            "A06:2021": OWASPMapping(
                category="A06:2021",
                title="Vulnerable and Outdated Components",
                description="Using components with known vulnerabilities.",
                examples=[
                    "Outdated libraries with CVEs",
                    "Unmaintained dependencies",
                    "Not scanning for vulnerabilities",
                ],
                mitigations=[
                    "Remove unused dependencies",
                    "Continuously inventory versions",
                    "Monitor security bulletins",
                ],
            ),
            "A07:2021": OWASPMapping(
                category="A07:2021",
                title="Identification and Authentication Failures",
                description="Confirmation of user identity, authentication, and session management failures.",
                examples=[
                    "Weak password policies",
                    "Credential stuffing attacks",
                    "Session fixation",
                ],
                mitigations=[
                    "Implement multi-factor authentication",
                    "Use secure session management",
                    "Implement password complexity requirements",
                ],
            ),
            "A08:2021": OWASPMapping(
                category="A08:2021",
                title="Software and Data Integrity Failures",
                description="Code and infrastructure that doesn't protect against integrity violations.",
                examples=[
                    "Using untrusted CDNs",
                    "Auto-update without integrity verification",
                    "Insecure CI/CD pipelines",
                ],
                mitigations=[
                    "Use digital signatures",
                    "Verify integrity of dependencies",
                    "Implement secure CI/CD pipelines",
                ],
            ),
            "A09:2021": OWASPMapping(
                category="A09:2021",
                title="Security Logging and Monitoring Failures",
                description="Insufficient logging and monitoring allowing breaches to go undetected.",
                examples=[
                    "No logging of security events",
                    "Logs not monitored",
                    "Inadequate alerting",
                ],
                mitigations=[
                    "Log all authentication and access control failures",
                    "Ensure logs can be monitored",
                    "Establish incident response plan",
                ],
            ),
            "A10:2021": OWASPMapping(
                category="A10:2021",
                title="Server-Side Request Forgery (SSRF)",
                description="Fetching remote resources without validating user-supplied URL.",
                examples=[
                    "Reading internal resources",
                    "Port scanning internal network",
                    "Accessing cloud metadata services",
                ],
                mitigations=[
                    "Sanitize and validate all user input",
                    "Use allowlists for URLs",
                    "Disable HTTP redirections",
                ],
            ),
        }
    
    def _initialize_sans_top25(self) -> Dict[str, Dict[str, Any]]:
        """Initialize SANS Top 25 CWE mappings (abbreviated)"""
        return {
            "CWE-89": {
                "name": "SQL Injection",
                "rank": 1,
                "score": 6.1,
                "description": "Improper Neutralization of Special Elements used in an SQL Command",
            },
            "CWE-78": {
                "name": "OS Command Injection",
                "rank": 2,
                "score": 5.9,
                "description": "Improper Neutralization of Special Elements used in an OS Command",
            },
            "CWE-79": {
                "name": "Cross-site Scripting",
                "rank": 3,
                "score": 5.5,
                "description": "Improper Neutralization of Input During Web Page Generation",
            },
            "CWE-787": {
                "name": "Out-of-bounds Write",
                "rank": 4,
                "score": 5.4,
                "description": "Writing data past the end or before the beginning of the intended buffer",
            },
            "CWE-20": {
                "name": "Improper Input Validation",
                "rank": 5,
                "score": 5.3,
                "description": "The product receives input but does not validate it properly",
            },
        }
    
    def _initialize_mitre_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize MITRE ATT&CK patterns (abbreviated)"""
        return {
            "T1190": {
                "name": "Exploit Public-Facing Application",
                "tactic": "Initial Access",
                "description": "Using software vulnerabilities in Internet-facing systems",
            },
            "T1059": {
                "name": "Command and Scripting Interpreter",
                "tactic": "Execution",
                "description": "Abusing command and script interpreters to execute commands",
            },
            "T1078": {
                "name": "Valid Accounts",
                "tactic": "Defense Evasion, Persistence, Privilege Escalation, Initial Access",
                "description": "Using legitimate credentials to gain access",
            },
        }
    
    def get_owasp_guidance(self, category: str) -> Optional[OWASPMapping]:
        """Get OWASP Top 10 guidance for a category"""
        return self.owasp_top10_2021.get(category)
    
    def get_cwe_info(self, cwe_id: str) -> Optional[Dict[str, Any]]:
        """Get CWE information from SANS Top 25"""
        return self.sans_top25.get(cwe_id)
    
    def get_mitre_attack_info(self, technique_id: str) -> Optional[Dict[str, Any]]:
        """Get MITRE ATT&CK technique information"""
        return self.mitre_attack_patterns.get(technique_id)
    
    def get_all_owasp_categories(self) -> List[str]:
        """Get all OWASP Top 10 categories"""
        return list(self.owasp_top10_2021.keys())
    
    def search_owasp_by_keyword(self, keyword: str) -> List[OWASPMapping]:
        """Search OWASP Top 10 by keyword"""
        results = []
        keyword_lower = keyword.lower()
        
        for mapping in self.owasp_top10_2021.values():
            if (
                keyword_lower in mapping.title.lower()
                or keyword_lower in mapping.description.lower()
            ):
                results.append(mapping)
        
        return results
