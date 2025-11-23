"""
Example: Using the Security Scanner

This demonstrates how to use the cybersecurity & secure-code contributor features.
"""

from agent_tester.security import (
    SecurityValidator,
    SecurityKnowledgeBase,
    SASTScanner,
    DependencyScanner,
    ConfigurationScanner,
)


def main():
    """Run security scan on the current repository"""
    
    print("=" * 70)
    print("🔒 Agent Tester - Security Scanner Example")
    print("=" * 70)
    print()
    
    # Initialize security validator
    validator = SecurityValidator()
    
    # Run comprehensive security scan
    print("📊 Running comprehensive security scan...")
    print()
    
    # Scan current repository
    report = validator.validate_repository(
        repository_path=".",
        branch="main",
        report_id="example-scan-001",
    )
    
    # Display summary
    print("=" * 70)
    print("SECURITY SCAN SUMMARY")
    print("=" * 70)
    print()
    
    summary = report.get_summary()
    print(f"Files Scanned:        {summary['files_scanned']}")
    print(f"Dependencies Checked: {summary['dependencies_checked']}")
    print(f"Scan Duration:        {summary['scan_duration']:.2f}s")
    print()
    
    print("Issues by Severity:")
    print(f"  🔴 Critical: {summary['critical']}")
    print(f"  🟠 High:     {summary['high']}")
    print(f"  🟡 Medium:   {summary['medium']}")
    print(f"  🔵 Low:      {summary['low']}")
    print(f"  ℹ️  Info:     {summary['info']}")
    print(f"  ━━━━━━━━━━━━━━━━━━")
    print(f"  📊 Total:    {summary['total_issues']}")
    print()
    
    # Display critical issues
    if report.critical_issues:
        print("=" * 70)
        print("🔴 CRITICAL ISSUES")
        print("=" * 70)
        print()
        
        for issue in report.critical_issues[:5]:  # Show first 5
            print(f"Title: {issue.title}")
            print(f"File:  {issue.file_path}:{issue.line_number or '?'}")
            print(f"Desc:  {issue.description}")
            print(f"Fix:   {issue.recommendation}")
            print("-" * 70)
            print()
    
    # Display high severity issues
    if report.high_issues:
        print("=" * 70)
        print("🟠 HIGH SEVERITY ISSUES")
        print("=" * 70)
        print()
        
        for issue in report.high_issues[:5]:  # Show first 5
            print(f"Title: {issue.title}")
            print(f"File:  {issue.file_path}:{issue.line_number or '?'}")
            print(f"Desc:  {issue.description}")
            print(f"Fix:   {issue.recommendation}")
            print("-" * 70)
            print()
    
    # Export reports
    print("=" * 70)
    print("EXPORTING REPORTS")
    print("=" * 70)
    print()
    
    # Markdown report
    markdown_file = "security_report.md"
    validator.export_report(report, format="markdown", output_file=markdown_file)
    print(f"✅ Markdown report saved: {markdown_file}")
    
    # JSON report
    json_file = "security_report.json"
    validator.export_report(report, format="json", output_file=json_file)
    print(f"✅ JSON report saved: {json_file}")
    print()
    
    # Demonstrate knowledge base
    print("=" * 70)
    print("SECURITY KNOWLEDGE BASE")
    print("=" * 70)
    print()
    
    kb = SecurityKnowledgeBase()
    
    # Show OWASP Top 10
    print("OWASP Top 10 2021 Categories:")
    for category in kb.get_all_owasp_categories():
        mapping = kb.get_owasp_guidance(category)
        print(f"  • {category}: {mapping.title}")
    print()
    
    # Search for injection guidance
    injection_guidance = kb.search_owasp_by_keyword("injection")
    if injection_guidance:
        print("OWASP Guidance for 'Injection':")
        for guidance in injection_guidance:
            print(f"  Category: {guidance.category}")
            print(f"  Title: {guidance.title}")
            print(f"  Mitigations:")
            for mitigation in guidance.mitigations:
                print(f"    - {mitigation}")
    print()
    
    # Final status
    print("=" * 70)
    if report.critical_issues or report.high_issues:
        print("❌ SECURITY SCAN FAILED - Critical/High issues found!")
        print("Please review and fix the issues before deploying.")
    else:
        print("✅ SECURITY SCAN PASSED - No critical/high issues found")
    print("=" * 70)


if __name__ == "__main__":
    main()
