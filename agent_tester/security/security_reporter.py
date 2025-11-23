"""
Security Reporter

Generates comprehensive security reports in various formats
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .models import SecurityReport, SecurityIssue, SecuritySeverity, SecurityChangeLog


class SecurityReporter:
    """Generates security reports in various formats"""

    def __init__(self):
        self.reports: List[SecurityReport] = []

    def generate_report(
        self,
        report: SecurityReport,
        output_format: str = "json",
        output_path: str = "security_report",
    ) -> str:
        """
        Generate security report in specified format

        Args:
            report: SecurityReport object
            output_format: 'json', 'html', 'markdown', or 'text'
            output_path: Base path for output file (extension added automatically)

        Returns:
            Path to generated report file
        """
        # Calculate summary before generating report
        report.calculate_summary()

        if output_format == "json":
            return self._generate_json_report(report, output_path)
        elif output_format == "html":
            return self._generate_html_report(report, output_path)
        elif output_format == "markdown":
            return self._generate_markdown_report(report, output_path)
        elif output_format == "text":
            return self._generate_text_report(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_json_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate JSON report"""
        file_path = f"{output_path}.json"

        with open(file_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

        return file_path

    def _generate_text_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate plain text report"""
        file_path = f"{output_path}.txt"

        lines = []
        lines.append("=" * 80)
        lines.append("SECURITY SCAN REPORT")
        lines.append("=" * 80)
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Scan Type: {report.scan_type}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Repository: {report.repository_path}")
        lines.append(f"Files Scanned: {report.files_scanned}")
        lines.append(f"Dependencies Checked: {report.dependencies_checked}")
        lines.append(f"Scan Duration: {report.scan_duration_seconds:.2f}s")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Issues: {report.summary.get('total_issues', 0)}")
        lines.append("")
        lines.append("By Severity:")
        for severity, count in report.summary.get("by_severity", {}).items():
            lines.append(f"  {severity.upper()}: {count}")
        lines.append(f"Exploitable Issues: {report.summary.get('exploitable_count', 0)}")
        lines.append(f"Files with Issues: {report.summary.get('files_with_issues', 0)}")
        lines.append("")

        # Issues by severity
        for severity in [
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
        ]:
            issues = report.get_by_severity(severity)
            if issues:
                lines.append(f"\n{severity.value.upper()} SEVERITY ISSUES ({len(issues)})")
                lines.append("-" * 80)
                for i, issue in enumerate(issues, 1):
                    lines.append(f"\n{i}. {issue.title}")
                    lines.append(f"   ID: {issue.issue_id}")
                    lines.append(f"   Category: {issue.category.value}")
                    if issue.file_path:
                        location = f"{issue.file_path}"
                        if issue.line_number:
                            location += f":{issue.line_number}"
                        lines.append(f"   Location: {location}")
                    lines.append(f"   Description: {issue.description}")
                    if issue.code_snippet:
                        lines.append(f"   Code: {issue.code_snippet}")
                    if issue.suggested_fix:
                        lines.append(f"   Fix: {issue.suggested_fix.description}")
                    if issue.exploitable:
                        lines.append("   ⚠️  EXPLOITABLE")

        # Recommendations
        if report.recommendations:
            lines.append("\n\nRECOMMENDATIONS")
            lines.append("-" * 80)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")

        lines.append("\n" + "=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)

        with open(file_path, "w") as f:
            f.write("\n".join(lines))

        return file_path

    def _generate_markdown_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate Markdown report"""
        file_path = f"{output_path}.md"

        lines = []
        lines.append("# Security Scan Report")
        lines.append("")
        lines.append(f"**Report ID:** {report.report_id}")
        lines.append(f"**Scan Type:** {report.scan_type}")
        lines.append(f"**Timestamp:** {report.timestamp}")
        lines.append(f"**Repository:** {report.repository_path}")
        lines.append(f"**Files Scanned:** {report.files_scanned}")
        lines.append(f"**Dependencies Checked:** {report.dependencies_checked}")
        lines.append(f"**Scan Duration:** {report.scan_duration_seconds:.2f}s")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Issues:** {report.summary.get('total_issues', 0)}")
        lines.append(f"- **Exploitable Issues:** {report.summary.get('exploitable_count', 0)}")
        lines.append(f"- **Files with Issues:** {report.summary.get('files_with_issues', 0)}")
        lines.append("")

        # Severity breakdown
        lines.append("### Issues by Severity")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for severity, count in report.summary.get("by_severity", {}).items():
            emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
                "info": "⚪",
            }.get(severity, "")
            lines.append(f"| {emoji} {severity.upper()} | {count} |")
        lines.append("")

        # Detailed issues
        for severity in [
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
        ]:
            issues = report.get_by_severity(severity)
            if issues:
                emoji = {
                    SecuritySeverity.CRITICAL: "🔴",
                    SecuritySeverity.HIGH: "🟠",
                    SecuritySeverity.MEDIUM: "🟡",
                    SecuritySeverity.LOW: "🔵",
                }.get(severity, "")

                lines.append(f"## {emoji} {severity.value.upper()} Severity Issues")
                lines.append("")
                for issue in issues:
                    lines.append(f"### {issue.title}")
                    if issue.exploitable:
                        lines.append("⚠️ **EXPLOITABLE**")
                    lines.append("")
                    lines.append(f"- **ID:** `{issue.issue_id}`")
                    lines.append(f"- **Category:** {issue.category.value}")
                    if issue.file_path:
                        location = f"`{issue.file_path}`"
                        if issue.line_number:
                            location += f" (line {issue.line_number})"
                        lines.append(f"- **Location:** {location}")
                    if issue.cve_id:
                        lines.append(f"- **CVE:** [{issue.cve_id}](https://nvd.nist.gov/vuln/detail/{issue.cve_id})")
                    lines.append("")
                    lines.append(f"**Description:** {issue.description}")
                    lines.append("")
                    if issue.code_snippet:
                        lines.append("**Code:**")
                        lines.append("```")
                        lines.append(issue.code_snippet)
                        lines.append("```")
                        lines.append("")
                    if issue.suggested_fix:
                        lines.append("**Suggested Fix:**")
                        lines.append(issue.suggested_fix.description)
                        lines.append("")
                    lines.append("---")
                    lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        with open(file_path, "w") as f:
            f.write("\n".join(lines))

        return file_path

    def _generate_html_report(self, report: SecurityReport, output_path: str) -> str:
        """Generate HTML report"""
        file_path = f"{output_path}.html"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Scan Report - {report.report_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .severity-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .critical {{ background: #dc3545; color: white; }}
        .high {{ background: #fd7e14; color: white; }}
        .medium {{ background: #ffc107; color: black; }}
        .low {{ background: #17a2b8; color: white; }}
        .info {{ background: #6c757d; color: white; }}
        .issue {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .issue.critical {{ border-left-color: #dc3545; }}
        .issue.high {{ border-left-color: #fd7e14; }}
        .issue.medium {{ border-left-color: #ffc107; }}
        .issue.low {{ border-left-color: #17a2b8; }}
        .code {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
        .fix {{
            background: #d4edda;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #28a745;
            margin-top: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Security Scan Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Scan Type:</strong> {report.scan_type}</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
        <p><strong>Repository:</strong> {report.repository_path}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{report.summary.get('total_issues', 0)}</div>
            <div>Total Issues</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report.files_scanned}</div>
            <div>Files Scanned</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report.dependencies_checked}</div>
            <div>Dependencies Checked</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report.scan_duration_seconds:.1f}s</div>
            <div>Scan Duration</div>
        </div>
    </div>

    <div class="summary">
        <h2>Severity Breakdown</h2>
        <p>
            <span class="severity-badge critical">CRITICAL: {report.summary.get('by_severity', {}).get('critical', 0)}</span>
            <span class="severity-badge high">HIGH: {report.summary.get('by_severity', {}).get('high', 0)}</span>
            <span class="severity-badge medium">MEDIUM: {report.summary.get('by_severity', {}).get('medium', 0)}</span>
            <span class="severity-badge low">LOW: {report.summary.get('by_severity', {}).get('low', 0)}</span>
            <span class="severity-badge info">INFO: {report.summary.get('by_severity', {}).get('info', 0)}</span>
        </p>
    </div>
"""

        # Add issues
        for severity in [
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
        ]:
            issues = report.get_by_severity(severity)
            if issues:
                html += f'<h2>{severity.value.upper()} Severity Issues ({len(issues)})</h2>\n'
                for issue in issues:
                    html += f'<div class="issue {severity.value}">\n'
                    html += f'<h3>{issue.title}</h3>\n'
                    if issue.exploitable:
                        html += '<span class="severity-badge critical">⚠️ EXPLOITABLE</span>\n'
                    html += f'<p><strong>ID:</strong> {issue.issue_id}</p>\n'
                    html += f'<p><strong>Category:</strong> {issue.category.value}</p>\n'
                    if issue.file_path:
                        location = issue.file_path
                        if issue.line_number:
                            location += f":{issue.line_number}"
                        html += f'<p><strong>Location:</strong> {location}</p>\n'
                    html += f'<p>{issue.description}</p>\n'
                    if issue.code_snippet:
                        html += f'<div class="code">{issue.code_snippet}</div>\n'
                    if issue.suggested_fix:
                        html += f'<div class="fix"><strong>Fix:</strong> {issue.suggested_fix.description}</div>\n'
                    html += '</div>\n'

        html += """
</body>
</html>
"""

        with open(file_path, "w") as f:
            f.write(html)

        return file_path

    def create_changelog_entry(
        self, current_report: SecurityReport, previous_report: Optional[SecurityReport] = None
    ) -> SecurityChangeLog:
        """Create a changelog entry comparing two reports"""
        changelog = SecurityChangeLog()

        if previous_report:
            current_issues = {issue.issue_id for issue in current_report.issues}
            previous_issues = {issue.issue_id for issue in previous_report.issues}

            new_issues = current_issues - previous_issues
            fixed_issues = previous_issues - current_issues

            changelog.issues_detected = len(new_issues)
            changelog.issues_fixed = len(fixed_issues)
            changelog.new_vulnerabilities = list(new_issues)
            changelog.fixed_vulnerabilities = list(fixed_issues)

        changelog.remaining_critical = len(current_report.get_critical_issues())

        return changelog
