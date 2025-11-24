"""
Command-Line Interface for Agent Tester Framework
===================================================

A Postman-like CLI for testing AI agents.
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from agent_tester import __version__
from agent_tester.models import TaskDefinition
from agent_tester.validators import (
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator,
)

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """
    🤖 Agent Tester - Testing framework for AI Agents
    
    As simple as Postman for APIs, but for testing AI Agents.
    """
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to test configuration file (YAML/JSON)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="test_report.html",
    help="Output path for HTML report",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(config: Optional[str], output: str, verbose: bool):
    """Run tests from a configuration file"""
    console.print(
        Panel.fit(
            "🚀 [bold blue]Agent Tester[/bold blue]\n"
            "Running your AI agent tests...",
            border_style="blue",
        )
    )

    if not config:
        console.print(
            "[red]Error:[/red] No configuration file specified. "
            "Use --config or -c to specify a test configuration."
        )
        console.print("\nExample: agent-tester run -c tests.yaml")
        sys.exit(1)

    # Load configuration
    config_path = Path(config)
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path) as f:
            test_config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path) as f:
            test_config = json.load(f)
    else:
        console.print("[red]Error:[/red] Unsupported file format. Use YAML or JSON.")
        sys.exit(1)

    console.print(f"📋 Loaded configuration from: [cyan]{config}[/cyan]")
    
    # Import suite and adapters
    from agent_tester import AgentTestSuite
    from agent_tester.adapters.openai_adapter import OpenAIAdapter
    
    # Parse test configuration
    test_name = test_config.get("name", "Agent Tests")
    tests = test_config.get("tests", [])
    
    if not tests:
        console.print("[yellow]⚠ No tests found in configuration[/yellow]")
        sys.exit(1)
    
    console.print(f"\n[bold]Test Suite:[/bold] {test_name}")
    console.print(f"[bold]Total Tests:[/bold] {len(tests)}\n")
    
    # Create task definitions from config
    task_definitions = []
    for test in tests:
        task = TaskDefinition(
            task_id=test.get("task_id", f"task_{len(task_definitions)}"),
            goal=test.get("goal", ""),
            constraints=test.get("constraints", []),
            expected_output_schema=test.get("expected_output_schema", {}),
            timeout_seconds=test.get("timeout_seconds", 300)
        )
        task_definitions.append(task)
    
    # Create adapter (try OpenAI first, fallback to mock)
    try:
        import os
        if os.getenv("OPENAI_API_KEY"):
            adapter = OpenAIAdapter()
            console.print("✅ Using OpenAI adapter\n")
        else:
            console.print("[yellow]⚠ OPENAI_API_KEY not set, using mock adapter[/yellow]\n")
            # Use a simple mock adapter
            from agent_tester.models import Trajectory, AgentMemory
            
            class MockAdapter:
                def __init__(self):
                    self.memory = AgentMemory(memory_id="mock_memory", max_size=100)
                    self.agent_id = "mock_agent"
                
                def execute_task(self, task):
                    import time
                    trajectory = Trajectory(trajectory_id=f"mock_{task.task_id}", task_id=task.task_id)
                    trajectory.complete()
                    return {
                        "output": {"status": "success", "result": f"Mock completion: {task.goal}"},
                        "execution_time": 0.5,
                        "trajectory": trajectory
                    }
            
            adapter = MockAdapter()
    except Exception as e:
        console.print(f"[red]Error creating adapter:[/red] {e}")
        sys.exit(1)
    
    # Run tests with progress bar
    suite = AgentTestSuite()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task(
            f"[cyan]Running {len(task_definitions)} tests...", total=len(task_definitions)
        )
        
        results = suite.run_all_tests(adapter, task_definitions)
        progress.update(task_progress, advance=len(task_definitions))
    
    # Display results
    console.print("\n" + "=" * 60)
    console.print("[bold]Test Results[/bold]")
    console.print("=" * 60 + "\n")
    
    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test ID", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Time (s)", justify="right")
    
    for result in results["test_results"]:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        status_style = "green" if result["passed"] else "red"
        table.add_row(
            result["task_id"],
            f"[{status_style}]{status}[/{status_style}]",
            f"{result['execution_time']:.2f}"
        )
    
    console.print(table)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total: {results['total_tests']}")
    console.print(f"  [green]Passed: {results['passed']}[/green]")
    console.print(f"  [red]Failed: {results['failed']}[/red]")
    console.print(f"  Pass Rate: {results['summary']['pass_rate']:.1f}%")
    
    # Generate HTML report
    suite.generate_html_report(results, output)
    console.print(f"\n📄 HTML report generated: [cyan]{output}[/cyan]")
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


@cli.command()
def init():
    """Initialize a new test configuration file"""
    console.print(
        Panel.fit(
            "📝 [bold green]Initialize Test Configuration[/bold green]\n"
            "Creating a sample test configuration...",
            border_style="green",
        )
    )

    sample_config = {
        "name": "My Agent Tests",
        "description": "Sample test suite for my AI agent",
        "tests": [
            {
                "task_id": "sample_task_1",
                "goal": "Answer a simple question",
                "constraints": [
                    {"name": "response_length", "type": "value_in_range", "min_value": 10, "max_value": 500}
                ],
                "expected_output_schema": {"required": ["answer"]},
                "timeout_seconds": 30,
            }
        ],
        "validators": {
            "task": {"strict_mode": False},
            "trajectory": {"max_actions": 20, "allow_backtracking": True},
            "memory": {"min_retention_score": 0.7},
        },
    }

    output_file = "agent_tests.yaml"
    with open(output_file, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)

    console.print(f"✅ Created sample configuration: [green]{output_file}[/green]")
    console.print("\nEdit this file to customize your tests, then run:")
    console.print(f"  [cyan]agent-tester run -c {output_file}[/cyan]")


@cli.command()
@click.argument("task_id")
@click.option("--goal", "-g", required=True, help="Task goal/objective")
@click.option("--timeout", "-t", default=300, help="Timeout in seconds")
def validate(task_id: str, goal: str, timeout: int):
    """Validate a single task (quick test)"""
    console.print(
        Panel.fit(
            f"🔍 [bold cyan]Validating Task[/bold cyan]\n"
            f"Task ID: {task_id}\n"
            f"Goal: {goal}",
            border_style="cyan",
        )
    )

    # Create task definition
    task = TaskDefinition(
        task_id=task_id,
        goal=goal,
        timeout_seconds=timeout,
    )

    # Try to create adapter
    try:
        import os
        from agent_tester.adapters.openai_adapter import OpenAIAdapter
        from agent_tester.validators.task_validator import TaskValidator
        
        if os.getenv("OPENAI_API_KEY"):
            adapter = OpenAIAdapter()
            console.print("✅ Using OpenAI adapter\n")
        else:
            console.print("[yellow]⚠ OPENAI_API_KEY not set, using mock adapter[/yellow]\n")
            # Simple mock adapter
            from agent_tester.models import Trajectory, AgentMemory
            import time as time_module
            
            class MockAdapter:
                def __init__(self):
                    self.memory = AgentMemory(memory_id="mock_memory", max_size=100)
                
                def execute_task(self, task):
                    trajectory = Trajectory(trajectory_id=f"mock_{task.task_id}", task_id=task.task_id)
                    trajectory.complete()
                    return {
                        "output": {"status": "success", "result": f"Mock completion: {task.goal}"},
                        "execution_time": 0.5,
                        "trajectory": trajectory
                    }
            
            adapter = MockAdapter()
        
        # Execute task
        with console.status("[bold green]Executing task..."):
            result = adapter.execute_task(task)
        
        # Validate result
        validator = TaskValidator()
        validation = validator.validate(
            result["output"],
            task,
            result["execution_time"]
        )
        
        # Display results
        console.print("\n" + "=" * 60)
        console.print("[bold]Validation Results[/bold]")
        console.print("=" * 60 + "\n")
        
        status = "✅ PASSED" if validation.passed else "❌ FAILED"
        status_color = "green" if validation.passed else "red"
        
        console.print(f"Status: [{status_color}]{status}[/{status_color}]")
        console.print(f"Goal Achieved: {'✅' if validation.goal_achieved else '❌'}")
        console.print(f"Execution Time: {result['execution_time']:.3f}s")
        console.print(f"\nOutput: {result['output']}")
        
        if validation.error_message:
            console.print(f"\n[red]Error:[/red] {validation.error_message}")
        
        sys.exit(0 if validation.passed else 1)
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception()
        sys.exit(1)


@cli.command()
def examples():
    """Show usage examples"""
    console.print(
        Panel.fit(
            "[bold magenta]Agent Tester - Usage Examples[/bold magenta]",
            border_style="magenta",
        )
    )

    examples_text = """
    [bold]1. Initialize a new test configuration:[/bold]
       agent-tester init

    [bold]2. Run tests from configuration:[/bold]
       agent-tester run -c my_tests.yaml

    [bold]3. Run tests with HTML report:[/bold]
       agent-tester run -c my_tests.yaml -o report.html

    [bold]4. Quick task validation:[/bold]
       agent-tester validate my_task --goal "Summarize this text"

    [bold]5. Python API usage:[/bold]
       [cyan]from agent_tester import TaskDefinition, TaskValidator
       
       task = TaskDefinition(
           task_id="test",
           goal="Do something"
       )
       
       validator = TaskValidator()
       result = validator.validate(output, task, time)[/cyan]
    """

    console.print(examples_text)


@cli.group()
def security():
    """🛡️  Security scanning and vulnerability detection commands"""
    pass


@security.command(name="scan")
@click.option(
    "--type",
    "-t",
    type=click.Choice(["full", "sast", "dependency", "secret", "config"]),
    default="full",
    help="Type of security scan to run",
)
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=".",
    help="Path to repository/directory to scan",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "html", "markdown", "text"]),
    default="html",
    help="Output report format",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="security_report",
    help="Output file path (without extension)",
)
def security_scan(type: str, path: str, format: str, output: str):
    """Run security scans on the codebase"""
    from agent_tester.security import SecurityOrchestrator

    console.print(
        Panel.fit(
            "🛡️  [bold blue]Security Scanner[/bold blue]\n"
            f"Running {type} security scan...",
            border_style="blue",
        )
    )

    orchestrator = SecurityOrchestrator(repository_path=path)

    # Run appropriate scan
    with console.status(f"[bold green]Scanning for vulnerabilities..."):
        if type == "full":
            report = orchestrator.run_full_scan()
        elif type == "sast":
            report = orchestrator.run_sast_scan()
        elif type == "dependency":
            report = orchestrator.run_dependency_scan()
        elif type == "secret":
            report = orchestrator.run_secret_scan()
        elif type == "config":
            report = orchestrator.run_config_scan()

    # Display summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Security Scan Results[/bold]")
    console.print("=" * 60 + "\n")

    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Severity", style="cyan")
    table.add_column("Count", justify="right")

    severity_colors = {
        "critical": "red",
        "high": "orange1",
        "medium": "yellow",
        "low": "blue",
        "info": "white",
    }

    for severity, count in report.summary.get("by_severity", {}).items():
        color = severity_colors.get(severity, "white")
        table.add_row(
            f"[{color}]{severity.upper()}[/{color}]",
            f"[{color}]{count}[/{color}]",
        )

    console.print(table)

    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Total Issues: {report.summary.get('total_issues', 0)}")
    console.print(f"  Exploitable: {report.summary.get('exploitable_count', 0)}")
    console.print(f"  Files Scanned: {report.files_scanned}")
    console.print(f"  Dependencies Checked: {report.dependencies_checked}")
    console.print(f"  Scan Duration: {report.scan_duration_seconds:.2f}s")

    # Generate report file
    report_file = orchestrator.generate_report_file(report, format, output)
    console.print(f"\n📄 Report generated: [cyan]{report_file}[/cyan]")

    # Show critical issues
    critical_issues = report.get_critical_issues()
    if critical_issues:
        console.print(
            f"\n[red bold]⚠️  {len(critical_issues)} CRITICAL issues found![/red bold]"
        )
        console.print("[red]Please review and fix immediately.[/red]")
        sys.exit(1)
    else:
        console.print("\n[green]✅ No critical issues found![/green]")
        sys.exit(0)


@security.command(name="report")
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=".",
    help="Path to repository",
)
def security_report(path: str):
    """Generate a quick security summary"""
    from agent_tester.security import SecurityOrchestrator

    console.print("🛡️  Generating security summary...\n")

    orchestrator = SecurityOrchestrator(repository_path=path)

    with console.status("[bold green]Scanning..."):
        report = orchestrator.run_full_scan()

    # Show top issues
    critical = report.get_critical_issues()
    if critical:
        console.print(f"\n[red bold]🔴 Critical Issues ({len(critical)}):[/red bold]")
        for i, issue in enumerate(critical[:5], 1):
            console.print(f"{i}. {issue.title}")
            if issue.file_path:
                console.print(f"   Location: {issue.file_path}:{issue.line_number or ''}")

    # Recommendations
    if report.recommendations:
        console.print(f"\n[bold]📋 Top Recommendations:[/bold]")
        for i, rec in enumerate(report.recommendations[:3], 1):
            console.print(f"{i}. {rec}")


@security.command(name="check-deps")
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=".",
    help="Path to repository",
)
def check_dependencies(path: str):
    """Check dependencies for known vulnerabilities"""
    from agent_tester.security import DependencyScanner

    console.print("📦 Checking dependencies for vulnerabilities...\n")

    scanner = DependencyScanner()

    # Scan requirements
    req_path = Path(path) / "requirements.txt"
    if req_path.exists():
        issues = scanner.scan_requirements(str(req_path))

        if issues:
            console.print(f"[yellow]Found {len(issues)} dependency issues:[/yellow]\n")
            for issue in issues:
                severity_color = {
                    "critical": "red",
                    "high": "orange1",
                    "medium": "yellow",
                    "low": "blue",
                }.get(issue.severity.value, "white")

                console.print(
                    f"[{severity_color}]{issue.severity.value.upper()}[/{severity_color}] - {issue.title}"
                )
                console.print(f"  {issue.description}")
                if issue.suggested_fix:
                    console.print(f"  Fix: {issue.suggested_fix.description}\n")
        else:
            console.print("[green]✅ No dependency vulnerabilities found![/green]")
    else:
        console.print("[yellow]No requirements.txt found[/yellow]")


@cli.command()
def version():
    """Show version information"""
    table = Table(show_header=False, box=None)
    table.add_row("[bold]Agent Tester[/bold]", f"v{__version__}")
    table.add_row("Python", f"{sys.version.split()[0]}")
    
    console.print(
        Panel.fit(
            table,
            title="Version Information",
            border_style="blue",
        )
    )


@cli.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=".",
    help="Path to repository or directory to scan (default: current directory)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for security report",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "json"], case_sensitive=False),
    default="markdown",
    help="Report format (default: markdown)",
)
@click.option(
    "--severity",
    "-s",
    type=click.Choice(["critical", "high", "medium", "low", "all"], case_sensitive=False),
    default="all",
    help="Minimum severity to report (default: all)",
)
def security(path: str, output: Optional[str], format: str, severity: str):
    """
    🔒 Run security scan on repository
    
    Performs comprehensive security analysis including:
    - Static Application Security Testing (SAST)
    - Dependency vulnerability scanning
    - Configuration security checks
    - Secret detection
    """
    from agent_tester.security import SecurityValidator
    
    console.print(Panel.fit(
        "[bold blue]Security Scanner[/bold blue]\n"
        "Cybersecurity & Secure-Code Contributor",
        border_style="blue",
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running security scans...", total=None)
        
        validator = SecurityValidator()
        report = validator.validate_repository(path)
        
        progress.update(task, description="[green]✓[/green] Security scan complete")
    
    # Display summary
    summary = report.get_summary()
    
    summary_table = Table(title="Security Scan Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    
    summary_table.add_row("Files Scanned", str(summary["files_scanned"]))
    summary_table.add_row("Dependencies Checked", str(summary["dependencies_checked"]))
    summary_table.add_row("Scan Duration", f"{summary['scan_duration']:.2f}s")
    summary_table.add_row("", "")  # Separator
    summary_table.add_row("[bold red]Critical Issues[/bold red]", f"[bold red]{summary['critical']}[/bold red]")
    summary_table.add_row("[bold yellow]High Issues[/bold yellow]", f"[bold yellow]{summary['high']}[/bold yellow]")
    summary_table.add_row("Medium Issues", str(summary["medium"]))
    summary_table.add_row("Low Issues", str(summary["low"]))
    summary_table.add_row("Info", str(summary["info"]))
    summary_table.add_row("[bold]Total Issues[/bold]", f"[bold]{summary['total_issues']}[/bold]")
    
    console.print(summary_table)
    
    # Filter by severity if needed
    if severity != "all":
        severity_levels = {
            "critical": ["critical"],
            "high": ["critical", "high"],
            "medium": ["critical", "high", "medium"],
            "low": ["critical", "high", "medium", "low"],
        }
        console.print(f"\n[dim]Filtering to show {severity.upper()} and above...[/dim]")
    
    # Display issues if any
    if summary["total_issues"] > 0:
        console.print("\n[bold yellow]⚠️  Security issues found![/bold yellow]")
        
        # Show critical issues
        if report.critical_issues:
            console.print("\n[bold red]🔴 CRITICAL ISSUES:[/bold red]")
            for issue in report.critical_issues[:5]:  # Show first 5
                console.print(f"  • {issue.title} ({issue.file_path}:{issue.line_number or '?'})")
        
        # Show high issues
        if report.high_issues:
            console.print("\n[bold yellow]🟠 HIGH SEVERITY ISSUES:[/bold yellow]")
            for issue in report.high_issues[:5]:  # Show first 5
                console.print(f"  • {issue.title} ({issue.file_path}:{issue.line_number or '?'})")
    else:
        console.print("\n[bold green]✅ No security issues found![/bold green]")
    
    # Export report
    if output:
        report_content = validator.export_report(report, format=format, output_file=output)
        console.print(f"\n[green]✓[/green] Report saved to: {output}")
    else:
        # Generate default filename
        default_output = f"security_report_{report.report_id}.{format.replace('markdown', 'md')}"
        validator.export_report(report, format=format, output_file=default_output)
        console.print(f"\n[green]✓[/green] Report saved to: {default_output}")
    
    # Exit with error code if critical or high issues found
    if report.critical_issues or report.high_issues:
        sys.exit(1)


def main():
    """Main entry point for the CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
