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

    # TODO: Implement actual test execution
    console.print("[yellow]⚠ Test execution not yet implemented[/yellow]")
    console.print(
        "This feature is coming soon. For now, use the Python API directly."
    )


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

    console.print("\n[yellow]⚠ Quick validation not yet implemented[/yellow]")
    console.print("This feature is coming soon. For now, use the Python API directly.")


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
