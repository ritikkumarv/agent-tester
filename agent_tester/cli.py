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
