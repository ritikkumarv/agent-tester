"""
Agent Test Suite - Orchestrates all validators for comprehensive testing
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from agent_tester.models import TaskDefinition
from agent_tester.validators.task_validator import TaskValidator
from agent_tester.validators.trajectory_validator import TrajectoryValidator
from agent_tester.validators.memory_validator import MemoryValidator

logger = logging.getLogger(__name__)


class AgentTestSuite:
    """
    Complete test suite for production agents

    Usage in CI/CD:
    ---------------
    from agent_tester import AgentTestSuite

    suite = AgentTestSuite()
    results = suite.run_all_tests(agent, test_cases)

    if not results["passed"]:
        print(f"Tests failed: {results['failures']}")
        sys.exit(1)
    """

    def __init__(
        self,
        task_validator: Optional[TaskValidator] = None,
        trajectory_validator: Optional[TrajectoryValidator] = None,
        memory_validator: Optional[MemoryValidator] = None,
    ):
        self.task_validator = task_validator or TaskValidator()
        self.trajectory_validator = trajectory_validator or TrajectoryValidator()
        self.memory_validator = memory_validator or MemoryValidator()

        self.test_results = []

    def run_all_tests(self, agent, test_cases: List[TaskDefinition]) -> Dict[str, Any]:
        """
        Run complete test suite on agent

        Args:
            agent: The agent to test (must have execute_task method)
            test_cases: List of test tasks

        Returns:
            Comprehensive test results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": getattr(agent, "agent_id", "unknown"),
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": [],
            "summary": {},
        }

        for task in test_cases:
            test_result = self._run_single_test(agent, task)
            results["test_results"].append(test_result)

            if test_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

        # Generate summary
        results["summary"] = {
            "pass_rate": (results["passed"] / results["total_tests"] * 100)
            if results["total_tests"] > 0
            else 0,
            "task_validation": self.task_validator.get_validation_report(),
            "avg_execution_time": sum(r["execution_time"] for r in results["test_results"])
            / len(results["test_results"])
            if results["test_results"]
            else 0,
        }

        return results

    def _run_single_test(self, agent, task: TaskDefinition) -> Dict[str, Any]:
        """Run a single test case"""
        execution_result = agent.execute_task(task)

        # Task validation
        task_result = self.task_validator.validate(
            execution_result["output"], task, execution_result["execution_time"]
        )

        # Trajectory validation
        trajectory_result = self.trajectory_validator.validate(
            execution_result["trajectory"]
        )

        # Memory validation (simplified - no conversation context)
        memory_result = self.memory_validator.validate(agent.memory, [])

        return {
            "task_id": task.task_id,
            "passed": all(
                [
                    task_result.passed,
                    trajectory_result.passed,
                    memory_result.passed,
                ]
            ),
            "task_validation": task_result.model_dump(),
            "trajectory_validation": trajectory_result.model_dump(),
            "memory_validation": memory_result.model_dump(),
            "execution_time": execution_result["execution_time"],
        }

    def generate_html_report(
        self, results: Dict[str, Any], output_path: str = "test_report.html"
    ):
        """Generate HTML test report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Agent Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2196F3; color: white; padding: 20px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f5f5f5; padding: 15px; border-radius: 5px; flex: 1; }}
        .passed {{ color: #4CAF50; }}
        .failed {{ color: #f44336; }}
        .test-result {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Agent Test Report</h1>
        <p>Generated: {results['timestamp']}</p>
        <p>Agent ID: {results['agent_id']}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <div style="font-size: 32px;">{results['total_tests']}</div>
        </div>
        <div class="metric">
            <h3 class="passed">Passed</h3>
            <div style="font-size: 32px;">{results['passed']}</div>
        </div>
        <div class="metric">
            <h3 class="failed">Failed</h3>
            <div style="font-size: 32px;">{results['failed']}</div>
        </div>
        <div class="metric">
            <h3>Pass Rate</h3>
            <div style="font-size: 32px;">{results['summary']['pass_rate']:.1f}%</div>
        </div>
    </div>
    
    <h2>Test Results</h2>
"""

        for test in results["test_results"]:
            status_class = "passed" if test["passed"] else "failed"
            status_text = "✓ PASSED" if test["passed"] else "✗ FAILED"

            html += f"""
    <div class="test-result">
        <h3 class="{status_class}">{status_text} - {test['task_id']}</h3>
        <p>Execution Time: {test['execution_time']:.3f}s</p>
        <details>
            <summary>Details</summary>
            <pre>{json.dumps(test, indent=2)}</pre>
        </details>
    </div>
"""

        html += """
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report generated: {output_path}")
