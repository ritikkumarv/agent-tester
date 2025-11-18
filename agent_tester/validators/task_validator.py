"""
Task Validator - Validates agent task completion
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from agent_tester.models import TaskDefinition

logger = logging.getLogger(__name__)


class TaskValidationResult(BaseModel):
    """Results from task validation"""

    task_id: str
    passed: bool
    goal_achieved: bool
    constraints_met: Dict[str, bool]
    output_valid: bool
    execution_time: float
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TaskValidator:
    """
    Validates agent task completion

    Example Usage:
        validator = TaskValidator()
        task = TaskDefinition(
            task_id="search_product",
            goal="Find top 3 laptops under $1000",
            constraints=[
                {"name": "budget", "check": "price < 1000"},
                {"name": "count", "check": "len(results) == 3"}
            ]
        )
        result = validator.validate(agent_output, task)
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_history: List[TaskValidationResult] = []

    def validate(
        self, agent_output: Dict[str, Any], task: TaskDefinition, execution_time: float
    ) -> TaskValidationResult:
        """
        Comprehensive task validation

        Args:
            agent_output: The agent's output/response
            task: Task definition with goals and constraints
            execution_time: Time taken to complete task

        Returns:
            TaskValidationResult with detailed validation info
        """
        # 1. Validate goal achievement
        goal_achieved = self._check_goal_achievement(agent_output, task.goal)

        # 2. Validate constraints
        constraints_met = {}
        for constraint in task.constraints:
            try:
                constraints_met[constraint["name"]] = self._check_constraint(
                    agent_output, constraint
                )
            except Exception as e:
                logger.warning(f"Error checking constraint {constraint['name']}: {e}")
                constraints_met[constraint["name"]] = False

        # 3. Validate output format
        output_valid = self._validate_output_schema(
            agent_output, task.expected_output_schema
        )

        # 4. Check execution time
        within_timeout = execution_time < task.timeout_seconds

        # Determine if task passed
        all_constraints_met = all(constraints_met.values())
        passed = goal_achieved and all_constraints_met and output_valid and within_timeout

        result = TaskValidationResult(
            task_id=task.task_id,
            passed=passed,
            goal_achieved=goal_achieved,
            constraints_met=constraints_met,
            output_valid=output_valid,
            execution_time=execution_time,
        )

        self.validation_history.append(result)
        return result

    def _check_goal_achievement(self, output: Dict, goal: str) -> bool:
        """Check if agent achieved the stated goal"""
        # Check for completion indicators
        if not output:
            return False

        # Check for explicit success/failure markers
        if "status" in output:
            return output["status"] in ["success", "completed"]

        # Check if output has substantial content
        if "result" in output or "data" in output:
            result_data = output.get("result") or output.get("data")
            return result_data is not None and len(str(result_data)) > 0

        return "output" in output and output["output"] is not None

    def _check_constraint(self, output: Dict, constraint: Dict) -> bool:
        """Validate a specific constraint"""
        constraint_type = constraint.get("type", "custom")

        if constraint_type == "budget":
            max_budget = constraint.get("max_value")
            actual = output.get("total_cost", 0)
            return actual <= max_budget

        elif constraint_type == "time":
            max_time = constraint.get("max_seconds")
            actual = output.get("execution_time", 0)
            return actual <= max_time

        elif constraint_type == "count":
            expected_count = constraint.get("expected")
            results = output.get("results", [])
            return len(results) == expected_count

        elif constraint_type == "format":
            required_fields = constraint.get("required_fields", [])
            return all(field in output for field in required_fields)

        # Custom constraint validation
        return constraint.get("validator", lambda x: True)(output)

    def _validate_output_schema(self, output: Dict, schema: Dict) -> bool:
        """Validate output matches expected schema"""
        if not schema:
            return True  # No schema requirements

        required_fields = schema.get("required", [])
        return all(field in output for field in required_fields)

    def get_validation_report(self) -> Dict[str, Any]:
        """Generate summary report of all validations"""
        total = len(self.validation_history)
        if total == 0:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

        passed = sum(1 for v in self.validation_history if v.passed)

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total * 100,
            "avg_execution_time": sum(v.execution_time for v in self.validation_history)
            / total,
        }
