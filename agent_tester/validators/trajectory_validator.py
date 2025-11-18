"""
Trajectory Validator - Analyzes agent execution trajectories
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from agent_tester.models import Trajectory, ActionType

logger = logging.getLogger(__name__)


class TrajectoryValidationResult(BaseModel):
    """Results from trajectory validation"""

    trajectory_id: str
    passed: bool
    is_efficient: bool
    has_loops: bool
    follows_best_practices: bool
    action_count: int
    optimal_action_count: int
    efficiency_score: float  # 0-100
    issues: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TrajectoryValidator:
    """
    Validates agent execution trajectories

    Example Usage:
        validator = TrajectoryValidator(max_actions=10, allow_backtracking=False)
        trajectory = agent.get_trajectory()
        result = validator.validate(trajectory)

        if not result.passed:
            print(f"Issues found: {result.issues}")
    """

    def __init__(
        self,
        max_actions: int = 20,
        allow_backtracking: bool = True,
        optimal_action_ratio: float = 1.5,
    ):
        self.max_actions = max_actions
        self.allow_backtracking = allow_backtracking
        self.optimal_action_ratio = optimal_action_ratio
        self.validation_history: List[TrajectoryValidationResult] = []

    def validate(
        self, trajectory: Trajectory, expected_pattern: Optional[List[str]] = None
    ) -> TrajectoryValidationResult:
        """
        Validate agent trajectory

        Args:
            trajectory: The agent's action sequence
            expected_pattern: Optional expected action pattern

        Returns:
            TrajectoryValidationResult with detailed analysis
        """
        issues = []

        # 1. Check action count
        action_count = len(trajectory.actions)
        if action_count > self.max_actions:
            issues.append(f"Too many actions: {action_count} > {self.max_actions}")

        # 2. Check for loops/redundancy
        has_loops = self._detect_loops(trajectory)
        if has_loops and not self.allow_backtracking:
            issues.append("Detected redundant action loops")

        # 3. Check for logical sequence
        follows_best_practices = self._check_action_sequence(trajectory, expected_pattern)
        if not follows_best_practices:
            issues.append("Action sequence doesn't follow best practices")

        # 4. Calculate efficiency
        optimal_count = self._estimate_optimal_actions(trajectory)
        efficiency_score = (optimal_count / action_count * 100) if action_count > 0 else 0
        is_efficient = action_count <= optimal_count * self.optimal_action_ratio

        if not is_efficient:
            issues.append(f"Inefficient: {action_count} actions vs optimal {optimal_count}")

        # 5. Check for failed actions
        failed_actions = [a for a in trajectory.actions if not a.success]
        if failed_actions:
            issues.append(f"{len(failed_actions)} actions failed")

        # Determine if passed
        passed = (
            action_count <= self.max_actions
            and follows_best_practices
            and is_efficient
            and (self.allow_backtracking or not has_loops)
        )

        result = TrajectoryValidationResult(
            trajectory_id=trajectory.trajectory_id,
            passed=passed,
            is_efficient=is_efficient,
            has_loops=has_loops,
            follows_best_practices=follows_best_practices,
            action_count=action_count,
            optimal_action_count=optimal_count,
            efficiency_score=efficiency_score,
            issues=issues,
        )

        self.validation_history.append(result)
        return result

    def _detect_loops(self, trajectory: Trajectory) -> bool:
        """Detect if agent is repeating actions"""
        action_signatures = []

        for action in trajectory.actions:
            # Create signature: action_type + tool_name + input_keys
            sig = f"{action.action_type.value}:{action.tool_name}:{sorted(action.input_data.keys())}"
            action_signatures.append(sig)

        # Check for consecutive duplicate actions (loop indicator)
        for i in range(len(action_signatures) - 2):
            if (
                action_signatures[i]
                == action_signatures[i + 1]
                == action_signatures[i + 2]
            ):
                return True

        return False

    def _check_action_sequence(
        self, trajectory: Trajectory, expected_pattern: Optional[List[str]]
    ) -> bool:
        """Check if actions follow logical sequence"""
        if not expected_pattern:
            # No specific pattern required - just check for basic logic
            return self._check_basic_logic(trajectory)

        # Match against expected pattern
        actual_pattern = [a.tool_name for a in trajectory.actions if a.tool_name]

        # Allow some flexibility - pattern should be subsequence
        pattern_idx = 0
        for action_tool in actual_pattern:
            if (
                pattern_idx < len(expected_pattern)
                and action_tool == expected_pattern[pattern_idx]
            ):
                pattern_idx += 1

        # Must match at least 80% of expected pattern
        return pattern_idx >= len(expected_pattern) * 0.8

    def _check_basic_logic(self, trajectory: Trajectory) -> bool:
        """Check for basic logical flow"""
        # Example logic: shouldn't write to memory before reading
        # shouldn't make decisions without gathering data first

        action_types = [a.action_type for a in trajectory.actions]

        # Check if agent made decision before gathering info
        if ActionType.DECISION in action_types:
            decision_idx = action_types.index(ActionType.DECISION)
            # Should have at least one tool call or memory read before deciding
            prior_actions = action_types[:decision_idx]
            has_info_gathering = (
                ActionType.TOOL_CALL in prior_actions
                or ActionType.MEMORY_READ in prior_actions
            )
            if not has_info_gathering:
                return False

        return True

    def _estimate_optimal_actions(self, trajectory: Trajectory) -> int:
        """Estimate minimum actions needed"""
        # Simplified heuristic: count unique tool calls + necessary LLM calls
        unique_tools = set()
        llm_calls = 0

        for action in trajectory.actions:
            if action.action_type == ActionType.TOOL_CALL and action.tool_name:
                unique_tools.add(action.tool_name)
            elif action.action_type == ActionType.LLM_CALL:
                llm_calls += 1

        # Estimate: unique tools + at least 1 LLM call
        return len(unique_tools) + max(1, llm_calls // 2)

    def visualize_trajectory(self, trajectory: Trajectory) -> str:
        """Generate ASCII visualization of trajectory"""
        lines = [
            f"Trajectory: {trajectory.trajectory_id}",
            f"Duration: {trajectory.get_duration():.2f}s",
            f"Actions: {len(trajectory.actions)}",
            "=" * 60,
        ]

        for i, action in enumerate(trajectory.actions, 1):
            status = "✓" if action.success else "✗"
            lines.append(
                f"{i}. {status} [{action.action_type.value}] {action.tool_name or 'N/A'} "
                f"({action.duration_ms:.0f}ms)"
            )

        return "\n".join(lines)
