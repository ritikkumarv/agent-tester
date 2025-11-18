"""
Production-Ready Agentic Testing Framework
===========================================

This is a real-world testing framework that can be used in production.
It demonstrates Task, Trajectory, and Memory Validation for AI agents.

Setup:
------
pip install openai anthropic langchain pytest pytest-asyncio pydantic

Usage:
------
# Run all tests
pytest test_agent_framework.py -v --html=report.html

# Run specific validation type
pytest test_agent_framework.py -k "task" -v
pytest test_agent_framework.py -k "trajectory" -v
pytest test_agent_framework.py -k "memory" -v

# Generate coverage report
pytest test_agent_framework.py --cov=agent_framework --cov-report=html
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from enum import Enum
import pytest
from pydantic import BaseModel, Field


# ============================================================================
# PART 1: TASK VALIDATION
# ============================================================================

class TaskStatus(Enum):
    """Task completion status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class TaskConstraint(BaseModel):
    """Represents a constraint that the agent must satisfy"""
    name: str
    description: str
    constraint_type: str  # Type of constraint: 'key_exists', 'value_equals', 'value_in_range', etc.
    expected_value: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def validate(self, output: Dict[str, Any]) -> bool:
        """
        Execute the validation logic safely without eval()
        
        Supported constraint types:
        - 'key_exists': Check if a key exists in output
        - 'value_equals': Check if output[name] equals expected_value
        - 'value_in_range': Check if min_value <= output[name] <= max_value
        - 'list_length': Check if len(output[name]) equals expected_value
        - 'not_empty': Check if output[name] is not empty
        """
        try:
            if self.constraint_type == 'key_exists':
                return self.name in output
            
            elif self.constraint_type == 'value_equals':
                return output.get(self.name) == self.expected_value
            
            elif self.constraint_type == 'value_in_range':
                value = output.get(self.name)
                if value is None:
                    return False
                if self.min_value is not None and value < self.min_value:
                    return False
                if self.max_value is not None and value > self.max_value:
                    return False
                return True
            
            elif self.constraint_type == 'list_length':
                value = output.get(self.name)
                if not isinstance(value, (list, tuple)):
                    return False
                return len(value) == self.expected_value
            
            elif self.constraint_type == 'not_empty':
                value = output.get(self.name)
                if value is None:
                    return False
                if isinstance(value, (str, list, dict)):
                    return len(value) > 0
                return bool(value)
            
            else:
                # Unknown constraint type
                return False
                
        except Exception:
            return False


class TaskDefinition(BaseModel):
    """Defines a task for the agent to complete"""
    task_id: str
    goal: str
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    expected_output_schema: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 300
    max_retries: int = 3


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
        self, 
        agent_output: Dict[str, Any], 
        task: TaskDefinition,
        execution_time: float
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
                constraints_met[constraint["name"]] = False
        
        # 3. Validate output format
        output_valid = self._validate_output_schema(
            agent_output, task.expected_output_schema
        )
        
        # 4. Check execution time
        within_timeout = execution_time < task.timeout_seconds
        
        # Determine if task passed
        all_constraints_met = all(constraints_met.values())
        passed = (
            goal_achieved and 
            all_constraints_met and 
            output_valid and 
            within_timeout
        )
        
        result = TaskValidationResult(
            task_id=task.task_id,
            passed=passed,
            goal_achieved=goal_achieved,
            constraints_met=constraints_met,
            output_valid=output_valid,
            execution_time=execution_time
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
            "avg_execution_time": sum(v.execution_time for v in self.validation_history) / total
        }


# ============================================================================
# PART 2: TRAJECTORY VALIDATION
# ============================================================================

class ActionType(Enum):
    """Types of actions an agent can take"""
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    DECISION = "decision"


class Action(BaseModel):
    """Represents a single action in agent trajectory"""
    action_id: str
    action_type: ActionType
    tool_name: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0
    success: bool = True
    error: Optional[str] = None


class Trajectory(BaseModel):
    """Sequence of actions taken by agent"""
    trajectory_id: str
    task_id: str
    actions: List[Action] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def add_action(self, action: Action):
        """Add action to trajectory"""
        self.actions.append(action)
    
    def complete(self):
        """Mark trajectory as complete"""
        self.end_time = time.time()
    
    def get_duration(self) -> float:
        """Get total trajectory duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


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
        optimal_action_ratio: float = 1.5
    ):
        self.max_actions = max_actions
        self.allow_backtracking = allow_backtracking
        self.optimal_action_ratio = optimal_action_ratio
        self.validation_history: List[TrajectoryValidationResult] = []
    
    def validate(
        self, 
        trajectory: Trajectory,
        expected_pattern: Optional[List[str]] = None
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
            issues.append(
                f"Inefficient: {action_count} actions vs optimal {optimal_count}"
            )
        
        # 5. Check for failed actions
        failed_actions = [a for a in trajectory.actions if not a.success]
        if failed_actions:
            issues.append(f"{len(failed_actions)} actions failed")
        
        # Determine if passed
        passed = (
            action_count <= self.max_actions and
            follows_best_practices and
            is_efficient and
            (self.allow_backtracking or not has_loops)
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
            issues=issues
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
            if action_signatures[i] == action_signatures[i+1] == action_signatures[i+2]:
                return True
        
        return False
    
    def _check_action_sequence(
        self, 
        trajectory: Trajectory,
        expected_pattern: Optional[List[str]]
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
            if pattern_idx < len(expected_pattern) and action_tool == expected_pattern[pattern_idx]:
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
                ActionType.TOOL_CALL in prior_actions or 
                ActionType.MEMORY_READ in prior_actions
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
            "=" * 60
        ]
        
        for i, action in enumerate(trajectory.actions, 1):
            status = "✓" if action.success else "✗"
            lines.append(
                f"{i}. {status} [{action.action_type.value}] {action.tool_name or 'N/A'} "
                f"({action.duration_ms:.0f}ms)"
            )
        
        return "\n".join(lines)


# ============================================================================
# PART 3: MEMORY VALIDATION
# ============================================================================

class MemoryEntry(BaseModel):
    """Represents a single memory entry"""
    key: str
    value: Any
    timestamp: float = Field(default_factory=time.time)
    access_count: int = 0
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMemory(BaseModel):
    """Agent's memory store"""
    memory_id: str
    entries: Dict[str, MemoryEntry] = Field(default_factory=dict)
    max_size: int = 100
    
    def store(self, key: str, value: Any, relevance: float = 1.0):
        """Store information in memory"""
        self.entries[key] = MemoryEntry(
            key=key,
            value=value,
            relevance_score=relevance
        )
        
        # Evict if over capacity
        if len(self.entries) > self.max_size:
            self._evict_least_relevant()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve from memory"""
        if key in self.entries:
            self.entries[key].access_count += 1
            return self.entries[key].value
        return None
    
    def _evict_least_relevant(self):
        """Remove least relevant/accessed entries"""
        if not self.entries:
            return
        
        # Score based on relevance and recency
        scored_entries = [
            (
                key,
                entry.relevance_score * (1 / (time.time() - entry.timestamp + 1))
            )
            for key, entry in self.entries.items()
        ]
        
        scored_entries.sort(key=lambda x: x[1])
        # Remove bottom 10%
        to_remove = max(1, len(scored_entries) // 10)
        for key, _ in scored_entries[:to_remove]:
            del self.entries[key]


class MemoryValidationResult(BaseModel):
    """Results from memory validation"""
    memory_id: str
    passed: bool
    context_retention_score: float  # 0-100
    consistency_score: float  # 0-100
    relevance_score: float  # 0-100
    within_capacity: bool
    memory_usage: int
    max_capacity: int
    issues: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MemoryValidator:
    """
    Validates agent memory management
    
    Example Usage:
        validator = MemoryValidator(min_retention_score=0.8)
        memory = agent.get_memory()
        conversation = agent.get_conversation_history()
        result = validator.validate(memory, conversation)
    """
    
    def __init__(
        self,
        min_retention_score: float = 0.7,
        min_relevance_score: float = 0.6
    ):
        self.min_retention_score = min_retention_score
        self.min_relevance_score = min_relevance_score
        self.validation_history: List[MemoryValidationResult] = []
    
    def validate(
        self,
        memory: AgentMemory,
        conversation_history: List[Dict[str, str]]
    ) -> MemoryValidationResult:
        """
        Validate agent memory
        
        Args:
            memory: Agent's memory state
            conversation_history: Conversation history to check against
            
        Returns:
            MemoryValidationResult with detailed analysis
        """
        issues = []
        
        # 1. Check context retention
        retention_score = self._check_context_retention(memory, conversation_history)
        if retention_score < self.min_retention_score:
            issues.append(
                f"Low context retention: {retention_score:.2%} < {self.min_retention_score:.2%}"
            )
        
        # 2. Check consistency
        consistency_score = self._check_consistency(memory)
        if consistency_score < 0.9:
            issues.append(f"Memory inconsistencies detected: {consistency_score:.2%}")
        
        # 3. Check relevance
        relevance_score = self._check_relevance(memory)
        if relevance_score < self.min_relevance_score:
            issues.append(
                f"Too much irrelevant data: {relevance_score:.2%} < {self.min_relevance_score:.2%}"
            )
        
        # 4. Check capacity
        within_capacity = len(memory.entries) <= memory.max_size
        if not within_capacity:
            issues.append(
                f"Memory overflow: {len(memory.entries)} > {memory.max_size}"
            )
        
        # Determine if passed
        passed = (
            retention_score >= self.min_retention_score and
            consistency_score >= 0.9 and
            relevance_score >= self.min_relevance_score and
            within_capacity
        )
        
        result = MemoryValidationResult(
            memory_id=memory.memory_id,
            passed=passed,
            context_retention_score=retention_score * 100,
            consistency_score=consistency_score * 100,
            relevance_score=relevance_score * 100,
            within_capacity=within_capacity,
            memory_usage=len(memory.entries),
            max_capacity=memory.max_size,
            issues=issues
        )
        
        self.validation_history.append(result)
        return result
    
    def _check_context_retention(
        self,
        memory: AgentMemory,
        conversation: List[Dict[str, str]]
    ) -> float:
        """Check if important context is retained"""
        if not conversation:
            return 1.0
        
        # Extract key entities/facts from conversation
        key_facts = self._extract_key_facts(conversation)
        
        if not key_facts:
            return 1.0
        
        # Check how many are in memory
        retained = sum(
            1 for fact in key_facts
            if any(fact.lower() in str(entry.value).lower() for entry in memory.entries.values())
        )
        
        return retained / len(key_facts)
    
    def _extract_key_facts(self, conversation: List[Dict[str, str]]) -> List[str]:
        """Extract important facts from conversation (simplified)"""
        facts = []
        
        # Look for user preferences, names, numbers, etc.
        keywords = ["prefer", "like", "want", "need", "name is", "budget"]
        
        for turn in conversation:
            if turn.get("role") == "user":
                content = turn.get("content", "").lower()
                if any(kw in content for kw in keywords):
                    facts.append(content)
        
        return facts
    
    def _check_consistency(self, memory: AgentMemory) -> float:
        """Check for contradictions in memory"""
        # Simplified: check for duplicate keys with different values
        contradictions = 0
        total_checks = 0
        
        # Check for semantic contradictions (simplified)
        entries_list = list(memory.entries.values())
        for i, entry1 in enumerate(entries_list):
            for entry2 in entries_list[i+1:]:
                total_checks += 1
                # Check if same key prefix but different values
                if (entry1.key.split('_')[0] == entry2.key.split('_')[0] and
                    entry1.value != entry2.value):
                    # Could be legitimate (e.g., updates) or contradiction
                    # Check timestamps - if close together, might be contradiction
                    if abs(entry1.timestamp - entry2.timestamp) < 60:
                        contradictions += 1
        
        if total_checks == 0:
            return 1.0
        
        return 1.0 - (contradictions / total_checks)
    
    def _check_relevance(self, memory: AgentMemory) -> float:
        """Check proportion of relevant memories"""
        if not memory.entries:
            return 1.0
        
        relevant_count = sum(
            1 for entry in memory.entries.values()
            if entry.relevance_score >= self.min_relevance_score
        )
        
        return relevant_count / len(memory.entries)


# ============================================================================
# PART 4: PYTEST TEST SUITE
# ============================================================================

class TestTaskValidation:
    """Test suite for task validation"""
    
    @pytest.fixture
    def task_validator(self):
        return TaskValidator()
    
    @pytest.fixture
    def sample_task(self):
        return TaskDefinition(
            task_id="test_search",
            goal="Search for products under budget",
            constraints=[
                {
                    "name": "budget",
                    "type": "budget",
                    "max_value": 1000
                },
                {
                    "name": "result_count",
                    "type": "count",
                    "expected": 3
                }
            ],
            expected_output_schema={
                "required": ["results", "total_cost"]
            }
        )
    
    def test_successful_task_completion(self, task_validator, sample_task):
        """Test validation of successful task"""
        agent_output = {
            "status": "success",
            "results": [
                {"name": "Product 1", "price": 300},
                {"name": "Product 2", "price": 400},
                {"name": "Product 3", "price": 250}
            ],
            "total_cost": 950
        }
        
        result = task_validator.validate(agent_output, sample_task, execution_time=2.5)
        
        assert result.passed is True
        assert result.goal_achieved is True
        assert all(result.constraints_met.values())
        assert result.output_valid is True
    
    def test_constraint_violation(self, task_validator, sample_task):
        """Test detection of constraint violations"""
        agent_output = {
            "status": "success",
            "results": [
                {"name": "Product 1", "price": 600},
                {"name": "Product 2", "price": 700}
            ],
            "total_cost": 1300  # Over budget!
        }
        
        result = task_validator.validate(agent_output, sample_task, execution_time=2.5)
        
        assert result.passed is False
        assert result.constraints_met["budget"] is False
        assert result.constraints_met["result_count"] is False
    
    def test_output_format_invalid(self, task_validator, sample_task):
        """Test detection of invalid output format"""
        agent_output = {
            "status": "success",
            # Missing required fields
        }
        
        result = task_validator.validate(agent_output, sample_task, execution_time=2.5)
        
        assert result.passed is False
        assert result.output_valid is False


class TestTrajectoryValidation:
    """Test suite for trajectory validation"""
    
    @pytest.fixture
    def trajectory_validator(self):
        return TrajectoryValidator(max_actions=10, allow_backtracking=False)
    
    @pytest.fixture
    def sample_trajectory(self):
        trajectory = Trajectory(
            trajectory_id="traj_001",
            task_id="task_001"
        )
        
        # Add sample actions
        trajectory.add_action(Action(
            action_id="act_1",
            action_type=ActionType.TOOL_CALL,
            tool_name="search",
            input_data={"query": "laptops"},
            output_data={"results": 10},
            duration_ms=150
        ))
        
        trajectory.add_action(Action(
            action_id="act_2",
            action_type=ActionType.LLM_CALL,
            input_data={"prompt": "analyze results"},
            duration_ms=300
        ))
        
        trajectory.add_action(Action(
            action_id="act_3",
            action_type=ActionType.DECISION,
            input_data={"options": ["option1", "option2"]},
            duration_ms=50
        ))
        
        trajectory.complete()
        return trajectory
    
    def test_efficient_trajectory(self, trajectory_validator, sample_trajectory):
        """Test validation of efficient trajectory"""
        result = trajectory_validator.validate(sample_trajectory)
        
        assert result.passed is True
        assert result.is_efficient is True
        assert result.has_loops is False
        assert len(result.issues) == 0
    
    def test_detect_loops(self, trajectory_validator):
        """Test detection of action loops"""
        trajectory = Trajectory(trajectory_id="traj_loop", task_id="task_001")
        
        # Add repeated actions (loop)
        for _ in range(4):
            trajectory.add_action(Action(
                action_id=f"act_{_}",
                action_type=ActionType.TOOL_CALL,
                tool_name="search",
                input_data={"query": "same query"},
                duration_ms=100
            ))
        
        result = trajectory_validator.validate(trajectory)
        
        assert result.has_loops is True
        assert result.passed is False  # Because allow_backtracking=False
    
    def test_too_many_actions(self, trajectory_validator):
        """Test detection of excessive actions"""
        trajectory = Trajectory(trajectory_id="traj_long", task_id="task_001")
        
        # Add more actions than allowed
        for i in range(15):
            trajectory.add_action(Action(
                action_id=f"act_{i}",
                action_type=ActionType.TOOL_CALL,
                tool_name=f"tool_{i}",
                duration_ms=100
            ))
        
        result = trajectory_validator.validate(trajectory)
        
        assert result.passed is False
        assert result.action_count > trajectory_validator.max_actions
        assert any("Too many actions" in issue for issue in result.issues)


class TestMemoryValidation:
    """Test suite for memory validation"""
    
    @pytest.fixture
    def memory_validator(self):
        return MemoryValidator(min_retention_score=0.7)
    
    @pytest.fixture
    def sample_memory(self):
        memory = AgentMemory(memory_id="mem_001", max_size=10)
        
        # Add some relevant memories
        memory.store("user_name", "Alice", relevance=1.0)
        memory.store("user_budget", 1000, relevance=0.9)
        memory.store("user_preference", "gaming laptops", relevance=0.95)
        memory.store("conversation_context", "Looking for laptop", relevance=0.85)
        
        return memory
    
    @pytest.fixture
    def sample_conversation(self):
        return [
            {"role": "user", "content": "Hi, my name is Alice"},
            {"role": "assistant", "content": "Hello Alice! How can I help?"},
            {"role": "user", "content": "I want to buy a gaming laptop under $1000"},
            {"role": "assistant", "content": "I'll search for gaming laptops in your budget"}
        ]
    
    def test_good_context_retention(self, memory_validator, sample_memory, sample_conversation):
        """Test validation of good context retention"""
        result = memory_validator.validate(sample_memory, sample_conversation)
        
        assert result.passed is True
        assert result.context_retention_score >= 70
        assert result.within_capacity is True
    
    def test_memory_overflow(self, memory_validator, sample_conversation):
        """Test detection of memory overflow"""
        memory = AgentMemory(memory_id="mem_overflow", max_size=5)
        
        # Add more than capacity
        for i in range(10):
            memory.store(f"key_{i}", f"value_{i}", relevance=0.8)
        
        result = memory_validator.validate(memory, sample_conversation)
        
        # Memory should auto-evict, so should still be within capacity
        assert result.within_capacity is True
        assert result.memory_usage <= memory.max_size
    
    def test_low_relevance_detection(self, memory_validator, sample_conversation):
        """Test detection of irrelevant memories"""
        memory = AgentMemory(memory_id="mem_irrelevant", max_size=10)
        
        # Add mostly irrelevant data
        memory.store("user_name", "Alice", relevance=1.0)
        memory.store("random_fact_1", "sky is blue", relevance=0.1)
        memory.store("random_fact_2", "grass is green", relevance=0.1)
        memory.store("random_fact_3", "water is wet", relevance=0.1)
        
        result = memory_validator.validate(memory, sample_conversation)
        
        assert result.relevance_score < 70
        assert any("irrelevant" in issue.lower() for issue in result.issues)


# ============================================================================
# PART 5: INTEGRATION TESTS & REAL-WORLD EXAMPLES
# ============================================================================

class MockAgent:
    """Mock agent for testing purposes"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory = AgentMemory(memory_id=f"mem_{agent_id}", max_size=50)
        self.current_trajectory: Optional[Trajectory] = None
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute a task and return output"""
        # Start trajectory
        self.current_trajectory = Trajectory(
            trajectory_id=f"traj_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # Simulate agent actions
            self._simulate_task_execution(task)
            
            # Generate output
            output = self._generate_output(task)
            output["status"] = "success"
            
        except Exception as e:
            output = {"status": "failed", "error": str(e)}
        
        finally:
            self.current_trajectory.complete()
        
        execution_time = time.time() - start_time
        
        return {
            "output": output,
            "execution_time": execution_time,
            "trajectory": self.current_trajectory
        }
    
    def _simulate_task_execution(self, task: TaskDefinition):
        """Simulate agent executing a task"""
        # Action 1: Read from memory
        self.current_trajectory.add_action(Action(
            action_id=f"act_{len(self.current_trajectory.actions)}",
            action_type=ActionType.MEMORY_READ,
            input_data={"keys": ["context"]},
            output_data={"context": "previous context"},
            duration_ms=10
        ))
        
        # Action 2: Tool call (search)
        self.current_trajectory.add_action(Action(
            action_id=f"act_{len(self.current_trajectory.actions)}",
            action_type=ActionType.TOOL_CALL,
            tool_name="search",
            input_data={"query": task.goal},
            output_data={"results": ["result1", "result2", "result3"]},
            duration_ms=150
        ))
        
        # Action 3: LLM call (analyze)
        self.current_trajectory.add_action(Action(
            action_id=f"act_{len(self.current_trajectory.actions)}",
            action_type=ActionType.LLM_CALL,
            input_data={"prompt": "analyze search results"},
            output_data={"analysis": "results look good"},
            duration_ms=300
        ))
        
        # Action 4: Write to memory
        self.current_trajectory.add_action(Action(
            action_id=f"act_{len(self.current_trajectory.actions)}",
            action_type=ActionType.MEMORY_WRITE,
            input_data={"key": "last_search", "value": task.goal},
            duration_ms=5
        ))
        
        # Store in memory
        self.memory.store("last_task", task.goal, relevance=0.9)
    
    def _generate_output(self, task: TaskDefinition) -> Dict[str, Any]:
        """Generate task output"""
        return {
            "results": [
                {"name": "Item 1", "price": 300},
                {"name": "Item 2", "price": 400},
                {"name": "Item 3", "price": 250}
            ],
            "total_cost": 950,
            "task_id": task.task_id
        }


class TestIntegration:
    """Integration tests combining all validation types"""
    
    @pytest.fixture
    def complete_test_setup(self):
        """Setup for full integration test"""
        agent = MockAgent("agent_001")
        task_validator = TaskValidator()
        trajectory_validator = TrajectoryValidator(max_actions=10)
        memory_validator = MemoryValidator()
        
        return {
            "agent": agent,
            "task_validator": task_validator,
            "trajectory_validator": trajectory_validator,
            "memory_validator": memory_validator
        }
    
    def test_complete_agent_validation(self, complete_test_setup):
        """Test complete agent validation pipeline"""
        setup = complete_test_setup
        agent = setup["agent"]
        
        # Define task
        task = TaskDefinition(
            task_id="integration_test",
            goal="Find products under $1000",
            constraints=[
                {"name": "budget", "type": "budget", "max_value": 1000},
                {"name": "count", "type": "count", "expected": 3}
            ],
            expected_output_schema={"required": ["results", "total_cost"]}
        )
        
        # Execute task
        result = agent.execute_task(task)
        
        # Validate Task
        task_result = setup["task_validator"].validate(
            result["output"],
            task,
            result["execution_time"]
        )
        
        # Validate Trajectory
        trajectory_result = setup["trajectory_validator"].validate(
            result["trajectory"]
        )
        
        # Validate Memory
        conversation = [
            {"role": "user", "content": "Find products under $1000"}
        ]
        memory_result = setup["memory_validator"].validate(
            agent.memory,
            conversation
        )
        
        # Assert all passed
        assert task_result.passed is True, f"Task validation failed: {task_result}"
        assert trajectory_result.passed is True, f"Trajectory validation failed: {trajectory_result.issues}"
        assert memory_result.passed is True, f"Memory validation failed: {memory_result.issues}"
        
        # Generate comprehensive report
        report = {
            "agent_id": agent.agent_id,
            "task": task_result.model_dump(),
            "trajectory": trajectory_result.model_dump(),
            "memory": memory_result.model_dump(),
            "overall_passed": all([
                task_result.passed,
                trajectory_result.passed,
                memory_result.passed
            ])
        }
        
        print("\n" + "="*60)
        print("INTEGRATION TEST REPORT")
        print("="*60)
        print(json.dumps(report, indent=2))
        print("="*60)


# ============================================================================
# PART 6: REAL-WORLD USAGE EXAMPLES & CI/CD INTEGRATION
# ============================================================================

class AgentTestSuite:
    """
    Complete test suite for production agents
    
    Usage in CI/CD:
    ---------------
    # In your CI pipeline (e.g., GitHub Actions, Jenkins)
    
    from agent_framework import AgentTestSuite
    
    suite = AgentTestSuite()
    results = suite.run_all_tests(agent)
    
    if not results["passed"]:
        print(f"Tests failed: {results['failures']}")
        sys.exit(1)
    """
    
    def __init__(
        self,
        task_validator: Optional[TaskValidator] = None,
        trajectory_validator: Optional[TrajectoryValidator] = None,
        memory_validator: Optional[MemoryValidator] = None
    ):
        self.task_validator = task_validator or TaskValidator()
        self.trajectory_validator = trajectory_validator or TrajectoryValidator()
        self.memory_validator = memory_validator or MemoryValidator()
        
        self.test_results = []
    
    def run_all_tests(self, agent: MockAgent, test_cases: List[TaskDefinition]) -> Dict[str, Any]:
        """
        Run complete test suite on agent
        
        Args:
            agent: The agent to test
            test_cases: List of test tasks
            
        Returns:
            Comprehensive test results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent.agent_id,
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": [],
            "summary": {}
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
            "pass_rate": (results["passed"] / results["total_tests"] * 100) if results["total_tests"] > 0 else 0,
            "task_validation": self.task_validator.get_validation_report(),
            "avg_execution_time": sum(
                r["execution_time"] for r in results["test_results"]
            ) / len(results["test_results"]) if results["test_results"] else 0
        }
        
        return results
    
    def _run_single_test(self, agent: MockAgent, task: TaskDefinition) -> Dict[str, Any]:
        """Run a single test case"""
        execution_result = agent.execute_task(task)
        
        # Task validation
        task_result = self.task_validator.validate(
            execution_result["output"],
            task,
            execution_result["execution_time"]
        )
        
        # Trajectory validation
        trajectory_result = self.trajectory_validator.validate(
            execution_result["trajectory"]
        )
        
        # Memory validation (simplified - no conversation context)
        memory_result = self.memory_validator.validate(
            agent.memory,
            []
        )
        
        return {
            "task_id": task.task_id,
            "passed": all([
                task_result.passed,
                trajectory_result.passed,
                memory_result.passed
            ]),
            "task_validation": task_result.model_dump(),
            "trajectory_validation": trajectory_result.model_dump(),
            "memory_validation": memory_result.model_dump(),
            "execution_time": execution_result["execution_time"]
        }
    
    def generate_html_report(self, results: Dict[str, Any], output_path: str = "test_report.html"):
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
        
        for test in results['test_results']:
            status_class = "passed" if test['passed'] else "failed"
            status_text = "✓ PASSED" if test['passed'] else "✗ FAILED"
            
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
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML report generated: {output_path}")


# ============================================================================
# PART 7: COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Command-line interface for running agent tests
    
    Usage:
        python agent_framework.py --agent-id my_agent --test-suite standard
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Testing Framework")
    parser.add_argument("--agent-id", default="test_agent", help="Agent ID to test")
    parser.add_argument("--test-suite", default="standard", choices=["quick", "standard", "comprehensive"])
    parser.add_argument("--output", default="test_report.html", help="Output report path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print(f"Starting agent validation tests for: {args.agent_id}")
    print(f"Test suite: {args.test_suite}")
    print("="*60)
    
    # Create agent
    agent = MockAgent(args.agent_id)
    
    # Define test cases based on suite
    test_cases = []
    
    if args.test_suite in ["quick", "standard", "comprehensive"]:
        test_cases.append(TaskDefinition(
            task_id="basic_search",
            goal="Search for products",
            constraints=[
                {"name": "budget", "type": "budget", "max_value": 1000}
            ],
            expected_output_schema={"required": ["results"]}
        ))
    
    if args.test_suite in ["standard", "comprehensive"]:
        test_cases.append(TaskDefinition(
            task_id="complex_query",
            goal="Find and compare products with filters",
            constraints=[
                {"name": "budget", "type": "budget", "max_value": 500},
                {"name": "count", "type": "count", "expected": 5}
            ],
            expected_output_schema={"required": ["results", "total_cost"]}
        ))
    
    if args.test_suite == "comprehensive":
        test_cases.append(TaskDefinition(
            task_id="multi_step",
            goal="Research, analyze, and recommend",
            constraints=[
                {"name": "budget", "type": "budget", "max_value": 2000}
            ],
            expected_output_schema={"required": ["results", "analysis", "recommendation"]}
        ))
    
    # Run tests
    suite = AgentTestSuite()
    results = suite.run_all_tests(agent, test_cases)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass Rate: {results['summary']['pass_rate']:.1f}%")
    print(f"Avg Execution Time: {results['summary']['avg_execution_time']:.3f}s")
    print("="*60)
    
    # Generate HTML report
    suite.generate_html_report(results, args.output)
    
    # Exit with appropriate code
    if results['failed'] > 0:
        logger.warning("Some tests failed!")
        return 1
    else:
        logger.info("All tests passed!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())