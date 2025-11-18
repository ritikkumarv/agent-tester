"""
Production-Ready Agentic Testing Framework
===========================================

BACKWARD COMPATIBILITY WRAPPER
-------------------------------
This file imports from the new agent_tester package for backward compatibility.
For new code, please use: from agent_tester import ...

Setup:
------
pip install -e .

Usage:
------
# Run all tests
pytest test_agent_framework.py -v --html=report.html

# Run specific validation type
pytest test_agent_framework.py -k "task" -v
pytest test_agent_framework.py -k "trajectory" -v  
pytest test_agent_framework.py -k "memory" -v

# Generate coverage report
pytest test_agent_framework.py --cov=agent_tester --cov-report=html
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
import pytest

# Import everything from the new package structure for backward compatibility
from agent_tester.models import (
    TaskStatus,
    TaskConstraint,
    TaskDefinition,
    ActionType,
    Action,
    Trajectory,
    MemoryEntry,
    AgentMemory,
)

from agent_tester.validators.task_validator import (
    TaskValidator,
    TaskValidationResult,
)

from agent_tester.validators.trajectory_validator import (
    TrajectoryValidator,
    TrajectoryValidationResult,
)

from agent_tester.validators.memory_validator import (
    MemoryValidator,
    MemoryValidationResult,
)

from agent_tester.suite import AgentTestSuite


# ============================================================================
# MOCK AGENT FOR TESTING
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


# ============================================================================
# PYTEST TEST SUITE
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
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Command-line interface for running agent tests
    
    Usage:
        python test_agent_framework.py --agent-id my_agent --test-suite standard
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
