"""
Simple Example: Testing a Mock Agent
=====================================

This example demonstrates the basic workflow of using Agent Tester
to validate an AI agent.
"""

import sys
sys.path.insert(0, '..')

from agent_tester import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator,
)

# Import MockAgent from the test file
import test_agent_framework
MockAgent = test_agent_framework.MockAgent


def main():
    print("🤖 Agent Tester - Simple Example")
    print("="*60)
    
    # 1. Create a mock agent for testing
    print("\n1️⃣  Creating mock agent...")
    agent = MockAgent(agent_id="demo_agent")
    
    # 2. Define a task
    print("2️⃣  Defining task...")
    task = TaskDefinition(
        task_id="demo_task",
        goal="Find top 3 products under $1000",
        constraints=[
            {"name": "budget", "type": "budget", "max_value": 1000},
            {"name": "count", "type": "count", "expected": 3}
        ],
        expected_output_schema={"required": ["results", "total_cost"]},
        timeout_seconds=30
    )
    
    # 3. Execute the task
    print("3️⃣  Executing task...")
    result = agent.execute_task(task)
    
    # 4. Validate results
    print("4️⃣  Validating results...")
    
    # Task validation
    task_validator = TaskValidator()
    task_result = task_validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    # Trajectory validation
    trajectory_validator = TrajectoryValidator(max_actions=10)
    trajectory_result = trajectory_validator.validate(result["trajectory"])
    
    # Memory validation
    memory_validator = MemoryValidator()
    memory_result = memory_validator.validate(agent.memory, [])
    
    # 5. Display results
    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS")
    print("="*60)
    
    print(f"\n✅ Task Validation:")
    print(f"   Passed: {task_result.passed}")
    print(f"   Goal Achieved: {task_result.goal_achieved}")
    print(f"   Constraints Met: {all(task_result.constraints_met.values())}")
    print(f"   Execution Time: {task_result.execution_time:.3f}s")
    
    print(f"\n🔍 Trajectory Validation:")
    print(f"   Passed: {trajectory_result.passed}")
    print(f"   Is Efficient: {trajectory_result.is_efficient}")
    print(f"   Action Count: {trajectory_result.action_count}")
    print(f"   Efficiency Score: {trajectory_result.efficiency_score:.1f}%")
    if trajectory_result.issues:
        print(f"   Issues: {trajectory_result.issues}")
    
    print(f"\n💾 Memory Validation:")
    print(f"   Passed: {memory_result.passed}")
    print(f"   Memory Usage: {memory_result.memory_usage}/{memory_result.max_capacity}")
    print(f"   Relevance Score: {memory_result.relevance_score:.1f}%")
    
    print("\n" + "="*60)
    overall_passed = all([
        task_result.passed,
        trajectory_result.passed,
        memory_result.passed
    ])
    if overall_passed:
        print("🎉 All validations PASSED!")
    else:
        print("❌ Some validations FAILED")
    print("="*60)


if __name__ == "__main__":
    main()
