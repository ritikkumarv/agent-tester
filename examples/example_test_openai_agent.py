"""
Quick Example: Testing a Real OpenAI Agent
===========================================

This script shows a simple end-to-end example of testing a real AI agent.

Run this after setting your OPENAI_API_KEY:
    $env:OPENAI_API_KEY = "your-key-here"
    python example_test_openai_agent.py
"""

import os
from test_real_agents import OpenAIAgentAdapter
from test_agent_framework import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator
)


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print('  $env:OPENAI_API_KEY = "sk-your-key-here"  # PowerShell')
        print('  export OPENAI_API_KEY="sk-your-key-here"  # Linux/Mac')
        return 1
    
    print("="*70)
    print("TESTING REAL OPENAI AGENT")
    print("="*70)
    
    # Define a simple task
    task = TaskDefinition(
        task_id="restaurant_recommendation",
        goal="Recommend 3 Italian restaurants in Seattle with outdoor seating. "
             "Include name, address, and average price per person.",
        constraints=[
            {"name": "count", "type": "count", "expected": 3},
            {"name": "format", "type": "format", "required_fields": ["results"]}
        ],
        expected_output_schema={
            "required": ["results"]
        },
        timeout_seconds=30
    )
    
    print(f"\n📋 Task: {task.goal}")
    print(f"⏱️  Timeout: {task.timeout_seconds}s")
    print("\n" + "-"*70)
    
    # Create OpenAI agent adapter
    print("\n🤖 Creating OpenAI Agent (gpt-4o-mini)...")
    adapter = OpenAIAgentAdapter(model="gpt-4o-mini")
    
    # Execute the task
    print("🚀 Executing task...")
    result = adapter.execute_task(task)
    
    print("\n" + "-"*70)
    print("📊 EXECUTION RESULTS")
    print("-"*70)
    print(f"Status: {result['output'].get('status', 'unknown')}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print(f"Actions Taken: {len(result['trajectory'].actions)}")
    
    # Print the agent's response
    print("\n🔍 Agent Response:")
    print("-"*70)
    if 'raw_response' in result['output']:
        print(result['output']['raw_response'][:500] + "...")
    else:
        import json
        print(json.dumps(result['output'], indent=2)[:500] + "...")
    
    # Validate the results
    print("\n" + "-"*70)
    print("✅ VALIDATION RESULTS")
    print("-"*70)
    
    # Task validation
    task_validator = TaskValidator()
    task_result = task_validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"\n📝 Task Validation:")
    print(f"  ✓ Passed: {task_result.passed}")
    print(f"  ✓ Goal Achieved: {task_result.goal_achieved}")
    print(f"  ✓ Output Valid: {task_result.output_valid}")
    print(f"  ✓ Constraints Met: {all(task_result.constraints_met.values())}")
    if not all(task_result.constraints_met.values()):
        print(f"    Failed constraints: {[k for k, v in task_result.constraints_met.items() if not v]}")
    
    # Trajectory validation
    trajectory_validator = TrajectoryValidator(max_actions=10)
    trajectory_result = trajectory_validator.validate(result["trajectory"])
    
    print(f"\n🛤️  Trajectory Validation:")
    print(f"  ✓ Efficient: {trajectory_result.is_efficient}")
    print(f"  ✓ Has Loops: {trajectory_result.has_loops}")
    print(f"  ✓ Action Count: {trajectory_result.action_count}/{trajectory_result.optimal_action_count} (optimal)")
    print(f"  ✓ Efficiency Score: {trajectory_result.efficiency_score:.1f}%")
    if trajectory_result.issues:
        print(f"  ⚠️  Issues: {', '.join(trajectory_result.issues)}")
    
    # Memory validation
    memory_validator = MemoryValidator()
    memory_result = memory_validator.validate(
        result["memory"],
        result["conversation"]
    )
    
    print(f"\n🧠 Memory Validation:")
    print(f"  ✓ Context Retention: {memory_result.context_retention_score:.1f}%")
    print(f"  ✓ Consistency: {memory_result.consistency_score:.1f}%")
    print(f"  ✓ Relevance: {memory_result.relevance_score:.1f}%")
    print(f"  ✓ Within Capacity: {memory_result.within_capacity}")
    print(f"  ✓ Memory Usage: {memory_result.memory_usage}/{memory_result.max_capacity}")
    
    # Overall summary
    print("\n" + "="*70)
    overall_passed = (
        task_result.passed and 
        trajectory_result.passed and 
        memory_result.passed
    )
    
    if overall_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
    else:
        print("⚠️  Some validations failed:")
        if not task_result.passed:
            print("   - Task validation failed")
        if not trajectory_result.passed:
            print("   - Trajectory validation failed")
        if not memory_result.passed:
            print("   - Memory validation failed")
    
    print("="*70)
    
    return 0 if overall_passed else 1


if __name__ == "__main__":
    exit(main())
