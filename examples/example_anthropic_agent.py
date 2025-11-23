"""
Anthropic (Claude) Agent Example
=================================

This example demonstrates how to use the Anthropic adapter with the Agent Tester framework.

Prerequisites:
--------------
1. Install Anthropic SDK:
   pip install anthropic

2. Set your Anthropic API key:
   export ANTHROPIC_API_KEY="your-api-key-here"

3. Run the example:
   python examples/example_anthropic_agent.py

Features Demonstrated:
----------------------
- Task execution with Claude
- Trajectory validation
- Task validation
- Memory validation
- Multi-task testing
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_tester import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator,
)
from agent_tester.adapters.anthropic_adapter import AnthropicAdapter


def example_simple_task():
    """Example: Simple task with Claude"""
    print("\n" + "="*60)
    print("Example 1: Simple Task Execution with Claude")
    print("="*60)
    
    # Create adapter
    adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
    
    # Define task
    task = TaskDefinition(
        task_id="simple_question",
        goal="What are the three main benefits of cloud computing?",
        expected_output_schema={"required": ["result"]},
        timeout_seconds=30
    )
    
    # Execute
    result = adapter.execute_task(task)
    
    # Validate
    task_validator = TaskValidator()
    validation = task_validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"\nTask: {task.goal}")
    print(f"Model: {result['output'].get('model', 'N/A')}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print(f"Validation Passed: {validation.passed}")
    print(f"Goal Achieved: {validation.goal_achieved}")
    print(f"Output: {result['output'].get('result', 'N/A')[:200]}...")


def example_structured_output():
    """Example: Task with structured JSON output"""
    print("\n" + "="*60)
    print("Example 2: Structured Output Task")
    print("="*60)
    
    adapter = AnthropicAdapter()
    
    task = TaskDefinition(
        task_id="sentiment_analysis",
        goal="Analyze the sentiment of this review: 'The product is amazing and exceeded my expectations!'",
        expected_output_schema={
            "required": ["sentiment", "confidence", "summary"]
        },
        timeout_seconds=30
    )
    
    result = adapter.execute_task(task)
    
    task_validator = TaskValidator()
    validation = task_validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"\nTask: {task.goal}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print(f"Validation Passed: {validation.passed}")
    print(f"Schema Valid: {validation.schema_valid}")
    print(f"Output: {result['output']}")


def example_with_constraints():
    """Example: Task with constraints"""
    print("\n" + "="*60)
    print("Example 3: Task with Constraints")
    print("="*60)
    
    adapter = AnthropicAdapter()
    
    task = TaskDefinition(
        task_id="summarize_constrained",
        goal="Summarize the key features of artificial intelligence",
        constraints=[
            {
                "name": "word_count",
                "type": "value_in_range",
                "min_value": 20,
                "max_value": 50
            }
        ],
        expected_output_schema={"required": ["summary"]},
        timeout_seconds=30
    )
    
    result = adapter.execute_task(task)
    
    task_validator = TaskValidator()
    validation = task_validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"\nTask: {task.goal}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print(f"Validation Passed: {validation.passed}")
    print(f"Constraints Met: {all(validation.constraints_met.values())}")
    print(f"Output: {result['output']}")


def example_trajectory_validation():
    """Example: Validate agent trajectory"""
    print("\n" + "="*60)
    print("Example 4: Trajectory Validation")
    print("="*60)
    
    adapter = AnthropicAdapter()
    
    task = TaskDefinition(
        task_id="trajectory_test",
        goal="List three programming languages and their primary use cases",
        expected_output_schema={"required": ["result"]},
        timeout_seconds=30
    )
    
    result = adapter.execute_task(task)
    
    # Validate trajectory
    trajectory_validator = TrajectoryValidator(max_actions=10)
    trajectory_validation = trajectory_validator.validate(result["trajectory"])
    
    print(f"\nTask: {task.goal}")
    print(f"Actions Taken: {len(result['trajectory'].actions)}")
    print(f"Trajectory Valid: {trajectory_validation.passed}")
    print(f"Efficient: {trajectory_validation.efficient}")
    print(f"Has Loops: {trajectory_validation.has_loops}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    
    # Print action details
    print("\nAction Sequence:")
    for i, action in enumerate(result['trajectory'].actions, 1):
        print(f"  {i}. {action.action_type} - {action.duration_ms:.2f}ms")


def example_multiple_models():
    """Example: Compare different Claude models"""
    print("\n" + "="*60)
    print("Example 5: Comparing Claude Models")
    print("="*60)
    
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307"
    ]
    
    task = TaskDefinition(
        task_id="model_comparison",
        goal="Write a creative haiku about testing",
        expected_output_schema={"required": ["result"]},
        timeout_seconds=30
    )
    
    print("\nComparing models on the same task:")
    print(f"Task: {task.goal}\n")
    
    for model in models:
        try:
            adapter = AnthropicAdapter(model=model)
            result = adapter.execute_task(task)
            
            print(f"Model: {model}")
            print(f"  Time: {result['execution_time']:.3f}s")
            print(f"  Output: {result['output'].get('result', 'N/A')}")
            print()
        except Exception as e:
            print(f"Model: {model}")
            print(f"  Error: {str(e)}")
            print()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Anthropic (Claude) Agent Testing Examples")
    print("="*60)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not set!")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        print("\nContinuing with examples (will use mock if SDK not available)...\n")
    
    try:
        # Run examples
        example_simple_task()
        example_structured_output()
        example_with_constraints()
        example_trajectory_validation()
        example_multiple_models()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
