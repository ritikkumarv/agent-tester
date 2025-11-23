"""
Example: Using the Cloud Adapter
=================================

This example demonstrates how to use the CloudAdapter to test
cloud-based AI agents with the Agent Tester framework.

Usage:
------
export CLOUD_API_ENDPOINT="https://your-cloud-endpoint.com/api"
export CLOUD_API_KEY="your-api-key"
export CLOUD_MODEL="your-model-name"

python examples/example_cloud_agent.py
"""

import os
from agent_tester.adapters.cloud_adapter import CloudAdapter
from agent_tester import TaskDefinition, TaskValidator


def main():
    """Run a complete example of cloud agent testing"""
    
    # For this example, we'll use mock credentials
    # In production, these would come from environment variables
    os.environ["CLOUD_API_ENDPOINT"] = "https://example-cloud-api.com/v1"
    os.environ["CLOUD_API_KEY"] = "example-api-key"
    os.environ["CLOUD_MODEL"] = "example-model"
    
    print("\n" + "="*70)
    print("Cloud Adapter Example - Agent Tester Framework")
    print("="*70)
    
    # Initialize the cloud adapter
    print("\n1. Initializing Cloud Adapter...")
    adapter = CloudAdapter(model="example-model")
    print(f"   ✓ Connected to: {adapter.api_endpoint}")
    print(f"   ✓ Using model: {adapter.model}")
    print(f"   ✓ Agent ID: {adapter.agent_id}")
    
    # Define a task
    print("\n2. Defining Task...")
    task = TaskDefinition(
        task_id="sentiment_analysis",
        goal="Analyze the sentiment of the following review: 'This product is amazing! I love it.'",
        constraints=[
            {
                "name": "confidence",
                "type": "value_in_range",
                "min_value": 0.0,
                "max_value": 1.0
            }
        ],
        expected_output_schema={
            "required": ["result", "sentiment"]
        },
        timeout_seconds=30
    )
    print(f"   ✓ Task ID: {task.task_id}")
    print(f"   ✓ Goal: {task.goal}")
    
    # Execute the task
    print("\n3. Executing Task...")
    result = adapter.execute_task(task)
    print(f"   ✓ Status: {result['output'].get('status', 'unknown')}")
    print(f"   ✓ Execution Time: {result['execution_time']:.3f}s")
    print(f"   ✓ Actions Taken: {len(result['trajectory'].actions)}")
    
    # Validate the result
    print("\n4. Validating Result...")
    validator = TaskValidator()
    validation = validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"   ✓ Validation Passed: {validation.passed}")
    print(f"   ✓ Goal Achieved: {validation.goal_achieved}")
    print(f"   ✓ Output Valid: {validation.output_valid}")
    
    if validation.constraints_met:
        print(f"   ✓ Constraints Met: {all(validation.constraints_met.values())}")
    
    # Show trajectory details
    print("\n5. Trajectory Analysis...")
    for i, action in enumerate(result['trajectory'].actions):
        print(f"   Action {i+1}: {action.action_type.value}")
        print(f"     - Duration: {action.duration_ms:.2f}ms")
        print(f"     - Success: {action.success}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Task: {task.task_id}")
    print(f"Status: {'✓ PASSED' if validation.passed else '✗ FAILED'}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print(f"Total Actions: {len(result['trajectory'].actions)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
