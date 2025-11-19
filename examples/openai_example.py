"""
OpenAI Agent Example
====================

This example shows how to use the Agent Tester framework with OpenAI.

Requirements:
    pip install openai
    export OPENAI_API_KEY="your-api-key"

Usage:
    python examples/openai_example.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_tester import TaskDefinition, TaskValidator
from agent_tester.adapters.openai_adapter import OpenAIAdapter


def main():
    print("🤖 Agent Tester - OpenAI Example")
    print("=" * 60)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\nFor this demo, we'll use a mock adapter instead.")
        print("=" * 60)
        
        # Use mock adapter for demo
        from agent_tester.models import Trajectory, AgentMemory
        import time
        
        class MockOpenAIAdapter:
            def __init__(self):
                self.memory = AgentMemory(memory_id="mock_memory", max_size=100)
            
            def execute_task(self, task):
                trajectory = Trajectory(
                    trajectory_id=f"mock_{task.task_id}",
                    task_id=task.task_id
                )
                trajectory.complete()
                return {
                    "output": {
                        "status": "success",
                        "sentiment": "positive",
                        "confidence": 0.85,
                        "summary": "This is a mock response. Set OPENAI_API_KEY for real responses."
                    },
                    "execution_time": 0.5,
                    "trajectory": trajectory
                }
        
        adapter = MockOpenAIAdapter()
        print("\n✅ Using mock adapter for demonstration")
    else:
        # Create OpenAI adapter
        print("\n1️⃣  Creating OpenAI adapter...")
        adapter = OpenAIAdapter(model="gpt-4o-mini")
        print("✅ OpenAI adapter created")
    
    # Define a task
    print("\n2️⃣  Defining task...")
    task = TaskDefinition(
        task_id="sentiment_analysis",
        goal="Analyze the sentiment of this text: 'I love using this testing framework! It makes my life so much easier.'",
        expected_output_schema={
            "required": ["sentiment", "confidence"]
        },
        timeout_seconds=30
    )
    print(f"✅ Task: {task.goal[:50]}...")
    
    # Execute the task
    print("\n3️⃣  Executing task...")
    result = adapter.execute_task(task)
    print(f"✅ Task executed in {result['execution_time']:.2f}s")
    
    # Validate results
    print("\n4️⃣  Validating results...")
    validator = TaskValidator()
    validation = validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    print(f"\n✅ Validation Status: {'PASSED ✓' if validation.passed else 'FAILED ✗'}")
    print(f"   Goal Achieved: {validation.goal_achieved}")
    print(f"   Output Valid: {validation.output_valid}")
    print(f"   Execution Time: {validation.execution_time:.3f}s")
    
    print(f"\n📤 Output:")
    for key, value in result["output"].items():
        print(f"   {key}: {value}")
    
    if validation.error_message:
        print(f"\n⚠️  Issues: {validation.error_message}")
    
    print("\n" + "=" * 60)
    if validation.passed:
        print("🎉 All validations PASSED!")
    else:
        print("❌ Validation FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
