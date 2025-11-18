"""
Complete Example: Azure AI Foundry Agent Testing
=================================================

This is a complete, runnable example showing how to:
1. Create an agent in Azure AI Foundry
2. Add custom tools to the agent
3. Test it with the testing framework
4. Validate all three aspects (Task, Trajectory, Memory)

Prerequisites:
--------------
1. Azure AI Foundry project with a deployed model
2. Environment variables set:
   - AZURE_AI_PROJECT_ENDPOINT
   - AZURE_AI_MODEL_DEPLOYMENT
3. Packages installed:
   pip install agent-framework-azure-ai --pre
   pip install azure-identity

Usage:
------
python example_azure_agent.py
"""

import os
import asyncio
from typing import Annotated
from test_azure_ai_agents import AzureAIFoundryAgentWithToolsAdapter
from test_agent_framework import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator
)


# ============================================================================
# DEFINE CUSTOM TOOLS FOR YOUR AGENT
# ============================================================================

def get_weather(
    location: Annotated[str, "The city or location to get weather for"]
) -> str:
    """Get current weather for a location"""
    # In a real application, this would call a weather API
    weather_data = {
        "Seattle": "Rainy, 55°F",
        "New York": "Sunny, 68°F",
        "London": "Cloudy, 60°F",
        "Tokyo": "Clear, 72°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def calculate_cost(
    items: Annotated[int, "Number of items"],
    price_per_item: Annotated[float, "Price per item in dollars"],
    tax_rate: Annotated[float, "Tax rate as decimal (e.g., 0.08 for 8%)"] = 0.08
) -> str:
    """Calculate total cost including tax"""
    subtotal = items * price_per_item
    tax = subtotal * tax_rate
    total = subtotal + tax
    return f"Subtotal: ${subtotal:.2f}, Tax: ${tax:.2f}, Total: ${total:.2f}"


def search_products(
    category: Annotated[str, "Product category to search"],
    max_price: Annotated[float, "Maximum price in dollars"]
) -> str:
    """Search for products in a category within budget"""
    # Mock product database
    products = {
        "laptop": [
            {"name": "Budget Laptop", "price": 499},
            {"name": "Mid-Range Laptop", "price": 899},
            {"name": "Premium Laptop", "price": 1499}
        ],
        "phone": [
            {"name": "Budget Phone", "price": 299},
            {"name": "Mid-Range Phone", "price": 599},
            {"name": "Premium Phone", "price": 1099}
        ]
    }
    
    results = []
    for product in products.get(category.lower(), []):
        if product["price"] <= max_price:
            results.append(f"{product['name']}: ${product['price']}")
    
    if results:
        return "Found: " + ", ".join(results)
    return f"No {category} found under ${max_price}"


# ============================================================================
# MAIN TESTING EXAMPLE
# ============================================================================

def main():
    print("="*70)
    print("AZURE AI FOUNDRY AGENT - COMPLETE TESTING EXAMPLE")
    print("="*70)
    
    # Step 1: Verify environment setup
    print("\n📋 Step 1: Checking environment...")
    
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT")
    
    if not endpoint or not deployment:
        print("\n❌ Error: Azure AI Foundry credentials not set!")
        print("\nPlease set the following environment variables:")
        print('  $env:AZURE_AI_PROJECT_ENDPOINT = "https://your-project.cognitiveservices.azure.com/"')
        print('  $env:AZURE_AI_MODEL_DEPLOYMENT = "gpt-4o-mini-deployment"')
        print("\nSee azure_ai_foundry_setup.md for detailed setup instructions.")
        return 1
    
    print(f"✅ Endpoint: {endpoint}")
    print(f"✅ Model: {deployment}")
    
    # Step 2: Create agent with tools
    print("\n🤖 Step 2: Creating Azure AI Foundry Agent with tools...")
    
    adapter = AzureAIFoundryAgentWithToolsAdapter(
        project_endpoint=endpoint,
        model_deployment_name=deployment,
        agent_instructions="You are a helpful shopping assistant with access to weather, cost calculation, and product search tools.",
        tools=[get_weather, calculate_cost, search_products]
    )
    
    print("✅ Agent created with 3 tools:")
    print("   - get_weather")
    print("   - calculate_cost")
    print("   - search_products")
    
    # Step 3: Define test tasks
    print("\n📝 Step 3: Defining test tasks...")
    
    tasks = [
        TaskDefinition(
            task_id="weather_query",
            goal="What's the weather like in Seattle?",
            constraints=[],
            expected_output_schema={"required": ["result"]},
            timeout_seconds=30
        ),
        TaskDefinition(
            task_id="shopping_task",
            goal="Find me a laptop under $1000 and calculate the total cost with 8% tax",
            constraints=[
                {"name": "budget", "type": "budget", "max_value": 1000}
            ],
            expected_output_schema={"required": ["result"]},
            timeout_seconds=60
        ),
        TaskDefinition(
            task_id="multi_step_task",
            goal="I need a phone under $700. Show me options and tell me the weather in New York.",
            constraints=[
                {"name": "budget", "type": "budget", "max_value": 700}
            ],
            expected_output_schema={"required": ["result"]},
            timeout_seconds=60
        )
    ]
    
    print(f"✅ Created {len(tasks)} test tasks")
    
    # Step 4: Create validators
    print("\n⚙️  Step 4: Setting up validators...")
    
    task_validator = TaskValidator()
    trajectory_validator = TrajectoryValidator(max_actions=15)
    memory_validator = MemoryValidator(min_retention_score=0.5)
    
    print("✅ Validators ready")
    
    # Step 5: Execute and validate each task
    print("\n🚀 Step 5: Executing tasks...")
    print("="*70)
    
    results = []
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*70}")
        print(f"TASK {i}/{len(tasks)}: {task.task_id}")
        print(f"{'='*70}")
        print(f"Goal: {task.goal}")
        print(f"\n⏳ Executing...")
        
        # Execute task
        result = adapter.execute_task(task)
        
        # Validate
        task_result = task_validator.validate(
            result["output"],
            task,
            result["execution_time"]
        )
        
        trajectory_result = trajectory_validator.validate(result["trajectory"])
        
        memory_result = memory_validator.validate(
            result["memory"],
            result["conversation"]
        )
        
        # Display results
        print(f"\n📊 Results:")
        print(f"-"*70)
        print(f"Status: {result['output'].get('status', 'unknown')}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"\n🤖 Agent Response:")
        print(f"-"*70)
        response = result['output'].get('result', result['output'].get('raw_response', str(result['output'])))
        print(response[:300] + ("..." if len(str(response)) > 300 else ""))
        
        print(f"\n✅ Validation:")
        print(f"-"*70)
        print(f"  Task Passed: {'✅' if task_result.passed else '❌'} ({task_result.passed})")
        print(f"  Goal Achieved: {'✅' if task_result.goal_achieved else '❌'} ({task_result.goal_achieved})")
        print(f"  Constraints Met: {'✅' if all(task_result.constraints_met.values()) else '❌'}")
        print(f"  Trajectory Efficient: {'✅' if trajectory_result.is_efficient else '❌'} ({trajectory_result.efficiency_score:.1f}%)")
        print(f"  Actions Count: {trajectory_result.action_count}/{trajectory_result.optimal_action_count}")
        print(f"  Memory Retention: {memory_result.context_retention_score:.1f}%")
        
        if trajectory_result.issues:
            print(f"  ⚠️  Issues: {', '.join(trajectory_result.issues)}")
        
        # Store results
        results.append({
            "task": task,
            "passed": task_result.passed and trajectory_result.passed and memory_result.passed,
            "execution_time": result["execution_time"],
            "task_validation": task_result,
            "trajectory_validation": trajectory_result,
            "memory_validation": memory_result
        })
    
    # Step 6: Summary
    print(f"\n{'='*70}")
    print("📈 FINAL SUMMARY")
    print(f"{'='*70}")
    
    total_tasks = len(results)
    passed_tasks = sum(1 for r in results if r["passed"])
    avg_time = sum(r["execution_time"] for r in results) / total_tasks
    
    print(f"\nTotal Tasks: {total_tasks}")
    print(f"Passed: {passed_tasks}/{total_tasks}")
    print(f"Pass Rate: {(passed_tasks/total_tasks)*100:.1f}%")
    print(f"Average Execution Time: {avg_time:.2f}s")
    
    print(f"\n{'='*70}")
    
    if passed_tasks == total_tasks:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Your Azure AI Foundry agent is working correctly!")
        print("✅ All validations passed!")
        print("\nNext steps:")
        print("  1. Try creating your own custom tools")
        print("  2. Test with more complex tasks")
        print("  3. Integrate into your CI/CD pipeline")
        print(f"{'='*70}\n")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print(f"\nFailed tasks: {total_tasks - passed_tasks}")
        print("\nReview the validation results above for details.")
        print(f"{'='*70}\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
