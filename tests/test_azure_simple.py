"""
Simple Synchronous Azure AI Test
"""
import os

def test_azure_connection():
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT")
    
    print("="*70)
    print("AZURE AI FOUNDRY CONNECTION TEST")
    print("="*70)
    print(f"\nEndpoint: {endpoint}")
    print(f"Deployment: {deployment}\n")
    
    if not endpoint or not deployment:
        print("ERROR: Environment variables not set!")
        print("\nRun these commands first:")
        print('$env:AZURE_AI_PROJECT_ENDPOINT = "your-endpoint"')
        print('$env:AZURE_AI_MODEL_DEPLOYMENT = "your-deployment"')
        return
    
    print("Testing Azure AI Foundry agent...")
    print("-"*70)
    
    # Import the adapter
    from test_azure_ai_agents import AzureAIFoundryAgentAdapter
    from test_agent_framework import TaskDefinition, TaskValidator
    
    # Create a simple task
    task = TaskDefinition(
        task_id="connection_test",
        goal="Say 'Hello! The connection is working!' and nothing else.",
        expected_output_schema={"required": ["result"]},
        timeout_seconds=30
    )
    
    print(f"Task: {task.goal}\n")
    
    try:
        # Create adapter
        print("Creating Azure AI agent adapter...")
        adapter = AzureAIFoundryAgentAdapter(
            project_endpoint=endpoint,
            model_deployment_name=deployment,
            agent_instructions="You are a helpful test agent."
        )
        
        print("Adapter created successfully")
        print("\nExecuting task...")
        print("-"*70)
        
        # Execute task
        result = adapter.execute_task(task)
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Status: {result['output'].get('status', 'unknown')}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"\nAgent Response:")
        print("-"*70)
        
        # Get the response
        if 'result' in result['output']:
            print(result['output']['result'])
        elif 'raw_response' in result['output']:
            print(result['output']['raw_response'])
        else:
            print(result['output'])
        
        print("-"*70)
        
        # Validate
        validator = TaskValidator()
        validation = validator.validate(
            result["output"],
            task,
            result["execution_time"]
        )
        
        print(f"\nValidation Results:")
        print(f"  Task Passed: {validation.passed}")
        print(f"  Goal Achieved: {validation.goal_achieved}")
        
        if validation.passed:
            print("\nSUCCESS: Azure AI Foundry connection is working!")
            print("\nYou can now:")
            print("  1. Run: python example_azure_agent.py")
            print("  2. Run: pytest test_azure_ai_agents.py -v")
            print("  3. Create your own agents with custom tools")
        else:
            print("\nWARNING: Task executed but validation had issues")
            print(f"  Constraints: {validation.constraints_met}")
        
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        print("\nFull error details:")
        print("-"*70)
        traceback.print_exc()

if __name__ == "__main__":
    test_azure_connection()
