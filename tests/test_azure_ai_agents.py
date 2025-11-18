"""
Azure AI Foundry Agent Testing Integration
===========================================

This module provides adapters for testing agents created with Microsoft Agent Framework
and deployed in Azure AI Foundry.

Prerequisites:
--------------
pip install agent-framework-azure-ai --pre
pip install azure-identity

Environment Variables:
----------------------
AZURE_AI_PROJECT_ENDPOINT - Your Azure AI Foundry project endpoint
AZURE_AI_MODEL_DEPLOYMENT - Your model deployment name
AZURE_AI_API_KEY - (Optional) API key if not using DefaultAzureCredential

Usage:
------
# Quick test
python test_azure_ai_agents.py

# With pytest
pytest test_azure_ai_agents.py -v

# Specific tests
pytest test_azure_ai_agents.py -k "azure" -v
"""

import os
import asyncio
import time
from typing import Dict, Any, List, Optional
import pytest
from test_agent_framework import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator,
    Trajectory,
    Action,
    ActionType,
    AgentMemory,
    MemoryEntry
)


# ============================================================================
# AZURE AI FOUNDRY AGENT ADAPTER
# ============================================================================

class AzureAIFoundryAgentAdapter:
    """
    Adapter for testing Azure AI Foundry agents built with Microsoft Agent Framework
    
    Example:
        from agent_framework import ChatAgent
        from agent_framework_azure_ai import AzureAIAgentClient
        from azure.identity.aio import DefaultAzureCredential
        
        adapter = AzureAIFoundryAgentAdapter(
            project_endpoint="https://your-project.cognitiveservices.azure.com/",
            model_deployment_name="gpt-4o-mini-deployment",
            agent_instructions="You are a helpful agent."
        )
        
        result = adapter.execute_task(task_definition)
    """
    
    def __init__(
        self,
        project_endpoint: Optional[str] = None,
        model_deployment_name: Optional[str] = None,
        api_key: Optional[str] = None,
        agent_instructions: str = "You are a helpful AI agent.",
        tools: Optional[List] = None
    ):
        """
        Args:
            project_endpoint: Azure AI Foundry project endpoint
            model_deployment_name: Name of deployed model
            api_key: Optional API key (uses DefaultAzureCredential if not provided)
            agent_instructions: System instructions for the agent
            tools: Optional list of tools/functions for the agent
        """
        # Get from environment if not provided
        self.project_endpoint = project_endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        self.model_deployment = model_deployment_name or os.getenv("AZURE_AI_MODEL_DEPLOYMENT")
        self.api_key = api_key or os.getenv("AZURE_AI_API_KEY")
        self.agent_instructions = agent_instructions
        self.tools = tools or []
        
        if not self.project_endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT not set")
        if not self.model_deployment:
            raise ValueError("AZURE_AI_MODEL_DEPLOYMENT not set")
        
        self.trajectory = None
        self.memory = AgentMemory(memory_id="azure_agent_memory", max_size=100)
        self.conversation_history = []
        self.agent_client = None
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task using Azure AI Foundry agent (synchronous wrapper)"""
        return asyncio.run(self._execute_task_async(task))
    
    async def _execute_task_async(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task using Azure AI Foundry agent (async)"""
        try:
            from agent_framework import ChatAgent
            from agent_framework_azure_ai import AzureAIAgentClient
            from azure.identity.aio import DefaultAzureCredential
        except ImportError as e:
            raise ImportError(
                "Microsoft Agent Framework not installed. Run: pip install agent-framework-azure-ai --pre"
            ) from e
        
        # Start trajectory tracking
        self.trajectory = Trajectory(
            trajectory_id=f"azure_agent_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # Create Azure AI Agent Client
            if self.api_key:
                # Use API key authentication
                client = AzureAIAgentClient(
                    project_endpoint=self.project_endpoint,
                    model_deployment_name=self.model_deployment,
                    api_key=self.api_key,
                    agent_name=f"Agent_{task.task_id}",
                )
            else:
                # Use DefaultAzureCredential
                client = AzureAIAgentClient(
                    project_endpoint=self.project_endpoint,
                    model_deployment_name=self.model_deployment,
                    async_credential=DefaultAzureCredential(),
                    agent_name=f"Agent_{task.task_id}",
                )
            
            # Build enhanced instructions with task constraints
            instructions = self._build_instructions(task)
            
            # Create agent
            async with ChatAgent(
                chat_client=client,
                instructions=instructions,
                tools=self.tools,
            ) as agent:
                # Create a thread for conversation
                thread = agent.get_new_thread()
                
                # Execute the task
                action_start = time.time()
                response_text = ""
                
                # Use streaming for better performance
                async for chunk in agent.run_stream(task.goal, thread=thread):
                    if chunk.text:
                        response_text += chunk.text
                
                action_duration = (time.time() - action_start) * 1000
                
                # Track the agent execution
                self.trajectory.add_action(Action(
                    action_id=f"agent_run_{len(self.trajectory.actions)}",
                    action_type=ActionType.LLM_CALL,
                    input_data={"goal": task.goal, "instructions": instructions},
                    output_data={"response": response_text},
                    duration_ms=action_duration,
                    success=True
                ))
                
                # Parse response to standard format
                output = self._parse_response(response_text, task)
                output["status"] = "success"
                
                # Store in memory
                self.memory.store("last_task", task.goal, relevance=1.0)
                self.memory.store("last_output", output, relevance=0.9)
                self.conversation_history.append({
                    "role": "user",
                    "content": task.goal
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
        
        except Exception as e:
            output = {"status": "failed", "error": str(e)}
            self.trajectory.add_action(Action(
                action_id=f"error_{len(self.trajectory.actions)}",
                action_type=ActionType.DECISION,
                success=False,
                error=str(e)
            ))
        
        finally:
            self.trajectory.complete()
        
        execution_time = time.time() - start_time
        
        return {
            "output": output,
            "execution_time": execution_time,
            "trajectory": self.trajectory,
            "memory": self.memory,
            "conversation": self.conversation_history
        }
    
    def _build_instructions(self, task: TaskDefinition) -> str:
        """Build enhanced instructions from task definition"""
        instructions = self.agent_instructions + "\n\n"
        instructions += f"Task Goal: {task.goal}\n\n"
        
        if task.constraints:
            instructions += "Constraints you must follow:\n"
            for constraint in task.constraints:
                constraint_name = constraint.get('name', 'constraint')
                constraint_type = constraint.get('type', 'custom')
                instructions += f"- {constraint_name}: {constraint}\n"
        
        if task.expected_output_schema.get('required'):
            instructions += "\nYour response must include these fields:\n"
            for field in task.expected_output_schema['required']:
                instructions += f"- {field}\n"
        
        return instructions
    
    def _parse_response(self, response: str, task: TaskDefinition) -> Dict[str, Any]:
        """Parse agent response into structured output"""
        import json
        
        # Try to extract JSON from response
        try:
            # Look for JSON in markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                # Try parsing entire response as JSON
                return json.loads(response)
        except:
            # Fallback: return raw response with result key
            return {"result": response, "raw_response": response}


# ============================================================================
# AZURE AI FOUNDRY AGENT WITH TOOLS
# ============================================================================

class AzureAIFoundryAgentWithToolsAdapter(AzureAIFoundryAgentAdapter):
    """
    Azure AI Foundry agent adapter with built-in tools
    
    Example:
        adapter = AzureAIFoundryAgentWithToolsAdapter(
            tools=[get_weather, search_database]
        )
        result = adapter.execute_task(task)
    """
    
    def __init__(self, **kwargs):
        """
        Initialize with tools
        
        Tools should be Python functions with type annotations:
        
        def get_weather(location: str) -> str:
            '''Get weather for location'''
            return f"Weather in {location}: Sunny, 72°F"
        """
        super().__init__(**kwargs)
        
        # Track tool calls in trajectory
        self.tool_call_count = 0
    
    async def _execute_task_async(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task with tool tracking"""
        result = await super()._execute_task_async(task)
        
        # Add tool usage to trajectory
        if self.tool_call_count > 0:
            self.trajectory.add_action(Action(
                action_id=f"tool_summary",
                action_type=ActionType.TOOL_CALL,
                tool_name="multiple_tools",
                metadata={"total_tool_calls": self.tool_call_count},
                success=True
            ))
        
        return result


# ============================================================================
# TEST SUITE FOR AZURE AI FOUNDRY AGENTS
# ============================================================================

class TestAzureAIFoundryAgents:
    """
    Test suite for Azure AI Foundry agents
    
    Set environment variables before running:
    - AZURE_AI_PROJECT_ENDPOINT: Your Azure AI Foundry project endpoint
    - AZURE_AI_MODEL_DEPLOYMENT: Your model deployment name
    - AZURE_AI_API_KEY: (Optional) Your API key
    """
    
    @pytest.fixture
    def task_validator(self):
        return TaskValidator()
    
    @pytest.fixture
    def trajectory_validator(self):
        return TrajectoryValidator(max_actions=20, allow_backtracking=True)
    
    @pytest.fixture
    def memory_validator(self):
        return MemoryValidator(min_retention_score=0.5)
    
    @pytest.fixture
    def simple_task(self):
        """Simple task for testing"""
        return TaskDefinition(
            task_id="simple_math",
            goal="Calculate the total cost: 3 items at $25 each, plus 8% sales tax",
            constraints=[
                {"name": "format", "type": "format", "required_fields": ["result"]}
            ],
            expected_output_schema={"required": ["result"]},
            timeout_seconds=30
        )
    
    @pytest.fixture
    def complex_task(self):
        """More complex task"""
        return TaskDefinition(
            task_id="recommendation",
            goal="Suggest 3 programming languages for building web applications, with pros and cons for each",
            constraints=[
                {"name": "count", "type": "count", "expected": 3}
            ],
            expected_output_schema={"required": ["results"]},
            timeout_seconds=60
        )
    
    @pytest.mark.skipif(
        not os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        reason="AZURE_AI_PROJECT_ENDPOINT not set"
    )
    def test_azure_agent_simple_task(
        self,
        simple_task,
        task_validator,
        trajectory_validator,
        memory_validator
    ):
        """Test Azure AI Foundry agent with a simple task"""
        # Create adapter
        adapter = AzureAIFoundryAgentAdapter(
            agent_instructions="You are a helpful math assistant."
        )
        
        # Execute task
        result = adapter.execute_task(simple_task)
        
        # Validate task completion
        task_result = task_validator.validate(
            result["output"],
            simple_task,
            result["execution_time"]
        )
        
        # Validate trajectory
        trajectory_result = trajectory_validator.validate(result["trajectory"])
        
        # Validate memory
        memory_result = memory_validator.validate(
            result["memory"],
            result["conversation"]
        )
        
        # Print results
        print("\n" + "="*60)
        print("AZURE AI FOUNDRY AGENT TEST RESULTS")
        print("="*60)
        print(f"Task Passed: {task_result.passed}")
        print(f"Goal Achieved: {task_result.goal_achieved}")
        print(f"Output: {result['output']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Trajectory Efficient: {trajectory_result.is_efficient}")
        print(f"Memory Score: {memory_result.context_retention_score:.1f}%")
        print("="*60)
        
        # Assertions
        assert result["output"]["status"] == "success"
        assert result["execution_time"] < simple_task.timeout_seconds
        assert len(result["trajectory"].actions) > 0
    
    @pytest.mark.skipif(
        not os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        reason="AZURE_AI_PROJECT_ENDPOINT not set"
    )
    def test_azure_agent_with_tools(
        self,
        simple_task,
        task_validator
    ):
        """Test Azure AI Foundry agent with custom tools"""
        from typing import Annotated
        
        # Define a custom tool
        def calculate_tax(
            amount: Annotated[float, "The amount to calculate tax on"],
            tax_rate: Annotated[float, "Tax rate as decimal (e.g., 0.08 for 8%)"] = 0.08
        ) -> str:
            """Calculate sales tax on an amount"""
            tax = amount * tax_rate
            total = amount + tax
            return f"Tax: ${tax:.2f}, Total: ${total:.2f}"
        
        # Create adapter with tools
        adapter = AzureAIFoundryAgentWithToolsAdapter(
            agent_instructions="You are a helpful assistant with access to calculation tools.",
            tools=[calculate_tax]
        )
        
        # Execute task
        result = adapter.execute_task(simple_task)
        
        # Validate
        task_result = task_validator.validate(
            result["output"],
            simple_task,
            result["execution_time"]
        )
        
        print("\n" + "="*60)
        print("AZURE AGENT WITH TOOLS TEST RESULTS")
        print("="*60)
        print(f"Task Passed: {task_result.passed}")
        print(f"Output: {result['output']}")
        print(f"Tool Calls: {adapter.tool_call_count}")
        print("="*60)
        
        assert result["output"]["status"] == "success"


# ============================================================================
# STANDALONE EXAMPLE
# ============================================================================

def example_azure_agent_test():
    """
    Standalone example of testing an Azure AI Foundry agent
    """
    print("="*70)
    print("AZURE AI FOUNDRY AGENT TESTING EXAMPLE")
    print("="*70)
    
    # Check environment
    if not os.getenv("AZURE_AI_PROJECT_ENDPOINT"):
        print("\nERROR: AZURE_AI_PROJECT_ENDPOINT not set")
        print("\nSet it with:")
        print('  $env:AZURE_AI_PROJECT_ENDPOINT = "https://your-project.cognitiveservices.azure.com/"')
        print('  $env:AZURE_AI_MODEL_DEPLOYMENT = "gpt-4o-mini-deployment"')
        return 1
    
    # Create task
    task = TaskDefinition(
        task_id="example_task",
        goal="Explain the benefits of using AI agents in 3 bullet points",
        constraints=[
            {"name": "format", "type": "format", "required_fields": ["result"]}
        ],
        expected_output_schema={"required": ["result"]},
        timeout_seconds=30
    )
    
    print(f"\n📋 Task: {task.goal}")
    print(f"⏱️  Timeout: {task.timeout_seconds}s\n")
    print("-"*70)
    
    # Create adapter
    print("\nCreating Azure AI Foundry Agent...")
    adapter = AzureAIFoundryAgentAdapter(
        agent_instructions="You are a helpful AI assistant."
    )
    
    # Execute
    print("Executing task...\n")
    result = adapter.execute_task(task)
    
    # Show results
    print("\n" + "-"*70)
    print("EXECUTION RESULTS")
    print("-"*70)
    print(f"Status: {result['output'].get('status', 'unknown')}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print(f"\n🔍 Agent Response:")
    print("-"*70)
    print(result['output'].get('result', result['output']))
    
    # Validate
    print("\n" + "-"*70)
    print("VALIDATION RESULTS")
    print("-"*70)
    
    validator = TaskValidator()
    validation = validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"Task Passed: {validation.passed}")
    print(f"Goal Achieved: {validation.goal_achieved}")
    print(f"Output Valid: {validation.output_valid}")
    print("="*70)
    
    return 0 if validation.passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(example_azure_agent_test())
