"""
Integration with Microsoft Agent Framework
===========================================

This shows how to test agents built with Microsoft's Agent Framework.

Setup:
------
pip install azure-ai-projects openai

Usage:
------
python test_agent_framework_integration.py
"""

from typing import Dict, Any
import time
from test_agent_framework import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidator,
    MemoryValidator,
    Trajectory,
    Action,
    ActionType,
    AgentMemory
)


class AgentFrameworkAdapter:
    """
    Adapter for Microsoft Agent Framework agents
    
    Example usage with agent-framework:
        from agent_framework import Agent
        
        agent = Agent(...)
        adapter = AgentFrameworkAdapter(agent)
        result = adapter.execute_task(task_definition)
    """
    
    def __init__(self, agent):
        """
        Args:
            agent: Agent Framework agent instance
        """
        self.agent = agent
        self.trajectory = None
        self.memory = AgentMemory(memory_id="agent_framework_memory", max_size=100)
        self.conversation_history = []
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task using Agent Framework agent"""
        # Start trajectory tracking
        self.trajectory = Trajectory(
            trajectory_id=f"agent_framework_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # Execute the agent
            action_start = time.time()
            
            # Agent Framework typically uses .run() or .execute()
            if hasattr(self.agent, 'run'):
                response = self.agent.run(task.goal)
            elif hasattr(self.agent, 'execute'):
                response = self.agent.execute(task.goal)
            else:
                raise AttributeError("Agent must have 'run' or 'execute' method")
            
            action_duration = (time.time() - action_start) * 1000
            
            # Track the agent execution
            self.trajectory.add_action(Action(
                action_id=f"agent_run_{len(self.trajectory.actions)}",
                action_type=ActionType.LLM_CALL,
                input_data={"goal": task.goal},
                output_data={"response": str(response)},
                duration_ms=action_duration,
                success=True
            ))
            
            # Parse response to standard format
            if isinstance(response, dict):
                output = response
            elif isinstance(response, str):
                output = {"result": response}
            else:
                output = {"result": str(response)}
            
            output["status"] = "success"
            
            # Store in memory
            self.memory.store("last_task", task.goal, relevance=1.0)
            self.memory.store("last_output", output, relevance=0.9)
            self.conversation_history.append({
                "role": "user",
                "content": task.goal
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


class AzureAIAgentAdapter:
    """
    Adapter for Azure AI Agents (from azure-ai-projects)
    
    Example:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        
        client = AIProjectClient(...)
        adapter = AzureAIAgentAdapter(client, agent_id="your-agent-id")
        result = adapter.execute_task(task_definition)
    """
    
    def __init__(self, project_client, agent_id: str):
        """
        Args:
            project_client: Azure AI Project Client
            agent_id: ID of the deployed agent
        """
        self.client = project_client
        self.agent_id = agent_id
        self.trajectory = None
        self.memory = AgentMemory(memory_id="azure_agent_memory", max_size=100)
        self.conversation_history = []
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task using Azure AI Agent"""
        # Start trajectory tracking
        self.trajectory = Trajectory(
            trajectory_id=f"azure_agent_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # Create a thread for the conversation
            action_start = time.time()
            thread = self.client.agents.create_thread()
            
            # Add user message
            self.client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=task.goal
            )
            
            # Run the agent
            run = self.client.agents.create_run(
                thread_id=thread.id,
                agent_id=self.agent_id
            )
            
            # Wait for completion
            while run.status in ["queued", "in_progress", "requires_action"]:
                time.sleep(1)
                run = self.client.agents.get_run(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            action_duration = (time.time() - action_start) * 1000
            
            # Get messages
            messages = self.client.agents.list_messages(thread_id=thread.id)
            
            # Extract assistant response
            assistant_messages = [
                msg for msg in messages.data 
                if msg.role == "assistant"
            ]
            
            if assistant_messages:
                response_content = assistant_messages[0].content[0].text.value
                output = {"result": response_content}
            else:
                output = {"result": "No response from agent"}
            
            # Track the action
            self.trajectory.add_action(Action(
                action_id=f"azure_agent_run_{len(self.trajectory.actions)}",
                action_type=ActionType.LLM_CALL,
                input_data={"goal": task.goal},
                output_data={"response": output["result"]},
                duration_ms=action_duration,
                success=run.status == "completed"
            ))
            
            output["status"] = "success" if run.status == "completed" else "failed"
            
            # Store in memory
            self.memory.store("last_task", task.goal, relevance=1.0)
            self.conversation_history.append({
                "role": "user",
                "content": task.goal
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": output["result"]
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_test_agent_framework():
    """
    Example of testing an Agent Framework agent
    """
    print("Testing Agent Framework Integration")
    print("="*70)
    
    # Mock agent for demonstration
    class MockAgentFrameworkAgent:
        def run(self, prompt):
            return {
                "response": "Here are 3 restaurant recommendations...",
                "results": [
                    {"name": "Italianissimo", "price": "$$$"},
                    {"name": "Pasta Palace", "price": "$$"},
                    {"name": "Roma Bistro", "price": "$$$"}
                ]
            }
    
    # Create task
    task = TaskDefinition(
        task_id="test_task",
        goal="Recommend 3 Italian restaurants",
        constraints=[
            {"name": "count", "type": "count", "expected": 3}
        ],
        expected_output_schema={"required": ["results"]}
    )
    
    # Create adapter
    mock_agent = MockAgentFrameworkAgent()
    adapter = AgentFrameworkAdapter(mock_agent)
    
    # Execute
    result = adapter.execute_task(task)
    
    # Validate
    validator = TaskValidator()
    validation = validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"Task Passed: {validation.passed}")
    print(f"Goal Achieved: {validation.goal_achieved}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print(f"Output: {result['output']}")
    print("="*70)


if __name__ == "__main__":
    example_test_agent_framework()
