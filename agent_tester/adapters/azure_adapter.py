"""
Azure AI Foundry Adapter Example
=================================

This example shows how to create an adapter for Azure AI Foundry agents.

Usage:
------
export AZURE_AI_PROJECT_ENDPOINT="https://your-project.cognitiveservices.azure.com/"
export AZURE_AI_MODEL_DEPLOYMENT="gpt-4o-mini"

python -m agent_tester.adapters.azure_adapter
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List

from agent_tester.models import (
    TaskDefinition,
    Trajectory,
    Action,
    ActionType,
    AgentMemory,
)

logger = logging.getLogger(__name__)


class AzureAIFoundryAdapter:
    """
    Adapter for Azure AI Foundry agents
    
    This adapter wraps Azure AI agents to work with the Agent Tester framework.
    """
    
    def __init__(
        self,
        project_endpoint: Optional[str] = None,
        model_deployment: Optional[str] = None,
        agent_instructions: str = "You are a helpful AI agent.",
    ):
        """
        Initialize Azure AI Foundry adapter
        
        Args:
            project_endpoint: Azure AI project endpoint
            model_deployment: Model deployment name
            agent_instructions: System instructions for the agent
        """
        self.project_endpoint = project_endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        self.model_deployment = model_deployment or os.getenv("AZURE_AI_MODEL_DEPLOYMENT")
        self.agent_instructions = agent_instructions
        
        if not self.project_endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT not set")
        if not self.model_deployment:
            raise ValueError("AZURE_AI_MODEL_DEPLOYMENT not set")
        
        self.trajectory = None
        self.memory = AgentMemory(memory_id="azure_agent_memory", max_size=100)
        self.agent_id = f"azure_agent_{int(time.time())}"
        
        logger.info(f"Initialized Azure AI Foundry adapter for {self.model_deployment}")
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """
        Execute a task using Azure AI Foundry
        
        Args:
            task: Task definition to execute
            
        Returns:
            Dict with output, execution_time, and trajectory
        """
        # Start trajectory tracking
        self.trajectory = Trajectory(
            trajectory_id=f"traj_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # TODO: Implement actual Azure AI Foundry execution
            # This is a placeholder showing the expected flow
            
            # Step 1: Create/get agent
            self._track_action(
                ActionType.TOOL_CALL,
                tool_name="create_agent",
                input_data={"instructions": self.agent_instructions},
                output_data={"agent_id": self.agent_id},
                duration_ms=50
            )
            
            # Step 2: Execute task
            self._track_action(
                ActionType.LLM_CALL,
                input_data={"task": task.goal},
                output_data={"response": "Task executed"},
                duration_ms=1000
            )
            
            # Generate output
            output = {
                "status": "success",
                "result": f"Completed: {task.goal}",
                "task_id": task.task_id
            }
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            output = {
                "status": "failed",
                "error": str(e)
            }
        
        finally:
            self.trajectory.complete()
        
        execution_time = time.time() - start_time
        
        return {
            "output": output,
            "execution_time": execution_time,
            "trajectory": self.trajectory
        }
    
    def _track_action(
        self,
        action_type: ActionType,
        tool_name: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0,
        success: bool = True
    ):
        """Track an action in the trajectory"""
        action = Action(
            action_id=f"act_{len(self.trajectory.actions)}",
            action_type=action_type,
            tool_name=tool_name,
            input_data=input_data or {},
            output_data=output_data or {},
            duration_ms=duration_ms,
            success=success
        )
        self.trajectory.add_action(action)


# Example usage
if __name__ == "__main__":
    from agent_tester import TaskValidator
    
    # Create adapter
    adapter = AzureAIFoundryAdapter()
    
    # Define task
    task = TaskDefinition(
        task_id="azure_example",
        goal="Summarize a customer review",
        expected_output_schema={"required": ["result"]},
        timeout_seconds=30
    )
    
    # Execute
    result = adapter.execute_task(task)
    
    # Validate
    validator = TaskValidator()
    validation = validator.validate(
        result["output"],
        task,
        result["execution_time"]
    )
    
    print(f"\n{'='*60}")
    print(f"Azure AI Foundry Adapter Example")
    print(f"{'='*60}")
    print(f"Task ID: {task.task_id}")
    print(f"Goal: {task.goal}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print(f"Actions Taken: {len(result['trajectory'].actions)}")
    print(f"Validation Passed: {validation.passed}")
    print(f"Goal Achieved: {validation.goal_achieved}")
    print(f"{'='*60}\n")
