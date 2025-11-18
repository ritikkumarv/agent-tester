"""
OpenAI Adapter Example
======================

This example shows how to create an adapter for OpenAI agents.

Usage:
------
export OPENAI_API_KEY="your-api-key"

python -m agent_tester.adapters.openai_adapter
"""

import os
import time
import logging
from typing import Dict, Any, Optional

from agent_tester.models import (
    TaskDefinition,
    Trajectory,
    Action,
    ActionType,
    AgentMemory,
)

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """
    Adapter for OpenAI agents
    
    This adapter wraps OpenAI API to work with the Agent Tester framework.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_message: str = "You are a helpful AI agent.",
    ):
        """
        Initialize OpenAI adapter
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            system_message: System message for the agent
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.system_message = system_message
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.trajectory = None
        self.memory = AgentMemory(memory_id="openai_agent_memory", max_size=100)
        self.agent_id = f"openai_agent_{int(time.time())}"
        
        logger.info(f"Initialized OpenAI adapter for {self.model}")
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """
        Execute a task using OpenAI
        
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
            # TODO: Implement actual OpenAI execution
            # This is a placeholder showing the expected flow
            
            # Track LLM call
            self._track_action(
                ActionType.LLM_CALL,
                input_data={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": task.goal}
                    ]
                },
                output_data={"response": f"Completed: {task.goal}"},
                duration_ms=800
            )
            
            # Generate output
            output = {
                "status": "success",
                "result": f"Completed: {task.goal}",
                "task_id": task.task_id,
                "model": self.model
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
    adapter = OpenAIAdapter(model="gpt-4o-mini")
    
    # Define task
    task = TaskDefinition(
        task_id="openai_example",
        goal="Write a haiku about testing",
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
    print(f"OpenAI Adapter Example")
    print(f"{'='*60}")
    print(f"Task ID: {task.task_id}")
    print(f"Goal: {task.goal}")
    print(f"Model: {result['output'].get('model', 'N/A')}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print(f"Actions Taken: {len(result['trajectory'].actions)}")
    print(f"Validation Passed: {validation.passed}")
    print(f"Goal Achieved: {validation.goal_achieved}")
    print(f"{'='*60}\n")
