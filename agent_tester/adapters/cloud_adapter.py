"""
Cloud Adapter Example
=====================

This example shows how to create an adapter for generic cloud-based AI agents.

Usage:
------
export CLOUD_API_ENDPOINT="https://your-cloud-endpoint.com/api"
export CLOUD_API_KEY="your-api-key"
export CLOUD_MODEL="your-model-name"

python -m agent_tester.adapters.cloud_adapter
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


class CloudAdapter:
    """
    Adapter for generic cloud-based AI agents
    
    This adapter provides a template for integrating with cloud-based AI services
    such as Google Cloud AI, AWS Bedrock, or custom cloud deployments.
    """
    
    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "default-model",
        system_message: str = "You are a helpful AI agent.",
    ):
        """
        Initialize Cloud adapter
        
        Args:
            api_endpoint: Cloud API endpoint URL
            api_key: API key for authentication
            model: Model to use
            system_message: System message for the agent
        """
        self.api_endpoint = api_endpoint or os.getenv("CLOUD_API_ENDPOINT")
        self.api_key = api_key or os.getenv("CLOUD_API_KEY")
        self.model = model or os.getenv("CLOUD_MODEL", "default-model")
        self.system_message = system_message
        
        if not self.api_endpoint:
            raise ValueError("CLOUD_API_ENDPOINT not set")
        if not self.api_key:
            raise ValueError("CLOUD_API_KEY not set")
        
        self.trajectory = None
        self.memory = AgentMemory(memory_id="cloud_agent_memory", max_size=100)
        self.agent_id = f"cloud_agent_{int(time.time())}"
        
        logger.info(f"Initialized Cloud adapter for {self.model} at {self.api_endpoint}")
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """
        Execute a task using cloud-based AI
        
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
            # Build system prompt
            system_prompt = self._build_system_prompt(task)
            
            # Track API initialization
            action_start = time.time()
            self._track_action(
                ActionType.TOOL_CALL,
                tool_name="initialize_cloud_client",
                input_data={"endpoint": self.api_endpoint, "model": self.model},
                output_data={"client_id": self.agent_id},
                duration_ms=(time.time() - action_start) * 1000
            )
            
            # Make cloud API call
            action_start = time.time()
            response_text = self._call_cloud_api(task.goal, system_prompt)
            action_duration = (time.time() - action_start) * 1000
            
            # Track LLM call
            self._track_action(
                ActionType.LLM_CALL,
                input_data={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task.goal}
                    ]
                },
                output_data={"response": response_text},
                duration_ms=action_duration
            )
            
            # Parse response into structured output
            output = self._parse_response(response_text, task)
            output["status"] = "success"
            output["task_id"] = task.task_id
            output["model"] = self.model
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            output = {
                "status": "failed",
                "error": str(e)
            }
            self._track_action(
                ActionType.DECISION,
                success=False,
                input_data={"error": str(e)}
            )
        
        finally:
            self.trajectory.complete()
        
        execution_time = time.time() - start_time
        
        return {
            "output": output,
            "execution_time": execution_time,
            "trajectory": self.trajectory
        }
    
    def _call_cloud_api(self, prompt: str, system_prompt: str) -> str:
        """
        Make API call to cloud service
        
        This is a mock implementation. In a real scenario, this would make
        an HTTP request to the cloud API endpoint.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            
        Returns:
            Response text from the API
        """
        # Mock implementation - in production this would call the actual API
        logger.info(f"Making cloud API call to {self.api_endpoint}")
        
        # Simulate API response
        return f"Cloud agent response: Task '{prompt}' completed successfully."
    
    def _build_system_prompt(self, task: TaskDefinition) -> str:
        """Build system prompt from task definition"""
        prompt = f"{self.system_message}\n\nYour goal: {task.goal}\n\n"
        
        if task.constraints:
            prompt += "Constraints:\n"
            for constraint in task.constraints:
                prompt += f"- {constraint.get('name', 'constraint')}: {constraint}\n"
        
        prompt += "\nProvide your response in JSON format"
        if task.expected_output_schema.get('required'):
            prompt += f" with the following required fields: {', '.join(task.expected_output_schema['required'])}"
        prompt += "."
        
        return prompt
    
    def _parse_response(self, response: str, task: TaskDefinition) -> Dict[str, Any]:
        """Parse API response into structured output"""
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
        except Exception:
            # Fallback: return raw response
            return {"result": response}
    
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
    
    # Set mock environment variables for testing
    os.environ["CLOUD_API_ENDPOINT"] = "https://example-cloud-api.com/v1"
    os.environ["CLOUD_API_KEY"] = "test-api-key"
    os.environ["CLOUD_MODEL"] = "test-model"
    
    # Create adapter
    adapter = CloudAdapter(model="test-model")
    
    # Define task
    task = TaskDefinition(
        task_id="cloud_example",
        goal="Write a haiku about cloud computing",
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
    print(f"Cloud Adapter Example")
    print(f"{'='*60}")
    print(f"Task ID: {task.task_id}")
    print(f"Goal: {task.goal}")
    print(f"Model: {result['output'].get('model', 'N/A')}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print(f"Actions Taken: {len(result['trajectory'].actions)}")
    print(f"Validation Passed: {validation.passed}")
    print(f"Goal Achieved: {validation.goal_achieved}")
    print(f"{'='*60}\n")
