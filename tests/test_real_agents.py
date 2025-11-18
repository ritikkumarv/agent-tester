"""
Real-World Agent Testing Examples
==================================

This file shows how to test actual AI agents using the testing framework.
It includes examples for:
1. OpenAI Agents SDK
2. LangChain Agents
3. Custom Agents
4. Agent Framework (Microsoft)

Setup:
------
pip install openai langchain langchain-openai anthropic

Usage:
------
pytest test_real_agents.py -v
pytest test_real_agents.py -k "openai" -v
pytest test_real_agents.py -k "langchain" -v
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
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
import pytest


# ============================================================================
# ADAPTER 1: OpenAI Agents SDK
# ============================================================================

class OpenAIAgentAdapter:
    """
    Adapter for testing OpenAI Agents
    
    Example:
        from openai import OpenAI
        import os
        
        # API key from environment variable (secure)
        client = OpenAI()  # Uses OPENAI_API_KEY env var
        
        adapter = OpenAIAgentAdapter(client)
        result = adapter.execute_task(task_definition)
    """
    
    def __init__(self, client=None, model: str = "gpt-4o-mini"):
        """
        Args:
            client: OpenAI client instance (optional, will create if not provided)
            model: Model to use for the agent
        """
        self.model = model
        self.trajectory = None
        self.memory = AgentMemory(memory_id="openai_memory", max_size=100)
        self.conversation_history = []
        
        # Initialize client
        if client:
            self.client = client
        else:
            # Will use OPENAI_API_KEY environment variable
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except ImportError:
                raise ImportError("OpenAI SDK not installed. Run: pip install openai")
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task using OpenAI chat completions"""
        # Start trajectory tracking
        self.trajectory = Trajectory(
            trajectory_id=f"openai_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # Build system prompt from task
            system_prompt = self._build_system_prompt(task)
            
            # Track LLM call
            action_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task.goal}
                ],
                temperature=0.7
            )
            action_duration = (time.time() - action_start) * 1000
            
            # Log action
            self.trajectory.add_action(Action(
                action_id=f"llm_call_{len(self.trajectory.actions)}",
                action_type=ActionType.LLM_CALL,
                input_data={"goal": task.goal},
                output_data={"response": response.choices[0].message.content},
                duration_ms=action_duration,
                success=True
            ))
            
            # Parse response into structured output
            output = self._parse_response(response.choices[0].message.content, task)
            
            # Store in memory
            self.memory.store("last_task", task.goal, relevance=1.0)
            self.memory.store("last_output", output, relevance=0.9)
            self.conversation_history.append({
                "role": "user",
                "content": task.goal
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            
            output["status"] = "success"
            
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
    
    def _build_system_prompt(self, task: TaskDefinition) -> str:
        """Build system prompt from task definition"""
        prompt = f"You are an AI assistant. Your goal: {task.goal}\n\n"
        
        if task.constraints:
            prompt += "Constraints:\n"
            for constraint in task.constraints:
                prompt += f"- {constraint.get('name', 'constraint')}: {constraint}\n"
        
        prompt += "\nProvide your response in JSON format with the following structure:\n"
        if task.expected_output_schema.get('required'):
            prompt += f"Required fields: {', '.join(task.expected_output_schema['required'])}\n"
        
        return prompt
    
    def _parse_response(self, response: str, task: TaskDefinition) -> Dict[str, Any]:
        """Parse LLM response into structured output"""
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
            # Fallback: return raw response
            return {"raw_response": response}


# ============================================================================
# ADAPTER 2: LangChain Agents
# ============================================================================

class LangChainAgentAdapter:
    """
    Adapter for testing LangChain Agents
    
    Example:
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        adapter = LangChainAgentAdapter(llm, tools=[search_tool, calculator_tool])
        result = adapter.execute_task(task_definition)
    """
    
    def __init__(self, llm=None, tools=None):
        """
        Args:
            llm: LangChain LLM instance
            tools: List of LangChain tools
        """
        self.llm = llm
        self.tools = tools or []
        self.trajectory = None
        self.memory = AgentMemory(memory_id="langchain_memory", max_size=100)
        self.conversation_history = []
        
        if not llm:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except ImportError:
                raise ImportError("LangChain OpenAI not installed. Run: pip install langchain-openai")
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task using LangChain agent"""
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import PromptTemplate
        
        # Start trajectory tracking
        self.trajectory = Trajectory(
            trajectory_id=f"langchain_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # Create agent with tools
            if self.tools:
                # Build prompt template
                template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
                
                prompt = PromptTemplate.from_template(template)
                agent = create_react_agent(self.llm, self.tools, prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=self.tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10
                )
                
                # Execute with callback to track actions
                callback_handler = self._create_callback_handler()
                response = agent_executor.invoke(
                    {"input": task.goal},
                    config={"callbacks": [callback_handler]}
                )
                
                output = {
                    "result": response.get("output", ""),
                    "intermediate_steps": [str(step) for step in response.get("intermediate_steps", [])]
                }
            else:
                # Simple LLM call without tools
                action_start = time.time()
                response = self.llm.invoke(task.goal)
                action_duration = (time.time() - action_start) * 1000
                
                self.trajectory.add_action(Action(
                    action_id=f"llm_call_{len(self.trajectory.actions)}",
                    action_type=ActionType.LLM_CALL,
                    input_data={"goal": task.goal},
                    output_data={"response": response.content},
                    duration_ms=action_duration,
                    success=True
                ))
                
                output = {"result": response.content}
            
            # Store in memory
            self.memory.store("last_task", task.goal, relevance=1.0)
            self.conversation_history.append({
                "role": "user",
                "content": task.goal
            })
            
            output["status"] = "success"
            
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
    
    def _create_callback_handler(self):
        """Create callback handler to track LangChain actions"""
        from langchain.callbacks.base import BaseCallbackHandler
        
        adapter = self
        
        class TrajectoryCallbackHandler(BaseCallbackHandler):
            def on_tool_start(self, serialized, input_str, **kwargs):
                adapter.trajectory.add_action(Action(
                    action_id=f"tool_{len(adapter.trajectory.actions)}",
                    action_type=ActionType.TOOL_CALL,
                    tool_name=serialized.get("name", "unknown"),
                    input_data={"input": input_str},
                    success=True
                ))
            
            def on_llm_start(self, serialized, prompts, **kwargs):
                adapter.trajectory.add_action(Action(
                    action_id=f"llm_{len(adapter.trajectory.actions)}",
                    action_type=ActionType.LLM_CALL,
                    input_data={"prompts": prompts},
                    success=True
                ))
        
        return TrajectoryCallbackHandler()


# ============================================================================
# ADAPTER 3: Custom Agent (Your Own Implementation)
# ============================================================================

class CustomAgentAdapter:
    """
    Template for wrapping your custom agent implementation
    
    Example:
        class MyAgent:
            def run(self, prompt):
                # Your agent logic here
                return {"result": "..."}
        
        my_agent = MyAgent()
        adapter = CustomAgentAdapter(my_agent)
        result = adapter.execute_task(task_definition)
    """
    
    def __init__(self, agent_instance):
        """
        Args:
            agent_instance: Your custom agent object
                           Must have a method that accepts a prompt/task
        """
        self.agent = agent_instance
        self.trajectory = None
        self.memory = AgentMemory(memory_id="custom_memory", max_size=100)
        self.conversation_history = []
    
    def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute task using your custom agent"""
        # Start trajectory tracking
        self.trajectory = Trajectory(
            trajectory_id=f"custom_{task.task_id}_{int(time.time())}",
            task_id=task.task_id
        )
        
        start_time = time.time()
        
        try:
            # Call your agent - adapt this to match your agent's API
            # Common patterns:
            # result = self.agent.run(task.goal)
            # result = self.agent.execute(task.goal)
            # result = self.agent.process({"goal": task.goal})
            
            action_start = time.time()
            
            # Example: if your agent has a 'run' method
            if hasattr(self.agent, 'run'):
                result = self.agent.run(task.goal)
            elif hasattr(self.agent, 'execute'):
                result = self.agent.execute(task.goal)
            elif hasattr(self.agent, '__call__'):
                result = self.agent(task.goal)
            else:
                raise AttributeError("Agent must have 'run', 'execute', or be callable")
            
            action_duration = (time.time() - action_start) * 1000
            
            # Track the action
            self.trajectory.add_action(Action(
                action_id=f"agent_call_{len(self.trajectory.actions)}",
                action_type=ActionType.DECISION,
                input_data={"goal": task.goal},
                output_data={"result": result},
                duration_ms=action_duration,
                success=True
            ))
            
            # Convert result to expected format
            if isinstance(result, dict):
                output = result
            else:
                output = {"result": result}
            
            output["status"] = "success"
            
            # Store in memory
            self.memory.store("last_task", task.goal, relevance=1.0)
            self.memory.store("last_output", output, relevance=0.9)
            
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
# TEST SUITE FOR REAL AGENTS
# ============================================================================

class TestRealAgents:
    """
    Test suite for real-world agents
    
    Set environment variables before running:
    - OPENAI_API_KEY: Your OpenAI API key
    - ANTHROPIC_API_KEY: Your Anthropic API key (if using Claude)
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
            goal="Calculate 15% tip on a $85 bill",
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
            task_id="product_search",
            goal="Find 3 laptop recommendations under $1000 with good battery life",
            constraints=[
                {"name": "count", "type": "count", "expected": 3},
                {"name": "budget", "type": "budget", "max_value": 1000}
            ],
            expected_output_schema={"required": ["results"]},
            timeout_seconds=60
        )
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_openai_agent_simple_task(
        self, 
        simple_task, 
        task_validator, 
        trajectory_validator, 
        memory_validator
    ):
        """Test OpenAI agent with a simple task"""
        # Create adapter
        adapter = OpenAIAgentAdapter(model="gpt-4o-mini")
        
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
        print("OPENAI AGENT TEST RESULTS")
        print("="*60)
        print(f"Task Passed: {task_result.passed}")
        print(f"Goal Achieved: {task_result.goal_achieved}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Trajectory Efficient: {trajectory_result.is_efficient}")
        print(f"Memory Score: {memory_result.context_retention_score:.1f}%")
        print("="*60)
        
        # Assertions (may need adjustment based on actual agent behavior)
        assert result["output"]["status"] == "success"
        assert result["execution_time"] < simple_task.timeout_seconds
        assert len(result["trajectory"].actions) > 0
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_langchain_agent(
        self,
        simple_task,
        task_validator,
        trajectory_validator
    ):
        """Test LangChain agent"""
        # Create adapter
        adapter = LangChainAgentAdapter()
        
        # Execute task
        result = adapter.execute_task(simple_task)
        
        # Validate
        task_result = task_validator.validate(
            result["output"],
            simple_task,
            result["execution_time"]
        )
        
        trajectory_result = trajectory_validator.validate(result["trajectory"])
        
        print("\n" + "="*60)
        print("LANGCHAIN AGENT TEST RESULTS")
        print("="*60)
        print(f"Task Passed: {task_result.passed}")
        print(f"Output: {result['output']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print("="*60)
        
        assert result["output"]["status"] == "success"
    
    def test_custom_agent_template(self, simple_task, task_validator):
        """Test custom agent adapter template"""
        # Example: Simple mock custom agent
        class MockCustomAgent:
            def run(self, prompt):
                return {"result": "12.75", "explanation": "15% of $85 is $12.75"}
        
        # Create adapter with your agent
        my_agent = MockCustomAgent()
        adapter = CustomAgentAdapter(my_agent)
        
        # Execute task
        result = adapter.execute_task(simple_task)
        
        # Validate
        task_result = task_validator.validate(
            result["output"],
            simple_task,
            result["execution_time"]
        )
        
        print("\n" + "="*60)
        print("CUSTOM AGENT TEST RESULTS")
        print("="*60)
        print(f"Task Passed: {task_result.passed}")
        print(f"Output: {result['output']}")
        print("="*60)
        
        assert result["output"]["status"] == "success"
        assert "result" in result["output"]


# ============================================================================
# BATCH TESTING MULTIPLE AGENTS
# ============================================================================

def compare_agents(task: TaskDefinition, agents: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare multiple agents on the same task
    
    Args:
        task: Task definition to test
        agents: Dict of {"agent_name": adapter_instance}
    
    Returns:
        Comparison results
    """
    validator = TaskValidator()
    results = {}
    
    print(f"\n{'='*60}")
    print(f"COMPARING AGENTS ON TASK: {task.task_id}")
    print(f"{'='*60}\n")
    
    for agent_name, adapter in agents.items():
        print(f"Testing {agent_name}...")
        
        try:
            result = adapter.execute_task(task)
            validation = validator.validate(
                result["output"],
                task,
                result["execution_time"]
            )
            
            results[agent_name] = {
                "passed": validation.passed,
                "execution_time": result["execution_time"],
                "goal_achieved": validation.goal_achieved,
                "output": result["output"]
            }
            
            print(f"  ✓ Completed in {result['execution_time']:.2f}s")
            
        except Exception as e:
            results[agent_name] = {
                "passed": False,
                "error": str(e)
            }
            print(f"  ✗ Failed: {e}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for agent_name, result in results.items():
        status = "✓ PASSED" if result.get("passed") else "✗ FAILED"
        exec_time = result.get("execution_time", "N/A")
        print(f"{agent_name:20s} {status:10s} {exec_time}s" if isinstance(exec_time, float) else f"{agent_name:20s} {status:10s} {exec_time}")
    
    print(f"{'='*60}\n")
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use this in a script
    """
    print("Real Agent Testing Examples")
    print("="*60)
    print("\nTo run these tests:")
    print("1. Set your API keys:")
    print("   $env:OPENAI_API_KEY='your-key-here'  # PowerShell")
    print("   export OPENAI_API_KEY='your-key-here'  # Linux/Mac")
    print("\n2. Run pytest:")
    print("   pytest test_real_agents.py -v")
    print("\n3. Run specific tests:")
    print("   pytest test_real_agents.py -k 'openai' -v")
    print("   pytest test_real_agents.py -k 'langchain' -v")
    print("="*60)
