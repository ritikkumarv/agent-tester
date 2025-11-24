"""
Tests for Anthropic (Claude) Adapter
=====================================

These tests validate the Anthropic adapter integration with the Agent Tester framework.

Run with:
    pytest tests/test_anthropic_adapter.py -v
    
For integration tests with real API:
    ANTHROPIC_API_KEY=your-key pytest tests/test_anthropic_adapter.py -v -m integration
"""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from agent_tester import TaskDefinition, TaskValidator, TrajectoryValidator
from agent_tester.adapters.anthropic_adapter import AnthropicAdapter


class TestAnthropicAdapterUnit:
    """Unit tests for Anthropic adapter (no API calls)"""
    
    def test_adapter_initialization(self):
        """Test adapter initializes correctly with API key"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            adapter = AnthropicAdapter()
            assert adapter.api_key == 'test-key'
            assert adapter.model == 'claude-3-5-sonnet-20241022'
            assert adapter.agent_id.startswith('anthropic_agent_')
    
    def test_adapter_initialization_custom_model(self):
        """Test adapter with custom model"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            adapter = AnthropicAdapter(model='claude-3-opus-20240229')
            assert adapter.model == 'claude-3-opus-20240229'
    
    def test_adapter_no_api_key_raises_error(self):
        """Test that missing API key raises ValueError"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
                AnthropicAdapter()
    
    def test_build_system_prompt_basic(self):
        """Test system prompt generation"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            adapter = AnthropicAdapter()
            task = TaskDefinition(
                task_id="test",
                goal="Test goal",
                expected_output_schema={"required": ["result"]},
                timeout_seconds=30
            )
            
            prompt = adapter._build_system_prompt(task)
            assert "Test goal" in prompt
            assert "JSON format" in prompt
            assert "result" in prompt
    
    def test_build_system_prompt_with_constraints(self):
        """Test system prompt with constraints"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            adapter = AnthropicAdapter()
            task = TaskDefinition(
                task_id="test",
                goal="Test goal",
                constraints=[
                    {"name": "word_count", "type": "value_in_range", "min_value": 10, "max_value": 50}
                ],
                expected_output_schema={"required": ["result"]},
                timeout_seconds=30
            )
            
            prompt = adapter._build_system_prompt(task)
            assert "Constraints:" in prompt
    
    def test_parse_response_json_block(self):
        """Test parsing JSON from markdown code block"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            adapter = AnthropicAdapter()
            task = TaskDefinition(
                task_id="test",
                goal="Test",
                expected_output_schema={"required": ["result"]},
                timeout_seconds=30
            )
            
            response = '```json\n{"result": "test value"}\n```'
            parsed = adapter._parse_response(response, task)
            assert parsed == {"result": "test value"}
    
    def test_parse_response_plain_json(self):
        """Test parsing plain JSON response"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            adapter = AnthropicAdapter()
            task = TaskDefinition(
                task_id="test",
                goal="Test",
                expected_output_schema={"required": ["result"]},
                timeout_seconds=30
            )
            
            response = '{"result": "test value"}'
            parsed = adapter._parse_response(response, task)
            assert parsed == {"result": "test value"}
    
    def test_parse_response_fallback(self):
        """Test fallback for non-JSON response"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            adapter = AnthropicAdapter()
            task = TaskDefinition(
                task_id="test",
                goal="Test",
                expected_output_schema={"required": ["result"]},
                timeout_seconds=30
            )
            
            response = "This is plain text"
            parsed = adapter._parse_response(response, task)
            assert parsed == {"result": "This is plain text"}
    
    def test_execute_task_success(self):
        """Test successful task execution"""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "Success"}')]
        mock_client.messages.create.return_value = mock_response
        
        # Mock the Anthropic module and class
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict('sys.modules', {'anthropic': mock_anthropic_module}):
                adapter = AnthropicAdapter()
                
                task = TaskDefinition(
                    task_id="test_task",
                    goal="Test goal",
                    expected_output_schema={"required": ["result"]},
                    timeout_seconds=30
                )
                
                result = adapter.execute_task(task)
                
                assert result["output"]["status"] == "success"
                assert result["output"]["result"] == "Success"
                assert "execution_time" in result
                assert "trajectory" in result
                assert len(result["trajectory"].actions) > 0
    
    def test_execute_task_handles_error(self):
        """Test error handling in task execution"""
        # Mock the Anthropic module to raise error
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.side_effect = Exception("API Error")
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict('sys.modules', {'anthropic': mock_anthropic_module}):
                adapter = AnthropicAdapter()
                
                task = TaskDefinition(
                    task_id="test_task",
                    goal="Test goal",
                    expected_output_schema={"required": ["result"]},
                    timeout_seconds=30
                )
                
                result = adapter.execute_task(task)
                
                assert result["output"]["status"] == "failed"
                assert "error" in result["output"]
    
    def test_trajectory_tracking(self):
        """Test that trajectory is tracked correctly"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "Test"}')]
        mock_client.messages.create.return_value = mock_response
        
        # Mock the Anthropic module
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict('sys.modules', {'anthropic': mock_anthropic_module}):
                adapter = AnthropicAdapter()
                
                task = TaskDefinition(
                    task_id="test",
                    goal="Test",
                    expected_output_schema={"required": ["result"]},
                    timeout_seconds=30
                )
                
                result = adapter.execute_task(task)
                
                trajectory = result["trajectory"]
                assert trajectory.task_id == "test"
                assert len(trajectory.actions) > 0
                assert trajectory.actions[0].action_type.value == "llm_call"


@pytest.mark.integration
@pytest.mark.anthropic
class TestAnthropicAdapterIntegration:
    """Integration tests with real Anthropic API (requires ANTHROPIC_API_KEY)"""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter for integration tests"""
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        return AnthropicAdapter()
    
    def test_real_api_simple_task(self, adapter):
        """Test with real API - simple question"""
        task = TaskDefinition(
            task_id="integration_simple",
            goal="What is 2+2?",
            expected_output_schema={"required": ["result"]},
            timeout_seconds=30
        )
        
        result = adapter.execute_task(task)
        
        assert result["output"]["status"] == "success"
        assert "result" in result["output"]
        assert result["execution_time"] < task.timeout_seconds
        
        # Validate with TaskValidator
        validator = TaskValidator()
        validation = validator.validate(
            result["output"],
            task,
            result["execution_time"]
        )
        assert validation.passed
    
    def test_real_api_structured_output(self, adapter):
        """Test with real API - structured output"""
        task = TaskDefinition(
            task_id="integration_structured",
            goal="Analyze sentiment: 'I love this product!'",
            expected_output_schema={
                "required": ["sentiment", "confidence"]
            },
            timeout_seconds=30
        )
        
        result = adapter.execute_task(task)
        
        assert result["output"]["status"] == "success"
        assert "sentiment" in result["output"]
        assert "confidence" in result["output"]
    
    def test_real_api_trajectory_validation(self, adapter):
        """Test trajectory validation with real API"""
        task = TaskDefinition(
            task_id="integration_trajectory",
            goal="List three colors",
            expected_output_schema={"required": ["result"]},
            timeout_seconds=30
        )
        
        result = adapter.execute_task(task)
        
        # Validate trajectory
        trajectory_validator = TrajectoryValidator(max_actions=10)
        validation = trajectory_validator.validate(result["trajectory"])
        
        assert validation.passed
        assert len(result["trajectory"].actions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
