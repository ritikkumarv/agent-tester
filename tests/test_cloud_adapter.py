"""
Tests for Cloud Adapter
========================

Test suite for the CloudAdapter to ensure it properly integrates
with the Agent Tester framework.
"""

import pytest
import os
from agent_tester.adapters.cloud_adapter import CloudAdapter
from agent_tester.models import TaskDefinition
from agent_tester import TaskValidator


@pytest.fixture
def cloud_adapter():
    """Create a CloudAdapter instance for testing"""
    # Set mock environment variables
    os.environ["CLOUD_API_ENDPOINT"] = "https://test-cloud-api.com/v1"
    os.environ["CLOUD_API_KEY"] = "test-api-key"
    os.environ["CLOUD_MODEL"] = "test-model"
    
    adapter = CloudAdapter(model="test-model")
    return adapter


@pytest.fixture
def sample_task():
    """Create a sample task for testing"""
    return TaskDefinition(
        task_id="test_task_1",
        goal="Analyze customer sentiment from a review",
        expected_output_schema={"required": ["result"]},
        timeout_seconds=30
    )


def test_cloud_adapter_initialization():
    """Test that CloudAdapter can be initialized correctly"""
    os.environ["CLOUD_API_ENDPOINT"] = "https://test-api.com"
    os.environ["CLOUD_API_KEY"] = "test-key"
    
    adapter = CloudAdapter(model="test-model")
    
    assert adapter.api_endpoint == "https://test-api.com"
    assert adapter.api_key == "test-key"
    assert adapter.model == "test-model"
    assert adapter.agent_id is not None


def test_cloud_adapter_missing_endpoint():
    """Test that CloudAdapter raises error when endpoint is missing"""
    # Clear environment variables
    if "CLOUD_API_ENDPOINT" in os.environ:
        del os.environ["CLOUD_API_ENDPOINT"]
    if "CLOUD_API_KEY" in os.environ:
        del os.environ["CLOUD_API_KEY"]
    
    with pytest.raises(ValueError, match="CLOUD_API_ENDPOINT not set"):
        CloudAdapter()


def test_cloud_adapter_missing_api_key():
    """Test that CloudAdapter raises error when API key is missing"""
    os.environ["CLOUD_API_ENDPOINT"] = "https://test-api.com"
    if "CLOUD_API_KEY" in os.environ:
        del os.environ["CLOUD_API_KEY"]
    
    with pytest.raises(ValueError, match="CLOUD_API_KEY not set"):
        CloudAdapter()


def test_execute_task(cloud_adapter, sample_task):
    """Test that execute_task returns expected structure"""
    result = cloud_adapter.execute_task(sample_task)
    
    # Check result structure
    assert "output" in result
    assert "execution_time" in result
    assert "trajectory" in result
    
    # Check output structure
    assert result["output"]["status"] == "success"
    assert result["output"]["task_id"] == sample_task.task_id
    assert result["output"]["model"] == "test-model"
    
    # Check trajectory
    assert result["trajectory"] is not None
    assert len(result["trajectory"].actions) >= 2  # At least init and LLM call


def test_trajectory_tracking(cloud_adapter, sample_task):
    """Test that trajectory actions are tracked correctly"""
    result = cloud_adapter.execute_task(sample_task)
    trajectory = result["trajectory"]
    
    # Check that actions were recorded
    assert len(trajectory.actions) > 0
    
    # Check that trajectory has the right task_id
    assert trajectory.task_id == sample_task.task_id
    
    # Check that actions have required fields
    for action in trajectory.actions:
        assert action.action_id is not None
        assert action.action_type is not None


def test_validation_with_cloud_adapter(cloud_adapter, sample_task):
    """Test that CloudAdapter results can be validated"""
    result = cloud_adapter.execute_task(sample_task)
    
    validator = TaskValidator()
    validation = validator.validate(
        result["output"],
        sample_task,
        result["execution_time"]
    )
    
    # Check validation passed
    assert validation.passed
    assert validation.goal_achieved


def test_build_system_prompt(cloud_adapter):
    """Test system prompt building with constraints"""
    task = TaskDefinition(
        task_id="test_prompt",
        goal="Summarize text",
        constraints=[
            {"name": "max_words", "type": "value_in_range", "max_value": 100}
        ],
        expected_output_schema={"required": ["summary", "word_count"]}
    )
    
    prompt = cloud_adapter._build_system_prompt(task)
    
    assert "Summarize text" in prompt
    assert "max_words" in prompt
    assert "summary" in prompt
    assert "word_count" in prompt


def test_parse_response_json(cloud_adapter, sample_task):
    """Test parsing JSON response"""
    import json
    
    # Test JSON in code block
    response1 = '```json\n{"result": "success", "data": "value"}\n```'
    parsed1 = cloud_adapter._parse_response(response1, sample_task)
    assert parsed1["result"] == "success"
    assert parsed1["data"] == "value"
    
    # Test plain JSON
    response2 = '{"result": "success"}'
    parsed2 = cloud_adapter._parse_response(response2, sample_task)
    assert parsed2["result"] == "success"


def test_parse_response_fallback(cloud_adapter, sample_task):
    """Test parsing non-JSON response falls back to raw text"""
    response = "This is plain text response"
    parsed = cloud_adapter._parse_response(response, sample_task)
    
    assert "result" in parsed
    assert parsed["result"] == response


def test_multiple_task_execution(cloud_adapter):
    """Test executing multiple tasks sequentially"""
    tasks = [
        TaskDefinition(
            task_id=f"task_{i}",
            goal=f"Task {i}",
            expected_output_schema={"required": ["result"]},
            timeout_seconds=30
        )
        for i in range(3)
    ]
    
    results = [cloud_adapter.execute_task(task) for task in tasks]
    
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result["output"]["status"] == "success"
        assert result["output"]["task_id"] == f"task_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
