"""
Comprehensive Test Suite Example
=================================

This example demonstrates a comprehensive test suite configuration
for the Agent Tester CLI.

Usage:
    1. Save this as comprehensive_tests.yaml
    2. Run: agent-tester run -c comprehensive_tests.yaml
"""

# Example YAML configuration shown below:

example_yaml = """
name: Comprehensive Agent Test Suite
description: A complete test suite demonstrating various testing scenarios

tests:
  # Test 1: Simple question answering
  - task_id: qa_test_1
    goal: "What is the capital of France?"
    expected_output_schema:
      required: ["answer"]
    timeout_seconds: 30
  
  # Test 2: Sentiment analysis
  - task_id: sentiment_test_1
    goal: "Analyze the sentiment of: 'This product exceeded my expectations!'"
    expected_output_schema:
      required: ["sentiment", "confidence"]
    timeout_seconds: 30
  
  # Test 3: Summarization
  - task_id: summarize_test_1
    goal: "Summarize the following: 'Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI is enabling unprecedented automation and insights.'"
    constraints:
      - name: summary_length
        type: value_in_range
        min_value: 20
        max_value: 100
    expected_output_schema:
      required: ["summary"]
    timeout_seconds: 45
  
  # Test 4: Data extraction
  - task_id: extraction_test_1
    goal: "Extract key information: 'John Smith, age 35, lives in New York and works as a software engineer.'"
    expected_output_schema:
      required: ["name", "age", "location", "occupation"]
    timeout_seconds: 30
  
  # Test 5: Simple math
  - task_id: math_test_1
    goal: "Calculate: What is 15% of 200?"
    expected_output_schema:
      required: ["answer", "calculation"]
    timeout_seconds: 20

validators:
  task:
    strict_mode: false
  trajectory:
    max_actions: 20
    allow_backtracking: true
  memory:
    min_retention_score: 0.7
"""

# Save the example to a file
if __name__ == "__main__":
    import os
    
    output_file = "comprehensive_tests.yaml"
    
    with open(output_file, 'w') as f:
        f.write(example_yaml.strip())
    
    print(f"✅ Created example test configuration: {output_file}")
    print("\nTo run these tests:")
    print(f"  agent-tester run -c {output_file}")
    print("\nNote: Set OPENAI_API_KEY for real API testing, or use mock adapter for demo")
