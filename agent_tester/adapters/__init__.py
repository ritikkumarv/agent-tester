"""
Adapters for different AI platforms
====================================

Adapters allow the framework to work with different AI agent platforms:
- Azure AI Foundry
- OpenAI
- Anthropic
- LangChain
- Custom agents

Each adapter implements a common interface for executing tasks and
collecting telemetry data (trajectory, memory state).

Example Usage:
--------------
from agent_tester.adapters.azure_adapter import AzureAIFoundryAdapter
from agent_tester.adapters.openai_adapter import OpenAIAdapter
from agent_tester.adapters.anthropic_adapter import AnthropicAdapter

# Azure
azure_adapter = AzureAIFoundryAdapter()
result = azure_adapter.execute_task(task)

# OpenAI
openai_adapter = OpenAIAdapter(model="gpt-4o-mini")
result = openai_adapter.execute_task(task)

# Anthropic
anthropic_adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
result = anthropic_adapter.execute_task(task)
"""

try:
    from agent_tester.adapters.azure_adapter import AzureAIFoundryAdapter
except ImportError:
    AzureAIFoundryAdapter = None

try:
    from agent_tester.adapters.openai_adapter import OpenAIAdapter
except ImportError:
    OpenAIAdapter = None

try:
    from agent_tester.adapters.anthropic_adapter import AnthropicAdapter
except ImportError:
    AnthropicAdapter = None

__all__ = [
    "AzureAIFoundryAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
]
