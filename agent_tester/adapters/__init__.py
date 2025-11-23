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

# Azure
azure_adapter = AzureAIFoundryAdapter()
result = azure_adapter.execute_task(task)

# OpenAI
openai_adapter = OpenAIAdapter(model="gpt-4o-mini")
result = openai_adapter.execute_task(task)
"""

try:
    from agent_tester.adapters.azure_adapter import AzureAIFoundryAdapter
except ImportError:
    AzureAIFoundryAdapter = None

try:
    from agent_tester.adapters.openai_adapter import OpenAIAdapter
except ImportError:
    OpenAIAdapter = None

__all__ = [
    "AzureAIFoundryAdapter",
    "OpenAIAdapter",
]
