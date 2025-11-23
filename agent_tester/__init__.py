"""
AI Agent Testing Framework
===========================

A production-grade testing framework for AI agents that validates:
- Task Completion
- Trajectory Efficiency  
- Memory Consistency

As simple to use as Postman for APIs.
"""

__version__ = "0.1.0"
__author__ = "Ritik Kumar"

from agent_tester.models import (
    TaskDefinition,
    TaskStatus,
    TaskConstraint,
    Action,
    ActionType,
    Trajectory,
    AgentMemory,
    MemoryEntry,
)

from agent_tester.validators.task_validator import TaskValidator, TaskValidationResult
from agent_tester.validators.trajectory_validator import (
    TrajectoryValidator,
    TrajectoryValidationResult,
)
from agent_tester.validators.memory_validator import MemoryValidator, MemoryValidationResult

from agent_tester.suite import AgentTestSuite

# Security module
from agent_tester.security import (
    SecurityValidator,
    SecurityReporter,
    SecurityIssue,
    Severity,
    IssueCategory,
    SASTScanner,
    DependencyScanner,
    ConfigurationScanner,
    SecurityKnowledgeBase,
)

__all__ = [
    # Core models
    "TaskDefinition",
    "TaskStatus",
    "TaskConstraint",
    "Action",
    "ActionType",
    "Trajectory",
    "AgentMemory",
    "MemoryEntry",
    # Validators
    "TaskValidator",
    "TaskValidationResult",
    "TrajectoryValidator",
    "TrajectoryValidationResult",
    "MemoryValidator",
    "MemoryValidationResult",
    # Test suite
    "AgentTestSuite",
    # Security
    "SecurityValidator",
    "SecurityReporter",
    "SecurityIssue",
    "Severity",
    "IssueCategory",
    "SASTScanner",
    "DependencyScanner",
    "ConfigurationScanner",
    "SecurityKnowledgeBase",
    # Version
    "__version__",
]
