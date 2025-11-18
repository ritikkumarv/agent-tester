"""
Core data models for the AI Agent Testing Framework
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ============================================================================
# TASK MODELS
# ============================================================================


class TaskStatus(Enum):
    """Task completion status"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class TaskConstraint(BaseModel):
    """Represents a constraint that the agent must satisfy"""

    name: str
    description: str
    constraint_type: str  # 'key_exists', 'value_equals', 'value_in_range', etc.
    expected_value: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def validate(self, output: Dict[str, Any]) -> bool:
        """
        Execute the validation logic safely without eval()

        Supported constraint types:
        - 'key_exists': Check if a key exists in output
        - 'value_equals': Check if output[name] equals expected_value
        - 'value_in_range': Check if min_value <= output[name] <= max_value
        - 'list_length': Check if len(output[name]) equals expected_value
        - 'not_empty': Check if output[name] is not empty
        """
        try:
            if self.constraint_type == "key_exists":
                return self.name in output

            elif self.constraint_type == "value_equals":
                return output.get(self.name) == self.expected_value

            elif self.constraint_type == "value_in_range":
                value = output.get(self.name)
                if value is None:
                    return False
                if self.min_value is not None and value < self.min_value:
                    return False
                if self.max_value is not None and value > self.max_value:
                    return False
                return True

            elif self.constraint_type == "list_length":
                value = output.get(self.name)
                if not isinstance(value, (list, tuple)):
                    return False
                return len(value) == self.expected_value

            elif self.constraint_type == "not_empty":
                value = output.get(self.name)
                if value is None:
                    return False
                if isinstance(value, (str, list, dict)):
                    return len(value) > 0
                return bool(value)

            else:
                # Unknown constraint type
                return False

        except Exception:
            return False


class TaskDefinition(BaseModel):
    """Defines a task for the agent to complete"""

    task_id: str
    goal: str
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    expected_output_schema: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 300
    max_retries: int = 3


# ============================================================================
# TRAJECTORY MODELS
# ============================================================================


class ActionType(Enum):
    """Types of actions an agent can take"""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    DECISION = "decision"


class Action(BaseModel):
    """Represents a single action in agent trajectory"""

    action_id: str
    action_type: ActionType
    tool_name: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0
    success: bool = True
    error: Optional[str] = None


class Trajectory(BaseModel):
    """Sequence of actions taken by agent"""

    trajectory_id: str
    task_id: str
    actions: List[Action] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_action(self, action: Action):
        """Add action to trajectory"""
        self.actions.append(action)

    def complete(self):
        """Mark trajectory as complete"""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get total trajectory duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


# ============================================================================
# MEMORY MODELS
# ============================================================================


class MemoryEntry(BaseModel):
    """Represents a single memory entry"""

    key: str
    value: Any
    timestamp: float = Field(default_factory=time.time)
    access_count: int = 0
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMemory(BaseModel):
    """Agent's memory store"""

    memory_id: str
    entries: Dict[str, MemoryEntry] = Field(default_factory=dict)
    max_size: int = 100

    def store(self, key: str, value: Any, relevance: float = 1.0):
        """Store information in memory"""
        self.entries[key] = MemoryEntry(key=key, value=value, relevance_score=relevance)

        # Evict if over capacity
        if len(self.entries) > self.max_size:
            self._evict_least_relevant()

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve from memory"""
        if key in self.entries:
            self.entries[key].access_count += 1
            return self.entries[key].value
        return None

    def _evict_least_relevant(self):
        """Remove least relevant/accessed entries"""
        if not self.entries:
            return

        # Score based on relevance and recency
        scored_entries = [
            (
                key,
                entry.relevance_score * (1 / (time.time() - entry.timestamp + 1)),
            )
            for key, entry in self.entries.items()
        ]

        scored_entries.sort(key=lambda x: x[1])
        # Remove bottom 10%
        to_remove = max(1, len(scored_entries) // 10)
        for key, _ in scored_entries[:to_remove]:
            del self.entries[key]
