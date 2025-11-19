"""
Memory Validator - Tests agent memory management
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field

from agent_tester.models import AgentMemory

logger = logging.getLogger(__name__)


class MemoryValidationResult(BaseModel):
    """Results from memory validation"""

    memory_id: str
    passed: bool
    context_retention_score: float  # 0-100
    consistency_score: float  # 0-100
    relevance_score: float  # 0-100
    within_capacity: bool
    memory_usage: int
    max_capacity: int
    issues: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MemoryValidator:
    """
    Validates agent memory management

    Example Usage:
        validator = MemoryValidator(min_retention_score=0.8)
        memory = agent.get_memory()
        conversation = agent.get_conversation_history()
        result = validator.validate(memory, conversation)
    """

    def __init__(
        self, min_retention_score: float = 0.7, min_relevance_score: float = 0.6
    ):
        self.min_retention_score = min_retention_score
        self.min_relevance_score = min_relevance_score
        self.validation_history: List[MemoryValidationResult] = []

    def validate(
        self, memory: AgentMemory, conversation_history: List[Dict[str, str]]
    ) -> MemoryValidationResult:
        """
        Validate agent memory

        Args:
            memory: Agent's memory state
            conversation_history: Conversation history to check against

        Returns:
            MemoryValidationResult with detailed analysis
        """
        issues = []

        # 1. Check context retention
        retention_score = self._check_context_retention(memory, conversation_history)
        if retention_score < self.min_retention_score:
            issues.append(
                f"Low context retention: {retention_score:.2%} < {self.min_retention_score:.2%}"
            )

        # 2. Check consistency
        consistency_score = self._check_consistency(memory)
        if consistency_score < 0.9:
            issues.append(f"Memory inconsistencies detected: {consistency_score:.2%}")

        # 3. Check relevance
        relevance_score = self._check_relevance(memory)
        if relevance_score < self.min_relevance_score:
            issues.append(
                f"Too much irrelevant data: {relevance_score:.2%} < {self.min_relevance_score:.2%}"
            )

        # 4. Check capacity
        within_capacity = len(memory.entries) <= memory.max_size
        if not within_capacity:
            issues.append(
                f"Memory overflow: {len(memory.entries)} > {memory.max_size}"
            )

        # Determine if passed
        passed = (
            retention_score >= self.min_retention_score
            and consistency_score >= 0.9
            and relevance_score >= self.min_relevance_score
            and within_capacity
        )

        result = MemoryValidationResult(
            memory_id=memory.memory_id,
            passed=passed,
            context_retention_score=retention_score * 100,
            consistency_score=consistency_score * 100,
            relevance_score=relevance_score * 100,
            within_capacity=within_capacity,
            memory_usage=len(memory.entries),
            max_capacity=memory.max_size,
            issues=issues,
        )

        self.validation_history.append(result)
        return result

    def _check_context_retention(
        self, memory: AgentMemory, conversation: List[Dict[str, str]]
    ) -> float:
        """Check if important context is retained"""
        if not conversation:
            return 1.0

        # Extract key entities/facts from conversation
        key_facts = self._extract_key_facts(conversation)

        if not key_facts:
            return 1.0

        # Check how many are in memory
        # Use fuzzy matching - check if key words from facts appear in memory values
        retained = 0
        for fact in key_facts:
            fact_lower = fact.lower()
            # Extract key words from fact (names, numbers, important terms)
            fact_words = [w for w in fact_lower.split() if len(w) > 3 or w.isdigit()]
            
            # Check if any significant word from the fact appears in memory
            for entry in memory.entries.values():
                value_str = str(entry.value).lower()
                if any(word in value_str or value_str in fact_lower for word in fact_words):
                    retained += 1
                    break

        return retained / len(key_facts)

    def _extract_key_facts(self, conversation: List[Dict[str, str]]) -> List[str]:
        """Extract important facts from conversation (simplified)"""
        facts = []

        # Look for user preferences, names, numbers, etc.
        keywords = ["prefer", "like", "want", "need", "name is", "budget"]

        for turn in conversation:
            if turn.get("role") == "user":
                content = turn.get("content", "").lower()
                if any(kw in content for kw in keywords):
                    facts.append(content)

        return facts

    def _check_consistency(self, memory: AgentMemory) -> float:
        """Check for contradictions in memory"""
        # Simplified: check for actual duplicate keys or obvious contradictions
        contradictions = 0
        total_checks = 0

        # Check for semantic contradictions (simplified)
        entries_list = list(memory.entries.values())
        for i, entry1 in enumerate(entries_list):
            for entry2 in entries_list[i + 1 :]:
                total_checks += 1
                # Check if exact same key with different values
                if (
                    entry1.key == entry2.key
                    and entry1.value != entry2.value
                ):
                    # Same key, different values could be legitimate updates
                    # Check timestamps - if very close together, might be contradiction
                    if abs(entry1.timestamp - entry2.timestamp) < 5:
                        contradictions += 1

        if total_checks == 0:
            return 1.0

        return 1.0 - (contradictions / total_checks)

    def _check_relevance(self, memory: AgentMemory) -> float:
        """Check proportion of relevant memories"""
        if not memory.entries:
            return 1.0

        relevant_count = sum(
            1
            for entry in memory.entries.values()
            if entry.relevance_score >= self.min_relevance_score
        )

        return relevant_count / len(memory.entries)
