"""Memory management for agent context allocation and scratchpad."""

from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class MemoryAllocation:
    """Configuration for memory allocation across temporal ranges."""
    
    def __init__(
        self,
        last_5_minutes: float = 0.75,
        last_30_minutes: float = 0.15,
        active_missions: float = 0.10,
    ):
        """Initialize memory allocation.
        
        Args:
            last_5_minutes: Percentage of context for last 5 minutes (0.0-1.0)
            last_30_minutes: Percentage for last 30 minutes (0.0-1.0)
            active_missions: Percentage for current mission context (0.0-1.0)
        """
        total = last_5_minutes + last_30_minutes + active_missions
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Memory allocation must sum to 1.0, got {total}")
        
        self.last_5_minutes = last_5_minutes
        self.last_30_minutes = last_30_minutes
        self.active_missions = active_missions


@dataclass
class ScratchpadEntry:
    """Entry in the agent's scratchpad."""
    content: str
    timestamp: float
    priority: int = 0  # Higher priority entries are kept longer


class Scratchpad:
    """Persistent scratchpad for agent to leave notes across interactions."""
    
    def __init__(self, max_entries: int = 100):
        """Initialize scratchpad.
        
        Args:
            max_entries: Maximum number of entries to store
        """
        self.max_entries = max_entries
        self.entries: list[ScratchpadEntry] = []
        self._current_time = 0.0
        
    def write(self, content: str, priority: int = 0) -> None:
        """Write a new entry to the scratchpad.
        
        Args:
            content: Content to write
            priority: Priority level (0=normal, 1=important, 2=critical)
        """
        entry = ScratchpadEntry(
            content=content,
            timestamp=self._current_time,
            priority=priority
        )
        self.entries.append(entry)
        
        # Trim if over capacity
        if len(self.entries) > self.max_entries:
            # Keep higher priority entries
            self.entries.sort(key=lambda e: (e.priority, e.timestamp), reverse=True)
            self.entries = self.entries[:self.max_entries]
        
        # Truncate content for logging
        content_preview = content[:50] + "..." if len(content) > 50 else content
        logger.debug("Added scratchpad entry: %s", content_preview)
        
    def read(self, limit: Optional[int] = None) -> list[str]:
        """Read all scratchpad entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of content strings, most recent first
        """
        entries = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)
        
        if limit is not None:
            entries = entries[:limit]
        
        return [entry.content for entry in entries]
    
    def read_with_metadata(self, limit: Optional[int] = None) -> list[ScratchpadEntry]:
        """Read all scratchpad entries with metadata.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of ScratchpadEntry objects, most recent first
        """
        entries = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)
        
        if limit is not None:
            entries = entries[:limit]
        
        return entries
    
    def clear(self) -> None:
        """Clear all scratchpad entries."""
        self.entries.clear()
        logger.debug("Cleared all scratchpad entries")
    
    def update_time(self, current_time: float) -> None:
        """Update the current time for timestamp calculations.
        
        Args:
            current_time: Current time in seconds
        """
        self._current_time = current_time


class MemoryManager:
    """Manages agent memory allocation across temporal ranges."""
    
    def __init__(
        self,
        total_context_budget: int = 256_000,
        allocation: Optional[MemoryAllocation] = None,
    ):
        """Initialize memory manager.
        
        Args:
            total_context_budget: Total tokens available for context
            allocation: Memory allocation configuration
        """
        self.total_context_budget = total_context_budget
        self.allocation = allocation or MemoryAllocation()
        self.scratchpad = Scratchpad()
        
    def allocate(self, allocation: Optional[MemoryAllocation] = None) -> Dict[str, int]:
        """Calculate token allocation across temporal ranges.
        
        Args:
            allocation: Optional override allocation configuration
            
        Returns:
            Dictionary mapping memory range to token count
        """
        alloc = allocation or self.allocation
        
        return {
            "last_5_minutes": int(self.total_context_budget * alloc.last_5_minutes),
            "last_30_minutes": int(self.total_context_budget * alloc.last_30_minutes),
            "active_missions": int(self.total_context_budget * alloc.active_missions),
        }
    
    def get_memory_budget(self, memory_type: str) -> int:
        """Get token budget for a specific memory type.
        
        Args:
            memory_type: Type of memory ("last_5_minutes", "last_30_minutes", "active_missions")
            
        Returns:
            Token budget for the memory type
        """
        budgets = self.allocate()
        return budgets.get(memory_type, 0)
    
    def update_allocation(
        self,
        last_5_minutes: Optional[float] = None,
        last_30_minutes: Optional[float] = None,
        active_missions: Optional[float] = None,
    ) -> None:
        """Update memory allocation configuration.
        
        Args:
            last_5_minutes: New percentage for last 5 minutes
            last_30_minutes: New percentage for last 30 minutes
            active_missions: New percentage for active missions
            
        Raises:
            ValueError: If percentages don't sum to 1.0
        """
        new_allocation = MemoryAllocation(
            last_5_minutes=last_5_minutes or self.allocation.last_5_minutes,
            last_30_minutes=last_30_minutes or self.allocation.last_30_minutes,
            active_missions=active_missions or self.allocation.active_missions,
        )
        self.allocation = new_allocation
        
        logger.info(
            "Updated memory allocation: 5min=%.1f%%, 30min=%.1f%%, missions=%.1f%%",
            new_allocation.last_5_minutes * 100,
            new_allocation.last_30_minutes * 100,
            new_allocation.active_missions * 100
        )
