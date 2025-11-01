"""
Agent Gatekeeper - Filters invalid actions and validates via ANN search.

Requires ≥3 shallow ANN hits to permit actions. Rejects explicitly invalid actions
like self-destruct regardless of ANN results.
"""

import logging
from typing import Any, Dict, List

from ..retrieval.ann_search import VectorSearch  # TYPE_CHECKING

logger = logging.getLogger(__name__)


class Gatekeeper:
    """
    Filters agent actions based on safety rules and ANN validation.

    Rejects explicitly invalid actions (e.g., self-destruct) and validates
    remaining actions via ANN search requiring ≥ min_hits shallow hits.
    """

    def __init__(self, ann_search: "VectorSearch", min_hits: int = 3):
        """
        Initialize gatekeeper with ANN search dependency.

        Args:
            ann_search: ANN search instance for vector validation
            min_hits: Minimum ANN hits required to permit actions (default: 3)
        """
        self.ann_search = ann_search
        self.min_hits = min_hits
        self.invalid_actions = {
            "self-destruct",
            "suicide",
            "quit",
            "exit",
            "die",
            "end_game",
            "game_over"
        }

    async def filter(self, valids: List[str], state: Dict[str, Any]) -> List[str]:
        """
        Filter valid actions through safety rules and ANN validation.

        Args:
            valids: List of potentially valid actions
            state: Current game state for ANN query

        Returns:
            Filtered list of actions that pass validation
        """
        # Reject explicitly invalid actions
        filtered = [action for action in valids if action not in self.invalid_actions]

        # Validate remaining actions via ANN search
        if filtered:
            query_vector = self._state_to_vector(state)
            try:
                hits = await self.ann_search.search_async(query_vector, k=self.min_hits)
                if len(hits) >= self.min_hits:
                    return filtered
                else:
                    logger.warning(f"Insufficient ANN hits ({len(hits)}) for actions: {filtered}")
            except Exception as e:
                logger.error(f"ANN search failed during action filtering: {e}")
                # On ANN failure, conservatively reject all actions
                return []
        else:
            logger.info("All actions filtered out by safety rules")

        return []

    def _state_to_vector(self, state: Dict[str, Any]) -> List[float]:
        """
        Convert game state to vector for ANN search.

        Uses simple concatenation of key state values. In production,
        this should be replaced with proper embedding generation.

        Args:
            state: Game state dictionary

        Returns:
            Vector representation of state
        """
        # Simple vectorization - concatenate key state elements
        vector_parts = []

        # Position
        if "player_x" in state and "player_y" in state:
            vector_parts.extend([float(state["player_x"]), float(state["player_y"])])

        # HP and floor
        if "player_hp" in state:
            vector_parts.append(float(state.get("player_hp", 0)))
        if "floor_number" in state:
            vector_parts.append(float(state["floor_number"]))

        # ASCII state (simplified)
        if "ascii" in state:
            ascii_str = state["ascii"][:100]  # Limit to first 100 chars
            vector_parts.extend([float(ord(c)) for c in ascii_str])

        # Pad to minimum length if needed
        while len(vector_parts) < 10:
            vector_parts.append(0.0)

        return vector_parts[:100]  # Cap at reasonable length