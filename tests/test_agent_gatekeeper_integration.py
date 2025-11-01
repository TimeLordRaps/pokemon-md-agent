"""Integration test for agent gatekeeper filtering."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agent.agent_core import AgentCore
from src.retrieval.ann_search import VectorSearch


@pytest.mark.asyncio
async def test_agent_gatekeeper_filters_actions():
    """Agent pipeline integrates gatekeeper to filter dangerous actions."""
    # Mock ANN search for gatekeeper
    ann_search = MagicMock(spec=VectorSearch)
    ann_search.search = AsyncMock(return_value=[{"id": 1, "score": 0.9}])

    # Create agent with gatekeeper
    agent = AgentCore(test_mode=True)

    # Replace gatekeeper with properly configured one
    agent.gatekeeper = agent.gatekeeper.__class__(ann_search, min_hits=3)

    # Mock Qwen to return actions including dangerous ones
    mock_response = "ATTACK | Fighting enemy\nUSE_ITEM | Using potion\nself-destruct | Bad idea"
    with patch.object(agent.qwen, 'generate_async', return_value=(mock_response, [])):
        # Create test state
        state = {
            "ascii": "Test dungeon with enemy",
            "ram": {"player_x": 10, "player_y": 10, "floor_number": 1}
        }

        # Reason with gatekeeper filtering
        decision = await agent.reason(state)

        # Gatekeeper should filter out self-destruct
        assert decision["action"] != "self-destruct"
        assert decision["action"] in ["ATTACK", "USE_ITEM", "WAIT"]


@pytest.mark.asyncio
async def test_agent_gatekeeper_insufficient_ann_hits():
    """Agent falls back when ANN hits insufficient."""
    # Mock ANN search with insufficient hits
    ann_search = MagicMock(spec=VectorSearch)
    ann_search.search = AsyncMock(return_value=[{"id": 1, "score": 0.5}])  # Only 1 hit

    # Create agent with gatekeeper requiring 3 hits
    agent = AgentCore(test_mode=True)
    agent.gatekeeper = agent.gatekeeper.__class__(ann_search, min_hits=3)

    # Mock Qwen response
    mock_response = "ATTACK | Fighting enemy"
    with patch.object(agent.qwen, 'generate_async', return_value=(mock_response, [])):
        state = {
            "ascii": "Test dungeon",
            "ram": {"player_x": 10, "player_y": 10, "floor_number": 1}
        }

        decision = await agent.reason(state)

        # Should fallback to random when insufficient hits
        assert decision["action"] == "random"
        assert "Gatekeeper filtered all actions" in decision["rationale"]