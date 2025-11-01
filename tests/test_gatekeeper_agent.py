"""Test Gatekeeper filter method for agent actions."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agent.gatekeeper import Gatekeeper


@pytest.mark.asyncio
async def test_gatekeeper_filter_rejects_self_destruct():
    """Gatekeeper rejects self-destruct actions."""
    ann_search = MagicMock()
    ann_search.search_async = AsyncMock(return_value=[
        {"id": 1, "score": 0.9},
        {"id": 2, "score": 0.85},
        {"id": 3, "score": 0.8}
    ])

    gatekeeper = Gatekeeper(ann_search)

    valids = ["move", "attack", "self-destruct", "use_item"]
    state = {"game_context": "dungeon_floor_1"}

    filtered = await gatekeeper.filter(valids, state)

    assert "self-destruct" not in filtered
    assert "move" in filtered
    assert "attack" in filtered
    assert len(filtered) == 3  # self-destruct rejected