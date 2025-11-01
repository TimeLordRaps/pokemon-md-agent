"""Tests for game reset and checkpoint recovery system."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.environment.game_reset_handler import (
    GameResetConfig,
    GameResetHandler,
    GameResetMode,
    GameState,
)


class TestGameResetMode:
    """Test GameResetMode enum."""

    def test_fresh_rom_mode(self) -> None:
        """Test FRESH_ROM mode value."""
        assert GameResetMode.FRESH_ROM.value == "fresh_rom"

    def test_checkpoint_load_mode(self) -> None:
        """Test CHECKPOINT_LOAD mode value."""
        assert GameResetMode.CHECKPOINT_LOAD.value == "checkpoint_load"

    def test_soft_reset_mode(self) -> None:
        """Test SOFT_RESET mode value."""
        assert GameResetMode.SOFT_RESET.value == "soft_reset"

    def test_title_screen_mode(self) -> None:
        """Test TITLE_SCREEN mode value."""
        assert GameResetMode.TITLE_SCREEN.value == "title_screen"


class TestGameResetConfig:
    """Test GameResetConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = GameResetConfig()
        assert config.reset_mode == GameResetMode.TITLE_SCREEN
        assert config.checkpoint_file is None
        assert config.rom_path is None
        assert config.auto_start_game is False

    def test_config_with_checkpoint(self) -> None:
        """Test configuration with checkpoint."""
        checkpoint_path = Path("save.ss0")
        config = GameResetConfig(
            reset_mode=GameResetMode.CHECKPOINT_LOAD,
            checkpoint_file=checkpoint_path
        )
        assert config.reset_mode == GameResetMode.CHECKPOINT_LOAD
        assert config.checkpoint_file == checkpoint_path

    def test_custom_button_sequence(self) -> None:
        """Test custom soft reset button sequence."""
        custom_buttons = ["Start", "Down", "A"]
        config = GameResetConfig(soft_reset_button_sequence=custom_buttons)
        assert config.soft_reset_button_sequence == custom_buttons

    def test_default_button_sequence(self) -> None:
        """Test default soft reset button sequence."""
        config = GameResetConfig()
        assert config.soft_reset_button_sequence is not None
        assert "Start" in config.soft_reset_button_sequence
        assert "A" in config.soft_reset_button_sequence


class TestGameState:
    """Test GameState dataclass."""

    def test_initial_state(self) -> None:
        """Test initial game state."""
        state = GameState()
        assert state.is_at_title_screen is False
        assert state.is_in_game is False
        assert state.current_hp is None
        assert state.dungeon_level is None
        assert state.reset_detected is False

    def test_state_with_values(self) -> None:
        """Test game state with values."""
        state = GameState(
            is_at_title_screen=True,
            is_in_game=False,
            current_hp=100,
            dungeon_level=3,
            position=(10, 20),
            frame_count=5000
        )
        assert state.is_at_title_screen is True
        assert state.current_hp == 100
        assert state.dungeon_level == 3
        assert state.position == (10, 20)


class TestGameResetHandler:
    """Test GameResetHandler class."""

    @pytest.fixture
    def mock_controller(self) -> Mock:
        """Create mock controller."""
        controller = Mock()
        controller.button_tap = Mock()
        controller.load_save = Mock()
        return controller

    @pytest.fixture
    def handler(self, mock_controller: Mock) -> GameResetHandler:
        """Create handler with mock controller."""
        config = GameResetConfig()
        return GameResetHandler(controller=mock_controller, config=config)

    def test_handler_initialization(self, handler: GameResetHandler) -> None:
        """Test handler initialization."""
        assert handler.controller is not None
        assert handler.config is not None
        assert handler.current_state is not None
        assert len(handler.reset_history) == 0

    def test_handler_without_controller(self) -> None:
        """Test handler without controller."""
        handler = GameResetHandler(controller=None)
        assert handler.controller is None
        assert handler.current_state is not None

    def test_update_game_state(self, handler: GameResetHandler) -> None:
        """Test updating game state."""
        handler.update_game_state(
            is_at_title=True,
            is_in_game=False,
            hp=100,
            dungeon_level=3,
            position=(10, 20),
            frame_count=5000,
            action="move_down"
        )

        assert handler.current_state.is_at_title_screen is True
        assert handler.current_state.is_in_game is False
        assert handler.current_state.current_hp == 100
        assert handler.current_state.dungeon_level == 3
        assert handler.current_state.position == (10, 20)
        assert handler.current_state.frame_count == 5000
        assert handler.current_state.last_action == "move_down"

    def test_detect_reset_via_frame_delta(self, handler: GameResetHandler) -> None:
        """Test detecting reset via frame count decrease."""
        result = handler.detect_reset(frame_delta=-5000)
        assert result is True
        assert handler.current_state.reset_detected is True

    def test_detect_no_reset_normal_delta(self, handler: GameResetHandler) -> None:
        """Test that normal frame deltas don't trigger reset detection."""
        result = handler.detect_reset(frame_delta=100)
        assert result is False

    def test_detect_reset_unexpected_title(self, handler: GameResetHandler) -> None:
        """Test detecting reset from unexpected title screen."""
        # Simulate being in game with an action
        handler.update_game_state(is_in_game=True, action="move_down")

        # Then detect at title with last action set
        handler.update_game_state(
            is_at_title=True,
            is_in_game=False,
            action="move_down"  # Last action still set
        )

        result = handler.detect_reset(frame_delta=10)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_title_screen_success(self, handler: GameResetHandler) -> None:
        """Test waiting for title screen successfully."""
        # Set state to at title after small delay
        async def set_title():
            await asyncio.sleep(0.1)
            handler.update_game_state(is_at_title=True)

        asyncio.create_task(set_title())
        result = await handler.wait_for_title_screen(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_title_screen_timeout(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test waiting for title screen timeout."""
        handler.update_game_state(is_at_title=False)
        result = await handler.wait_for_title_screen(timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_reset_to_title_screen_success(
        self,
        mock_controller: Mock,
        handler: GameResetHandler
    ) -> None:
        """Test resetting to title screen."""
        # Mock wait_for_title_screen to succeed
        with patch.object(
            handler,
            'wait_for_title_screen',
            new_callable=AsyncMock,
            return_value=True
        ):
            result = await handler.reset_to_title_screen()

        assert result is True
        assert handler.current_state.is_in_game is False
        assert len(handler.reset_history) == 1
        assert handler.reset_history[0]["mode"] == "soft_reset"

    @pytest.mark.asyncio
    async def test_reset_to_title_screen_no_controller(self) -> None:
        """Test reset fails without controller."""
        handler = GameResetHandler(controller=None)
        result = await handler.reset_to_title_screen()
        assert result is False

    @pytest.mark.asyncio
    async def test_reset_in_progress_flag(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test reset in progress flag prevents concurrent resets."""
        handler._reset_in_progress = True
        result = await handler.reset_to_title_screen()
        assert result is False

    @pytest.mark.asyncio
    async def test_load_checkpoint_success(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "save.ss0"
            checkpoint_path.write_text("dummy save data")

            config = GameResetConfig(
                reset_mode=GameResetMode.CHECKPOINT_LOAD,
                checkpoint_file=checkpoint_path
            )
            handler.config = config

            result = await handler.load_checkpoint(checkpoint_path)

        assert result is True
        assert handler.current_state.is_in_game is True
        assert len(handler.reset_history) == 1

    @pytest.mark.asyncio
    async def test_load_checkpoint_not_found(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test loading nonexistent checkpoint."""
        nonexistent = Path("/nonexistent/save.ss0")
        result = await handler.load_checkpoint(nonexistent)
        assert result is False

    @pytest.mark.asyncio
    async def test_load_checkpoint_no_controller(self) -> None:
        """Test checkpoint load fails without controller."""
        handler = GameResetHandler(controller=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "save.ss0"
            checkpoint_path.write_text("dummy")
            result = await handler.load_checkpoint(checkpoint_path)

        assert result is False

    @pytest.mark.asyncio
    async def test_start_new_game(self, handler: GameResetHandler) -> None:
        """Test starting a new game."""
        handler.update_game_state(is_at_title=True)

        result = await handler.start_new_game(timeout=0.1)

        assert result is True
        assert handler.current_state.is_in_game is True
        handler.controller.button_tap.assert_called()

    @pytest.mark.asyncio
    async def test_start_new_game_not_at_title(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test starting game fails if not at title screen."""
        handler.update_game_state(is_at_title=False)
        result = await handler.start_new_game()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_new_game_no_controller(self) -> None:
        """Test starting game fails without controller."""
        handler = GameResetHandler(controller=None)
        handler.update_game_state(is_at_title=True)
        result = await handler.start_new_game()
        assert result is False

    @pytest.mark.asyncio
    async def test_perform_reset_title_screen_mode(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test perform_reset with TITLE_SCREEN mode."""
        with patch.object(
            handler,
            'reset_to_title_screen',
            new_callable=AsyncMock,
            return_value=True
        ):
            result = await handler.perform_reset(GameResetMode.TITLE_SCREEN)

        assert result is True

    @pytest.mark.asyncio
    async def test_perform_reset_checkpoint_mode(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test perform_reset with CHECKPOINT_LOAD mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "save.ss0"
            checkpoint_path.write_text("dummy")

            handler.config = GameResetConfig(
                reset_mode=GameResetMode.CHECKPOINT_LOAD,
                checkpoint_file=checkpoint_path
            )

            with patch.object(
                handler,
                'load_checkpoint',
                new_callable=AsyncMock,
                return_value=True
            ):
                result = await handler.perform_reset()

        assert result is True

    @pytest.mark.asyncio
    async def test_perform_reset_with_auto_start(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test perform_reset with auto_start_game enabled."""
        handler.config = GameResetConfig(
            reset_mode=GameResetMode.TITLE_SCREEN,
            auto_start_game=True
        )

        with patch.object(
            handler,
            'reset_to_title_screen',
            new_callable=AsyncMock,
            return_value=True
        ):
            with patch.object(
                handler,
                'start_new_game',
                new_callable=AsyncMock,
                return_value=True
            ):
                result = await handler.perform_reset()

        assert result is True

    @pytest.mark.asyncio
    async def test_perform_reset_missing_checkpoint(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test perform_reset fails if checkpoint required but not specified."""
        handler.config = GameResetConfig(
            reset_mode=GameResetMode.CHECKPOINT_LOAD,
            checkpoint_file=None
        )

        result = await handler.perform_reset()
        assert result is False

    def test_get_reset_status(self, handler: GameResetHandler) -> None:
        """Test getting reset status."""
        handler.update_game_state(
            is_at_title=True,
            is_in_game=False,
            hp=100,
            dungeon_level=2,
            frame_count=3000
        )

        handler.reset_history.append({"mode": "soft_reset", "success": True})

        status = handler.get_reset_status()

        assert status["current_state"]["at_title_screen"] is True
        assert status["current_state"]["in_game"] is False
        assert status["current_state"]["hp"] == 100
        assert status["current_state"]["dungeon_level"] == 2
        assert status["total_resets"] == 1
        assert status["reset_mode"] == "title_screen"

    @pytest.mark.asyncio
    async def test_recovery_flow_success(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test full recovery flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "save.ss0"
            checkpoint_path.write_text("dummy")

            handler.config = GameResetConfig(
                reset_mode=GameResetMode.CHECKPOINT_LOAD,
                checkpoint_file=checkpoint_path
            )

            with patch.object(
                handler,
                'reset_to_title_screen',
                new_callable=AsyncMock,
                return_value=True
            ):
                with patch.object(
                    handler,
                    'load_checkpoint',
                    new_callable=AsyncMock,
                    return_value=True
                ):
                    result = await handler.recovery_flow()

        assert result is True

    @pytest.mark.asyncio
    async def test_recovery_flow_reset_fails(
        self,
        handler: GameResetHandler
    ) -> None:
        """Test recovery flow fails if reset fails."""
        with patch.object(
            handler,
            'reset_to_title_screen',
            new_callable=AsyncMock,
            return_value=False
        ):
            result = await handler.recovery_flow()

        assert result is False

    def test_button_sequence_execution(
        self,
        mock_controller: Mock,
        handler: GameResetHandler
    ) -> None:
        """Test that soft reset executes correct button sequence."""
        handler.config = GameResetConfig(
            soft_reset_button_sequence=["Start", "A", "B"]
        )

        # Mock wait_for_title_screen to succeed
        with patch.object(
            handler,
            'wait_for_title_screen',
            new_callable=AsyncMock,
            return_value=True
        ):
            asyncio.run(handler.reset_to_title_screen())

        # Check each button was tapped
        calls = mock_controller.button_tap.call_args_list
        assert len(calls) >= 3
        assert calls[0][0][0] == "Start"
        assert calls[1][0][0] == "A"
        assert calls[2][0][0] == "B"
