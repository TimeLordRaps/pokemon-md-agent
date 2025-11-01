"""Test feedback loop analyzers."""

import pytest
import tempfile
import json
from pathlib import Path
from src.feedback.screenshot_analyzer import ScreenshotAnalyzer, SpriteDetectionResult
from src.feedback.ensemble_analyzer import EnsembleConsensusAnalyzer, ModelOutput
from src.feedback.context_analyzer import ContextWindowAnalyzer, ContextSnapshot
from src.feedback.skill_analyzer import SkillFormationAnalyzer, SkillDiscovery
from src.feedback.metrics_analyzer import PerformanceMetricsAnalyzer, GameMetrics
from src.feedback.enhanced_logging import EnhancedLogger


class TestScreenshotAnalyzer:
    """Test screenshot analyzer functionality."""

    def test_screenshot_analyzer_init(self):
        """Test screenshot analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ScreenshotAnalyzer(log_dir=Path(tmpdir))
            assert analyzer.logger is not None
            assert len(analyzer.frames) == 0

    def test_add_sprite_detection(self):
        """Test adding sprite detection results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ScreenshotAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_sprite_detection(
                frame_number=1,
                screenshot_hash="abc123",
                sprites_detected=5,
                confidence_scores=[0.9, 0.85, 0.8, 0.75, 0.7]
            )

            assert len(analyzer.frames) == 1
            assert analyzer.frames[0].frame_number == 1
            assert analyzer.frames[0].sprite_detection.sprites_detected == 5
            assert analyzer.frames[0].sprite_detection.mean_confidence == pytest.approx(0.8, abs=0.01)

    def test_screenshot_statistics(self):
        """Test computing screenshot statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ScreenshotAnalyzer(log_dir=Path(tmpdir))

            for i in range(5):
                analyzer.add_sprite_detection(
                    frame_number=i,
                    screenshot_hash=f"hash{i}",
                    sprites_detected=3 + i,
                    confidence_scores=[0.8 + 0.01*j for j in range(3+i)]
                )

            stats = analyzer.compute_statistics()
            assert stats["total_frames_analyzed"] == 5
            assert stats["sprite_detection"]["total_frames"] == 5

    def test_screenshot_save_analysis(self):
        """Test saving screenshot analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ScreenshotAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_sprite_detection(
                frame_number=1,
                screenshot_hash="abc123",
                sprites_detected=3,
                confidence_scores=[0.9, 0.85, 0.8]
            )

            output_path = analyzer.save_analysis()
            assert output_path.exists()

            # Check file format
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                data = json.loads(lines[0])
                assert data["sprite_detection"]["sprites_detected"] == 3


class TestEnsembleConsensusAnalyzer:
    """Test ensemble consensus analyzer functionality."""

    def test_ensemble_analyzer_init(self):
        """Test ensemble analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EnsembleConsensusAnalyzer(log_dir=Path(tmpdir))
            assert analyzer.logger is not None
            assert len(analyzer.model_outputs) == 0

    def test_add_model_output(self):
        """Test adding model outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EnsembleConsensusAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_model_output(
                frame_number=1,
                model_name="model_1",
                model_variant="qwen3-vl-2b-instruct",
                prediction="move_left",
                confidence=0.9
            )

            assert 1 in analyzer.model_outputs
            assert len(analyzer.model_outputs[1]) == 1
            assert analyzer.model_outputs[1][0].prediction == "move_left"

    def test_compute_consensus(self):
        """Test computing ensemble consensus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EnsembleConsensusAnalyzer(log_dir=Path(tmpdir))

            # Add 3 model outputs for frame 1
            for i in range(3):
                analyzer.add_model_output(
                    frame_number=1,
                    model_name=f"model_{i}",
                    model_variant="qwen3-vl-2b-instruct",
                    prediction="move_left" if i < 2 else "move_right",
                    confidence=0.9
                )

            decision = analyzer.compute_consensus(
                frame_number=1,
                winning_action="move_left"
            )

            assert decision is not None
            assert decision.winning_action == "move_left"
            assert decision.consensus_strength == 2/3
            assert not decision.unanimous

    def test_ensemble_statistics(self):
        """Test computing ensemble statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EnsembleConsensusAnalyzer(log_dir=Path(tmpdir))

            for frame in range(3):
                for i in range(3):
                    analyzer.add_model_output(
                        frame_number=frame,
                        model_name=f"model_{i}",
                        model_variant="qwen3-vl-2b-instruct",
                        prediction="action_a",
                        confidence=0.9
                    )
                analyzer.compute_consensus(frame, "action_a")

            stats = analyzer.compute_statistics()
            assert stats["total_frames"] == 3
            assert stats["mean_consensus_strength"] == 1.0  # All unanimous


class TestContextWindowAnalyzer:
    """Test context window analyzer functionality."""

    def test_context_analyzer_init(self):
        """Test context analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ContextWindowAnalyzer(log_dir=Path(tmpdir))
            assert analyzer.logger is not None
            assert len(analyzer.snapshots) == 0

    def test_add_context_snapshot(self):
        """Test adding context snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ContextWindowAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_context_snapshot(
                frame_number=1,
                context_length=1000,
                memory_silos_used=3,
                silo_distribution={"silo_0": 400, "silo_1": 300, "silo_2": 300},
                game_state_tokens=100,
                retrieval_tokens=500,
                instruction_tokens=400,
                entities_in_context=["entity_1", "entity_2"]
            )

            assert 1 in analyzer.snapshots
            assert analyzer.snapshots[1].context_length == 1000

    def test_context_statistics(self):
        """Test computing context statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ContextWindowAnalyzer(log_dir=Path(tmpdir))

            for i in range(3):
                analyzer.add_context_snapshot(
                    frame_number=i,
                    context_length=1000 + 100*i,
                    memory_silos_used=3,
                    silo_distribution={"silo_0": 500, "silo_1": 300, "silo_2": 200},
                    game_state_tokens=100,
                    retrieval_tokens=500,
                    instruction_tokens=400
                )

            stats = analyzer.compute_statistics()
            assert stats["total_snapshots"] == 3
            assert stats["context_length"]["mean"] > 1000


class TestSkillFormationAnalyzer:
    """Test skill formation analyzer functionality."""

    def test_skill_analyzer_init(self):
        """Test skill analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SkillFormationAnalyzer(log_dir=Path(tmpdir))
            assert analyzer.logger is not None
            assert len(analyzer.skill_discoveries) == 0

    def test_add_skill_discovery(self):
        """Test adding skill discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SkillFormationAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_skill_discovery(
                skill_id="skill_move_left",
                skill_name="Move Left",
                actions_sequence=["press_left"],
                frames_range=(0, 10),
                context_tags=["dungeon"],
                related_entities=["player"]
            )

            assert "skill_move_left" in analyzer.skill_discoveries
            discovery = analyzer.skill_discoveries["skill_move_left"]
            assert discovery.skill_name == "Move Left"

    def test_skill_execution(self):
        """Test recording skill execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SkillFormationAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_skill_discovery(
                skill_id="skill_move",
                skill_name="Move",
                actions_sequence=["press_left"],
                frames_range=(0, 10)
            )

            analyzer.add_skill_execution(
                frame_number=5,
                skill_id="skill_move",
                skill_name="Move",
                success=True,
                predicted_outcome="player_moved",
                actual_outcome="player_moved"
            )

            assert len(analyzer.skill_executions) == 1
            assert analyzer.skill_executions[0].success


class TestPerformanceMetricsAnalyzer:
    """Test performance metrics analyzer functionality."""

    def test_metrics_analyzer_init(self):
        """Test metrics analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = PerformanceMetricsAnalyzer(log_dir=Path(tmpdir))
            assert analyzer.logger is not None
            assert len(analyzer.frame_metrics) == 0

    def test_add_frame_metrics(self):
        """Test adding frame metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = PerformanceMetricsAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_frame_metrics(
                frame_number=1,
                dungeon_floor=1,
                player_hp=50,
                max_hp=100,
                player_level=1,
                experience_points=100,
                inventory_items=5,
                money=1000,
                total_damage_dealt=50,
                total_damage_taken=10,
                enemies_defeated=2
            )

            assert 1 in analyzer.frame_metrics
            assert analyzer.frame_metrics[1].player_hp == 50

    def test_add_run_summary(self):
        """Test adding run summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = PerformanceMetricsAnalyzer(log_dir=Path(tmpdir))

            analyzer.add_run_summary(
                run_id="run_001",
                duration_frames=1000,
                max_floor_reached=5,
                final_level=10,
                total_experience=5000,
                enemies_defeated=50,
                traps_avoided=10,
                total_damage_dealt=500,
                total_damage_taken=100,
                final_status="success"
            )

            assert "run_001" in analyzer.run_summaries
            assert analyzer.run_summaries["run_001"].final_status == "success"


class TestEnhancedLogger:
    """Test enhanced logger with feedback integration."""

    def test_enhanced_logger_init(self):
        """Test enhanced logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EnhancedLogger(
                logger_name="test",
                log_dir=Path(tmpdir),
                enable_feedback_logging=True
            )

            assert logger.screenshot_analyzer is not None
            assert logger.ensemble_analyzer is not None
            assert logger.session_id is not None

    def test_enhanced_logger_sprite_logging(self):
        """Test logging sprite detection through enhanced logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EnhancedLogger(
                logger_name="test",
                log_dir=Path(tmpdir)
            )

            logger.log_sprite_detection(
                frame_number=1,
                screenshot_hash="abc123",
                sprites_detected=3,
                confidence_scores=[0.9, 0.85, 0.8]
            )

            assert logger.event_count == 1
            assert len(logger.screenshot_analyzer.frames) == 1

    def test_enhanced_logger_save_analyses(self):
        """Test saving all analyses through enhanced logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EnhancedLogger(
                logger_name="test",
                log_dir=Path(tmpdir)
            )

            logger.log_sprite_detection(
                frame_number=1,
                screenshot_hash="abc123",
                sprites_detected=3,
                confidence_scores=[0.9, 0.85, 0.8]
            )

            logger.log_model_output(
                frame_number=1,
                model_name="model_1",
                model_variant="qwen3-vl-2b",
                prediction="move_left",
                confidence=0.9
            )

            paths = logger.save_all_analyses()
            assert "screenshot" in paths
            assert "ensemble" in paths
            assert paths["screenshot"].exists()

    def test_enhanced_logger_session_summary(self):
        """Test getting session summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EnhancedLogger(
                logger_name="test",
                log_dir=Path(tmpdir)
            )

            logger.log_frame_metrics(
                frame_number=1,
                dungeon_floor=1,
                player_hp=50,
                max_hp=100,
                player_level=1,
                experience_points=100,
                inventory_items=5,
                money=1000,
                total_damage_dealt=50,
                total_damage_taken=10
            )

            summary = logger.get_session_summary()
            assert summary["session_id"] is not None
            assert summary["frame_count"] == 1
            assert summary["event_count"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
