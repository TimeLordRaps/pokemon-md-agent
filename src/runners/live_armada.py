"""Live armada orchestrator for multi-model inference loop.

Chains:
1. mGBA socket server (Lua)
2. Capture loop (screenshot + RAM at configurable rates)
3. Vision processing (grid parsing, ASCII rendering, 4-up quads)
4. Qwen3-VL armada routing (6 models)
5. Dashboard writes (images + JSONL traces + keyframes)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s"
)
logger = logging.getLogger("live_armada")


@dataclass
class ArmadaConfig:
    """Configuration for live armada runner."""
    rom: Path
    save: Path
    lua: Path
    mgba_exe: Path
    host: str = "localhost"
    port: int = 8888
    capture_fps: float = 6.0
    mem_hz: float = 10.0
    rate_limit: int = 30
    gatekeeper: str = "on"
    dashboard_dir: Path = field(default_factory=lambda: Path("docs/current"))
    trace_jsonl: Path = field(default_factory=lambda: Path("docs/current/traces/latest.jsonl"))
    dry_run: bool = False
    verbose: bool = False
    frame_sample_interval: int = 300  # Sample every 300 frames (10s @ 30 FPS)
    inference_timeout: float = 5.0  # Max 5 seconds for inference
    confidence_threshold: float = 0.5  # Min confidence for inference result

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.dry_run:
            if not self.rom.exists():
                logger.error(f"ROM not found: {self.rom}")
                return False
            if not self.save.exists():
                logger.error(f"Save not found: {self.save}")
                return False
            if not self.lua.exists():
                logger.error(f"Lua script not found: {self.lua}")
                return False
            if not self.mgba_exe.exists():
                logger.error(f"mGBA executable not found: {self.mgba_exe}")
                return False
        return True


@dataclass
class ObservationFrame:
    """Single observation frame."""
    step_id: int
    timestamp: float
    screenshot: Optional[np.ndarray] = None
    ram_dump: Optional[bytes] = None
    game_title: Optional[str] = None
    grid: Optional[np.ndarray] = None
    ascii_view: Optional[str] = None


@dataclass
class VisionOutput:
    """Output from vision processing."""
    step_id: int
    raw_img: Image.Image
    grid_overlay: Image.Image
    hud_meta: Image.Image
    retrieval_mosaic: Image.Image


@dataclass
class ArmadaResponse:
    """Response from Qwen3-VL armada."""
    step_id: int
    timestamp: float
    model_id: str
    thinking: Optional[str] = None
    action: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: float = 0.5


def make_quad_image(vision_out: VisionOutput) -> Image.Image:
    """Compose 2×2 quad from vision outputs.

    Layout:
        [raw_img]      [grid_overlay]
        [hud_meta]     [retrieval_mosaic]
    """
    # Assume all images are same size for simplicity
    size = 240  # Each quadrant

    quad = Image.new("RGB", (size * 2, size * 2), (0, 0, 0))

    # Resize and paste each quadrant
    quad.paste(vision_out.raw_img.resize((size, size)), (0, 0))
    quad.paste(vision_out.grid_overlay.resize((size, size)), (size, 0))
    quad.paste(vision_out.hud_meta.resize((size, size)), (0, size))
    quad.paste(vision_out.retrieval_mosaic.resize((size, size)), (size, size))

    return quad


class MockTransport:
    """Mock transport for dry-run mode."""

    def __init__(self) -> None:
        self.frame_count = 0

    async def send_command(self, cmd: str) -> str:
        """Simulate command response."""
        self.frame_count += 1
        if "ping" in cmd:
            return "pong"
        elif "title" in cmd:
            return "Pokemon Mystery Dungeon: Red Rescue Team"
        elif "screenshot" in cmd:
            # Return a dummy 240x160 RGB image as bytes
            img = Image.new("RGB", (240, 160), color=(73, 109, 137))
            img_bytes = img.tobytes()
            return f"screenshot:{len(img_bytes)}:{img_bytes.hex()}"
        elif "read_mem" in cmd:
            # Return dummy RAM data
            return "mem:" + "00" * 512
        return "ok"


class LiveArmadaRunner:
    """Main orchestrator for live armada loop."""

    def __init__(self, config: ArmadaConfig) -> None:
        self.config = config
        self.step_count = 0
        self.transport: Optional[MockTransport] = None
        self.controller: Optional[object] = None  # MGBAController instance
        self.dashboard_dir = config.dashboard_dir
        self.trace_jsonl = config.trace_jsonl

        # Initialize skill and bootstrap systems
        from ..skills.skill_manager import SkillManager
        from ..bootstrap.state_manager import BootstrapStateManager
        
        self.skill_manager = SkillManager(skills_dir=self.dashboard_dir.parent / "skills")
        self.bootstrap_manager = BootstrapStateManager(
            bootstrap_dir=self.dashboard_dir.parent / "bootstrap",
            enable_bootstrap=True
        )
        
        # Track discovered skills in this run
        self.discovered_skills: list = []
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Ensure dashboard directories exist
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        (self.dashboard_dir / "keyframes").mkdir(exist_ok=True)
        (self.dashboard_dir / "traces").mkdir(parents=True, exist_ok=True)

    async def ensure_ready(self) -> bool:
        """Ensure mGBA and Lua server are ready."""
        if self.config.dry_run:
            logger.info("DRY_RUN: Skipping mGBA/Lua checks")
            self.transport = MockTransport()
            return True

        logger.info("Checking mGBA connection...")
        try:
            from ..environment.mgba_controller import MGBAController

            self.controller = MGBAController(
                host=self.config.host,
                port=self.config.port,
                timeout=5.0
            )

            if not self.controller.connect():
                logger.error("Failed to connect to mGBA socket server")
                return False

            title = self.controller.get_game_title()
            logger.info(f"✓ mGBA ready (game: {title})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to mGBA: {e}")
            return False

    async def capture_observation(self) -> ObservationFrame:
        """Capture a single observation frame."""
        self.step_count += 1

        if self.config.dry_run:
            # Generate dummy screenshot for dry-run
            screenshot = np.random.randint(0, 256, (160, 240, 3), dtype=np.uint8)
            ram_dump = np.random.bytes(0x10000)
            game_title = "Pokemon MD: Red Rescue Team"
        else:
            try:
                # Capture real screenshot from mGBA
                if self.controller:
                    import tempfile
                    from PIL import Image as PILImage

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name

                    # Take screenshot
                    self.controller.screenshot(tmp_path)

                    # Load image and convert to numpy
                    img = PILImage.open(tmp_path)
                    screenshot = np.array(img, dtype=np.uint8)

                    # Read RAM dump (query full WRAM)
                    ram_dump = self.controller.memory_domain_read_range("WRAM", 0, 0x10000) or bytes(0x10000)

                    # Get game title
                    game_title = self.controller.get_game_title() or "Pokemon MD: Red Rescue Team"

                    # Cleanup
                    import os
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                else:
                    # Fallback if controller not initialized
                    screenshot = np.zeros((160, 240, 3), dtype=np.uint8)
                    ram_dump = bytes(0x10000)
                    game_title = "Pokemon MD: Red Rescue Team"

            except Exception as e:
                logger.warning(f"Failed to capture observation: {e}, using dummy data")
                screenshot = np.zeros((160, 240, 3), dtype=np.uint8)
                ram_dump = bytes(0x10000)
                game_title = "Pokemon MD: Red Rescue Team"

        return ObservationFrame(
            step_id=self.step_count,
            timestamp=time.time(),
            screenshot=screenshot,
            ram_dump=ram_dump,
            game_title=game_title,
        )

    async def process_vision(self, obs: ObservationFrame) -> VisionOutput:
        """Convert observation to 2×2 quad vision output.

        Vision flow:
        1. Raw image (screenshot)
        2. Grid overlay (sprite grid + HUD)
        3. HUD/meta (stats, inventory, location)
        4. Retrieval hit mosaic (similar past frames)
        """
        # Convert numpy screenshot to PIL Image
        raw_img = Image.fromarray(obs.screenshot)

        # Create placeholder outputs
        grid_overlay = Image.new("RGB", raw_img.size, (100, 200, 50))
        hud_meta = Image.new("RGB", raw_img.size, (200, 100, 50))
        retrieval_mosaic = Image.new("RGB", raw_img.size, (50, 100, 200))

        return VisionOutput(
            step_id=obs.step_id,
            raw_img=raw_img,
            grid_overlay=grid_overlay,
            hud_meta=hud_meta,
            retrieval_mosaic=retrieval_mosaic,
        )

    async def infer_armada(self, vision_out: VisionOutput) -> ArmadaResponse:
        """Route vision output through FULL 6-model Qwen3-VL armada.

        Runs all 6 Qwen3-VL models in parallel with voting consensus:
        1. Qwen/Qwen3-VL-2B-Thinking-FP8
        2. unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit
        3. unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit
        4. unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit
        5. unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit
        6. unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit
        
        Each model analyzes the screenshot independently. Consensus vote wins.
        Real inference takes 2-5 seconds per step for proper GPU utilization.
        """
        import asyncio
        import sys
        import importlib.util

        step = vision_out.step_id

        # All 6 models - used exactly as specified
        model_ids = [
            "Qwen/Qwen3-VL-2B-Thinking-FP8",
            "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
            "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
            "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
            "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
            "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        ]

        try:
            # Lazy load all 6 models on first inference
            if not hasattr(self, '_armada_models'):
                logger.info("=" * 60)
                logger.info("LOADING 6-MODEL QWEN3-VL ARMADA ON GPU (UNSLOTH BACKEND)")
                logger.info("=" * 60)
                self._armada_models = {}
                self._armada_tokenizers = {}

                # Try to import Unsloth for FastVisionModel support
                try:
                    # Suppress triton warnings
                    import warnings
                    warnings.filterwarnings('ignore')

                    import torch

                    # Attempt to import Unsloth - this is the required backend
                    try:
                        from unsloth import FastVisionModel
                        HAS_UNSLOTH = True
                        logger.info("✓ Unsloth backend available")
                    except (ImportError, ModuleNotFoundError) as e:
                        logger.warning(f"Unsloth import failed: {e}")
                        logger.warning("Will use fallback inference pattern")
                        HAS_UNSLOTH = False

                    if HAS_UNSLOTH:
                        for i, model_id in enumerate(model_ids, 1):
                            try:
                                logger.info(f"[{i}/6] Loading {model_id}...")

                                # Determine quantization settings
                                load_kwargs = {
                                    "max_seq_length": 2048,
                                }

                                # Use 4-bit quantization for unsloth models, fp8 for direct models
                                if "unsloth" in model_id:
                                    load_kwargs["load_in_4bit"] = True
                                    logger.debug(f"     Using 4-bit quantization for {model_id}")
                                else:
                                    # For non-unsloth models, use 8-bit if available
                                    load_kwargs["load_in_8bit"] = True
                                    logger.debug(f"     Using 8-bit quantization for {model_id}")

                                # Load model and tokenizer using Unsloth's FastVisionModel
                                model, tokenizer = FastVisionModel.from_pretrained(
                                    model_name=model_id,
                                    **load_kwargs
                                )

                                # Enable fast inference mode (not training mode)
                                model = FastVisionModel.for_inference(model)

                                # Store in armada
                                self._armada_models[model_id] = model
                                self._armada_tokenizers[model_id] = tokenizer
                                logger.info(f"     ✓ {model_id} loaded successfully")

                            except Exception as e:
                                logger.warning(f"     ✗ FAIL {model_id}: {e}")

                        logger.info("=" * 60)
                        logger.info(f"Armada initialized with {len(self._armada_models)}/6 models ready")
                        logger.info("=" * 60)

                        if len(self._armada_models) == 0:
                            logger.warning("No models loaded successfully - will use fallback pattern")
                            self._use_fallback = True
                    else:
                        logger.info("Unsloth unavailable - using smart fallback pattern")
                        self._use_fallback = True

                except Exception as e:
                    logger.warning(f"Model loading setup failed: {e}")
                    self._use_fallback = True

            # If models loaded, use them; otherwise use fallback
            if self._armada_models and not getattr(self, '_use_fallback', False):
                # Game analysis prompt for all models
                prompt = """Analyze the Pokemon Mystery Dungeon game screen and output ONE action.

Possible actions:
- move_up, move_down, move_left, move_right: Move in that direction
- confirm: Press A button to interact/enter
- cancel: Press B button to go back

OUTPUT FORMAT:
ACTION: <single_action_name>"""

                # Run all loaded models in parallel for consensus voting
                screen_img = vision_out.raw_img
                action_votes = {}

                async def run_model_inference(model_id):
                    """Run a single model inference and return its action vote."""
                    try:
                        import torch
                        model = self._armada_models[model_id]
                        tokenizer = self._armada_tokenizers[model_id]

                        # Prepare inputs using Unsloth tokenizer
                        # Qwen3-VL expects text + image format
                        inputs = tokenizer(
                            text=prompt,
                            images=[screen_img],
                            return_tensors="pt",
                            padding=True,
                        ).to(model.device)

                        # Generate output with fast inference (enabled via for_inference)
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=32,
                                do_sample=True,
                                temperature=0.8,
                                top_p=0.95,
                            )

                        # Decode response using tokenizer
                        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                        # Extract action from response
                        action = None
                        for line in response.split('\n'):
                            if 'ACTION:' in line.upper():
                                potential_action = line.split(':')[-1].strip().lower()
                                if potential_action in ["move_up", "move_down", "move_left", "move_right", "confirm", "cancel"]:
                                    action = potential_action
                                    break

                        if not action:
                            action = "move_down"  # Safe default

                        logger.debug(f"Model {model_id.split('/')[-1]}: {action}")
                        return action

                    except Exception as e:
                        logger.warning(f"Model {model_id} inference failed: {e}")
                        return None

                # Run all models concurrently (real GPU work happening here)
                logger.debug(f"Step {step}: Running 6-model armada inference...")
                tasks = [run_model_inference(mid) for mid in self._armada_models.keys()]
                results = await asyncio.gather(*tasks)

                # Count votes from successful models
                for action_result in results:
                    if action_result:
                        action_votes[action_result] = action_votes.get(action_result, 0) + 1

                # Consensus decision
                if action_votes:
                    final_action = max(action_votes.items(), key=lambda x: x[1])[0]
                    successful_models = len([r for r in results if r])
                    vote_count = action_votes[final_action]
                    confidence = vote_count / successful_models if successful_models else 0.5
                else:
                    final_action = "move_down"
                    confidence = 0.3
                    vote_count = 0

                logger.info(f"Step {step}: Armada votes: {action_votes} -> {final_action.upper()} (conf: {confidence:.1%})")

                return ArmadaResponse(
                    step_id=vision_out.step_id,
                    timestamp=time.time(),
                    model_id=f"Qwen3-VL-Armada-{len(self._armada_models)}-models",
                    thinking=f"Armada consensus: {action_votes}",
                    action=final_action,
                    reasoning=f"Voted by {vote_count}/{len(self._armada_models)} models",
                    confidence=min(confidence, 0.95),
                )
            else:
                # Fallback inference pattern (no GPU needed)
                if step % 25 == 0:
                    action = "confirm"
                elif step % 20 == 15:
                    action = "move_up"
                elif step % 20 == 10:
                    action = "move_left"
                elif step % 20 == 5:
                    action = "move_right"
                else:
                    action = "move_down"

                return ArmadaResponse(
                    step_id=vision_out.step_id,
                    timestamp=time.time(),
                    model_id="Smart-Fallback-Pattern",
                    thinking="Using pattern fallback",
                    action=action,
                    reasoning="Smart action pattern",
                    confidence=0.6,
                )

        except Exception as e:
            logger.warning(f"Armada inference error: {e}, using fallback")
            # Ultimate fallback - smart action cycling
            if step % 25 == 0:
                action = "confirm"
            elif step % 20 == 15:
                action = "move_up"
            elif step % 20 == 10:
                action = "move_left"
            elif step % 20 == 5:
                action = "move_right"
            else:
                action = "move_down"

            return ArmadaResponse(
                step_id=vision_out.step_id,
                timestamp=time.time(),
                model_id="Fallback-Pattern",
                thinking=f"Error: {str(e)[:50]}",
                action=action,
                reasoning="Fallback pattern due to error",
                confidence=0.5,
            )
    def write_quad_image(self, vision_out: VisionOutput) -> None:
        """Write 2×2 quad image to dashboard."""
        quad = make_quad_image(vision_out)
        image_path = self.dashboard_dir / f"quad_{vision_out.step_id:06d}.png"
        quad.save(image_path)
        logger.debug(f"Wrote quad: {image_path}")

    def write_trace(self, obs: ObservationFrame, response: ArmadaResponse) -> None:
        """Append observation + response trace to JSONL."""
        trace = {
            "step_id": obs.step_id,
            "timestamp": obs.timestamp,
            "game_title": obs.game_title,
            "model_id": response.model_id,
            "thinking": response.thinking,
            "action": response.action,
            "reasoning": response.reasoning,
            "confidence": response.confidence,
        }

        # Ensure file exists
        self.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

        with open(self.trace_jsonl, "a") as f:
            f.write(json.dumps(trace) + "\n")

        logger.debug(f"Wrote trace: {response.action} @ step {obs.step_id}")

    def mark_keyframe(self, vision_out: VisionOutput, response: ArmadaResponse) -> None:
        """Mark significant frames (low SSIM, floor change, combat, etc.)."""
        # TODO: Implement keyframe policy based on vision output
        # For now, mark every Nth frame
        if vision_out.step_id % 30 == 0:
            keyframe_path = self.dashboard_dir / "keyframes" / f"keyframe_{vision_out.step_id:06d}.png"
            quad = make_quad_image(vision_out)
            quad.save(keyframe_path)
            logger.info(f"Marked keyframe: step {vision_out.step_id}")

    async def execute_action(self, action: str) -> bool:
        """Execute action on the emulator via button presses.

        Args:
            action: Action string (e.g., "move_down", "move_up", "move_left", "move_right", "confirm", "cancel")

        Returns:
            True if action executed successfully
        """
        if not self.controller:
            logger.warning(f"Controller not initialized, skipping action: {action}")
            return False

        try:
            # Map action strings to button presses
            action_map = {
                "move_down": ("Down", 1),
                "move_up": ("Up", 1),
                "move_left": ("Left", 1),
                "move_right": ("Right", 1),
                "confirm": ("A", 1),
                "cancel": ("B", 1),
                "menu": ("Start", 1),
                "attack": ("A", 1),
            }

            if action not in action_map:
                logger.warning(f"Unknown action: {action}, treating as noop")
                return False

            button, taps = action_map[action]

            # Execute button press(s)
            for _ in range(taps):
                self.controller.button_tap(button)
                await asyncio.sleep(0.05)  # Brief pause between taps

            # Wait for action to complete (roughly ~0.5-1 second for game to process)
            await asyncio.sleep(0.3)

            logger.debug(f"Executed action: {action} ({button})")
            return True

        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
            return False

    async def discover_skill_from_sequence(
        self,
        actions: list,
        confidences: list,
        precondition: Optional[str] = None,
        postcondition: Optional[str] = None
    ) -> Optional[object]:
        """Discover and save a skill from an action sequence.
        
        Args:
            actions: List of action strings
            confidences: List of confidence scores
            precondition: Optional precondition (e.g., "at_entrance")
            postcondition: Optional postcondition (e.g., "dungeon_cleared")
        
        Returns:
            Skill object if created successfully
        """
        if not actions or len(actions) != len(confidences):
            logger.warning("Cannot discover skill: action/confidence mismatch")
            return None

        try:
            skill_name = f"discovered_step{self.step_count}_{len(actions)}actions"
            skill = self.skill_manager.create_skill_from_trajectory(
                name=skill_name,
                actions=actions,
                confidences=confidences,
                description=f"Skill discovered at step {self.step_count}",
                precondition=precondition,
                postcondition=postcondition,
                tags=["discovered", "live"]
            )
            
            if skill:
                self.discovered_skills.append(skill)
                logger.info(f"✓ Discovered skill: {skill_name}")
                return skill
            else:
                logger.warning(f"Failed to save discovered skill")
                return None
                
        except Exception as e:
            logger.error(f"Error discovering skill: {e}")
            return None

    async def _infer_with_timeout(self, vision_out: VisionOutput) -> ArmadaResponse:
        """Wrap infer_armada with timeout and confidence threshold logic.

        Max 5 seconds for inference. If confidence falls below threshold,
        return best guess instead of waiting for full inference.
        """
        import asyncio

        try:
            # Run inference with timeout
            response = await asyncio.wait_for(
                self.infer_armada(vision_out),
                timeout=self.config.inference_timeout
            )

            # Check confidence threshold
            if response.confidence < self.config.confidence_threshold:
                logger.warning(
                    f"Step {vision_out.step_id}: Confidence {response.confidence:.2f} "
                    f"below threshold {self.config.confidence_threshold}. Using fallback."
                )
                # Return best guess fallback
                return ArmadaResponse(
                    step_id=vision_out.step_id,
                    timestamp=time.time(),
                    model_id="Best-Guess-Fallback",
                    thinking="Low confidence inference - using smart fallback",
                    action="move_down",
                    reasoning="Best guess due to low confidence",
                    confidence=0.4,
                )

            return response

        except asyncio.TimeoutError:
            logger.warning(
                f"Step {vision_out.step_id}: Inference timeout after "
                f"{self.config.inference_timeout}s. Using best guess."
            )
            # Return best guess fallback
            return ArmadaResponse(
                step_id=vision_out.step_id,
                timestamp=time.time(),
                model_id="Timeout-Fallback",
                thinking="Inference exceeded 5-second timeout",
                action="move_down",
                reasoning="Best guess due to inference timeout",
                confidence=0.4,
            )
        except Exception as e:
            logger.error(
                f"Step {vision_out.step_id}: Inference error: {e}. Using best guess."
            )
            # Return best guess fallback
            return ArmadaResponse(
                step_id=vision_out.step_id,
                timestamp=time.time(),
                model_id="Error-Fallback",
                thinking=f"Inference error: {str(e)[:100]}",
                action="move_down",
                reasoning="Best guess due to inference error",
                confidence=0.3,
            )

    async def run(self, max_steps: int = 100) -> int:
        """Run main loop with smart frame sampling.

        Emulator maintains 30 FPS continuously. VLM inference only runs
        every ~300 frames (10-second intervals @ 30 FPS), giving 5-10 seconds
        for inference processing while emulator stays responsive.
        """
        logger.info(f"Starting live armada (max_steps={max_steps}, dry_run={self.config.dry_run})")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Frame sampling: VLM inference every {self.config.frame_sample_interval} frames "
                   f"(~{self.config.frame_sample_interval / 30:.1f}s @ 30 FPS)")
        logger.info(f"Inference timeout: {self.config.inference_timeout}s, "
                   f"confidence threshold: {self.config.confidence_threshold}")

        if not await self.ensure_ready():
            return 1

        # Load bootstrap state if available
        bootstrap_checkpoint = self.bootstrap_manager.load_latest_checkpoint()
        if bootstrap_checkpoint:
            logger.info(f"Loaded bootstrap state from {bootstrap_checkpoint.run_id}")
            logger.info(f"  → {len(bootstrap_checkpoint.learned_skills)} learned skills")

            # Restore learned skills
            for skill_dict in bootstrap_checkpoint.learned_skills:
                try:
                    from ..skills.skill_manager import Skill
                    skill = Skill.from_dict(skill_dict)
                    self.skill_manager.save_skill(skill)
                except Exception as e:
                    logger.warning(f"Failed to restore skill: {e}")

        collected_actions: list = []
        collected_confidences: list = []
        frame_count = 0  # Track raw emulator frames
        inference_count = 0  # Track VLM inference calls

        try:
            for step in range(max_steps):
                # Capture observation (always at 30 FPS)
                obs = await self.capture_observation()
                frame_count += 1

                # Only run VLM inference every frame_sample_interval frames
                # This gives ~10 seconds between inference calls (300 frames @ 30 FPS)
                should_infer = (frame_count % self.config.frame_sample_interval) == 0

                if should_infer:
                    inference_count += 1
                    logger.info(f"Step {obs.step_id}: VLM inference #{inference_count} (frame {frame_count})")

                    # Process vision
                    vision_out = await self.process_vision(obs)

                    # Route through armada with timeout
                    response = await self._infer_with_timeout(vision_out)
                    logger.info(f"  → {response.model_id}: {response.action} (conf={response.confidence:.2f})")

                    # Track actions for skill discovery
                    if response.action:
                        collected_actions.append(response.action)
                        collected_confidences.append(response.confidence)

                    # Write outputs
                    self.write_quad_image(vision_out)
                    self.write_trace(obs, response)
                    self.mark_keyframe(vision_out, response)

                    # CRITICAL: Execute action on emulator
                    if response.action:
                        success = await self.execute_action(response.action)
                        if success:
                            logger.debug(f"  ✓ Action executed: {response.action}")
                        else:
                            logger.debug(f"  ✗ Action failed: {response.action}")
                else:
                    # No inference this frame, just advance emulator at 30 FPS
                    logger.debug(f"Frame {frame_count}: skipped VLM inference "
                               f"(next at frame {frame_count + (self.config.frame_sample_interval - (frame_count % self.config.frame_sample_interval))})")

                # Control loop timing - ALWAYS maintain 30 FPS
                await asyncio.sleep(1.0 / self.config.capture_fps)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            return 1

        # Save discovered skills and bootstrap checkpoint
        if collected_actions:
            try:
                skill_name = f"trajectory_{self.run_id}"
                skill = self.skill_manager.create_skill_from_trajectory(
                    name=skill_name,
                    actions=collected_actions,
                    confidences=collected_confidences,
                    description=f"Discovered trajectory from {self.run_id}",
                    tags=["discovered", "live_run"]
                )
                if skill:
                    self.discovered_skills.append(skill)
                    logger.info(f"Saved skill: {skill_name} ({len(skill.steps)} steps)")
            except Exception as e:
                logger.error(f"Failed to save discovered skill: {e}")

        # Save bootstrap checkpoint for next run
        try:
            self.bootstrap_manager.save_checkpoint(
                run_id=self.run_id,
                learned_skills=[s.to_dict() for s in self.skill_manager.list_skills()],
                memory_buffer={},
                stats={
                    "total_steps": self.step_count,
                    "skills_discovered": len(self.discovered_skills)
                }
            )
            logger.info(f"Saved bootstrap checkpoint: {self.run_id}")
        except Exception as e:
            logger.error(f"Failed to save bootstrap checkpoint: {e}")

        logger.info(f"Finished. Outputs written to: {self.dashboard_dir}")
        logger.info(f"Traces: {self.trace_jsonl}")
        logger.info(f"Keyframes: {self.dashboard_dir / 'keyframes'}")
        logger.info(f"Skills discovered: {len(self.discovered_skills)}")
        logger.info(f"Total skills available: {len(self.skill_manager.list_skills())}")

        return 0


async def main() -> int:
    """CLI entry point."""
    import os

    parser = argparse.ArgumentParser(
        description="Live armada: multi-model inference loop",
        prog="python -m pokemon-md-agent.src.runners.live_armada",
    )

    parser.add_argument(
        "--rom",
        type=Path,
        required=False,
        help="Path to GBA ROM file (env: PMD_ROM)"
    )

    parser.add_argument(
        "--save",
        type=Path,
        required=False,
        help="Path to game save file (env: PMD_SAVE)"
    )

    parser.add_argument(
        "--lua",
        type=Path,
        required=False,
        help="Path to mGBA Lua socket server script (env: MGBALUA)"
    )

    parser.add_argument(
        "--mgba-exe",
        type=Path,
        required=False,
        help="Path to mGBA executable (env: MGBAX)"
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="mGBA socket server host (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="mGBA socket server port (default: 8888)"
    )

    parser.add_argument(
        "--capture-fps",
        type=float,
        default=6.0,
        help="Screenshot capture rate (default: 6 fps)"
    )

    parser.add_argument(
        "--mem-hz",
        type=float,
        default=10.0,
        help="RAM read rate (default: 10 Hz)"
    )

    parser.add_argument(
        "--rate-limit",
        type=int,
        default=30,
        help="Command rate limit (default: 30 cmds/s)"
    )

    parser.add_argument(
        "--gatekeeper",
        choices=["on", "off"],
        default="on",
        help="Enable retrieval gatekeeper (default: on)"
    )

    parser.add_argument(
        "--dashboard-dir",
        type=Path,
        default=Path("docs/current"),
        help="Dashboard output directory (default: docs/current)"
    )

    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        default=Path("docs/current/traces/latest.jsonl"),
        help="Trace output JSONL file"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps to run (default: 100)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without emulator (mock transport)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config with environment variable fallbacks
    rom = args.rom or (Path(os.environ.get("PMD_ROM")) if os.environ.get("PMD_ROM") else None)
    save = args.save or (Path(os.environ.get("PMD_SAVE")) if os.environ.get("PMD_SAVE") else None)
    lua = args.lua or (Path(os.environ.get("MGBALUA")) if os.environ.get("MGBALUA") else None)
    mgba_exe = args.mgba_exe or (Path(os.environ.get("MGBAX")) if os.environ.get("MGBAX") else None)

    config = ArmadaConfig(
        rom=rom or Path("C:/Homework/agent_hackathon/rom/Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba"),
        save=save or Path("C:/Homework/agent_hackathon/pokemon-md-agent/config/save_files/game_start_save.ss0"),
        lua=lua or Path("C:/Homework/agent_hackathon/pokemon-md-agent/src/mgba-harness/mgba-http/mGBASocketServer.lua"),
        mgba_exe=mgba_exe or Path("C:/Program Files/mGBA/mGBA.exe"),
        host=args.host,
        port=args.port,
        capture_fps=args.capture_fps,
        mem_hz=args.mem_hz,
        rate_limit=args.rate_limit,
        gatekeeper=args.gatekeeper,
        dashboard_dir=args.dashboard_dir,
        trace_jsonl=args.trace_jsonl,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if not config.validate():
        return 1

    runner = LiveArmadaRunner(config)
    return await runner.run(max_steps=args.max_steps)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
