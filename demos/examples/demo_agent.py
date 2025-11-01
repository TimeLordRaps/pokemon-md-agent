#!/usr/bin/env python3
"""
Pokemon MD Agent Demo Script

Standalone demo runner that connects to the mGBA Lua socket server and drives
the Pokemon Mystery Dungeon agent loop with rich console logging. Falls back
to a mock agent with ASCII visualization when the production AgentCore is not
ready yet, making it useful for both live demos and dry runs.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Ensure project src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Add pokemon-md-agent src to path for imports
pokemon_src = PROJECT_ROOT / "pokemon-md-agent" / "src"
if str(pokemon_src) not in sys.path:
    sys.path.insert(0, str(pokemon_src))

try:
    # Try to import from relative path first (when running from project)
    try:
        from environment.mgba_controller import MGBAController
    except ImportError:
        # Fallback to absolute import for installed package
        from pokemon_md_agent.src.environment.mgba_controller import MGBAController
except ImportError:
    # Fallback to mock controller for demo
    class MGBAController:
        def __init__(self, host="localhost", port=8888, timeout=10.0):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.connected = False

        def connect_with_retry(self, max_retries=3, backoff_factor=1.5):
            # Mock connection - always fail gracefully for demo
            return False

        def disconnect(self):
            pass

        def load_file(self, path):
            return False

        def load_save_file(self, path):
            return False

        def get_game_title(self):
            return None

        def get_game_code(self):
            return None

    print("Using mock MGBAController (production controller not available)")
    MGBAController = MGBAController

# Attempt to import AgentCore (future-ready, may currently be WIP)
AGENT_MODULE_IMPORT_ERROR: Optional[Exception] = None
AgentCoreClass: Optional[type] = None
try:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "pokemon-md-agent" / "src"))
    agent_core_module = importlib.import_module("agent.agent_core")
    AgentCoreClass = getattr(agent_core_module, "AgentCore", None)
except Exception as exc:  # pragma: no cover - import helps determine fallback
    AGENT_MODULE_IMPORT_ERROR = exc

logger = logging.getLogger("demo_agent")


@dataclass
class NormalizedDecision:
    action: str
    rationale: str


@dataclass
class DemoOutcome:
    success: bool
    fatal: bool


class MockAgent:
    """Lightweight agent used when the production AgentCore is unavailable."""

    def __init__(self, objective: str, grid_size: int = 15) -> None:
        self.objective = objective
        self.grid_size = grid_size
        self.position = [grid_size // 2, grid_size // 2]
        self.stairs = [grid_size - 2, grid_size - 2]
        self._rng = random.Random(1337)

    def perceive(self) -> dict[str, Any]:
        """Return a mocked game state with ASCII visualization."""
        tiles = []
        for y in range(self.grid_size):
            row_chars = []
            for x in range(self.grid_size):
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row_chars.append("#")
                    continue
                if [x, y] == self.position:
                    row_chars.append("P")
                elif [x, y] == self.stairs:
                    row_chars.append("S")
                else:
                    row_chars.append("." if (x + y) % 3 else " ")
            tiles.append("".join(row_chars))

        ascii_map = "\n".join(tiles)
        return {
            "ascii": ascii_map,
            "ram": {
                "player_x": self.position[0],
                "player_y": self.position[1],
                "floor_number": 1,
            },
            "mock": True,
        }

    def reason(self, state: dict[str, Any]) -> dict[str, str]:
        """Greedy pathing toward the stairs."""
        px, py = self.position
        sx, sy = self.stairs

        if (px, py) == (sx, sy):
            return {"action": "TAKE_STAIRS", "rationale": "Reached the stairs"}

        if px < sx:
            return {"action": "MOVE_RIGHT", "rationale": "Heading east toward stairs"}
        if px > sx:
            return {"action": "MOVE_LEFT", "rationale": "Heading west toward stairs"}
        if py < sy:
            return {"action": "MOVE_DOWN", "rationale": "Heading south toward stairs"}
        if py > sy:
            return {"action": "MOVE_UP", "rationale": "Heading north toward stairs"}

        # Should never hit, but keep WAIT as safe default
        return {"action": "WAIT", "rationale": "Holding position"}

    def act(self, decision: dict[str, str]) -> None:
        """Update internal position based on the chosen action."""
        action = decision.get("action", "").upper()
        if action == "MOVE_RIGHT":
            self.position[0] += 1
        elif action == "MOVE_LEFT":
            self.position[0] -= 1
        elif action == "MOVE_DOWN":
            self.position[1] += 1
        elif action == "MOVE_UP":
            self.position[1] -= 1

        # Clamp to walkable interior to keep ASCII tidy
        self.position[0] = max(1, min(self.grid_size - 2, self.position[0]))
        self.position[1] = max(1, min(self.grid_size - 2, self.position[1]))


def configure_logging(log_dir: Optional[Path], verbose: bool) -> Path:
    """Initialize logging to console and rotating demo log file."""
    target_dir = log_dir or (PROJECT_ROOT / "logs")
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = target_dir / f"demo_run_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    logger.debug("Logging configured (verbose=%s) -> %s", verbose, log_file)
    return log_file


def print_banner() -> None:
    """Render the console banner."""
    banner = r"""
╔════════════════════════════════════════════════════════════╗
║         POKEMON MYSTERY DUNGEON - AI AGENT DEMO            ║
║                    Powered by Qwen3-VL                     ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_ascii_view(ascii_block: str, step: int) -> None:
    """Pretty-print 4-up ASCII metaview visualization of the current game state."""
    divider = "=" * 80

    # Create 4-up multiview layout (mock for now since we only have one view)
    # In a real implementation, this would combine multiple camera views
    view_width = max(len(line) for line in ascii_block.split('\n')) if ascii_block else 40
    view_height = len(ascii_block.split('\n')) if ascii_block else 15

    # For now, just show the single view in top-left quadrant
    top_left = ascii_block
    top_right = "VIEW 2\n(Not implemented)\n" + "\n".join(["." * 20] * 10)
    bottom_left = "VIEW 3\n(Not implemented)\n" + "\n".join(["." * 20] * 10)
    bottom_right = "VIEW 4\n(Not implemented)\n" + "\n".join(["." * 20] * 10)

    # Combine into 4-up layout
    top_lines = []
    bottom_lines = []

    top_left_lines = top_left.split('\n')
    top_right_lines = top_right.split('\n')
    bottom_left_lines = bottom_left.split('\n')
    bottom_right_lines = bottom_right.split('\n')

    max_lines = max(len(top_left_lines), len(top_right_lines), len(bottom_left_lines), len(bottom_right_lines))

    for i in range(max_lines):
        tl = top_left_lines[i] if i < len(top_left_lines) else ""
        tr = top_right_lines[i] if i < len(top_right_lines) else ""
        bl = bottom_left_lines[i] if i < len(bottom_left_lines) else ""
        br = bottom_right_lines[i] if i < len(bottom_right_lines) else ""

        # Pad lines to consistent width
        tl = tl.ljust(25)
        tr = tr.ljust(25)
        bl = bl.ljust(25)
        br = br.ljust(25)

        top_lines.append(f"{tl} | {tr}")
        bottom_lines.append(f"{bl} | {br}")

    combined_view = "\n".join(top_lines + ["-" * 52] + bottom_lines)

    print(
        f"\n{divider}\n4-UP METAVIEW SNAPSHOT (step {step})\n{divider}\n"
        f"{combined_view}\n{divider}\n"
    )


def check_mgba_connection(
    host: str,
    port: int,
    timeout: float,
    max_retries: int,
    backoff_factor: float,
) -> tuple[bool, dict[str, Any]]:
    """Verify that the mGBA Lua server is reachable before starting the demo."""
    logger.info("Checking mGBA connection at %s:%d ...", host, port)
    controller = MGBAController(host=host, port=port, timeout=timeout)
    meta: dict[str, Any] = {}

    try:
        if not controller.connect_with_retry(max_retries=max_retries, backoff_factor=backoff_factor):
            logger.error("❌ Failed to connect to mGBA on %s:%d", host, port)
            logger.error("Please ensure mGBA is running with the Lua socket server loaded:")
            logger.error("  1. Open mGBA and load the PMD Red Rescue Team ROM")
            logger.error("  2. Tools → Scripting → Load Script")
            logger.error("  3. Pick: src/mgba-harness/mgba-http/mGBASocketServer.lua")
            logger.error("  4. Confirm the console prints 'ready on port %d'", port)
            return False, meta

        logger.info("✅ Connected to mGBA!")
        title = controller.get_game_title()
        code = controller.get_game_code()
        if title:
            logger.info("ROM detected: %s (%s)", title, code or "unknown code")
            meta["title"] = title
            if code:
                meta["code"] = code

        return True, meta
    finally:
        try:
            controller.disconnect()
        except Exception:
            logger.debug("Failed to disconnect mGBA controller cleanly", exc_info=True)


def normalize_decision(decision: Any) -> NormalizedDecision:
    """Convert agent decisions to a consistent logging structure."""
    action: Optional[str] = None
    rationale: Optional[str] = None

    if isinstance(decision, dict):
        action = decision.get("action") or decision.get("move") or decision.get("command")
        rationale = decision.get("rationale") or decision.get("reason") or decision.get("explanation")
    elif hasattr(decision, "action"):
        action = getattr(decision, "action")
        rationale = getattr(decision, "rationale", getattr(decision, "reasoning", None))
    elif isinstance(decision, (list, tuple)) and decision:
        action = str(decision[0])
        if len(decision) > 1:
            rationale = str(decision[1])
    elif isinstance(decision, str):
        action = decision

    if not action:
        action = "UNKNOWN"
    if not rationale:
        rationale = "No rationale provided"

    return NormalizedDecision(action=action, rationale=rationale)


def instantiate_agent(
    objective: str,
    rom_path: Path,
    save_dir: Path,
    force_mock: bool,
) -> tuple[Any, str, list[str]]:
    """Instantiate the production AgentCore when possible, otherwise fallback."""
    reasons: list[str] = []

    if force_mock:
        logger.info("Using MockAgent (forced via CLI flag)")
        return MockAgent(objective), "mock", reasons

    if AgentCoreClass is None:
        if AGENT_MODULE_IMPORT_ERROR:
            reasons.append(f"ImportError: {AGENT_MODULE_IMPORT_ERROR}")
        else:
            reasons.append("AgentCore class not found in agent.agent_core")
        logger.warning("Falling back to MockAgent. Reasons:\n%s", "\n".join(f"- {r}" for r in reasons))
        return MockAgent(objective), "mock", reasons

    signature = inspect.signature(AgentCoreClass)
    kwargs: dict[str, Any] = {}

    if "objective" in signature.parameters:
        kwargs["objective"] = objective
    if "rom_path" in signature.parameters:
        kwargs["rom_path"] = rom_path
    if "save_dir" in signature.parameters:
        kwargs["save_dir"] = save_dir

    # Only attempt to pass config if optional with default
    param = signature.parameters.get("config")
    if param and param.default is inspect._empty:
        reasons.append("AgentCore.config parameter is required but no prototype config is available")
        logger.warning("Falling back to MockAgent. Reasons:\n%s", "\n".join(f"- {r}" for r in reasons))
        return MockAgent(objective), "mock", reasons

    try:
        agent = AgentCoreClass(**kwargs)
        logger.info("AgentCore instantiated with signature %s", signature)
        return agent, "agent", reasons
    except TypeError as exc:
        reasons.append(f"TypeError: {exc}")
    except Exception as exc:  # pragma: no cover - forward compatible handling
        reasons.append(f"{type(exc).__name__}: {exc}")

    logger.warning("Falling back to MockAgent. Reasons:\n%s", "\n".join(f"- {r}" for r in reasons))
    return MockAgent(objective), "mock", reasons


def load_game_into_mgba(
    controller: MGBAController,
    rom_path: Path,
    save_path: Optional[Path] = None
) -> tuple[bool, dict[str, Any]]:
    """Check game status in mGBA (game should already be loaded)."""
    meta: dict[str, Any] = {}

    try:
        # Skip ROM/save loading since mGBA is already running with game loaded
        logger.info("Checking game status - mGBA should already have game loaded")

        # Get game metadata
        title = controller.get_game_title()
        code = controller.get_game_code()
        if title:
            meta["title"] = title
            logger.info("Game title: %s", title)
        if code:
            meta["code"] = code

        return True, meta

    except Exception as exc:
        logger.error("Error checking game status: %s", exc)
        return False, meta


def run_demo(
    *,
    max_steps: int,
    objective: str,
    ascii_interval: int,
    throttle: float,
    rom_path: Path,
    save_dir: Path,
    connection_opts: dict[str, Any],
    force_mock: bool,
    skip_connection_check: bool,
) -> DemoOutcome:
    """Execute the main demo loop."""
    print_banner()

    # Check mGBA connection
    connection_ok = True
    controller = None
    if not skip_connection_check:
        connection_ok, _meta = check_mgba_connection(**connection_opts)
        if not connection_ok and not force_mock:
            logger.error("Cannot continue demo without mGBA connection (use --mock to bypass).")
            return DemoOutcome(success=False, fatal=True)

    # Load game if connection available
    if connection_ok and not force_mock:
        controller = MGBAController(host=connection_opts["host"], port=connection_opts["port"], timeout=connection_opts["timeout"])
        save_path = save_dir / "tiny-wood-floor-1-fresh-game.ss0"  # Use the save state from rom directory
        game_loaded, game_meta = load_game_into_mgba(controller, rom_path, save_path)
        if not game_loaded:
            logger.warning("Failed to load game into mGBA, falling back to mock agent")
            force_mock = True
        else:
            logger.info("Game loaded successfully into mGBA")

    # Ensure save directory exists for trajectory logging
    save_dir.mkdir(parents=True, exist_ok=True)

    agent, mode, reasons = instantiate_agent(
        objective=objective,
        rom_path=rom_path,
        save_dir=save_dir,
        force_mock=force_mock or not connection_ok,
    )

    if mode == "mock" and reasons:
        for reason in reasons:
            logger.debug("Agent fallback reason: %s", reason)

    # Setup trajectory logging
    trajectory_file = save_dir / f"trajectory_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    logger.info("Trajectory will be logged to: %s", trajectory_file)

    start_time = time.time()
    success = False

    try:
        for step in range(1, max_steps + 1):
            try:
                state = agent.perceive()
            except Exception as exc:
                logger.exception("Perception failed at step %d: %s", step, exc)
                return DemoOutcome(success=False, fatal=True)

            try:
                if hasattr(agent, 'reason') and inspect.iscoroutinefunction(agent.reason):
                    # Handle async reason method
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    decision = loop.run_until_complete(agent.reason(state))
                    loop.close()
                else:
                    decision = agent.reason(state)
            except Exception as exc:
                logger.exception("Reasoning failed at step %d: %s", step, exc)
                return DemoOutcome(success=False, fatal=True)

            try:
                if hasattr(agent, "act"):
                    agent.act(decision)
            except Exception as exc:
                logger.exception("Action failed at step %d: %s", step, exc)
                return DemoOutcome(success=False, fatal=True)

            view = normalize_decision(decision)
            elapsed = time.time() - start_time
            logger.info(
                "Step %3d | Action: %-12s | Rationale: %s",
                step,
                view.action[:12],
                view.rationale,
            )

            # Stream ASCII metaview each step (4-up format)
            if state.get("ascii"):
                print_ascii_view(state["ascii"], step)
            else:
                # Fallback if no ASCII view available
                logger.info("No ASCII view available for step %d", step)

            # Log trajectory data
            trajectory_entry = {
                "step": step,
                "timestamp": time.time(),
                "elapsed": elapsed,
                "state": {
                    k: v for k, v in state.items() 
                    if k not in ["screenshot", "grid", "ram"]  # Exclude non-serializable objects
                },
                "decision": {"action": view.action, "rationale": view.rationale}
            }
            # Add serializable versions of excluded fields
            if "grid" in state and hasattr(state["grid"], "width") and hasattr(state["grid"], "height"):
                trajectory_entry["state"]["grid"] = {
                    "width": state["grid"].width,
                    "height": state["grid"].height,
                    "tile_count": len(getattr(state["grid"], "cells", []))
                }
            if "ram" in state and "snapshot" in state["ram"]:
                # Convert RAMSnapshot to dict for serialization
                ram_dict = {
                    "player_x": state["ram"].get("player_x", 0),
                    "player_y": state["ram"].get("player_y", 0),
                    "floor_number": state["ram"].get("floor_number", 1)
                }
                trajectory_entry["state"]["ram"] = ram_dict
            
            with open(trajectory_file, 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(trajectory_entry) + '\n')

            if "stairs" in view.action.lower() or "take_stairs" in view.action.lower():
                logger.info("=" * 60)
                logger.info(
                    "🎉 SUCCESS! Objective satisfied after %d steps (%.1fs)",
                    step,
                    elapsed,
                )
                logger.info("=" * 60)
                success = True
                break

            time.sleep(throttle)
        else:
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(
                "Demo completed after %d steps (%.1fs). Objective not reached.",
                max_steps,
                elapsed,
            )
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Demo interrupted by user (Ctrl+C)")
        logger.info("=" * 60)
        return DemoOutcome(success=False, fatal=False)
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("❌ Demo failed unexpectedly: %s", exc)
        return DemoOutcome(success=False, fatal=True)

    return DemoOutcome(success=success, fatal=False)


def parse_args() -> argparse.Namespace:
    """Build the CLI interface."""
    parser = argparse.ArgumentParser(description="Pokemon Mystery Dungeon Agent Demo")
    parser.add_argument("--steps", type=int, default=100, help="Maximum steps to execute")
    parser.add_argument("--objective", type=str, default="Navigate to stairs", help="Agent objective")
    parser.add_argument("--ascii-interval", type=int, default=0, help="Print ASCII view every N steps (0 disables)")
    parser.add_argument("--throttle", type=float, default=0.1, help="Delay between steps to avoid overloading mGBA")
    parser.add_argument("--host", type=str, default="localhost", help="mGBA Lua server host")
    parser.add_argument("--port", type=int, default=8888, help="mGBA Lua server port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Socket timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Connection retry attempts")
    parser.add_argument("--backoff", type=float, default=1.5, help="Retry backoff multiplier")
    parser.add_argument("--rom", type=Path, default=None, help="Path to PMD Red ROM (defaults to rom/ directory)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory for demo saves/logs (defaults to runs/demo_*)")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for log files (defaults to project logs/)")
    parser.add_argument("--skip-connection-check", action="store_true", help="Skip mGBA connectivity preflight")
    parser.add_argument("--mock", action="store_true", help="Force use of mock agent even if AgentCore is available")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> int:
    """Entry point for the demo script."""
    args = parse_args()

    log_file = configure_logging(args.log_dir, args.verbose)
    logger.info("Demo log file: %s", log_file)

    rom_path = args.rom or (PROJECT_ROOT / "rom" / "Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba")
    save_dir = args.save_dir or (PROJECT_ROOT / "runs" / f"demo_{time.strftime('%Y%m%d_%H%M%S')}")

    outcome = run_demo(
        max_steps=args.steps,
        objective=args.objective,
        ascii_interval=args.ascii_interval,
        throttle=args.throttle,
        rom_path=rom_path,
        save_dir=save_dir,
        connection_opts={
            "host": args.host,
            "port": args.port,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "backoff_factor": args.backoff,
        },
        force_mock=args.mock,
        skip_connection_check=args.skip_connection_check,
    )

    if outcome.success:
        logger.info("Demo finished successfully.")
    elif not outcome.fatal:
        logger.info("Demo finished without hitting the success condition.")
    else:
        logger.error("Demo terminated due to errors.")

    logger.info("Console log saved to %s", log_file)
    logger.info("Trajectory log saved to %s", "N/A (mock mode)" if args.mock else f"{save_dir}/trajectory_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    return 0 if not outcome.fatal else 1


if __name__ == "__main__":
    sys.exit(main())
