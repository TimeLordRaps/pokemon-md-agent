"""Command-line interface for mgba-harness operations.

Provides manual control and debugging tools for the mgba emulator
via the Lua Socket API.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path for relative imports
project_root = Path(__file__).parents[2]  # cli.py -> mgba-harness -> src -> pokemon-md-agent
sys.path.insert(0, str(project_root))

from ..environment.mgba_controller import MGBAController
from ..environment.save_manager import SaveManager


def encode_cmd(cmd: str, *args) -> str:
    """Encode command and arguments in colon-delimited format.

    Prefers colon-format for commands with arguments (e.g., "screenshot:480:320:2").
    This format is compatible with both the original comma-delimited parser
    and the new dual-format parser in mGBASocketServer.lua.

    Args:
        cmd: Command name (e.g., "screenshot")
        args: Command arguments

    Returns:
        Formatted command string

    Example:
        >>> encode_cmd("screenshot", 480, 320, 2)
        'screenshot:480:320:2'
        >>> encode_cmd("ping")
        'ping'
    """
    if not args:
        return cmd
    return f"{cmd}:{':'.join(map(str, args))}"


class MGBACLI:
    """Command-line interface for mgba operations."""
    
    def __init__(self):
        """Initialize CLI with controller."""
        self.controller = MGBAController()
        self.save_manager: Optional[SaveManager] = None
        
    def connect(self) -> bool:
        """Connect to mgba server.
        
        Returns:
            True if connection successful
        """
        print("Connecting to mgba...")
        
        if not self.controller.connect():
            print("Failed to connect to mgba!")
            print("Make sure mgba is running with --http-server and mGBASocketServer.lua loaded")
            return False
        
        print(f"✓ Connected successfully")
        print(f"  Game: {self.controller._game_title}")
        print(f"  Code: {self.controller._game_code}")
        print(f"  Memory domains: {self.controller._memory_domains}")
        
        # Initialize save manager
        save_dir = Path.home() / ".cache" / "pmd-red" / "saves"
        self.save_manager = SaveManager(self.controller, save_dir)
        
        return True
    
    def cmd_ping(self, args) -> None:
        """Ping the mgba server.
        
        Args:
            args: Command arguments
        """
        print("Pinging mgba server...")
        
        start_time = time.time()
        title = self.controller.get_game_title()
        end_time = time.time()
        
        if title:
            latency = (end_time - start_time) * 1000
            print(f"✓ Pong! Game: {title}")
            print(f"  Latency: {latency:.1f}ms")
        else:
            print("✗ No response")
    
    def cmd_title(self, args) -> None:
        """Get game title and info.
        
        Args:
            args: Command arguments
        """
        print("Getting game info...")
        
        title = self.controller.get_game_title()
        code = self.controller.get_game_code()
        frame = self.controller.current_frame()
        
        if title:
            print(f"✓ Title: {title}")
            print(f"  Code: {code or 'Unknown'}")
            print(f"  Frame: {frame}")
        else:
            print("✗ Failed to get game info")
    
    def cmd_tap(self, args) -> None:
        """Tap a button.
        
        Args:
            args: Command arguments (button)
        """
        if not args.button:
            print("Error: button name required")
            return
        
        print(f"Tapping button: {args.button}")
        
        if self.controller.button_tap(args.button):
            print(f"✓ {args.button} tapped")
        else:
            print(f"✗ Failed to tap {args.button}")
    
    def cmd_hold(self, args) -> None:
        """Hold a button for duration.
        
        Args:
            args: Command arguments (button, duration_ms)
        """
        if not args.button:
            print("Error: button name required")
            return
        
        duration = args.duration or 500
        print(f"Holding button: {args.button} for {duration}ms")
        
        if self.controller.button_hold(args.button, duration):
            print(f"✓ {args.button} held for {duration}ms")
        else:
            print(f"✗ Failed to hold {args.button}")
    
    def cmd_screenshot(self, args) -> None:
        """Take a screenshot.
        
        Args:
            args: Command arguments (path)
        """
        output_path = args.path or f"screenshot_{int(time.time())}.png"
        output_path = Path(output_path)
        
        print(f"Taking screenshot: {output_path}")
        
        if self.controller.screenshot(str(output_path)):
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"✓ Screenshot saved ({size} bytes)")
            else:
                print("✗ Screenshot file not created")
        else:
            print("✗ Failed to take screenshot")
    
    def cmd_read(self, args) -> None:
        """Read memory range.
        
        Args:
            args: Command arguments (domain, address, length)
        """
        domain = args.domain or "WRAM"
        address = int(args.address, 16) if args.address else 0
        length = args.length or 16
        
        print(f"Reading memory: {domain} + 0x{address:08X} ({length} bytes)")
        
        data = self.controller.memory_domain_read_range(domain, address, length)
        
        if data:
            print(f"✓ Read {len(data)} bytes:")
            
            # Print as hex
            hex_str = " ".join(f"{b:02X}" for b in data)
            print(f"  Hex: {hex_str}")
            
            # Print as ASCII (printable only)
            ascii_str = "".join(chr(b) if 32 <= b <= 126 else "." for b in data)
            print(f"  ASCII: {ascii_str}")
        else:
            print("✗ Failed to read memory")
    
    def cmd_memory_domains(self, args) -> None:
        """List memory domains.
        
        Args:
            args: Command arguments
        """
        print("Available memory domains:")
        
        domains = self.controller.get_memory_domains()
        
        if domains:
            for domain in domains:
                print(f"  - {domain}")
        else:
            print("  ✗ No domains available")
    
    def cmd_save(self, args) -> None:
        """Save state to slot.
        
        Args:
            args: Command arguments (slot, description)
        """
        if not self.save_manager:
            print("✗ Save manager not initialized")
            return
        
        slot = args.slot
        description = args.description
        
        print(f"Saving to slot {slot}...")
        
        if self.save_manager.save_slot(slot, description):
            print(f"✓ Saved to slot {slot}")
            
            # Show slot info
            slot_info = self.save_manager.get_slot_info(slot)
            if slot_info:
                print(f"  Frame: {slot_info.frame}")
                print(f"  Time: {slot_info.timestamp}")
        else:
            print(f"✗ Failed to save slot {slot}")
    
    def cmd_load(self, args) -> None:
        """Load state from slot.
        
        Args:
            args: Command arguments (slot)
        """
        if not self.save_manager:
            print("✗ Save manager not initialized")
            return
        
        slot = args.slot
        print(f"Loading from slot {slot}...")
        
        if self.save_manager.load_slot(slot):
            print(f"✓ Loaded from slot {slot}")
        else:
            print(f"✗ Failed to load slot {slot}")
    
    def cmd_list_slots(self, args) -> None:
        """List all save slots.
        
        Args:
            args: Command arguments
        """
        if not self.save_manager:
            print("✗ Save manager not initialized")
            return
        
        print("Save slots:")
        
        slots = self.save_manager.list_slots()
        
        if slots:
            for slot_info in slots:
                print(f"  Slot {slot_info.slot:2d}: {slot_info.description or 'No description'}")
                print(f"    Path: {slot_info.path}")
                print(f"    Time: {time.ctime(slot_info.timestamp)}")
                if slot_info.frame:
                    print(f"    Frame: {slot_info.frame}")
        else:
            print("  No save slots found")
    
    def cmd_reset(self, args) -> None:
        """Reset to title screen.
        
        Args:
            args: Command arguments
        """
        if not self.save_manager:
            print("✗ Save manager not initialized")
            return
        
        print("Resetting to title screen...")
        
        if self.save_manager.reset_to_title_screen():
            print("✓ Reset to title screen")
        else:
            print("✗ Failed to reset")
    
    def run_profile(self, profile_path: Path) -> None:
        """Run a button sequence profile.
        
        Args:
            profile_path: Path to profile JSON file
        """
        import json
        
        if not profile_path.exists():
            print(f"✗ Profile file not found: {profile_path}")
            return
        
        print(f"Running profile: {profile_path}")
        
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            steps = profile.get("steps", [])
            print(f"Profile has {len(steps)} steps")
            
            for i, step in enumerate(steps):
                button = step.get("button")
                action = step.get("action", "tap")
                duration = step.get("duration", 500)
                delay = step.get("delay", 0.5)
                
                print(f"Step {i+1}/{len(steps)}: {action} {button}")
                
                if action == "tap":
                    self.controller.button_tap(button)
                elif action == "hold":
                    self.controller.button_hold(button, duration)
                
                if delay > 0:
                    time.sleep(delay)
            
            print("✓ Profile completed")
            
        except Exception as e:
            print(f"✗ Profile failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="mgba-harness CLI for Pokemon MD agent",
        prog="mgba-harness"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="mgba server host (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="mgba server port (default: 8888)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ping command
    ping_parser = subparsers.add_parser("ping", help="Ping mgba server")
    
    # title command
    title_parser = subparsers.add_parser("title", help="Get game title and info")
    
    # tap command
    tap_parser = subparsers.add_parser("tap", help="Tap a button")
    tap_parser.add_argument("button", help="Button name (A, B, Start, etc.)")
    
    # hold command
    hold_parser = subparsers.add_parser("hold", help="Hold a button")
    hold_parser.add_argument("button", help="Button name")
    hold_parser.add_argument("--duration", type=int, default=500, help="Duration in ms")
    
    # screenshot command
    screenshot_parser = subparsers.add_parser("screenshot", help="Take screenshot")
    screenshot_parser.add_argument("path", nargs="?", help="Output file path")
    
    # read command
    read_parser = subparsers.add_parser("read", help="Read memory range")
    read_parser.add_argument("--domain", help="Memory domain (default: WRAM)")
    read_parser.add_argument("--address", help="Start address (hex, e.g., 0x2000000)")
    read_parser.add_argument("--length", type=int, help="Number of bytes")
    
    # memory-domains command
    mem_parser = subparsers.add_parser("memory-domains", help="List memory domains")
    
    # save command
    save_parser = subparsers.add_parser("save", help="Save state to slot")
    save_parser.add_argument("slot", type=int, help="Slot number")
    save_parser.add_argument("--description", help="Slot description")
    
    # load command
    load_parser = subparsers.add_parser("load", help="Load state from slot")
    load_parser.add_argument("slot", type=int, help="Slot number")
    
    # list-slots command
    list_parser = subparsers.add_parser("list-slots", help="List all save slots")
    
    # reset command
    reset_parser = subparsers.add_parser("reset", help="Reset to title screen")
    
    # profile command
    profile_parser = subparsers.add_parser("profile", help="Run button profile")
    profile_parser.add_argument("path", help="Profile JSON file path")
    
    args = parser.parse_args()
    
    # Create CLI and connect
    cli = MGBACLI()
    cli.controller.host = args.host
    cli.controller.port = args.port
    
    if not cli.connect():
        return 1
    
    # Execute command
    try:
        if args.command == "ping":
            cli.cmd_ping(args)
        elif args.command == "title":
            cli.cmd_title(args)
        elif args.command == "tap":
            cli.cmd_tap(args)
        elif args.command == "hold":
            cli.cmd_hold(args)
        elif args.command == "screenshot":
            cli.cmd_screenshot(args)
        elif args.command == "read":
            cli.cmd_read(args)
        elif args.command == "memory-domains":
            cli.cmd_memory_domains(args)
        elif args.command == "save":
            cli.cmd_save(args)
        elif args.command == "load":
            cli.cmd_load(args)
        elif args.command == "list-slots":
            cli.cmd_list_slots(args)
        elif args.command == "reset":
            cli.cmd_reset(args)
        elif args.command == "profile":
            cli.run_profile(Path(args.path))
        else:
            parser.print_help()
        
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cli.controller.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
