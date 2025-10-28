"""ASCII renderer for creating text-based representations of game state.

This module creates deterministic ASCII art representations of:
- Environment + entities (with species codes)
- Map only (no entities)
- Environment + grid overlay (every 5 tiles)
- Meta HUD (HP/Belly/PP/missions)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from .grid_parser import GridFrame, TileType
from ..environment.ram_decoders import RAMSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ASCIIRenderOptions:
    """Options for ASCII rendering."""
    width: int = 80  # Character width
    height: int = 24  # Character height
    show_grid_indices: bool = False  # Show tile indices every 5 tiles
    show_entities: bool = True  # Show entities on map
    show_items: bool = True  # Show items on map
    show_traps: bool = True  # Show traps on map
    legend_width: int = 20  # Width reserved for legend
    show_meta: bool = True  # Show meta information
    use_species_codes: bool = True  # Use 2-letter species codes


class ASCIIRenderer:
    """Creates ASCII representations of game state."""
    
    def __init__(self, options: Optional[ASCIIRenderOptions] = None):
        """Initialize ASCII renderer.
        
        Args:
            options: Rendering options
        """
        self.options = options or ASCIIRenderOptions()
        
        # Species code mapping (2-letter abbreviations)
        self.species_codes = {
            1: "Ba", 2: "Iv", 3: "Ve",  # Bulbasaur line
            4: "Cm", 5: "Cl", 6: "Cz",  # Charmander line
            7: "Sq", 8: "Wt", 9: "Bl",  # Squirtle line
            10: "Ca", 11: "Me", 12: "Bu",  # Caterpie line
            13: "We", 14: "Ka", 15: "Be",  # Weedle line
            16: "Pi", 17: "Po", 18: "Pt",  # Pidgey line
            19: "Ra", 20: "Rt",  # Rattata line
            # Add more as needed
        }
        
        # Item symbols
        self.item_symbols = {
            1: "S",  # Stick
            2: "I",  # Iron Thorn
            3: "G",  # Silver Spike
            4: "B",  # Bullet Seed
            9: "A",  # Apple
            10: "a",  # Great Apple
            11: "O",  # Orange
            13: "o",  # Large Orange
            19: "T",  # Training Seed
            20: "O",  # Oran Berry
            # Add more as needed
        }
        
        logger.info("ASCIIRenderer initialized")
    
    def render_environment_with_entities(
        self,
        grid: GridFrame,
        snapshot: RAMSnapshot,
        output_path: Optional[Path] = None,
    ) -> str:
        """Render environment map with entities overlaid.
        
        Args:
            grid: Grid frame
            snapshot: RAM snapshot
            output_path: Optional output file path
            
        Returns:
            ASCII string representation
        """
        lines = []
        
        # Header
        lines.append("=" * (self.options.width - self.options.legend_width))
        lines.append(f"DUNGEON: {snapshot.player_state.floor_number} | TURN: {snapshot.player_state.turn_counter}")
        lines.append("=" * (self.options.width - self.options.legend_width))
        
        # Main map area
        map_area_height = self.options.height - 8  # Reserve space for header/legend
        map_lines = self._render_map_area(grid, snapshot, show_entities=True)
        
        # Add grid indices if requested
        if self.options.show_grid_indices:
            map_lines = self._add_grid_indices(map_lines)
        
        # Append map lines
        for _i, line in enumerate(map_lines[:map_area_height]):
            lines.append(line.ljust(self.options.width - self.options.legend_width))
        
        # Add legend
        lines.extend(self._render_legend())
        
        # Add meta information
        if self.options.show_meta:
            lines.extend(self._render_meta(snapshot))
        
        ascii_text = "\n".join(lines)
        
        # Save to file if requested
        if output_path:
            self._save_to_file(ascii_text, output_path)
        
        return ascii_text
    
    def render_map_only(
        self,
        grid: GridFrame,
        output_path: Optional[Path] = None,
    ) -> str:
        """Render map only (no entities).
        
        Args:
            grid: Grid frame
            output_path: Optional output file path
            
        Returns:
            ASCII string representation
        """
        lines = []
        
        # Header
        lines.append("=" * (self.options.width - self.options.legend_width))
        lines.append("MAP ONLY (No Entities)")
        lines.append("=" * (self.options.width - self.options.legend_width))
        
        # Map area
        map_area_height = self.options.height - 5
        map_lines = self._render_map_area(grid, None, show_entities=False)
        
        # Add grid indices if requested
        if self.options.show_grid_indices:
            map_lines = self._add_grid_indices(map_lines)
        
        # Append map lines
        for _i, line in enumerate(map_lines[:map_area_height]):
            lines.append(line.ljust(self.options.width - self.options.legend_width))
        
        # Add legend
        lines.extend(self._render_legend())
        
        ascii_text = "\n".join(lines)
        
        # Save to file if requested
        if output_path:
            self._save_to_file(ascii_text, output_path)
        
        return ascii_text
    
    def render_environment_with_grid(
        self,
        grid: GridFrame,
        snapshot: RAMSnapshot,
        output_path: Optional[Path] = None,
    ) -> str:
        """Render environment with grid indices overlaid.
        
        Args:
            grid: Grid frame
            snapshot: RAM snapshot
            output_path: Optional output file path
            
        Returns:
            ASCII string representation
        """
        # Create a copy of options with grid indices enabled
        options = ASCIIRenderOptions(**self.options.__dict__)
        options.show_grid_indices = True
        
        # Temporarily use modified options
        original_options = self.options
        self.options = options
        
        try:
            ascii_text = self.render_environment_with_entities(grid, snapshot, output_path)
        finally:
            self.options = original_options
        
        return ascii_text
    
    def render_meta(
        self,
        snapshot: RAMSnapshot,
        output_path: Optional[Path] = None,
    ) -> str:
        """Render meta HUD information only.
        
        Args:
            snapshot: RAM snapshot
            output_path: Optional output file path
            
        Returns:
            ASCII string representation
        """
        lines = []
        
        # Header
        lines.append("=" * self.options.width)
        lines.append("HUD METADATA")
        lines.append("=" * self.options.width)
        
        # Add meta information
        lines.extend(self._render_meta(snapshot))
        
        ascii_text = "\n".join(lines)
        
        # Save to file if requested
        if output_path:
            self._save_to_file(ascii_text, output_path)
        
        return ascii_text
    
    def _render_map_area(
        self,
        grid: GridFrame,
        snapshot: Optional[RAMSnapshot],
        show_entities: bool,
    ) -> List[str]:
        """Render the main map area.
        
        Args:
            grid: Grid frame
            snapshot: RAM snapshot
            show_entities: Whether to show entities
            
        Returns:
            List of ASCII lines
        """
        lines = []
        
        for y in range(min(grid.height, self.options.height - 10)):
            line_chars = []
            
            for x in range(min(grid.width, self.options.width - self.options.legend_width - 5)):
                # Get base tile
                tile_char = self._tile_to_char(grid.tiles[y][x].tile_type)
                
                # Overlay entities if requested
                if show_entities and snapshot:
                    entity_char = self._get_entity_char_at(grid, x, y, snapshot)
                    if entity_char:
                        tile_char = entity_char
                    
                    item_char = self._get_item_char_at(grid, x, y, snapshot)
                    if item_char:
                        tile_char = item_char
                
                line_chars.append(tile_char)
            
            lines.append("".join(line_chars))
        
        return lines
    
    def _tile_to_char(self, tile_type: TileType) -> str:
        """Convert tile type to character.
        
        Args:
            tile_type: Tile type enum
            
        Returns:
            Character representation
        """
        char_map = {
            TileType.WALL: "#",
            TileType.FLOOR: ".",
            TileType.WATER: "~",
            TileType.LAVA: "^",
            TileType.STAIRS: ">",
            TileType.TRAP: "!",
            TileType.ITEM: "*",
            TileType.MONSTER: "M",
            TileType.SHOP: "$",
            TileType.UNKNOWN: "?",
        }
        
        return char_map.get(tile_type, "?")
    
    def _get_entity_char_at(
        self,
        _grid: GridFrame,
        x: int,
        y: int,
        snapshot: RAMSnapshot,
    ) -> Optional[str]:
        """Get character for entity at position.
        
        Args:
            grid: Grid frame
            x: X coordinate
            y: Y coordinate
            snapshot: RAM snapshot
            
        Returns:
            Character or None
        """
        # Check player position
        if (x == snapshot.player_state.player_tile_x and 
            y == snapshot.player_state.player_tile_y):
            return "@"  # Player
        
        # Check partner position
        if (x == snapshot.player_state.partner_tile_x and 
            y == snapshot.player_state.partner_tile_y):
            return "P"  # Partner
        
        # Check other entities
        for entity in snapshot.entities:
            if entity.tile_x == x and entity.tile_y == y and entity.visible:
                if self.options.use_species_codes:
                    # Get species code
                    code = self.species_codes.get(entity.species_id, f"{entity.species_id:02}")
                    return code[:2]  # 2-character code
                else:
                    # Use generic monster symbol with affiliation indicator
                    if entity.affiliation == 0:  # Ally
                        return "A"
                    else:  # Enemy
                        return "E"
        
        return None
    
    def _get_item_char_at(
        self,
        _grid: GridFrame,
        x: int,
        y: int,
        snapshot: RAMSnapshot,
    ) -> Optional[str]:
        """Get character for item at position.
        
        Args:
            grid: Grid frame
            x: X coordinate
            y: Y coordinate
            snapshot: RAM snapshot
            
        Returns:
            Character or None
        """
        for item in snapshot.items:
            if item.tile_x == x and item.tile_y == y:
                symbol = self.item_symbols.get(item.item_id, "?")
                return symbol
        
        return None
    
    def _add_grid_indices(self, map_lines: List[str]) -> List[str]:
        """Add grid indices to map lines.
        
        Args:
            map_lines: Original map lines
            
        Returns:
            Map lines with grid indices
        """
        if not map_lines:
            return map_lines
        
        lines_with_indices = []
        
        for y, line in enumerate(map_lines):
            # Add Y coordinate every 5 rows
            if y % 5 == 0:
                index_line = f"{y:2d}|" + line
            else:
                index_line = "   " + line
            
            # Add X coordinates on first line
            if y == 0:
                x_coords = "   "
                for x in range(0, len(line), 5):
                    x_coords += f"{x:2d}   "
                lines_with_indices.append(x_coords)
            
            lines_with_indices.append(index_line)
        
        return lines_with_indices
    
    def _render_legend(self) -> List[str]:
        """Render the legend section.
        
        Returns:
            List of legend lines
        """
        lines = []
        lines.append("")
        lines.append("LEGEND:")
        lines.append("-" * 18)
        lines.append("@ = Player")
        lines.append("P = Partner")
        lines.append("# = Wall")
        lines.append(". = Floor")
        lines.append("~ = Water")
        lines.append("^ = Lava")
        lines.append("> = Stairs")
        lines.append("* = Item")
        lines.append("! = Trap")
        lines.append("## = Pokemon (2-letter code)")
        lines.append("")
        
        return lines
    
    def _render_meta(self, snapshot: RAMSnapshot) -> List[str]:
        """Render meta information section.
        
        Args:
            snapshot: RAM snapshot
            
        Returns:
            List of meta lines
        """
        lines = []
        
        lines.append("")
        lines.append("STATUS:")
        lines.append("-" * 18)
        
        # Player info
        lines.append(f"Player HP: {snapshot.party_status.leader_hp}/{snapshot.party_status.leader_hp_max}")
        lines.append(f"Player Belly: {snapshot.party_status.leader_belly}")
        
        # Partner info
        lines.append(f"Partner HP: {snapshot.party_status.partner_hp}/{snapshot.party_status.partner_hp_max}")
        lines.append(f"Partner Belly: {snapshot.party_status.partner_belly}")
        
        # Dungeon info
        lines.append(f"Floor: {snapshot.player_state.floor_number}")
        lines.append(f"Dungeon ID: {snapshot.player_state.dungeon_id}")
        lines.append(f"Turn: {snapshot.player_state.turn_counter}")
        
        # Position info
        lines.append(f"Pos: ({snapshot.player_state.player_tile_x}, {snapshot.player_state.player_tile_y})")
        
        # Entity count
        lines.append(f"Enemies: {len([e for e in snapshot.entities if e.affiliation != 0])}")
        lines.append(f"Items: {len(snapshot.items)}")
        
        lines.append("")
        
        return lines
    
    def _save_to_file(self, ascii_text: str, output_path: Path) -> None:
        """Save ASCII text to file.
        
        Args:
            ascii_text: ASCII text to save
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ascii_text)
        
        logger.info("Saved ASCII render to %s", output_path)
    
    def create_multi_view_output(
        self,
        grid: GridFrame,
        snapshot: RAMSnapshot,
        output_dir: Path,
        prefix: str = "scene",
    ) -> Dict[str, Path]:
        """Create all four view variants and return their paths.
        
        Args:
            grid: Grid frame
            snapshot: RAM snapshot
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping view name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # 1. Environment + entities
        env_path = output_dir / f"{prefix}_environment.txt"
        self.render_environment_with_entities(grid, snapshot, env_path)
        paths["environment"] = env_path
        
        # 2. Map only
        map_path = output_dir / f"{prefix}_map_only.txt"
        self.render_map_only(grid, map_path)
        paths["map_only"] = map_path
        
        # 3. Environment + grid
        grid_path = output_dir / f"{prefix}_env_grid.txt"
        self.render_environment_with_grid(grid, snapshot, grid_path)
        paths["env_grid"] = grid_path
        
        # 4. Meta HUD
        meta_path = output_dir / f"{prefix}_meta.txt"
        self.render_meta(snapshot, meta_path)
        paths["meta"] = meta_path
        
        logger.info("Created %d ASCII view files in %s", len(paths), output_dir)
        
        return paths
