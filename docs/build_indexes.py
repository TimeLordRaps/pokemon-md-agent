#!/usr/bin/env python3
"""
Dashboard Index Builder

Builds navigation indexes for the PMD-Red Agent dashboard documentation.
Generates species, items, and dungeons index pages with links to individual entries.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class IndexEntry:
    """Represents an entry in a dashboard index."""
    name: str
    path: str
    description: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DashboardIndexer:
    """Builds indexes for dashboard documentation."""

    def __init__(self, docs_root: Path):
        self.docs_root = docs_root
        self.species_dir = docs_root / "species"
        self.items_dir = docs_root / "items"
        self.dungeons_dir = docs_root / "dungeons"

    def build_species_index(self) -> List[IndexEntry]:
        """Build index of Pokemon species."""
        species = []

        # Read species data from config if available
        config_path = Path("../../config/addresses/pmd_red_us_v1.json")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Extract species from RAM addresses
                species_addresses = config.get("species", {})
                for species_name, addr_info in species_addresses.items():
                    if isinstance(addr_info, dict) and "description" in addr_info:
                        species.append(IndexEntry(
                            name=species_name,
                            path=f"species/{species_name.lower().replace(' ', '_')}.md",
                            description=addr_info["description"],
                            metadata={"address": addr_info.get("address")}
                        ))
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback to basic species list
        if not species:
            basic_species = [
                "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
                "Squirtle", "Wartortle", "Blastoise", "Pikachu", "Raichu", "Eevee",
                "Vaporeon", "Jolteon", "Flareon", "Snorlax", "Mew", "Mewtwo"
            ]
            species = [
                IndexEntry(
                    name=name,
                    path=f"species/{name.lower()}.md",
                    description=f"Pokemon species: {name}"
                ) for name in basic_species
            ]

        return sorted(species, key=lambda x: x.name)

    def build_items_index(self) -> List[IndexEntry]:
        """Build index of items."""
        items = []

        # Basic item categories
        item_categories = {
            "Consumables": ["Potion", "Super Potion", "Hyper Potion", "Max Potion", "Full Restore"],
            "Held Items": ["Power Band", "Special Band", "Defense Scarf", "Zinc Band"],
            "TMs": ["TM01 Focus Punch", "TM02 Dragon Claw", "TM03 Water Pulse"],
            "Key Items": ["Treasure Bag", "Key", "Music Box", "Link Box"]
        }

        for category, item_list in item_categories.items():
            for item in item_list:
                items.append(IndexEntry(
                    name=item,
                    path=f"items/{item.lower().replace(' ', '_')}.md",
                    description=f"{category}: {item}",
                    metadata={"category": category}
                ))

        return sorted(items, key=lambda x: x.name)

    def build_dungeons_index(self) -> List[IndexEntry]:
        """Build index of dungeons."""
        dungeons = []

        # Basic dungeon list
        dungeon_list = [
            ("Tiny Woods", "Beginner dungeon with basic Pokemon"),
            ("Thunderwave Cave", "Electric-type dungeon"),
            ("Mt. Steel", "Steel-type mountain dungeon"),
            ("Sinister Woods", "Dark-type forest dungeon"),
            ("Silent Chasm", "Ghost-type dungeon"),
            ("Mt. Thunder", "Electric-type mountain dungeon"),
            ("Great Canyon", "Large canyon dungeon"),
            ("Lapis Cave", "Water-type cave dungeon"),
            ("Mt. Blaze", "Fire-type volcano dungeon"),
            ("Frosty Forest", "Ice-type forest dungeon"),
            ("Mt. Freeze", "Ice-type mountain dungeon"),
            ("Magma Cavern", "Fire-type cave dungeon"),
            ("Sky Tower", "Flying-type tower dungeon"),
            ("Uproar Forest", "Normal-type forest dungeon"),
            ("Howling Forest", "Dark-type forest dungeon"),
            ("Stormy Sea", "Water-type sea dungeon"),
            ("Silver Trench", "Water-type deep sea dungeon"),
            ("Meteor Cave", "Rock-type cave dungeon"),
            ("Fiery Field", "Fire-type field dungeon"),
            ("Lightning Field", "Electric-type field dungeon"),
            ("Northwind Field", "Ice-type field dungeon"),
            ("Mt. Faraway", "Final mountain dungeon")
        ]

        for name, description in dungeon_list:
            dungeons.append(IndexEntry(
                name=name,
                path=f"dungeons/{name.lower().replace(' ', '_').replace('.', '')}.md",
                description=description
            ))

        return sorted(dungeons, key=lambda x: x.name)

    def generate_index_markdown(self, title: str, entries: List[IndexEntry]) -> str:
        """Generate markdown for an index page."""
        lines = [f"# {title}", ""]

        # Group entries by first letter for large indexes
        if len(entries) > 20:
            current_letter = ""
            for entry in entries:
                first_letter = entry.name[0].upper()
                if first_letter != current_letter:
                    current_letter = first_letter
                    lines.append(f"## {current_letter}")
                    lines.append("")
        else:
            lines.append("## Entries")
            lines.append("")

        for entry in entries:
            lines.append(f"- **[{entry.name}]({entry.path})**")
            if entry.description:
                lines.append(f"  - {entry.description}")
            lines.append("")

        return "\n".join(lines)

    def build_main_index(self) -> str:
        """Build the main dashboard index."""
        lines = [
            "# PMD-Red Agent Dashboard",
            "",
            "Interactive documentation and data explorer for Pokemon Mystery Dungeon: Red Rescue Team.",
            "",
            "## Navigation",
            "",
            "- [Species Index](species/index.md) - Pokemon species information",
            "- [Items Index](items/index.md) - Items and equipment",
            "- [Dungeons Index](dungeons/index.md) - Dungeon information",
            "",
            "## Data Sources",
            "",
            "- Live agent observations and trajectories",
            "- Game memory analysis and RAM decoding",
            "- Vision processing and sprite detection",
            "- RAG system retrieval and embeddings",
            "",
            "## Configuration",
            "",
            "Dashboard updates are controlled by the agent configuration:",
            "",
            "```python",
            "# In agent_core.py AgentConfig",
            "dashboard = DashboardConfig(",
            "    enabled=True,",
            "    branch='gh-pages',",
            "    site_root='docs/',",
            "    flush_seconds=30.0,",
            "    max_batch_bytes=8*1024*1024,  # 8MB",
            "    max_files_per_minute=30",
            ")",
            "```",
            "",
            "---",
            "",
            "*Generated by PMD-Red Agent*"
        ]

        return "\n".join(lines)

    def build_all_indexes(self):
        """Build all index files."""
        # Create directories
        for dir_path in [self.species_dir, self.items_dir, self.dungeons_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Build and write indexes
        indexes = [
            ("index.md", self.build_main_index()),
            ("species/index.md", self.generate_index_markdown("Pokemon Species", self.build_species_index())),
            ("items/index.md", self.generate_index_markdown("Items", self.build_items_index())),
            ("dungeons/index.md", self.generate_index_markdown("Dungeons", self.build_dungeons_index()))
        ]

        for rel_path, content in indexes:
            full_path = self.docs_root / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Generated {rel_path}")


def main():
    """Main entry point."""
    docs_root = Path(__file__).parent / "docs"

    indexer = DashboardIndexer(docs_root)
    indexer.build_all_indexes()

    print(f"Dashboard indexes built in {docs_root}")


if __name__ == "__main__":
    main()