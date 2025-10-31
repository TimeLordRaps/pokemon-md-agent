"""Lightweight test harness for WRAMDecoderV2."""
from __future__ import annotations

import struct
from pathlib import Path

from decoder_v2 import DecoderConfig, WRAMDecoderV2, load_default_decoder


def build_synthetic_dump() -> bytes:
    """Create a synthetic dump to validate structural assumptions."""
    buffer = bytearray(512)
    base_offset = 0x0120
    struct_size = 32

    def write_entity(slot: int, species: int, x: int, y: int, hp: int, hp_max: int) -> None:
        offset = base_offset + slot * struct_size
        struct.pack_into("<H", buffer, offset, species)
        struct.pack_into("<H", buffer, offset + 2, x)
        struct.pack_into("<H", buffer, offset + 4, y)
        struct.pack_into("<H", buffer, offset + 6, hp)
        struct.pack_into("<H", buffer, offset + 8, hp_max)

    write_entity(0, 1, 5, 6, 35, 40)
    write_entity(1, 25, 10, 14, 20, 25)
    write_entity(2, 150, 20, 18, 88, 88)
    # Leave remaining slots empty (species=0 by default).

    return bytes(buffer)


def run_synthetic_test() -> None:
    print("=== Synthetic Test ===")
    data = build_synthetic_dump()
    decoder = WRAMDecoderV2(DecoderConfig(max_entities=10))
    entities = decoder.decode_entities(data)
    for entity in entities:
        print(
            f"[slot {entity.slot_index:02d}] species={entity.species_id:03d} "
            f"pos=({entity.position_x},{entity.position_y}) "
            f"hp={entity.hp_current}/{entity.hp_max} "
            f"offset=0x{entity.raw_offset:04X}"
        )
    assert len(entities) == 3, "Synthetic dump should return 3 entities"
    print("PASS Synthetic test passed\n")


def run_real_dumps() -> None:
    sandbox = Path(__file__).resolve().parent
    dumps = sorted((sandbox / "sample_wram_dumps").glob("*.bin"))
    if not dumps:
        print("⚠️  No real dump files found; synthetic test only.")
        return

    for dump_path in dumps:
        print(f"=== Decoding {dump_path.name} ===")
        decoder = load_default_decoder()
        data = dump_path.read_bytes()
        entities = decoder.decode_entities(data)
        print(f"Found {len(entities)} entities")
        for entity in entities:
            print(
                f"  [slot {entity.slot_index:02d}] species={entity.species_id:03d} "
                f"pos=({entity.position_x},{entity.position_y}) "
                f"hp={entity.hp_current}/{entity.hp_max} "
                f"offset=0x{entity.raw_offset:04X}"
            )
        print()


def main() -> None:
    run_synthetic_test()
    run_real_dumps()


if __name__ == "__main__":
    main()
