"""Analyze WRAM dumps to identify entity structure offsets."""
from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class EntityRecord:
    """Simple container for decoded entity data."""

    offset: int
    species_id: int
    pos_x: int
    pos_y: int
    hp_current: int
    hp_max: int


@dataclass
class Candidate:
    """Candidate description for an entity array."""

    base_offset: int
    struct_size: int
    records: List[EntityRecord]
    scanned_slots: int
    leading_empty_slots: int

    @property
    def score(self) -> int:
        return len(self.records)

    @property
    def sort_key(self) -> Tuple[int, int, int]:
        # Higher score preferred, fewer leading empties preferred, larger structs last.
        return (self.score, -self.leading_empty_slots, -self.struct_size)


def hex_dump(data: bytes, offset: int, length: int = 32) -> None:
    """Print a hex/ASCII dump for manual inspection."""
    for i in range(0, length, 16):
        chunk = data[offset + i : offset + i + 16]
        hex_part = " ".join(f"{b:02X}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"{offset + i:04X}: {hex_part:<48} {ascii_part}")


def _is_plausible_entity(
    species_id: int,
    pos_x: int,
    pos_y: int,
    hp_current: int,
    hp_max: int,
) -> bool:
    if species_id == 0:
        return True  # empty slot is allowed; handled upstream
    return (
        1 <= species_id <= 386
        and 0 <= pos_x <= 64
        and 0 <= pos_y <= 64
        and 1 <= hp_current <= 999
        and hp_current <= hp_max <= 999
    )


def _extract_candidate(
    data: bytes,
    *,
    base_offset: int,
    struct_size: int,
    max_entities: int,
) -> Optional[Candidate]:
    records: List[EntityRecord] = []
    empty_slots = 0
    leading_empty = 0
    scanned_slots = 0

    for idx in range(max_entities):
        offset = base_offset + idx * struct_size
        if offset + 10 > len(data):
            break

        species_id = struct.unpack_from("<H", data, offset)[0]
        pos_x = struct.unpack_from("<H", data, offset + 2)[0]
        pos_y = struct.unpack_from("<H", data, offset + 4)[0]
        hp_current = struct.unpack_from("<H", data, offset + 6)[0]
        hp_max = struct.unpack_from("<H", data, offset + 8)[0]
        scanned_slots += 1

        if species_id == 0:
            empty_slots += 1
            if not records:
                leading_empty += 1
            # Allow a couple of empty slots, but give up if nothing looks valid.
            if empty_slots > 3 and not records:
                return None
            continue

        if not _is_plausible_entity(species_id, pos_x, pos_y, hp_current, hp_max):
            if records:
                break
            return None

        empty_slots = 0
        records.append(
            EntityRecord(
                offset=offset,
                species_id=species_id,
                pos_x=pos_x,
                pos_y=pos_y,
                hp_current=hp_current,
                hp_max=hp_max,
            )
        )

    if not records:
        return None

    return Candidate(
        base_offset=base_offset,
        struct_size=struct_size,
        records=records,
        scanned_slots=scanned_slots,
        leading_empty_slots=leading_empty,
    )


def scan_for_entity_candidates(
    data: bytes,
    *,
    struct_sizes: Iterable[int] = (32, 48, 64),
    max_entities: int = 20,
    alignment: int = 2,
) -> List[Candidate]:
    """Return sorted list of plausible entity array candidates."""
    candidates: List[Candidate] = []

    for struct_size in struct_sizes:
        for offset in range(0, len(data) - struct_size, alignment):
            candidate = _extract_candidate(
                data,
                base_offset=offset,
                struct_size=struct_size,
                max_entities=max_entities,
            )
            if candidate:
                candidates.append(candidate)

    # Deduplicate by base offset/struct size pair keeping the best.
    dedup: Dict[Tuple[int, int], Candidate] = {}
    for cand in candidates:
        key = (cand.base_offset, cand.struct_size)
        prev = dedup.get(key)
        if (
            prev is None
            or cand.score > prev.score
            or (cand.score == prev.score and cand.leading_empty_slots < prev.leading_empty_slots)
        ):
            dedup[key] = cand

    return sorted(dedup.values(), key=lambda c: c.sort_key, reverse=True)


def analyze_dump(path: Path) -> Dict[str, Optional[int]]:
    print("=" * 60)
    print(f"Analyzing: {path.name}")
    print("=" * 60)

    data = path.read_bytes()
    print(f"Size: {len(data)} bytes\n")

    candidates = scan_for_entity_candidates(data)
    if not candidates:
        print("⚠️  No entity candidates found.")
        print("First 256 bytes for manual inspection:")
        hex_dump(data, 0, min(256, len(data)))
        return {"entity_array_base": None, "entity_struct_size": None, "max_entities": None}

    for idx, cand in enumerate(candidates[:5], start=1):
        print(
            f"Candidate {idx}: base=0x{cand.base_offset:04X}, struct_size={cand.struct_size}, "
            f"leading_empties={cand.leading_empty_slots}"
        )
        print(f"  Records decoded: {cand.score} (scanned {cand.scanned_slots} slots)")
        for record in cand.records[:5]:
            print(
                f"    @0x{record.offset:04X} -> species={record.species_id:03d} "
                f"pos=({record.pos_x:02d},{record.pos_y:02d}) "
                f"hp={record.hp_current}/{record.hp_max}"
            )
        print("  Hex dump around first record:")
        hex_dump(data, cand.records[0].offset, 32)
        print()

    best = candidates[0]
    return {
        "entity_array_base": best.base_offset,
        "entity_struct_size": best.struct_size,
        "max_entities": best.scanned_slots,
    }


def consolidate_results(results: Dict[str, Dict[str, Optional[int]]]) -> Dict[str, Optional[int]]:
    bases = Counter()
    struct_sizes = Counter()
    max_entities_values = []

    for dump_name, metrics in results.items():
        base = metrics.get("entity_array_base")
        size = metrics.get("entity_struct_size")
        max_entities = metrics.get("max_entities")

        if base is not None:
            bases[base] += 1
        if size is not None:
            struct_sizes[size] += 1
        if max_entities:
            max_entities_values.append(max_entities)

    summary = {
        "entity_array_base": bases.most_common(1)[0][0] if bases else None,
        "entity_struct_size": struct_sizes.most_common(1)[0][0] if struct_sizes else None,
        "max_entities": int(sum(max_entities_values) / len(max_entities_values))
        if max_entities_values
        else None,
    }

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    sandbox = Path(__file__).resolve().parent
    dumps = sorted((sandbox / "sample_wram_dumps").glob("*.bin"))

    if not dumps:
        print("⚠️  No dump files found in sample_wram_dumps/.")
        return

    per_dump_results: Dict[str, Dict[str, Optional[int]]] = {}
    for dump_path in dumps:
        per_dump_results[dump_path.name] = analyze_dump(dump_path)

    summary = consolidate_results(per_dump_results)

    output_json = sandbox / "analysis_summary.json"
    payload = {"dumps": per_dump_results, "summary": summary}
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"\n✓ Wrote summary to {output_json}")


if __name__ == "__main__":
    main()
