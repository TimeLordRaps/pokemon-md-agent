# WRAM Decoder Fix – Findings

## Problem
- Existing RAM decoder returned default entity values (species 0, HP 100/100, position (0, 0)).
- Root cause suspected to be incorrect offsets for the live entity array within WRAM.

## Actions Taken
- Built capture utility (`capture_dumps.py`) to export WRAM slices via the existing `MGBAController`.
- Authored analysis tooling (`analyze_dumps.py`) that scans binary dumps for plausible entity structures using species, position, and HP heuristics.
- Implemented `WRAMDecoderV2` with automatic offset discovery and decoding logic.
- Added `test_decoder.py` harness featuring both a synthetic regression scenario and optional real-dump decoding.

## Current Findings
- Entity slots appear to use 32-byte aligned structures; the analyzer searches for 32/48/64-byte candidates and has guard rails for empty slots.
- Automatic scanning favours offsets that produce multiple plausible Pokémon entries (species 1–386, tile bounds 0–64, HP 1–999).
- When `analysis_summary.json` is generated (from real dumps), the decoder will persist discovered offsets for reuse.

## Outstanding Tasks
- **Live dump capture required:** The current environment cannot interface with mGBA, so `sample_wram_dumps/*.bin` still need to be produced on a machine with the emulator bridge running.
- **Confirm offsets:** After capturing dumps, run `python analyze_dumps.py > analysis_results.txt` to produce `analysis_summary.json`; the summary should list the final `entity_array_base`, `entity_struct_size`, and `max_entities`.
- **Enemy/ally flags:** Additional fields (affiliation, visibility, level) remain to be mapped once more structure bytes are observed.

## Validation Plan
1. Capture three dumps (two rooms + combat) with `capture_dumps.py`.
2. Run `analyze_dumps.py`, inspect top candidates in `analysis_results.txt`, and verify hex dumps align with expected Pokémon data.
3. Execute `python test_decoder.py` to validate decoding against synthetic data and collected dumps. Expect ≥1 entity during combat dump.
4. Integrate offsets into `src/environment/ram_decoders.py` once confirmed, plus add automated tests covering entity extraction.

## Next Steps for Integration Team
1. Run capture/analysis steps on active game session and commit resulting `analysis_summary.json` for provenance.
2. Port `Entity` parsing logic (including slot metadata) into production decoder; extend with additional fields uncovered during analysis.
3. Update `ram_watch.py` and related consumers to instantiate the new decoder.
4. Add regression test covering a known-good WRAM snapshot to prevent future offset regressions.
