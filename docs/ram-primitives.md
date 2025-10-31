# RAM Primitives for PMD Red Rescue Team (USA/Australia)

This document describes the memory addresses and data structures used for reading game state from Pokemon Mystery Dungeon: Red Rescue Team (USA, Australia) ROM.

**Source:** [Data Crystal RAM Map](https://datacrystal.tcrf.net/wiki/Pokémon_Mystery_Dungeon:_Red_Rescue_Team:RAM_map)

## Address Table

### Player 1 Stats (VERIFIED)

| Name | Address | Type | Size | Description |
|------|---------|------|------|-------------|
| Level | 0x02004199 | u8 | 1 | Player level |
| IQ | 0x0200419C | u16 | 2 | Intelligence Quotient |
| **HP** | **0x0200419E** | **u16** | **2** | **Current HP (VERIFIED: 30/30)** |
| **Max HP** | **0x020041A0** | **u16** | **2** | **Maximum HP (VERIFIED: 30)** |
| Attack | 0x020041A4 | u8 | 1 | Attack stat |
| Sp. Attack | 0x020041A5 | u8 | 1 | Special Attack stat |
| Defense | 0x020041A6 | u8 | 1 | Defense stat |
| Sp. Defense | 0x020041A7 | u8 | 1 | Special Defense stat |
| Experience | 0x020041A8 | u32 | 4 | Experience Points |
| **Belly** | **0x020042CC** | **u8** | **1** | **Current Belly (VERIFIED: 100/100)** |
| **Max Belly** | **0x020042D0** | **u8** | **1** | **Maximum Belly (VERIFIED: 100)** |
| HP Clone | 0x0201BD1A | u16 | 2 | HP mirror/backup value |

### Dungeon State (VERIFIED)

| Name | Address | Type | Size | Description |
|------|---------|------|------|-------------|
| **Floor** | **0x02004139** | **u8** | **1** | **Current floor number (VERIFIED: 1, set to 0xFF to escape)** |
| Turn Counter | 0x02004156 | u8 | 1 | Turn counter (cycles 0x00-0x23) |
| Turns Remaining | 0x0200415A | u16 | 2 | Turns remaining before forced exit |
| Background Music | 0x02007504 | u16 | 2 | Current BGM ID |

### Position (NEEDS VERIFICATION)

| Name | Address | Type | Size | Description |
|------|---------|------|------|-------------|
| Player X | 0x020041F8 | u16 | 2 | X coordinate on current floor (NEEDS VERIFICATION) |
| Player Y | 0x020041FC | u16 | 2 | Y coordinate on current floor (NEEDS VERIFICATION) |

### Other Party Members

| Name | Address | Type | Size | Description |
|------|---------|------|------|-------------|
| P2 Health | 0x020043A6 | u16 | 2 | Player 2 HP |
| P3 Health | 0x020045AE | u16 | 2 | Player 3 HP |
| P4 Health | 0x020047B6 | u16 | 2 | Player 4 HP |
| Wild Pokemon HP | 0x02004BC6 | u16 | 2 | Wild Pokemon health |

### Save Data

| Name | Address | Type | Size | Description |
|------|---------|------|------|-------------|
| Money | 0x02038C08 | u32 | 4 | Money on hand |
| Bank Money | 0x02038C0C | u32 | 4 | Money in bank |
| Team Name | 0x02038C10 | string | 10 | Team name |
| Rescue Points | 0x02038C1C | u32 | 4 | Rescue points |
| Friend Areas | 0x02038C28 | u8[57] | 57 | Friend areas purchased (1 byte each) |
| Time (Hours) | 0x02038C80 | u16 | 2 | Time played—hours |
| Time (Min/Sec) | 0x02038C82 | bytes | 3 | Time played—minutes/seconds |
| Item Storage | 0x020389FA | u16[475] | 950 | Item storage amounts (2 bytes each) |

## Memory Domains

When using the mGBA memory API, addresses must be converted to domain offsets:

- **EWRAM (wram)**: Base `0x02000000`, Size `256KB`
  - Example: Address `0x0200419E` → Domain `wram`, Offset `0x419E`
- **IWRAM (iwram)**: Base `0x03000000`, Size `32KB`
  - Example: Address `0x03000100` → Domain `iwram`, Offset `0x100`

## RAM-First Design Principles

1. **Primary Source**: RAM reads are the authoritative source for game state
2. **Vision Assistance**: Vision used to supplement RAM when addresses are unknown
3. **Validation**: Vision detections validated against known RAM values
4. **Fallback Chain**: RAM → Vision → Conservative Action → Log Warning
5. **Performance**: RAM reads prioritized for speed-critical decisions

## Common RAM Patterns

### Health Management
- Check HP/Belly ratios for food decisions
- Monitor status effects for cure timing
- Track max values for percentage calculations

### Navigation
- Position changes indicate movement success
- Floor changes trigger map updates
- Dungeon ID changes indicate transitions

### Inventory Management
- Count items before bulk operations
- Check specific slots for critical items
- Monitor money for shop decisions

### Team Management
- Track team size for formation decisions
- Monitor leader index for control flow
- Check individual member status

## Verification Status

✅ **VERIFIED** (tested with live game at Tiny Woods F1):
- HP: 30/30
- Max HP: 30
- Belly: 100/100
- Max Belly: 100
- Floor: 1

⚠️ **NEEDS VERIFICATION**:
- Player X/Y position coordinates
- Status effects address
- Other party member stats
