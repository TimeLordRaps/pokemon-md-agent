import sys,json,struct
sys.path.insert(0,'src')
from environment.mgba_controller import MGBAController

config = json.load(open('config/addresses/pmd_red_us_v1.json'))['addresses']
controller = MGBAController()

if not controller.connect_with_retry():
    print('Failed to connect to mGBA')
    sys.exit(1)


def read_bytes(domain, address, length):
    data = controller.memory_domain_read_range(domain, address, length)
    if data is None:
        raise RuntimeError(f'Read failed for {domain}@{address}')
    return data


def read_uint(domain, address, size):
    data = read_bytes(domain, address, size)
    if size == 1:
        return data[0]
    return int.from_bytes(data, 'little')

print('--- Player State ---')
player = config['player_state']
for key in ['floor_number', 'dungeon_id', 'turn_counter', 'player_tile_x', 'player_tile_y', 'partner_tile_x', 'partner_tile_y', 'room_flag']:
    info = player[key]
    val = read_uint(info['domain'], info['address'], info['size'])
    print(f"{key}: {val}")

print('\n--- Party Status ---')
party = config['party_status']
for label, prefix in [('Leader', 'leader'), ('Partner', 'partner')]:
    print(label)
    for field in ['hp', 'hp_max', 'belly']:
        info = party[f'{prefix}_{field}']
        val = read_uint(info['domain'], info['address'], info['size'])
        print(f"  {field}: {val}")

print('\n--- Map Data ---')
map_data = config['map_data']
for key in ['camera_origin_x', 'camera_origin_y', 'weather_state', 'turn_phase', 'stairs_x', 'stairs_y']:
    info = map_data[key]
    val = read_uint(info['domain'], info['address'], info['size'])
    print(f"{key}: {val}")

entities = config['entities']
monster_count = read_uint(entities['monster_count']['domain'], entities['monster_count']['address'], entities['monster_count']['size'])
print('\nmonster_count:', monster_count)
monster_ptr = read_uint(entities['monster_list_ptr']['domain'], entities['monster_list_ptr']['address'], entities['monster_list_ptr']['size'])
print('monster_list_ptr:', hex(monster_ptr))
struct_size = entities['monster_struct_size']['value']
fields = entities['monster_fields']

if monster_count > 0 and monster_ptr != 0:
    wram_offset = monster_ptr - 0x02000000
    print('monster list WRAM offset:', hex(wram_offset))
    data = read_bytes('WRAM', wram_offset, struct_size)

    def get_field(name, size):
        off = fields[name]['offset']
        if size == 1:
            return data[off]
        if size == 2:
            return int.from_bytes(data[off:off+2], 'little')
        return int.from_bytes(data[off:off+4], 'little')

    print(' first monster species:', get_field('species_id', 2), 'level:', get_field('level', 1), 'hp:', get_field('hp_current', 2))
else:
    print('No monsters decoded (pointer zero or count 0)')

controller.disconnect()
