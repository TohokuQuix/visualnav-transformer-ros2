"""Map frozen base-frame maneuver paths using the verified nominal anchor pose."""
import argparse
import json
import math
from pathlib import Path


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'changed fixed Path: {path}')
    else:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state-folder', type=Path, required=True)
    p.add_argument('--nominal-map', type=Path, required=True)
    a = p.parse_args()
    base = json.loads((a.state_folder / 'nominal-base-path.json').read_text())
    fixed = json.loads(a.nominal_map.read_text())
    dx, dy = base[1][0]-base[0][0], base[1][1]-base[0][1]
    mx, my = fixed[1][0]-fixed[0][0], fixed[1][1]-fixed[0][1]
    yaw = math.atan2(my, mx) - math.atan2(dy, dx)
    c, s = math.cos(yaw), math.sin(yaw)
    for name in ('nominal', 'slight_left', 'slight_right', 'strong_left', 'strong_right'):
        points = json.loads((a.state_folder / f'{name}-base-path.json').read_text())
        transformed = [[float(fixed[0][0]+c*x-s*y), float(fixed[0][1]+s*x+c*y)] for x,y in points]
        if name == 'nominal' and max(abs(v-w) for a,b in zip(transformed, fixed) for v,w in zip(a,b)) > 1e-8:
            raise ValueError('nominal map transformation did not reproduce frozen Path')
        if name == 'nominal':
            transformed = fixed
        write_once(a.state_folder / f'{name}-map-path.json', transformed)
    print('SMOKE_MAP_PATHS_READY', a.state_folder)

if __name__ == '__main__':
    main()
