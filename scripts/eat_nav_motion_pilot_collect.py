"""Sequential paired Arc collection over independent train/validation/test seeds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from eat_nav_motion_arc_stage_b import verify_existing, write_atomic
from eat_nav_motion_collect_host import sha


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO/'config/eat_nav_motion_pilot_v1.json'
HOST = REPO/'scripts/eat_nav_motion_collect_host.py'
SIM = REPO/'scripts/eat_nav_motion_collect_sim.py'


def name(seed, angle):
    suffix = 'direct' if angle == 0 else ('minus' if angle < 0 else 'plus') + str(abs(angle))
    return f'c-pilot-random-rough-seed{seed}-{suffix}'


def main():
    cfg = json.loads(CONFIG.read_text())
    if cfg.get('execution_gate') != 'phase_b_bank_frozen_for_pilot':
        raise ValueError('Phase C pilot blocked until Arc Bank feasibility review')
    parent = Path(cfg['persistent_parent'])
    control = parent/'pilot-control'
    control.mkdir(parents=True, exist_ok=True)
    frozen = {'schema': cfg['schema'], 'config_sha256': sha(CONFIG),
              'host_source_sha256': sha(HOST), 'simulator_source_sha256': sha(SIM),
              'split_by_base_terrain_seed': cfg['split_by_base_terrain_seed']}
    file = control/'frozen.json'
    if file.exists():
        if json.loads(file.read_text()) != frozen:
            raise ValueError('pilot collection source/config drift')
    else:
        write_atomic(file, frozen)
    schedule = [(split, seed, angle) for split in ('train', 'validation', 'test')
                for seed in cfg['split_by_base_terrain_seed'][split]
                for angle in cfg['arc_angles_deg']]
    done = []
    for split, seed, angle in schedule:
        root = parent/name(seed, angle)
        if not root.exists():
            reference = parent/name(seed, 0)/'sim-output'
            command = ['/usr/bin/python3', str(HOST), '--root', str(root),
                       '--arc-angle', str(angle), '--seed', str(seed),
                       '--terrain-type', cfg['terrain_type'], '--terrain-level', str(cfg['terrain_level'])]
            if angle:
                command.extend(['--reference-prepath', str(reference)])
            with (control/f'{seed}-{angle:+d}.log').open('x') as stream:
                result = subprocess.run(command, env={**os.environ, 'PYTHONPATH': str(REPO/'src')},
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                write_atomic(control/'status.json', {'stage': 'stopped',
                             'completed': len(done), 'total': len(schedule),
                             'failed_seed': seed, 'failed_angle_deg': angle,
                             'first_failure': f'host_exit_{result.returncode}'})
                return 1
        seal = verify_existing(root, angle)
        if json.loads((root/'manifest.json').read_text())['seed'] != seed:
            raise ValueError('pilot seed drift')
        done.append({'split': split, 'seed': seed, **seal})
        write_atomic(control/'status.json', {'stage': 'running', 'completed': len(done),
                     'total': len(schedule), 'last_seed': seed, 'last_angle_deg': angle})
    write_atomic(control/'completed.json', {'schema': cfg['schema'], 'records': done})
    write_atomic(control/'status.json', {'stage': 'complete', 'completed': len(done),
                 'total': len(schedule)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
