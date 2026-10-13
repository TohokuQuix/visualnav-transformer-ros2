"""Resume the frozen Stage B bank in a detached process, stopping at first failure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from eat_nav_motion_collect_host import verify_sim_seal, sha


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO/'config/eat_nav_motion_vnext_stage_b.json'
HOST = REPO/'scripts/eat_nav_motion_collect_host.py'


def name(angle):
    suffix = 'direct' if angle == 0 else ('minus' if angle < 0 else 'plus') + str(abs(angle))
    return f'b-stage-random-rough-seed2026101101-{suffix}'


def write_atomic(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as out:
        json.dump(value, out, sort_keys=True, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def verify_existing(root, angle):
    decision = json.loads((root/'decision.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    if not decision['pass'] or manifest['angle_deg'] != angle or \
            json.loads((root/'native-exit.json').read_text())['exit_code'] != 0:
        raise ValueError(f'previous Arc process invalid: {root}')
    verify_sim_seal(root/'sim-output')
    outer = json.loads((root/'outer-seal.json').read_text())
    ledger = json.loads((root/'outer-sha-ledger.json').read_text())
    actual = {str(path.relative_to(root)): sha(path) for path in root.rglob('*')
              if path.is_file() and path.name not in ('outer-sha-ledger.json', 'outer-seal.json')}
    if actual != ledger or sha(root/'outer-sha-ledger.json') != outer['ledger_sha256']:
        raise ValueError(f'outer seal invalid: {root}')
    return {'angle_deg': angle, 'root': str(root), 'outer_seal_sha256': sha(root/'outer-seal.json')}


def main():
    cfg = json.loads(CONFIG.read_text())
    if cfg['ordered_angles_deg'] != [0, -30, 30, -60, 60, -90, 90,
                                     -120, 120, -150, 150, -180, 180]:
        raise ValueError('Arc schedule drift')
    parent = Path(cfg['persistent_parent'])
    control = parent/'stage-b-control'
    control.mkdir(parents=True, exist_ok=True)
    frozen = {'schema': cfg['schema'], 'config_sha256': sha(CONFIG),
              'host_source_sha256': sha(HOST),
              'simulator_source_sha256': sha(REPO/'scripts/eat_nav_motion_collect_sim.py'),
              'ordered_angles_deg': cfg['ordered_angles_deg']}
    frozen_path = control/'frozen.json'
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != frozen:
            raise ValueError('frozen Stage B source/config changed')
    else:
        write_atomic(frozen_path, frozen)
    records = []
    reference = parent/cfg['reference_basename']/'sim-output'
    for angle in cfg['ordered_angles_deg']:
        root = parent/name(angle)
        if root.exists():
            records.append(verify_existing(root, angle))
            write_atomic(control/'status.json', {'stage': 'running', 'completed': len(records),
                         'total': len(cfg['ordered_angles_deg']), 'last_angle_deg': angle})
            continue
        command = ['/usr/bin/python3', str(HOST), '--root', str(root),
                   '--arc-angle', str(angle), '--seed', str(cfg['seed']),
                   '--terrain-type', cfg['terrain_type'], '--terrain-level', str(cfg['terrain_level']),
                   '--reference-prepath', str(reference)]
        started = time.monotonic()
        with (control/f'angle-{angle:+d}.log').open('x') as logfile:
            result = subprocess.run(command, env={**os.environ, 'PYTHONPATH': str(REPO/'src')},
                                    stdout=logfile, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            write_atomic(control/'status.json', {'stage': 'stopped', 'completed': len(records),
                         'total': len(cfg['ordered_angles_deg']), 'failed_angle_deg': angle,
                         'first_failure': f'host_exit_{result.returncode}',
                         'elapsed_seconds': time.monotonic() - started})
            return 1
        records.append(verify_existing(root, angle))
        write_atomic(control/'status.json', {'stage': 'running', 'completed': len(records),
                     'total': len(cfg['ordered_angles_deg']), 'last_angle_deg': angle,
                     'last_wall_seconds': time.monotonic() - started})
    write_atomic(control/'completed.json', {'schema': cfg['schema'], 'records': records,
                 'completed': len(records)})
    write_atomic(control/'status.json', {'stage': 'complete', 'completed': len(records),
                 'total': len(cfg['ordered_angles_deg'])})
    return 0


if __name__ == '__main__':
    sys.exit(main())
