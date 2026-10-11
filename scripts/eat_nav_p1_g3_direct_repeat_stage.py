"""Freeze a new persistent 12-process G3 direct-repeat input bundle offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from eat_nav_p1_g1_lifecycle import verify_seal
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import build_schedule


BASE = Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1')
G2 = BASE / 'g2-attempt-0'
OLD_G3 = BASE / 'g3-attempt-0'
G1_SEAL = 'c5712eb6bc816f49222f01218bb593f116a38a00f8ce07eb79bd5b637e420609'
G2_SEAL = '73ec7b6cac44dfdb717a8f8a8dd2a4cce4f19f9dc8d7b565df6055a0f251f1b0'
PROTOCOL = 'eatnav_p1_longhorizon_g3_direct_repeat_v1'
INPUTS = ('assignment.json', 'freeze.json', 'statistical-config.json',
          'source-contract.json', 'anchor-fixtures.json', 'eat_nav.yaml')
MODULES = ('p1_g3_shared_dem.py', 'p1_dem_capture.py', 'p1_longhorizon_protocol.py',
           'p1_g3_direct_repeat.py')


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def stage_bundle(repo: Path, g2: Path, old_g3: Path, root: Path,
                 manifest: dict) -> Path:
    repo, g2, old_g3, root = map(Path, (repo, g2, old_g3, root))
    if manifest.get('schema') != PROTOCOL or manifest.get('execution_mode') != 'direct_repeat_four_env':
        raise ValueError('direct-repeat protocol mismatch')
    source = repo / 'scripts/eat_nav_p1_g3_direct_repeat_sim.py'
    host = repo / 'scripts/eat_nav_p1_g3_direct_repeat_host.py'
    module = repo / 'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py'
    repeat_module = repo / 'src/visualnav_transformer/eat_nav/p1_g3_direct_repeat.py'
    old_assignment = old_g3 / 'g3-assignment.json'
    for path, key in ((source, 'g3_simulator_source_sha256'),
                      (host, 'g3_host_source_sha256'),
                      (module, 'shared_dem_module_sha256'),
                      (repeat_module, 'direct_repeat_module_sha256'),
                      (old_assignment, 'source_reference_assignment_sha256')):
        if digest(path) != manifest[key]:
            raise ValueError('source or original assignment SHA drift: ' + str(path))
    schedule = build_schedule(json.loads((g2 / 'input/assignment.json').read_text()))
    old_rows = json.loads(old_assignment.read_text())['trials']
    frozen_direct = [row for row in old_rows if row.get('candidate_slot') == 'direct_duplicate_0']
    if len(frozen_direct) != 4 or any(row['path_sha256'] !=
            next(process['trials'][0]['path_sha256'] for process in schedule['processes']
                 if process['anchor_id'] == row['anchor_id'] and
                    process['batch_index'] == row['batch_index']) for row in frozen_direct):
        raise ValueError('G3 frozen direct Path SHA differs from statistical assignment')
    if root.exists():
        raise FileExistsError('direct-repeat root already exists')
    root.mkdir(parents=True)
    durable_json(root / 'g3-assignment.json', schedule)
    schedule_sha = digest(root / 'g3-assignment.json')
    if manifest['assignment_sha256'] not in ('filled by staging', schedule_sha):
        raise ValueError('direct-repeat schedule SHA drift')
    frozen = {**manifest, 'assignment_sha256': schedule_sha,
              'process_count': 12, 'trial_count': 48}
    durable_json(root / 'manifest.json', frozen)
    for process in schedule['processes']:
        batch = root / f"batch-{process['anchor_id']}-{process['batch_index']}-repeat-{process['repeat_index']}"
        batch.mkdir()
        shutil.copy2(source, batch / 'source.py')
        shutil.copy2(host, batch / 'host-runner.py')
        shutil.copytree(g2 / 'src', batch / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        target = batch / 'src/visualnav_transformer/eat_nav'
        for name in MODULES:
            shutil.copy2(repo / 'src/visualnav_transformer/eat_nav' / name, target / name)
        inputs = batch / 'input'
        inputs.mkdir()
        for name in INPUTS:
            shutil.copy2(g2 / 'input' / name, inputs / name)
        shutil.copy2(root / 'g3-assignment.json', inputs / 'g3-assignment.json')
        durable_json(inputs / 'g3-contract.json', frozen)
        paths = (batch / 'source.py', batch / 'host-runner.py',
                 *(target / name for name in MODULES), *sorted(inputs.iterdir()))
        ledger = {str(path.relative_to(batch)): digest(path) for path in paths}
        durable_json(batch / 'input-sha-ledger.json', ledger)
        verify_stage(batch, frozen)
    durable_json(root / 'stage-complete.json', {
        'schema': PROTOCOL, 'process_count': 12, 'trial_count': 48,
        'manifest_sha256': digest(root / 'manifest.json'),
        'assignment_sha256': schedule_sha,
        'batch_input_ledgers': {batch.name: digest(batch / 'input-sha-ledger.json')
                                for batch in sorted(root.glob('batch-*'))},
    })
    return root


def verify_stage(batch: Path, manifest: dict) -> dict:
    ledger = json.loads((batch / 'input-sha-ledger.json').read_text())
    for name, expected in ledger.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(batch / relative) != expected:
            raise ValueError('direct-repeat staged SHA mismatch: ' + name)
    if (ledger['source.py'] != manifest['g3_simulator_source_sha256'] or
            ledger['host-runner.py'] != manifest['g3_host_source_sha256'] or
            ledger['src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py'] !=
            manifest['shared_dem_module_sha256'] or
            ledger['src/visualnav_transformer/eat_nav/p1_g3_direct_repeat.py'] !=
            manifest['direct_repeat_module_sha256'] or
            ledger['input/g3-assignment.json'] != manifest['assignment_sha256'] or
            json.loads((batch / 'input/g3-contract.json').read_text()) != manifest):
        raise ValueError('direct-repeat staged contract SHA mismatch')
    return {'pass': True, 'file_count': len(ledger),
            'input_sha_ledger_sha256': digest(batch / 'input-sha-ledger.json')}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.root.resolve().parent != BASE.resolve():
        raise ValueError('direct-repeat root must be a fresh persistent subdirectory')
    if digest(BASE / 'g1-attempt-2/host-seal.json') != G1_SEAL or \
            digest(G2 / 'host-seal.json') != G2_SEAL:
        raise ValueError('G1/G2 original seal drift')
    verify_seal(BASE / 'g1-attempt-2')
    verify_seal(G2)
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'docs/eat-nav/p1-g3-direct-repeat-v1-manifest.json').read_text())
    if digest(Path(__file__)) != manifest['staging_source_sha256']:
        raise ValueError('direct-repeat staging source SHA drift')
    result = stage_bundle(repo, G2, OLD_G3, args.root, manifest)
    print(result / 'stage-complete.json')


if __name__ == '__main__':
    main()
