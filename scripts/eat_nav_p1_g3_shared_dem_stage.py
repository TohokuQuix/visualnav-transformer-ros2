"""Prepare a fresh, sealed-input G3 v2 candidate-free bundle; never execute it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from eat_nav_p1_g1_lifecycle import verify_seal

BASE = Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1')
G2 = BASE/'g2-attempt-0'
OLD_G3 = BASE/'g3-attempt-0'
G1_SEAL = 'c5712eb6bc816f49222f01218bb593f116a38a00f8ce07eb79bd5b637e420609'
G2_SEAL = '73ec7b6cac44dfdb717a8f8a8dd2a4cce4f19f9dc8d7b565df6055a0f251f1b0'
PROTOCOL = 'eatnav_p1_longhorizon_g3_shared_dem_v2'
INPUTS = ('assignment.json', 'freeze.json', 'statistical-config.json',
          'source-contract.json', 'anchor-fixtures.json', 'eat_nav.yaml')


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def stage_bundle(repo: Path, g2: Path, old_g3: Path, root: Path,
                 manifest: dict) -> Path:
    """Copy only frozen inputs into a *new* root; no simulator/router side effect."""
    repo, g2, old_g3, root = map(Path, (repo, g2, old_g3, root))
    if manifest['schema'] != PROTOCOL or manifest['execution_mode'] != 'candidate_free_20_step_dem':
        raise ValueError('G3 v2 diagnostic protocol/phase mismatch')
    source = repo/'scripts/eat_nav_p1_g3_shared_dem_sim.py'
    host = repo/'scripts/eat_nav_p1_g3_shared_dem_host.py'
    module = repo/'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py'
    if (digest(source) != manifest['g3_simulator_source_sha256'] or
            digest(host) != manifest['g3_host_source_sha256'] or
            digest(module) != manifest['shared_dem_module_sha256'] or
            digest(old_g3/'g3-assignment.json') != manifest['assignment_sha256']):
        raise ValueError('G3 v2 source or assignment SHA drift')
    if root.exists():
        raise FileExistsError('G3 v2 root already exists')
    root.mkdir(parents=True, mode=0o775)
    durable_json(root/'manifest.json', manifest)
    batch = root/'batch-0'
    batch.mkdir()
    shutil.copy2(source, batch/'source.py')
    shutil.copy2(host, batch/'host-runner.py')
    shutil.copytree(g2/'src', batch/'src', ignore=shutil.ignore_patterns('__pycache__'))
    dest_module = batch/'src/visualnav_transformer/eat_nav'
    for name in ('p1_g3_shared_dem.py', 'p1_dem_capture.py', 'p1_longhorizon_protocol.py'):
        shutil.copy2(repo/'src/visualnav_transformer/eat_nav'/name, dest_module/name)
    inputs = batch/'input'; inputs.mkdir()
    for name in INPUTS:
        shutil.copy2(g2/'input'/name, inputs/name)
    shutil.copy2(old_g3/'g3-assignment.json', inputs/'g3-assignment.json')
    durable_json(inputs/'g3-contract.json', manifest)
    ledger = {str(path.relative_to(batch)): digest(path) for path in
              (batch/'source.py', batch/'host-runner.py',
               *(dest_module/name for name in ('p1_g3_shared_dem.py',
                   'p1_dem_capture.py', 'p1_longhorizon_protocol.py')),
               *sorted(inputs.iterdir()))}
    durable_json(batch/'input-sha-ledger.json', ledger)
    verify_stage(batch, manifest)
    return root


def verify_stage(root: Path, manifest: dict) -> dict:
    root = Path(root)
    ledger = json.loads((root/'input-sha-ledger.json').read_text())
    for name, expected in ledger.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or digest(root/name) != expected:
            raise ValueError('G3 v2 staged input SHA mismatch:' + name)
    if (ledger['source.py'] != manifest['g3_simulator_source_sha256'] or
            ledger['host-runner.py'] != manifest['g3_host_source_sha256'] or
            ledger['src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py'] !=
                manifest['shared_dem_module_sha256'] or
            ledger['input/g3-assignment.json'] != manifest['assignment_sha256']):
        raise ValueError('G3 v2 staged source/assignment provenance mismatch')
    if json.loads((root/'input/g3-contract.json').read_text()) != manifest:
        raise ValueError('G3 v2 manifest drift')
    return {'pass': True, 'file_count': len(ledger),
            'input_sha_ledger_sha256': digest(root/'input-sha-ledger.json')}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.root.resolve().parent != BASE.resolve():
        raise ValueError('G3 v2 root must be a fresh persistent subdirectory')
    if digest(BASE/'g1-attempt-2/host-seal.json') != G1_SEAL or \
            digest(G2/'host-seal.json') != G2_SEAL:
        raise ValueError('G1/G2 original seal drift')
    verify_seal(BASE/'g1-attempt-2'); verify_seal(G2)
    manifest = json.loads((Path(__file__).resolve().parents[1]/
        'docs/eat-nav/p1-g3-shared-dem-v2-manifest.json').read_text())
    result = stage_bundle(Path(__file__).resolve().parents[1], G2, OLD_G3,
                          args.root, manifest)
    print(result/'batch-0')


if __name__ == '__main__':
    main()
