"""Create one authorized ROS-free attempt from a verified, immutable preparation root.

This stages bytes only; it never starts Docker, Isaac, a follower, or a router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from visualnav_transformer.eat_nav.p1_g3_rosfree_path_contract import (
    HOST_BIND, CONTAINER_BIND, SCHEMA,
)


PROCESS_ID = 'batch-first-differential-0-period-0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_preparation(root):
    root = Path(root)
    marker = json.loads((root/'stage-complete.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    batch = root/PROCESS_ID
    ledger = json.loads((batch/'input-sha-ledger.json').read_text())
    if (manifest['runtime_authorized'] or manifest['authorized_process_ids'] or
            marker['runtime_authorized'] or
            sha(root/'manifest.json') != marker['manifest_sha256'] or
            sha(root/'g3-assignment.json') != marker['assignment_sha256'] or
            sha(batch/'input-sha-ledger.json') != marker['batch_input_ledger_sha256'] or
            sha(batch/'source.py') != manifest['g3_simulator_source_sha256'] or
            sha(batch/'host-runner.py') != manifest['g3_host_source_sha256']):
        raise ValueError('preparation identity mismatch')
    for relative, expected in ledger.items():
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts or sha(batch/relative) != expected:
            raise ValueError('preparation source/input SHA mismatch: '+str(relative))
    return manifest, marker, ledger


def create_attempt(preparation, destination, *, host_bind_source=HOST_BIND,
                   container_bind_destination=CONTAINER_BIND):
    preparation = Path(preparation)
    destination = Path(destination)
    manifest, marker, old_ledger = verify_preparation(preparation)
    if (manifest['schema'] != SCHEMA or
            destination.parent != Path(host_bind_source) or
            destination.parent.is_symlink() or
            not destination.name.startswith('.eatnav-p1-g3-rosfree-direct-core-v2-attempt-')):
        raise ValueError('attempt must use the frozen ROS-free v2 bind root')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(preparation, destination)
    batch = destination/PROCESS_ID
    new_manifest = dict(manifest)
    new_manifest.update({
        'runtime_authorized': True,
        'runtime_scope': 'single_direct_only_process',
        'authorized_process_ids': [PROCESS_ID],
        'automatic_continuation': False,
        'preparation_manifest_sha256': marker['manifest_sha256'],
        'preparation_stage_marker_sha256': sha(preparation/'stage-complete.json'),
        'attempt_id': destination.name,
        'host_bind_source': str(host_bind_source),
        'container_bind_destination': str(container_bind_destination),
        'attempt_host_root': str(destination),
        'attempt_container_root': str(Path(container_bind_destination)/destination.name),
        'authorization_source_sha256': sha(Path(__file__)),
    })
    for path in (destination/'manifest.json', batch/'input/g3-contract.json'):
        path.unlink()
        durable_json(path, new_manifest)
    ledger = {relative: sha(batch/relative) for relative in old_ledger}
    (batch/'input-sha-ledger.json').unlink()
    durable_json(batch/'input-sha-ledger.json', ledger)
    (destination/'stage-complete.json').unlink()
    durable_json(destination/'stage-complete.json', {
        'schema': new_manifest['schema'], 'runtime_authorized': True,
        'process_id': PROCESS_ID,
        'manifest_sha256': sha(destination/'manifest.json'),
        'assignment_sha256': sha(destination/'g3-assignment.json'),
        'batch_input_ledger_sha256': sha(batch/'input-sha-ledger.json'),
        'preparation_stage_marker_sha256': sha(preparation/'stage-complete.json'),
    })
    durable_json(destination/'attempt-provenance.json', {
        'schema': new_manifest['schema'], 'attempt_id': destination.name,
        'preparation_root': str(preparation),
        'preparation_manifest_sha256': marker['manifest_sha256'],
        'preparation_stage_marker_sha256': sha(preparation/'stage-complete.json'),
        'attempt_manifest_sha256': sha(destination/'manifest.json'),
        'attempt_assignment_sha256': sha(destination/'g3-assignment.json'),
        'simulator_source_sha256': sha(batch/'source.py'),
        'host_source_sha256': sha(batch/'host-runner.py'),
        'authorization_source_sha256': sha(Path(__file__)),
        'source_input_file_count': len(ledger),
        'authorized_process_ids': [PROCESS_ID],
        'host_bind_source': str(host_bind_source),
        'container_bind_destination': str(container_bind_destination),
    })
    verify_preparation(preparation)
    assert all(sha(batch/path) == value for path, value in ledger.items())
    return json.loads((destination/'attempt-provenance.json').read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_attempt(args.preparation, args.destination), sort_keys=True))


if __name__ == '__main__':
    main()
