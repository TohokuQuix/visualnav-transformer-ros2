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
from eat_nav_p1_g1_lifecycle import verify_seal
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_evidence import validate_output

from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_path_contract import (
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
    selected=marker['process_id']
    if selected not in [f'batch-first-differential-0-period-{i}' for i in range(3)] or \
            manifest['runtime_pilot_process_id'] != selected or \
            manifest.get('prepared_period_index',0) != int(selected.rsplit('-',1)[-1]):
        raise ValueError('preparation period identity mismatch')
    batch = root/selected
    ledger = json.loads((batch/'input-sha-ledger.json').read_text())
    if (manifest['runtime_authorized'] or manifest['authorized_process_ids'] or
            marker['runtime_authorized'] or
            sha(root/'manifest.json') != marker['manifest_sha256'] or
            sha(root/'g3-assignment.json') != marker['assignment_sha256'] or
            sha(batch/'input-sha-ledger.json') != marker['batch_input_ledger_sha256'] or
            sha(batch/'source.py') != manifest['g3_simulator_source_sha256'] or
            sha(batch/'host-runner.py') != manifest['g3_host_source_sha256'] or
            sha(batch/'host-runner.py') != manifest['staged_host_runner_sha256'] or
            sha(batch/'eat_nav_p1_owner_wrapper.py') != manifest['owner_wrapper_sha256'] or
            sha(batch/'eat_nav_p1_owner_wrapper.py') != manifest['staged_owner_wrapper_sha256']):
        raise ValueError('preparation identity mismatch')
    for relative, expected in ledger.items():
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts or sha(batch/relative) != expected:
            raise ValueError('preparation source/input SHA mismatch: '+str(relative))
    return manifest, marker, ledger


def verify_period_reference(root, *, expected_period, assignment_sha):
    """Read-only verification of a sealed earlier period before continuation."""
    root=Path(root)
    selected=f'batch-first-differential-0-period-{expected_period}'
    batch=root/selected
    try:
        manifest=json.loads((root/'manifest.json').read_text())
        assignment=json.loads((root/'g3-assignment.json').read_text())
        seal=verify_seal(batch)
        status=json.loads((batch/'rosfree-host-status.json').read_text())
        complete=json.loads((batch/'output/complete.json').read_text())
        assigned=assignment['processes'][expected_period]['trials']
        validation=validate_output(batch/'output',manifest,assigned)
        if (seal['stage']!='G3_rosfree_crossover_protocol_valid' or
                manifest['assignment_sha256']!=assignment_sha or
                sha(root/'g3-assignment.json')!=assignment_sha or
                manifest['authorized_process_ids']!=[selected] or
                complete['execution_index']!=expected_period or
                status['failure'] is not None or status['native_exit']!=0 or
                not validation['pass']):
            raise ValueError('sealed reference protocol mismatch')
        snapshot=batch/'output/prepath-crossover-snapshot.json'
        if sha(snapshot)!=complete['prepath_crossover_snapshot_sha256']:
            raise ValueError('sealed reference snapshot SHA mismatch')
        return {'host_root':str(root),'process_id':selected,
                'manifest_sha256':sha(root/'manifest.json'),
                'assignment_sha256':assignment_sha,
                'host_seal_sha256':sha(batch/'host-seal.json'),
                'snapshot_sha256':sha(snapshot),
                'simulator_source_sha256':manifest['g3_simulator_source_sha256'],
                'follower_core_sha256':manifest['cpp_core_source_sha256']}
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f'period-{expected_period} sealed reference invalid: {exc}') from exc


def create_attempt(preparation, destination, *, host_bind_source=HOST_BIND,
                   container_bind_destination=CONTAINER_BIND,
                   reference_period_zero=None, prior_period_one=None):
    preparation = Path(preparation)
    destination = Path(destination)
    manifest, marker, old_ledger = verify_preparation(preparation)
    period=manifest.get('prepared_period_index',0)
    selected=marker['process_id']
    if period:
        if reference_period_zero is None:
            raise ValueError('sealed period-0 reference required')
        reference=verify_period_reference(reference_period_zero,
            expected_period=0,assignment_sha=manifest['assignment_sha256'])
        if (reference['simulator_source_sha256']!=manifest['g3_simulator_source_sha256'] or
                reference['follower_core_sha256']!=manifest['cpp_core_source_sha256']):
            raise ValueError('period-0 frozen simulator/core source mismatch')
        prior=None
        if period==2:
            if prior_period_one is None:
                raise ValueError('sealed period-1 reference required')
            prior=verify_period_reference(prior_period_one,
                expected_period=1,assignment_sha=manifest['assignment_sha256'])
    elif reference_period_zero is not None or prior_period_one is not None:
        raise ValueError('period zero cannot use earlier-period references')
    if (manifest['schema'] != SCHEMA or
            destination.parent != Path(host_bind_source) or
            destination.parent.is_symlink() or
            not destination.name.startswith('.eatnav-p1-g3-rosfree-crossover-v1-attempt-')):
        raise ValueError('attempt must use the frozen ROS-free crossover bind root')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(preparation, destination)
    batch = destination/selected
    new_manifest = dict(manifest)
    new_manifest.update({
        'runtime_authorized': True,
        'runtime_scope': 'single_crossover_period_process',
        'authorized_process_ids': [selected],
        'authorized_period_index':period,
        'periods_1_2_authorized':bool(period),
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
    if period:
        new_manifest['sealed_period_zero_reference']=reference
    if period==2:
        new_manifest['sealed_period_one_reference']=prior
    for path in (destination/'manifest.json', batch/'input/g3-contract.json'):
        path.unlink()
        durable_json(path, new_manifest)
    ledger = {relative: sha(batch/relative) for relative in old_ledger}
    (batch/'input-sha-ledger.json').unlink()
    durable_json(batch/'input-sha-ledger.json', ledger)
    (destination/'stage-complete.json').unlink()
    durable_json(destination/'stage-complete.json', {
        'schema': new_manifest['schema'], 'runtime_authorized': True,
        'process_id': selected,
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
        'authorized_process_ids': [selected],
        'authorized_period_index':period,
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
    parser.add_argument('--reference-period-zero', type=Path)
    parser.add_argument('--prior-period-one', type=Path)
    args = parser.parse_args()
    print(json.dumps(create_attempt(args.preparation, args.destination,
        reference_period_zero=args.reference_period_zero,
        prior_period_one=args.prior_period_one), sort_keys=True))


if __name__ == '__main__':
    main()
