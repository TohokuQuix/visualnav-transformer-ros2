"""Offline freeze/stage of the 18-process P1 G3 Path crossover protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from visualnav_transformer.eat_nav.p1_g3_direct_repeat import SEEDS
from visualnav_transformer.eat_nav.p1_g3_crossover_runtime import (
    SCHEMA, STRICT_ENV_FIELDS, STRICT_RUN_FIELDS, DIAGNOSTIC_ENV_FIELDS,
    validate_crossover_assignment,
)
from visualnav_transformer.eat_nav.p1_path_crossover_v2 import build_crossover


BASE = Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1')
OLD = BASE/'g3-direct-repeat-v1-attempt-0'
SOURCE_MODULES = ('p1_g3_shared_dem.py', 'p1_dem_capture.py',
                  'p1_longhorizon_protocol.py', 'p1_g3_direct_repeat.py',
                  'p1_g3_crossover_runtime.py', 'p1_path_crossover_v2.py')
INPUTS = ('assignment.json', 'freeze.json', 'statistical-config.json',
          'source-contract.json', 'anchor-fixtures.json', 'eat_nav.yaml')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, obj) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def assignment_from_design(design: dict, source_contract: dict) -> dict:
    validate_crossover_assignment(design)
    paths = {int(p['bearing_deg']): p['path_sha256'] for p in source_contract['paths']}
    roles = {'direct': 0, 'left_20': 20, 'right_20': -20}
    if any(bearing not in paths for bearing in roles.values()):
        raise ValueError('frozen source Path missing')
    processes = []
    for process in design['processes']:
        group, seed = process['state_group'], process['seed']
        block = list(SEEDS[group]).index(seed)
        execution = process['execution_index']
        trials = []
        for row in process['trials']:
            slot, role = row['env_id'], row['path_role']
            trials.append({'slot': slot, 'env_id': slot,
                           'trial_id': f'{group}-seed-{seed}-period-{execution}-env-{slot}',
                           'candidate_slot': f'crossover_{slot}',
                           'condition': 'A', 'path_role': role,
                           'bearing_deg': roles[role], 'path_sha256': paths[roles[role]]})
        processes.append({'anchor_id': group, 'batch_index': block, 'batch_seed': seed,
                          'execution_index': execution,
                          'assignment_period': process['assignment_period'],
                          'trials': trials})
    return {'schema': SCHEMA+'_assignment', 'processes': processes,
            'pilot_process_ids': [f'batch-first-differential-0-period-{i}' for i in range(3)],
            'independent_seed_conditions': 6, 'process_count': 18, 'trial_count': 72}


def manifest_for(repo: Path, old_root: Path, assignment_sha: str, assignment: dict) -> dict:
    old = json.loads((repo/'docs/eat-nav/p1-g3-direct-repeat-v1-manifest.json').read_text())
    old_stage = json.loads((old_root/'stage-complete.json').read_text())
    if old_stage['manifest_sha256'] != sha(old_root/'manifest.json'):
        raise ValueError('old direct-repeat frozen manifest drift')
    module = repo/'src/visualnav_transformer/eat_nav'
    material_review=json.loads((repo/'docs/eat-nav/p1-g3-path-crossover-v2-analysis.json').read_text())
    result = {**old, 'schema': SCHEMA, 'execution_mode': 'path_crossover_four_env',
              'assignment_sha256': assignment_sha, 'process_count': 18,
              'trial_count': 72, 'automatic_continuation_after_pilot': False,
              'pilot_process_ids': assignment['pilot_process_ids'],
              'runtime_scope': 'not_authorized_in_this_implementation',
              'g3_simulator_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_crossover_sim.py'),
              'g3_host_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_crossover_host.py'),
              'analysis_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_crossover_analyze.py'),
              'staging_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_crossover_stage.py'),
              'crossover_runtime_module_sha256': sha(module/'p1_g3_crossover_runtime.py'),
              'path_crossover_module_sha256': sha(module/'p1_path_crossover_v2.py'),
              'cross_period_prepath': {
                  'strict_run_fields': list(STRICT_RUN_FIELDS),
                  'strict_env_fields': list(STRICT_ENV_FIELDS),
                  'diagnostic_env_fields': list(DIAGNOSTIC_ENV_FIELDS),
                  'strict_numeric_rule': 'exact_value_equality',
                  'native_contact_and_Critic': 'diagnostic_only',
                  'host_approval_before_Path': True},
              'live_material': {'terrain_composed_USD_binding_required': True,
                                'robot_composed_USD_binding_required': True,
                                'robot_effective_PhysX_tensor_required': True,
                                'robot_effective_expected_per_shape': [0.8, 0.6, 0.0]},
              'processes': assignment['processes'],
              'old_direct_manifest_sha256': sha(old_root/'manifest.json')}
    result.pop('pilot_process_id', None)
    result['declared_terrain_material_sha256']=material_review['material']['canonical_declared_sha256']
    result['external_router_runtime_sidecar_required']=True
    # Inherited SHA fields point to unchanged source modules and frozen inputs.
    for name,key in (('p1_g3_shared_dem.py','shared_dem_module_sha256'),
                     ('p1_g3_direct_repeat.py','direct_repeat_module_sha256')):
        if sha(module/name)!=result[key]:
            raise ValueError('inherited module SHA drift: '+name)
    return result


def freeze(repo: Path, old_root: Path, assignment_path: Path,
           manifest_path: Path) -> tuple[dict, dict]:
    source = old_root/'batch-first-differential-0-repeat-0/input/source-contract.json'
    old_manifest = json.loads((old_root/'manifest.json').read_text())
    if sha(source)!=old_manifest['source_input_sha256']['source-contract.json']:
        raise ValueError('frozen source Path SHA drift')
    assignment = assignment_from_design(build_crossover(SEEDS), json.loads(source.read_text()))
    write_new(assignment_path, assignment)
    manifest = manifest_for(repo, old_root, sha(assignment_path), assignment)
    write_new(manifest_path, manifest)
    return assignment, manifest


def stage(repo: Path, old_root: Path, assignment_path: Path,
          manifest_path: Path, root: Path) -> dict:
    if root.exists():
        raise FileExistsError('crossover root must be new')
    assignment = json.loads(assignment_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha(assignment_path)!=manifest['assignment_sha256'] or \
            manifest_for(repo, old_root, sha(assignment_path), assignment)!=manifest:
        raise ValueError('crossover frozen source/assignment drift')
    root.mkdir(parents=True)
    shutil.copy2(assignment_path,root/'g3-assignment.json')
    shutil.copy2(manifest_path,root/'manifest.json')
    reference=old_root/'batch-first-differential-0-repeat-0'
    ledgers={}
    for process in assignment['processes']:
        name=(f"batch-{process['anchor_id']}-{process['batch_index']}-"
              f"period-{process['execution_index']}")
        batch=root/name
        batch.mkdir()
        shutil.copy2(repo/'scripts/eat_nav_p1_g3_crossover_sim.py',batch/'source.py')
        shutil.copy2(repo/'scripts/eat_nav_p1_g3_crossover_host.py',batch/'host-runner.py')
        shutil.copytree(reference/'src',batch/'src',ignore=shutil.ignore_patterns('__pycache__'))
        module=batch/'src/visualnav_transformer/eat_nav'
        for source_name in SOURCE_MODULES:
            shutil.copy2(repo/'src/visualnav_transformer/eat_nav'/source_name,module/source_name)
        inputs=batch/'input'; inputs.mkdir()
        for input_name in INPUTS:
            source=reference/'input'/input_name
            if sha(source)!=manifest['source_input_sha256'][input_name]:
                raise ValueError('old sealed input SHA drift: '+input_name)
            shutil.copy2(source,inputs/input_name)
        shutil.copy2(assignment_path,inputs/'g3-assignment.json')
        write_new(inputs/'g3-contract.json',manifest)
        paths=[batch/'source.py',batch/'host-runner.py',
               *(module/name for name in SOURCE_MODULES),*sorted(inputs.iterdir())]
        ledger={str(path.relative_to(batch)):sha(path) for path in paths}
        write_new(batch/'input-sha-ledger.json',ledger)
        ledgers[name]=sha(batch/'input-sha-ledger.json')
    write_new(root/'stage-complete.json',{'schema':SCHEMA,'process_count':18,
              'trial_count':72,'assignment_sha256':sha(assignment_path),
              'manifest_sha256':sha(manifest_path),'batch_input_ledgers':ledgers})
    return {'process_count':18,'trial_count':72,'ledgers':len(ledgers)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--stage-root',type=Path)
    parser.add_argument('--old-root',type=Path,default=OLD)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if args.freeze:
        freeze(repo,args.old_root,args.assignment,args.manifest)
    if args.stage_root is not None:
        stage(repo,args.old_root,args.assignment,args.manifest,args.stage_root)


if __name__=='__main__':
    main()
