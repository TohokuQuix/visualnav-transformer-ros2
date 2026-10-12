"""Freeze the three-period ROS-free crossover and stage period zero only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_evidence import SCHEMA
from visualnav_transformer.eat_nav.p1_g3_material_gate_v2 import SCHEMA as MATERIAL_SCHEMA
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment


REFERENCE = Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0')
SOURCE_MODULES = ('p1_g3_shared_dem.py','p1_dem_capture.py','p1_longhorizon_protocol.py',
                  'p1_g3_direct_repeat.py','p1_g3_crossover_runtime.py',
                  'p1_path_crossover_v2.py','p1_g3_material_diagnostic.py',
                  'p1_g3_material_gate_v2.py','p1_g3_assignment.py',
                  'p1_follower_core.cpp','p1_follower_core.py',
                  'p1_g3_rosfree_control.py','p1_g3_rosfree_crossover_evidence.py',
                  'p1_g3_rosfree_crossover_path_contract.py',
                  'p1_g3_rosfree_crossover_contract.py')
INPUTS = ('assignment.json','freeze.json','statistical-config.json',
          'source-contract.json','anchor-fixtures.json','eat_nav.yaml')
PROCESS_ID = 'batch-first-differential-0-period-0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def build_assignment(base):
    processes=[dict(row) for row in base['processes'][:3]]
    roles=(('direct','direct','left_20','right_20'),
           ('left_20','right_20','direct','direct'),
           ('right_20','left_20','right_20','left_20'))
    if len(processes)!=3:
        raise ValueError('frozen crossover schedule missing')
    for i,process in enumerate(processes):
        if (process['anchor_id'],process['batch_index'],process['execution_index'],
            process['batch_seed'],process['assignment_period']) != \
                ('first-differential',0,i,2026101101,i):
            raise ValueError('frozen crossover period identity drift')
        rows=process['trials']
        validate_frozen_assignment(rows,rows,stage='prelaunch')
        if tuple(row['path_role'] for row in rows)!=roles[i]:
            raise ValueError('frozen Path role schedule drift')
    for env in range(4):
        if {process['trials'][env]['path_role'] for process in processes} != \
                {'direct','left_20','right_20'}:
            raise ValueError('Path and env are confounded')
    return {'schema':SCHEMA+'_assignment','processes':processes,
            'process_count':3,'trial_count':12,
            'meaning':'frozen_balanced_crossover_period_zero_only_authorized'}


def build_manifest(repo: Path, reference: Path, assignment_sha: str, assignment: dict):
    repo,reference=Path(repo),Path(reference)
    base=json.loads((repo/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    source=repo/'src/visualnav_transformer/eat_nav'
    input_root=reference/'batch-first-differential-0-repeat-0/input'
    for name,expected in base['source_input_sha256'].items():
        if name in INPUTS and sha(input_root/name)!=expected:
            raise ValueError('sealed direct-repeat input drift: '+name)
    keys=('batch_seeds','control_dt_s','declared_terrain_material_sha256',
          'direct_duplicate_gates','episode_length_s','live_material',
          'max_control_steps','native_dem_equality','physics_envs',
          'policy_action_mode','post_arrival_hold_steps','resource_cap',
          'rl_checkpoint_sha256','settling_steps','shared_dem_source_env',
          'source_input_sha256','unreached_control_step_cap')
    manifest={key:base[key] for key in keys}
    manifest.update({'schema':SCHEMA,'material_schema':MATERIAL_SCHEMA,
        'execution_mode':'path_crossover_four_env_rosfree',
        'control_backend':'ros_free_cpp_core','external_router_required':False,
        'runtime_scope':'offline_preparation_only','runtime_authorized':False,
        'authorized_process_ids':[],'automatic_continuation':False,
        'process_count':3,'trial_count':12,'processes':assignment['processes'],
        'runtime_pilot_process_id':PROCESS_ID,'periods_1_2_authorized':False,
        'assignment_sha256':assignment_sha,
        'g3_simulator_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_rosfree_crossover_sim.py'),
        'g3_host_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_rosfree_crossover_host.py'),
        'staged_host_runner_sha256':sha(repo/'scripts/eat_nav_p1_g3_rosfree_crossover_host.py'),
        'staged_owner_wrapper_sha256':sha(repo/'scripts/eat_nav_p1_owner_wrapper.py'),
        'preflight_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_rosfree_crossover_preflight.py'),
        'analysis_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_rosfree_crossover_analyze.py'),
        'staging_source_sha256':sha(Path(__file__)),
        'owner_wrapper_sha256':sha(repo/'scripts/eat_nav_p1_owner_wrapper.py'),
        'parent_crossover_manifest_sha256':sha(repo/'docs/eat-nav/p1-g3-crossover-v2-manifest.json')})
    key_map={'p1_g3_shared_dem.py':'shared_dem_module_sha256',
        'p1_g3_direct_repeat.py':'direct_repeat_module_sha256',
        'p1_g3_crossover_runtime.py':'crossover_runtime_module_sha256',
        'p1_path_crossover_v2.py':'path_crossover_module_sha256',
        'p1_g3_material_diagnostic.py':'material_diagnostic_module_sha256',
        'p1_g3_material_gate_v2.py':'material_gate_module_sha256',
        'p1_follower_core.cpp':'cpp_core_source_sha256',
        'p1_follower_core.py':'python_core_source_sha256',
        'p1_g3_rosfree_control.py':'rosfree_control_source_sha256',
        'p1_g3_rosfree_crossover_evidence.py':'rosfree_crossover_evidence_source_sha256',
        'p1_g3_rosfree_crossover_path_contract.py':'rosfree_path_contract_source_sha256',
        'p1_g3_rosfree_crossover_contract.py':'rosfree_crossover_contract_source_sha256'}
    manifest.update({key:sha(source/name) for name,key in key_map.items()})
    return manifest


def stage_one(repo, reference, root, assignment, manifest):
    repo,reference,root=Path(repo),Path(reference),Path(root)
    if root.exists():raise FileExistsError('fresh ROS-free evidence root required')
    batch=root/PROCESS_ID;batch.mkdir(parents=True)
    write_new(root/'manifest.json',manifest)
    write_new(root/'g3-assignment.json',assignment)
    shutil.copy2(repo/'scripts/eat_nav_p1_g3_rosfree_crossover_sim.py',batch/'source.py')
    shutil.copy2(repo/'scripts/eat_nav_p1_g3_rosfree_crossover_host.py',batch/'host-runner.py')
    shutil.copy2(repo/'scripts/eat_nav_p1_owner_wrapper.py',
                 batch/'eat_nav_p1_owner_wrapper.py')
    old=reference/'batch-first-differential-0-repeat-0'
    shutil.copytree(old/'src',batch/'src',ignore=shutil.ignore_patterns('__pycache__'))
    module=batch/'src/visualnav_transformer/eat_nav'
    for name in SOURCE_MODULES:
        shutil.copy2(repo/'src/visualnav_transformer/eat_nav'/name,module/name)
    input_dir=batch/'input';input_dir.mkdir()
    for name in INPUTS:
        shutil.copy2(old/'input'/name,input_dir/name)
    write_new(input_dir/'g3-assignment.json',assignment)
    write_new(input_dir/'g3-contract.json',manifest)
    paths=[batch/'source.py',batch/'host-runner.py',
           batch/'eat_nav_p1_owner_wrapper.py',
           *sorted(path for path in (batch/'src').rglob('*') if path.is_file()),
           *sorted(input_dir.iterdir())]
    write_new(batch/'input-sha-ledger.json',
              {str(path.relative_to(batch)):sha(path) for path in paths})
    write_new(root/'stage-complete.json',{'schema':SCHEMA,'runtime_authorized':False,
        'process_id':PROCESS_ID,'manifest_sha256':sha(root/'manifest.json'),
        'assignment_sha256':sha(root/'g3-assignment.json'),
        'batch_input_ledger_sha256':sha(batch/'input-sha-ledger.json')})
    return {'root':str(root),'manifest_sha256':sha(root/'manifest.json'),
            'assignment_sha256':sha(root/'g3-assignment.json'),
            'input_ledger_sha256':sha(batch/'input-sha-ledger.json')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest-output',type=Path,required=True)
    parser.add_argument('--stage-root',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    base=json.loads((repo/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment=build_assignment(base)
    serialized=json.dumps(assignment,sort_keys=True,indent=2,allow_nan=False)+'\n'
    assignment_sha=hashlib.sha256(serialized.encode()).hexdigest()
    manifest=build_manifest(repo,REFERENCE,assignment_sha,assignment)
    write_new(args.manifest_output,manifest)
    print(json.dumps(stage_one(repo,REFERENCE,args.stage_root,assignment,manifest),sort_keys=True))


if __name__=='__main__':main()
