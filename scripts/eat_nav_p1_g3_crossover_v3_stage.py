"""Freeze and stage only the first remaining P1 G3 crossover v3 process."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from eat_nav_p1_g3_crossover_v2_stage import INPUTS, SOURCE_MODULES, write_new
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment


SCHEMA = 'eatnav_p1_g3_path_crossover_v3_assignment_validator'
TARGET = ('first-differential', 1, 0)
TARGET_NAME = 'batch-first-differential-1-period-0'


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _target(process: dict) -> bool:
    return (process['anchor_id'], process['batch_index'], process['execution_index']) == TARGET


def make_manifest(repo: Path, old_root: Path, assignment_path: Path) -> dict:
    repo, old_root = Path(repo), Path(old_root)
    base_path = repo/'docs/eat-nav/p1-g3-crossover-v2-manifest.json'
    base = json.loads(base_path.read_text())
    assignment = json.loads(Path(assignment_path).read_text())
    if sha(assignment_path) != base['assignment_sha256'] or \
            assignment['processes'] != base['processes'] or \
            sha(repo/'scripts/eat_nav_p1_g3_crossover_v2_sim.py') != base['g3_simulator_source_sha256']:
        raise ValueError('frozen v2 assignment or simulator source drift')
    source = repo/'src/visualnav_transformer/eat_nav'
    for name,key in (('p1_g3_shared_dem.py','shared_dem_module_sha256'),
                     ('p1_g3_direct_repeat.py','direct_repeat_module_sha256'),
                     ('p1_g3_crossover_runtime.py','crossover_runtime_module_sha256'),
                     ('p1_path_crossover_v2.py','path_crossover_module_sha256'),
                     ('p1_g3_material_gate_v2.py','material_gate_module_sha256'),
                     ('p1_g3_material_diagnostic.py','material_diagnostic_module_sha256')):
        if sha(source/name) != base[key]:
            raise ValueError('frozen v2 module drift: '+name)
    first = [p for p in assignment['processes'] if _target(p)]
    if len(first) != 1:
        raise ValueError('first remaining process missing')
    validate_frozen_assignment(first[0]['trials'], first[0]['trials'], stage='prelaunch')
    old_stage = json.loads((old_root/'stage-complete.json').read_text())
    if old_stage['manifest_sha256'] != sha(old_root/'manifest.json'):
        raise ValueError('direct-repeat source stage drift')
    return {**base,
            'host_protocol_version':SCHEMA,
            'g3_host_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_crossover_v3_host.py'),
            'g3_assignment_module_sha256':sha(source/'p1_g3_assignment.py'),
            'staging_source_sha256':sha(repo/'scripts/eat_nav_p1_g3_crossover_v3_stage.py'),
            'parent_v2_manifest_sha256':sha(base_path),
            'runtime_scope':'authorized_first_remaining_process_only',
            'authorized_process_id':TARGET_NAME,
            'authorized_process_count':1,
            'pilot_authorized_max_processes':1,
            'pilot_process_ids':[TARGET_NAME],
            'automatic_continuation_after_pilot':False}


def stage_one(repo: Path, old_root: Path, assignment_path: Path,
              manifest: dict, root: Path) -> dict:
    repo, old_root, root = Path(repo), Path(old_root), Path(root)
    if root.exists():
        raise FileExistsError('fresh v3 root required')
    if manifest != make_manifest(repo, old_root, assignment_path):
        raise ValueError('v3 host/assignment/manifest freeze mismatch')
    reference = old_root/'batch-first-differential-0-repeat-0'
    batch = root/TARGET_NAME
    batch.mkdir(parents=True)
    write_new(root/'manifest.json',manifest)
    shutil.copy2(assignment_path,root/'g3-assignment.json')
    shutil.copy2(repo/'scripts/eat_nav_p1_g3_crossover_v2_sim.py',batch/'source.py')
    shutil.copy2(repo/'scripts/eat_nav_p1_g3_crossover_v3_host.py',batch/'host-runner.py')
    shutil.copytree(reference/'src',batch/'src',ignore=shutil.ignore_patterns('__pycache__'))
    module=batch/'src/visualnav_transformer/eat_nav'
    for name in (*SOURCE_MODULES,'p1_g3_assignment.py'):
        shutil.copy2(repo/'src/visualnav_transformer/eat_nav'/name,module/name)
    inputs=batch/'input';inputs.mkdir()
    for name in INPUTS:
        source=reference/'input'/name
        if sha(source)!=manifest['source_input_sha256'][name]:
            raise ValueError('sealed direct-repeat input drift: '+name)
        shutil.copy2(source,inputs/name)
    shutil.copy2(assignment_path,inputs/'g3-assignment.json')
    write_new(inputs/'g3-contract.json',manifest)
    ledger={str(path.relative_to(batch)):sha(path) for path in
            [batch/'source.py',batch/'host-runner.py',
             *(module/name for name in (*SOURCE_MODULES,'p1_g3_assignment.py')),
             *sorted(inputs.iterdir())]}
    write_new(batch/'input-sha-ledger.json',ledger)
    write_new(root/'stage-complete.json',{'schema':SCHEMA,
              'authorized_process_count':1,'authorized_process_id':TARGET_NAME,
              'assignment_sha256':sha(assignment_path),
              'manifest_sha256':sha(root/'manifest.json'),
              'batch_input_ledger_sha256':sha(batch/'input-sha-ledger.json')})
    return {'batch_name':TARGET_NAME,'manifest_sha256':sha(root/'manifest.json'),
            'input_ledger_sha256':sha(batch/'input-sha-ledger.json')}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--manifest-output',type=Path,required=True)
    parser.add_argument('--stage-root',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    manifest=make_manifest(repo,args.old_root,args.assignment)
    write_new(args.manifest_output,manifest)
    result=stage_one(repo,args.old_root,args.assignment,manifest,args.stage_root)
    print(result)


if __name__=='__main__':main()
