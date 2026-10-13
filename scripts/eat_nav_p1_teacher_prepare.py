"""Freeze one Direct-only ROS-free teacher-capture diagnostic attempt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from eat_nav_p1_g3_rosfree_crossover_prepare import (
    REFERENCE, build_assignment, build_manifest, stage_one, write_new,
)


PROCESS_ID = 'batch-first-differential-0-period-0'
TEACHER_SCHEMA = 'eatnav_p1_teacher_capture_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + '\n').encode()).hexdigest()


def build_teacher_assignment(base):
    assignment = copy.deepcopy(build_assignment(base))
    first = assignment['processes'][0]['trials']
    direct = first[0]
    for row in first:
        row['bearing_deg'] = 0
        row['path_role'] = 'direct'
        row['path_sha256'] = direct['path_sha256']
    assignment['schema'] = TEACHER_SCHEMA + '_assignment'
    assignment['meaning'] = 'one_direct_only_logging_diagnostic; period_1_2_not_authorized'
    return assignment


def build_teacher_manifest(repo, assignment, attempt_id):
    repo = Path(repo)
    if not attempt_id.startswith('.eatnav-p1-teacher-capture-v1-attempt-') or '/' in attempt_id:
        raise ValueError('teacher_attempt_identity')
    manifest = build_manifest(repo, REFERENCE, canonical_sha(assignment), assignment)
    module = repo/'src/visualnav_transformer/eat_nav'
    manifest.update({
        'teacher_capture_schema': TEACHER_SCHEMA,
        'teacher_runtime_authorized': True,
        'runtime_scope': 'one_direct_only_teacher_logging_diagnostic',
        'runtime_authorized': True,
        'authorized_process_ids': [PROCESS_ID],
        'authorized_period_index': 0,
        'automatic_continuation': False,
        'attempt_id': attempt_id,
        'attempt_host_root': str(Path('/home/kojima/lib/onix_isaac_lab')/attempt_id),
        'attempt_container_root': str(Path('/workspace/isaaclab_onix')/attempt_id),
        'host_bind_source': '/home/kojima/lib/onix_isaac_lab',
        'container_bind_destination': '/workspace/isaaclab_onix',
        'g3_simulator_source_sha256': sha(repo/'scripts/eat_nav_p1_teacher_sim.py'),
        'g3_host_source_sha256': sha(repo/'scripts/eat_nav_p1_teacher_host.py'),
        'staged_host_runner_sha256': sha(repo/'scripts/eat_nav_p1_teacher_host.py'),
        'preflight_source_sha256': sha(repo/'scripts/eat_nav_p1_teacher_preflight.py'),
        'rosfree_path_contract_source_sha256': sha(module/'p1_teacher_path_contract.py'),
        'teacher_capture_source_sha256': sha(module/'p1_teacher_capture.py'),
        'teacher_runtime_source_sha256': sha(module/'p1_teacher_runtime.py'),
        'teacher_chunks_source_sha256': sha(module/'p1_teacher_chunks.py'),
        'teacher_prepare_source_sha256': sha(__file__),
        'logging_on': True,
        'reward_sum_atol': 1e-5,
        'reward_sum_rtol': 1e-5,
        'logging_chunk_rows': 128,
        'substep_contact_mode': 'physx_direct_per_5ms_step_fail_closed',
    })
    return manifest


def stage_teacher(repo, root, assignment, manifest):
    repo, root = Path(repo), Path(root)
    if root.exists():
        raise FileExistsError(root)
    stage_one(repo, REFERENCE, root, assignment, manifest)
    batch = root/PROCESS_ID
    replacements = (
        (repo/'scripts/eat_nav_p1_teacher_sim.py',batch/'source.py'),
        (repo/'scripts/eat_nav_p1_teacher_host.py',batch/'host-runner.py'),
    )
    module = batch/'src/visualnav_transformer/eat_nav'
    for name in ('p1_teacher_capture.py','p1_teacher_runtime.py',
                 'p1_teacher_chunks.py','p1_teacher_path_contract.py'):
        replacements += ((repo/'src/visualnav_transformer/eat_nav'/name,module/name),)
    for source, dest in replacements:
        shutil.copy2(source,dest)
    ledger = {str(path.relative_to(batch)):sha(path)
              for path in (batch/'source.py',batch/'host-runner.py',
                           batch/'eat_nav_p1_owner_wrapper.py',
                           *sorted(p for p in (batch/'src').rglob('*') if p.is_file()),
                           *sorted((batch/'input').iterdir()))}
    tmp = batch/'input-sha-ledger.json.tmp'
    with tmp.open('x') as stream:
        json.dump(ledger,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(batch/'input-sha-ledger.json')
    marker = {'schema':TEACHER_SCHEMA,'runtime_authorized':True,
              'process_id':PROCESS_ID,'manifest_sha256':sha(root/'manifest.json'),
              'assignment_sha256':sha(root/'g3-assignment.json'),
              'batch_input_ledger_sha256':sha(batch/'input-sha-ledger.json')}
    tmp = root/'stage-complete.json.tmp'
    with tmp.open('x') as stream:
        json.dump(marker,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(root/'stage-complete.json')
    return {'root':str(root),'manifest_sha256':sha(root/'manifest.json'),
            'assignment_sha256':sha(root/'g3-assignment.json'),
            'input_ledger_sha256':sha(batch/'input-sha-ledger.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    base = json.loads((repo/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = build_teacher_assignment(base)
    manifest = build_teacher_manifest(repo,assignment,args.root.name)
    if args.root != Path(manifest['attempt_host_root']):
        raise ValueError('persistent attempt root contract')
    print(json.dumps(stage_teacher(repo,args.root,assignment,manifest),sort_keys=True))


if __name__ == '__main__':
    main()
