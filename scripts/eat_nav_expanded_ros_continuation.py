"""Prepare a new E1 continuation root with read-only references to 15 sealed runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from eat_nav_e1_artifact_recovery import verify_seal
from eat_nav_g0e_gpu_probe import atomic,sha


OLD=Path('/tmp/eatnav-expanded-feasibility-x2-retry-20261005')
ROOT=Path('/tmp/eatnav-expanded-feasibility-ros-continuation-20261005')
X2=Path('/tmp/eatnav-expanded-feasibility-ros-x2-20261005')
PRIOR_ACTIVE_SEC=2337.05
FROZEN_FILES=('selected-states.json','selected-states.npz','candidate-paths.jsonl.gz',
    'run-manifest.json','input-freeze.json','runtime-storage-forecast.json',
    'offline-e0-audit.json','train-only-path-nearest-neighbor.json',
    'e1-gates.yaml','e1-safety-stops.yaml','eat_nav_expanded_sim_v2.py',
    'runtime-source-freeze-v2.json','config-restoration-freeze.json',
    'external-router-provenance.json')
DERIVED_REUSED=('anchor-gate-expanded-train-000.json',
    'anchor-gate-expanded-train-001.json',
    'duplicate-gate-expanded-train-000.json',
    'duplicate-gate-expanded-train-001.json',
    'body-noise-floor-expanded-train-000.json',
    'body-noise-floor-expanded-train-001.json',
    'group-complete-expanded-train-000.json',
    'map-path-freeze-state-000.json','map-path-freeze-state-001.json')


def row_split(rows):
    if len(rows)!=50 or [r['order'] for r in rows]!=list(range(50)):
        raise ValueError('frozen 50-row order differs')
    if rows[15]['process_id']!='expanded-train-001-left_2':
        raise ValueError('row 16 identity differs')
    return rows[:15],rows[15:]


def reused_run_bytes(old,rows):
    return sum(path.stat().st_size for row in rows
        for path in (Path(old)/'runs'/row['process_id']).rglob('*') if path.is_file())


def prepare(root=ROOT,old=OLD,x2=X2):
    root,old,x2=Path(root),Path(old),Path(x2)
    if root.exists():raise FileExistsError(root)
    readiness=json.loads((x2/'router-readiness.json').read_text())
    if readiness['gate']['status']!='pass' or len(readiness['samples'])!=3:
        raise ValueError('candidate-free X2 not passed')
    rows=json.loads((old/'run-manifest.json').read_text())['rows']
    reused,pending=row_split(rows)
    evidence=[]
    for row in reused:
        run=old/'runs'/row['process_id']
        if not verify_seal(run):raise ValueError('reused seal invalid '+row['process_id'])
        marker=json.loads((old/'run-stage-markers'/f"{row['process_id']}.json").read_text())
        if marker['status']!='complete' or marker['run_complete_sha256']!=sha(run/'run-complete.json'):
            raise ValueError('reused completion invalid '+row['process_id'])
        if row['role']!='anchor_only' and json.loads((run/'protocol-valid.json').read_text())['status']!='valid':
            raise ValueError('reused protocol invalid '+row['process_id'])
        evidence.append({'order':row['order'],'run_id':row['process_id'],
            'source_root':str(old),'relative_path':str(Path('runs')/row['process_id']),
            'run_complete_sha256':sha(run/'run-complete.json'),
            'seal_sha256':sha(run/'sealed-tree-manifest.json'),
            'stage_marker_sha256':sha(old/'run-stage-markers'/f"{row['process_id']}.json"),
            'seed':row['seed'],'state_group_id':row['state_group_id'],
            'path_id':row['path_id']})
    failed=old/'runs'/pending[0]['process_id']
    if (failed/'run-complete.json').exists() or (failed/'live-0.trace.json.gz').exists():
        raise ValueError('excluded row 16 has a completed artifact')
    root.mkdir()
    copied={}
    for name in (*FROZEN_FILES,*DERIVED_REUSED):
        source=old/name;target=root/name
        shutil.copy2(source,target)
        if sha(source)!=sha(target):raise ValueError('copied SHA differs '+name)
        copied[name]=sha(target)
    shutil.copytree(old/'inputs',root/'inputs')
    input_files={}
    for source in (old/'inputs').rglob('*'):
        if source.is_file():
            relative=source.relative_to(old)
            if sha(source)!=sha(root/relative):raise ValueError('copied input SHA differs '+str(relative))
            input_files[str(relative)]=sha(source)
    (root/'runs').mkdir();(root/'run-stage-markers').mkdir()
    for row in reused:
        name=row['process_id']
        (root/'runs'/name).symlink_to(old/'runs'/name,target_is_directory=True)
        (root/'run-stage-markers'/f'{name}.json').symlink_to(
            old/'run-stage-markers'/f'{name}.json')
        if not verify_seal(root/'runs'/name):raise ValueError('reuse reference seal invalid '+name)
    shutil.copy2(x2/'router-readiness.json',root/'router-readiness.json')
    atomic(root/'x2-readiness-gate.json',{'status':'pass','stable_snapshots':3,
        'candidate_path_published':False,'accepted_pre_path_commands':0,
        'readiness_sha256':sha(root/'router-readiness.json')})
    bytes_reused=reused_run_bytes(old,reused)
    atomic(root/'continuation-reuse-ledger.json',{
        'protocol_valid_reused':15,'measurement_invalid_excluded':1,
        'reused_run_bytes':bytes_reused,'prior_active_elapsed_sec':PRIOR_ACTIVE_SEC,
        'active_budget_remaining_sec':14400-PRIOR_ACTIVE_SEC,
        'original_manifest_sha256':sha(old/'run-manifest.json'),
        'reused':evidence,'excluded':{'run_id':pending[0]['process_id'],
            'source_root':str(old),'relative_path':str(Path('runs')/pending[0]['process_id']),
            'outcome_counted':False},'copied_frozen_sha256':copied,
        'copied_input_sha256':input_files,'new_pending_run_ids':[r['process_id'] for r in pending],
        'raw_reused_runs_copied':False})
    atomic(root/'continuation-manifest.json',{'original_manifest_sha256':sha(old/'run-manifest.json'),
        'reuse_orders':list(range(15)),'excluded_old_order':15,
        'new_order':list(range(15,50)),'rows':pending,
        'candidate_reselection':False})
    atomic(root/'status.json',{'stage':'CONTINUATION_READY_NO_NEW_RUN',
        'protocol_valid_reused':15,'new_processes_completed':0,'next_run':pending[0]['process_id']})
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--x2',type=Path,default=X2)
    args=p.parse_args();print(prepare(root=args.root,x2=args.x2))
