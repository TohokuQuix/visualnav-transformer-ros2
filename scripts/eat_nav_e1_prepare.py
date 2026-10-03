"""Freeze the approved E1 continuation without modifying G0-E artifacts."""
from __future__ import annotations

from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

ROOT=Path('/tmp/eatnav-e1-feasibility-20261003')
G0E=Path('/tmp/eatnav-g0e-formal-anchor-20261003')
BANK=Path('/tmp/eatnav-candidate-bank-g0e')
RAW_MANIFEST=Path('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def copy_frozen(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('frozen copied input SHA differs '+str(target))
        return
    tmp=target.with_name(target.name+'.tmp')
    shutil.copyfile(source,tmp)
    if sha(source)!=sha(tmp):raise ValueError('frozen input copy SHA differs')
    tmp.replace(target)


def verify_anchor_reuse(root=G0E):
    results=json.loads((root/'formal-anchor-results.json').read_text())
    if results['status']!='pass':raise ValueError('G0-E formal anchors did not pass')
    found=[]
    for index in range(2):
        folder=root/'runs'/f'anchor-{index:03d}'
        completed=json.loads((folder/'complete.marker.json').read_text())
        final=json.loads((folder/'final-result.json').read_text())
        native=json.loads((folder/'anchor-0.marker.json').read_text())
        if (completed['status']!='pass' or final['status']!='pass' or
                completed['result_sha256']!=sha(folder/'final-result.json') or
                completed['cleanup_sha256']!=sha(folder/'cleanup-result.json') or
                final['anchor_result_sha256']!=sha(root/f'formal-anchor-{index:03d}-result.json') or
                sha(folder/'anchor-0.json') not in native['outputs_sha256'].values()):
            raise ValueError('completed G0-E anchor SHA differs')
        result=json.loads((root/f'formal-anchor-{index:03d}-result.json').read_text())
        if result['status']!='pass' or result['dem_history_gate']['status']!='pass':
            raise ValueError('G0-E anchor gate differs')
        found.append(f'g0e-train-{index:03d}')
    return found


def build_continuation(original):
    rows=original['rows']
    if len(rows)!=20 or [r['order'] for r in rows]!=list(range(20)):
        raise ValueError('20-row original process order differs')
    roles=Counter(r['role'] for r in rows)
    if roles!={'anchor_only':2,'nominal_duplicate':6,'candidate_outcome':12}:
        raise ValueError('original process roles differ')
    pending=copy.deepcopy([r for r in rows if r['role']!='anchor_only'])
    done=copy.deepcopy([r for r in rows if r['role']=='anchor_only'])
    if [r['order'] for r in done]!=[0,10] or any(r['seed']!=123 for r in rows):
        raise ValueError('state or seed order differs')
    result={'schema_version':'eatnav_e1_continuation_v1','original_manifest_sha256':sha(G0E/'e1-process-manifest.json'),
        'completed_anchor_rows':done,'pending_rows':pending,
        'counts':{'completed_anchor_only':2,'pending_direct_repeats':6,
            'pending_nondirect_outcomes':12,'pending_total':18,'unique_candidates':14},
        'execution_authorized_by_user':True,'new_anchor_replay':False,'candidate_outcomes_before_start':0}
    validate_frozen_identity(original,result)
    return result


def validate_frozen_identity(original,continuation):
    if continuation['completed_anchor_rows']!=[r for r in original['rows'] if r['role']=='anchor_only'] or \
            continuation['pending_rows']!=[r for r in original['rows'] if r['role']!='anchor_only']:
        raise ValueError('continuation identity differs from original manifest')
    rows=continuation['pending_rows']
    if len(rows)!=18 or len({r['process_id'] for r in rows})!=18:
        raise ValueError('continuation process identity differs')
    for sid in ('g0e-train-000','g0e-train-001'):
        group=[r for r in rows if r['state_group_id']==sid]
        if len(group)!=9 or [r['path_id'] for r in group]!=[
                'direct_reference']*3+['left_1','left_2','left_3','right_1','right_2','right_3']:
            raise ValueError('candidate order differs')
        if len({r['map_path_sha256'] for r in group})!=7 or len({r['map_path_sha256'] for r in group[:3]})!=1:
            raise ValueError('Path SHA identity differs')
    return True


def raw_actions_for_suffix(manifest,states):
    wanted={step for state in states for step in range(state['step']+1,state['step']+121)}
    maximum=max(wanted);actions={step:{} for step in wanted}
    for chunk in manifest['chunks']:
        path=RAW_MANIFEST.parent/chunk['file']
        if sha(path)!=chunk['sha256']:raise ValueError('raw collection chunk SHA differs')
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);step=int(row['step'])
                if step not in wanted:continue
                env=int(row['env_id'])
                if env in actions[step]:raise ValueError('duplicate raw suffix action')
                actions[step][env]=row['action']
        if all(len(actions[step])==100 for step in wanted):break
    if any(len(actions[step])!=100 for step in wanted):
        raise ValueError('missing recorded 100-env suffix actions')
    return {i:np.asarray([[actions[s][env] for env in range(100)]
             for s in range(state['step']+1,state['step']+121)],dtype=np.float32)
             for i,state in enumerate(states)}


def main():
    if ROOT.exists():raise FileExistsError('E1 preparation already performed; no overwrite')
    verify_anchor_reuse()
    decision=json.loads((G0E/'g0e-final-decision.json').read_text())
    if decision['status']!='G0-E pass / E1 awaiting separate approval' or decision['E1_started']:
        raise ValueError('G0-E decision differs')
    original=json.loads((G0E/'e1-process-manifest.json').read_text())
    continuation=build_continuation(original)
    with gzip.open(G0E/'map-path-identities.jsonl.gz','rt') as stream:
        paths={(r['state_group_id'],r['candidate_id']):json.loads(json.dumps(r)) for r in map(json.loads,stream)}
    if len(paths)!=14:raise ValueError('14 frozen map Paths required')
    source=json.loads((BANK/'inputs/selected-states.json').read_text())
    states=source['train']
    if [s['state_id'] for s in states]!=verify_anchor_reuse():raise ValueError('selected state identity differs')
    raw_manifest=json.loads(RAW_MANIFEST.read_text())
    actions=raw_actions_for_suffix(raw_manifest,states)
    ROOT.mkdir(parents=True)
    copy_frozen(G0E/'e1-process-manifest.json',ROOT/'original-20-row-manifest.json')
    copy_frozen(G0E/'e1-gates.yaml',ROOT/'e1-gates.yaml')
    copy_frozen(G0E/'e1-safety-stops.yaml',ROOT/'e1-safety-stops.yaml')
    copy_frozen(BANK/'inputs/selected-states.json',ROOT/'inputs/selected-states.json')
    copy_frozen(BANK/'g0e-selected-states.npz',ROOT/'inputs/g0e-selected-states.npz')
    copy_frozen(G0E/'map-path-identities.jsonl.gz',ROOT/'map-path-identities.jsonl.gz')
    atomic(ROOT/'continuation-manifest.json',continuation)
    atomic(ROOT/'completed-pending-identity.json',{'completed':[r['process_id'] for r in continuation['completed_anchor_rows']],
        'pending':[r['process_id'] for r in continuation['pending_rows']],
        'completed_anchor_result_sha256':[sha(G0E/f'formal-anchor-{i:03d}-result.json') for i in range(2)]})
    formal=yaml.safe_load((G0E/'formal-anchor-config.yaml').read_text())
    with gzip.open(BANK/'g0e-candidate-bank.jsonl.gz','rt') as stream:
        bank={(r['state_id'],r['candidate_id']):r for r in map(json.loads,stream)}
    for i,state in enumerate(states):
        folder=ROOT/'inputs'/f'state-{i:03d}';folder.mkdir(parents=True)
        previous=BANK/'inputs'/f'state-{i:03d}'
        for name in ('prefix-actions.npz','logged-anchor.json','recorded-frames.json','anchor-config.yaml'):
            copy_frozen(previous/name,folder/name)
        path=folder/'suffix-actions-120.npz'
        with path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream,actions=actions[i])
        path.with_suffix('.npz.tmp').replace(path)
        if actions[i].shape!=(120,100,6) or not np.isfinite(actions[i]).all():
            raise ValueError('suffix action shape/finite differs')
        local_cfg=yaml.safe_load((folder/'anchor-config.yaml').read_text())
        local_cfg['control_steps']=120
        local_cfg['scope']='e1_live_candidate'
        local_cfg['anchor_limits']=formal['anchor_limits']
        (folder/'config-120.yaml').write_text(yaml.safe_dump(local_cfg,sort_keys=True))
        for entry in [r for r in continuation['pending_rows'] if r['state_group_id']==state['state_id']]:
            candidate=bank[(state['state_id'],entry['path_id'])]
            frozen=paths[(state['state_id'],entry['path_id'])]
            local=np.asarray(candidate['waypoints_xy_m'],dtype=np.float32)
            mapped=np.asarray(frozen['map_waypoints_xy_m'],dtype=np.float32)
            if hashlib.sha256(local.tobytes()).hexdigest()!=entry['path_sha256'] or \
                    hashlib.sha256(mapped.tobytes()).hexdigest()!=entry['map_path_sha256']:
                raise ValueError('frozen candidate Path SHA differs')
            base=[[0.,0.]]+local.astype(float).tolist()
            map_path=[list(map(float,frozen['anchor_pose_xy_yaw'][:2]))]+mapped.astype(float).tolist()
            name=entry['process_id']
            atomic(folder/f'{name}-base-path.json',base)
            atomic(folder/f'{name}-map-path.json',map_path)
            atomic(folder/f'{name}-goal-metadata.json',{'state_id':state['state_id'],'run_id':name,
                'mission_logged_goal':state['logged_goal'],'goal_xy':base[-1],
                'base_path':base,'base_path_sha256':sha(folder/f'{name}-base-path.json'),
                'candidate_id':entry['path_id'],'amplitude_m':entry['amplitude_m'],
                'map_path_8point_sha256':entry['map_path_sha256'],
                'robot_local_8point_sha256':entry['path_sha256']})
    atomic(ROOT/'input-freeze.json',{'source_manifest_sha256':sha(RAW_MANIFEST),
        'split_hash':json.loads((G0E/'formal-anchor-provenance.json').read_text())['split_hash'],
        'G0E_manifest_sha256':sha(G0E/'e1-process-manifest.json'),
        'continuation_sha256':sha(ROOT/'continuation-manifest.json'),
        'map_paths_sha256':sha(ROOT/'map-path-identities.jsonl.gz'),
        'suffix_action_sha256':{f'state-{i:03d}':sha(ROOT/'inputs'/f'state-{i:03d}'/'suffix-actions-120.npz') for i in range(2)}})
    atomic(ROOT/'status.json',{'stage':'E1_PREPARED_NO_ROLLOUT','completed_anchor_only':2,
        'completed_new_processes':0,'pending_new_processes':18,'E1_live_started':False})
    print('E1_PREPARED',ROOT,flush=True)


if __name__=='__main__':main()
