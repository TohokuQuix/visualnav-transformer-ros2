"""Freeze raw replay inputs and offline E0 geometry for five new StateGroups."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from eat_nav_expanded_feasibility_preflight import CONFIG,sha,write_once
from visualnav_transformer.eat_nav.candidate_bank_centerline import audit_candidate,generate_bank
from visualnav_transformer.eat_nav.candidate_bank_e05 import urdf_collision_bounds

E0=Path('/tmp/eatnav-vnt-goal-ood-simulator-v2/docs/quality-identifiability-gate/candidate-coverage/e0-centerline/e0-centerline-config.yaml')
E0_SHA='7a521fa9077d45fc23f0ce6fa5e2b639248190811280859ab4f75aa0f8661181'
BANK_SHA='c68815526efc8150b229025bfaa196d9ff9edc58076b2042ae5295e5ec985f02'


def frozen_bank():
    cfg=yaml.safe_load(CONFIG.read_text())
    if sha(cfg['source_g0e_bank'])!=BANK_SHA:
        raise ValueError('frozen E1 candidate bank SHA differs')
    with gzip.open(cfg['source_g0e_bank'],'rt') as stream:
        rows=[json.loads(line) for line in stream if True]
    first=[r for r in rows if r['state_id']=='g0e-train-000']
    if len(rows)!=14 or len(first)!=7 or [r['candidate_id'] for r in first]!=cfg['candidate_ids']:
        raise ValueError('frozen Bezier bank identity differs')
    other=[r for r in rows if r['state_id']=='g0e-train-001']
    if any(a['waypoints_xy_m']!=b['waypoints_xy_m'] or a['dense_centerline_xy_m']!=b['dense_centerline_xy_m']
           for a,b in zip(first,other)):
        raise ValueError('old E1 bank changed between StateGroups')
    return first


def local_to_map(base,anchor_map_pose):
    points=np.asarray(base,dtype=np.float64);pose=np.asarray(anchor_map_pose,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or pose.shape!=(3,) or not np.isfinite(points).all() or not np.isfinite(pose).all():
        raise ValueError('finite base Path and anchor XY/yaw required')
    c,s=math.cos(pose[2]),math.sin(pose[2])
    return points@np.array([[c,s],[-s,c]])+pose[:2]


def validate_frame(row,episode_key,stamp_ns,env_id,*,expected_axis='row=+world_y,col=+world_x'):
    if ([row.get('session_id',episode_key[0]),row.get('env_id',episode_key[1]),row['episode_id']]!=list(episode_key)
        or row['pre_reset'] or row['timestamp_ns']!=stamp_ns or row.get('env_id',env_id)!=env_id):
        raise ValueError('DEM frame identity/timestamp differs')
    d=row['dem'];array=np.asarray(d['transformed'],dtype=np.float32)
    mask=array[1]
    if (array.shape!=(2,20,20) or not np.isfinite(array).all() or
        not np.logical_or(mask==0,mask==1).all() or d['axis']!=expected_axis or
        abs(float(d['resolution_m'])-.1)>1e-12 or
        np.asarray(d['origin_xy']).shape!=(2,) or not np.isfinite(d['origin_xy']).all()):
        raise ValueError('DEM frame axis/resolution/mask/finite differs')
    return True


def raw_replay_rows(manifest,states):
    max_anchor=max(s['step'] for s in states)
    max_suffix=max_anchor+120
    actions={};frames=[{} for _ in states];commands=[{} for _ in states];anchors={}
    frame_steps=[set(s['step']-15+i*5 for i in range(4)) for s in states]
    source=json.loads(Path(manifest).read_text())
    for chunk in source['chunks']:
        file=Path(manifest).parent/chunk['file']
        if sha(file)!=chunk['sha256']:raise ValueError('source raw chunk SHA differs '+str(file))
        with gzip.open(file,'rt') as stream:
            for line in stream:
                row=json.loads(line);step=int(row['step'])
                if step>max_suffix:continue
                env=int(row['env_id'])
                actions.setdefault(step,{})
                if env in actions[step]:raise ValueError('duplicate source action step/env')
                action=np.asarray(row['action'],dtype=np.float32)
                if action.shape!=(6,) or not np.isfinite(action).all():raise ValueError('nonfinite source action')
                actions[step][env]=action
                for i,s in enumerate(states):
                    if env!=s['env_id'] or step>s['step']:continue
                    commands[i][step]=row['command']
                    if step in frame_steps[i]:
                        validate_frame(row,s['episode_key'],(step+1)*50_000_000,env)
                        frames[i][step]=row
                    if step==s['step']:
                        anchors[i]=row
        if (len(anchors)==len(states) and all(len(frames[i])==4 for i in range(len(states)))
            and all(len(actions.get(step,{}))==100 for step in range(max_suffix+1))):break
    if (len(anchors)!=len(states) or any(len(frames[i])!=4 for i in range(len(states)))
        or any(len(actions.get(step,{}))!=100 for step in range(max_suffix+1))):
        raise ValueError('raw prefix/suffix/DEM support incomplete')
    all_actions=np.asarray([[actions[s][e] for e in range(100)] for s in range(max_suffix+1)],dtype=np.float32)
    return all_actions,frames,commands,anchors


def main():
    cfg=yaml.safe_load(CONFIG.read_text());root=Path(cfg['output_root'])
    if json.loads((root/'status.json').read_text())['stage']!='X1_SELECTED_NO_ROLLOUT':
        raise ValueError('X1 not frozen')
    if sha(E0)!=E0_SHA:raise ValueError('frozen E0 config SHA differs')
    e0=yaml.safe_load(E0.read_text())
    bank=frozen_bank();goal=np.asarray(cfg['goal_robot_local_xy_m'])
    generated=generate_bank('cubic_bezier',goal,cfg['signed_amplitudes_m'][1:4],
        parameter_samples=e0['generator']['curve_parameter_samples'],point_count=8)
    for old,new in zip(bank,generated):
        if old['candidate_id']!=new['candidate_id'] or not np.allclose(old['waypoints_xy_m'],new['waypoints'],atol=0,rtol=0):
            raise ValueError('frozen seven candidate geometry generation differs')
    states=json.loads((root/'selected-states.json').read_text())['train']
    with np.load(root/'selected-states.npz') as data:dems=np.asarray(data['dem'],dtype=np.float32)
    bounds=urdf_collision_bounds(e0['inputs']['urdf'],articulated=True)
    audits=[];bank_rows=[]
    for i,state in enumerate(states):
        row=[]
        for generated_row in generated:
            result=audit_candidate(generated_row,goal,dems[i],bounds,{**e0,'_family':'cubic_bezier'})
            row.append(result)
            original=next(x for x in bank if x['candidate_id']==result['candidate_id'])
            if result['waypoints_xy_m']!=original['waypoints_xy_m']:
                raise ValueError('E0 audited candidate Path differs from frozen bank')
            bank_rows.append({**original,'state_id':state['state_id']})
        audits.append({'state_id':state['state_id'],'terrain':state['terrain'],
            'candidate_audit':row,'hard_valid':all(not x['type_A_reasons'] for x in row)})
    write_once(root/'offline-e0-audit.json',{'groups':audits,'e0_config_sha256':sha(E0),
        'old_bank_sha256':sha(cfg['source_g0e_bank']),
        'all_hard_valid':all(x['hard_valid'] for x in audits),
        'footprint_and_curvature_diagnostic_only':True})
    with (root/'candidate-paths.jsonl.gz').open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
        for row in bank_rows:stream.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode())
    if not all(x['hard_valid'] for x in audits):
        write_once(root/'status.json.pending',{'stage':'X3_E0_HARD_INVALID_STOP','rollout_started':False})
        raise ValueError('E0 hard invalid; no live rollout')
    all_actions,frames,commands,anchors=raw_replay_rows(cfg['source_manifest'],states)
    for i,state in enumerate(states):
        folder=root/'inputs'/f'state-{i:03d}';folder.mkdir(parents=True,exist_ok=False)
        step=state['step'];command=np.asarray([commands[i][s] for s in range(step+1)],dtype=np.float32)
        if command.shape!=(step+1,3) or not np.isfinite(command).all():raise ValueError('prefix command incomplete')
        with (folder/'prefix-actions.npz').open('wb') as stream:
            np.savez_compressed(stream,actions=all_actions[:step+1],selected_env_commands=command)
        with (folder/'suffix-actions-120.npz').open('wb') as stream:
            np.savez_compressed(stream,actions=all_actions[step+1:step+121])
        write_once(folder/'logged-anchor.json',anchors[i])
        write_once(folder/'recorded-frames.json',[frames[i][s] for s in sorted(frames[i])])
        state_cfg={'schema_version':'eatnav_expanded_anchor_v1','scope':'anchor_only_then_live_candidate',
            'anchor_limits':{'position_m':.02,'orientation_rad':.03,'dem_rms_m':.03,'mask_iou_min':.95},
            'source_state':{'split':'train','state_index':i,'env_index':state['env_id'],
                'episode_id':0,'anchor_step':step},'control_steps':120}
        (folder/'config-120.yaml').write_text(yaml.safe_dump(state_cfg,sort_keys=True))
        for row in (x for x in bank_rows if x['state_id']==state['state_id']):
            base=[[0.,0.]]+row['waypoints_xy_m']
            write_once(folder/f"{row['candidate_id']}-base-path.json",base)
        dem=np.asarray([row['dem']['transformed'] for row in json.loads((folder/'recorded-frames.json').read_text())],dtype=np.float32)
        if hashlib.sha256(dem.tobytes()).hexdigest()!=state['dem_sha256']:
            raise ValueError('raw four DEM frames differ from selected window')
    write_once(root/'input-freeze.json',{'selected_states_sha256':sha(root/'selected-states.json'),
        'selected_dem_sha256':sha(root/'selected-states.npz'),
        'candidate_bank_sha256':sha(root/'candidate-paths.jsonl.gz'),
        'offline_audit_sha256':sha(root/'offline-e0-audit.json'),
        'per_state':[{p.name:sha(p) for p in sorted((root/'inputs'/f'state-{i:03d}').iterdir())}
                     for i in range(len(states))],
        'source_manifest_sha256':sha(cfg['source_manifest'])})
    write_once(root/'status.json.pending',{'stage':'X3_E0_INPUTS_READY_NO_ROLLOUT',
        'state_count':len(states),'candidate_count':len(bank_rows),'rollout_started':False})
    print('X3_E0_INPUTS_READY',[(x['state_id'],x['terrain']) for x in states],flush=True)


if __name__=='__main__':main()
