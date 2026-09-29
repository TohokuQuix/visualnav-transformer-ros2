"""Search complete-only train for independent-episode teacher support of bank winners."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import yaml
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView

ROOT=Path('/tmp/eatnav-candidate-coverage')
OUT=ROOT/'feasible-transfer'
BASE=ROOT/'baseline-worktree-8e8f4fd'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write_once(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise FileExistsError(f'existing result differs: {path}')
        return
    temp=path.with_name(path.name+'.tmp');temp.write_text(content);temp.replace(path)


def angle(a,b):
    return math.degrees(math.acos(float(np.clip(np.dot(a,b)/(max(np.linalg.norm(a),1e-9)*max(np.linalg.norm(b),1e-9)),-1,1))))


def run():
    started=time.time()
    config_path=BASE/'docs/quality-identifiability-gate/candidate-coverage/feasible-transfer/analysis-config.yaml'
    spec=yaml.safe_load(config_path.read_text())
    prior_path=BASE/'docs/quality-identifiability-gate/relative-quality/relative-quality-multi-match-threshold.json'
    prior=json.loads(prior_path.read_text())['geometry']
    for current,old in [('max_dem_rmse_m','max_dem_rmse_m'),('min_mask_iou','min_mask_iou'),
                        ('goal_distance_m','goal_distance_m'),('goal_direction_deg','goal_direction_deg')]:
        if spec['teacher_match'][current]!=prior[old]:raise ValueError('teacher threshold changed')
    gate=json.loads((OUT/'stage-0-complete.json').read_text())
    if gate['input_sha256']['config']!=sha(config_path):raise ValueError('Stage 0 config SHA mismatch')
    prior_stage=json.loads((OUT/'stage-1a-complete.json').read_text())
    if prior_stage['status']!='complete':raise ValueError('Stage 1A not complete')
    state_selection=json.loads((ROOT/'stage0/frozen/pilot-selection.json').read_text())['selected']
    comparisons=json.loads((OUT/'nomad-oracle-regret.json').read_text())['bank_winner_comparisons']
    manifest=Path('/tmp/eatnav-extended/output/manifest.json')
    cfg=load_eat_config(BASE/'docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
    view=CompleteOnlyView(EATDataset(manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(view)!=49598:raise ValueError('complete-only train count changed')
    windows=[view.dataset.windows[i] for i in view.indices]
    goals=np.asarray([w['goal'] for w in windows],dtype=np.float32)
    radii=np.linalg.norm(goals,axis=1)
    episodes=[tuple(w['episode_key']) for w in windows]
    result=[]
    for comp in comparisons:
        state=next(s for s in state_selection if s['state_id']==comp['state_id'])
        own=tuple(state['episode_key']);target_goal=np.asarray(state['goal'],dtype=float)
        target_radius=float(np.linalg.norm(target_goal))
        offline=json.loads((ROOT/'stage1'/f"{state['state_id']}.json").read_text())
        winner=next(c for c in offline['candidates'] if c['candidate_id']==comp['bank_candidate_id'])
        target_path=np.asarray(winner['points'],dtype=np.float32)
        if target_path.shape!=(8,2):raise ValueError('target trajectory shape changed')
        state_window=windows[state['view_index']]
        target_dem=np.asarray(state_window['dem'],dtype=np.float32)
        if hashlib.sha256(target_dem.tobytes()).hexdigest()!=state['dem_sha256']:
            raise ValueError('target DEM changed')
        goal_candidates=np.flatnonzero(np.abs(radii-target_radius)<=spec['teacher_match']['goal_distance_m'])
        direction=[]
        for i in goal_candidates:
            if episodes[i]!=own and angle(goals[i],target_goal)<=spec['teacher_match']['goal_direction_deg']:
                direction.append(int(i))
        near_dem=[];near_path=[];near_motion=[];details=[]
        target_mask=target_dem[:,1]>.5
        target_initial=target_path[0]
        for i in direction:
            w=windows[i];dem=np.asarray(w['dem'],dtype=np.float32)
            mask=dem[:,1]>.5
            both=target_mask&mask
            union=target_mask|mask
            iou=float(both.sum()/max(union.sum(),1))
            height_rmse=float(np.sqrt(np.mean((target_dem[:,0][both]-dem[:,0][both])**2))) if both.any() else float('inf')
            if iou<spec['teacher_match']['min_mask_iou'] or height_rmse>spec['teacher_match']['max_dem_rmse_m']:
                continue
            near_dem.append(i)
            increments=np.asarray(w['action'],dtype=np.float32)
            path=np.cumsum(increments,axis=0)
            rms=float(np.sqrt(np.mean((path-target_path)**2)))
            if rms<=spec['teacher_match']['path_rms_m']:near_path.append(i)
            motion_deg=angle(increments[0],target_initial)
            if rms<=spec['teacher_match']['path_rms_m'] and motion_deg<=spec['teacher_match']['initial_motion_direction_deg']:
                near_motion.append(i)
                details.append({'view_index':i,'dataset_index':int(view.indices[i]),
                    'episode_key':list(episodes[i]),'path_rms_m':rms,'initial_motion_direction_deg':motion_deg,
                    'dem_rmse_m':height_rmse,'mask_iou':iou,'goal_distance_delta_m':float(abs(radii[i]-target_radius)),
                    'goal_direction_delta_deg':angle(goals[i],target_goal)})
        if comp['same_command_mode'] and comp['same_realized_mode']:
            classification='same_mode_outcome_variation'
        elif near_motion:
            classification='supported_but_not_generated'
        elif near_dem:
            classification='unsupported_by_dataset'
        else:
            classification='insufficient_evidence'
        result.append({'state_id':state['state_id'],'terrain':state['terrain_type'],
            'bank_candidate_id':comp['bank_candidate_id'],'nearest_nomad_candidate_id':comp['nearest_nomad_candidate_id'],
            'generator':'bank','seed':None,'classification':classification,
            'same_command_mode':comp['same_command_mode'],'same_realized_mode':comp['same_realized_mode'],
            'counts':{'complete_train':len(view),'goal_distance_only':len(goal_candidates),
                'goal_direction_and_different_episode':len(direction),'dem_goal_match':len(near_dem),
                'trajectory_match':len(near_path),'trajectory_and_motion_match':len(near_motion),
                'distinct_support_episodes':len(set(episodes[i] for i in near_motion))},
            'matched_teacher_windows':details[:100],
            'matched_teacher_total':len(details)})
    write_once(OUT/'teacher-support-analysis.json',{'states':result,'complete_only_train_window_count':len(view),
        'threshold_source_sha256':sha(prior_path),'manifest_sha256':sha(manifest),
        'selection_sha256':sha(ROOT/'stage0/frozen/pilot-selection.json'),
        'elapsed_sec':time.time()-started,'validation_used':False,'heldout_used':False})
    write_once(OUT/'stage-1b-complete.json',{'stage':'1b','status':'complete',
        'source_sha256':sha(OUT/'stage-1a-complete.json'),
        'output_sha256':sha(OUT/'teacher-support-analysis.json'),'unix_sec':time.time()})
    print('FEASIBLE_TRANSFER_STAGE1B_COMPLETE',len(result),'states',round(time.time()-started,1),'sec',flush=True)

if __name__=='__main__':run()
