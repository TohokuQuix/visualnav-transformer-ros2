"""Frozen held-out EAT trajectory casebook stages; run inside the Isaac container."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset, _nearest, _records
from visualnav_transformer.eat_nav.quality_gate_evaluation import episode_cluster_ci

BASE=Path('/tmp/eatnav-train')
ROOT=BASE/'casebook'
OUT=ROOT/'output'
SOURCE=BASE/'quality-gate-output'
MANIFEST=Path('/tmp/eatnav-extended/output/manifest.json')
QUALITY=BASE/'quality.yaml'
BASELINE=BASE/'baseline.yaml'
CONFIG=BASE/'casebook-config.yaml'
CHECK_Q=BASE/'pilot-quality-best.pt'
CHECK_0=BASE/'baseline-run/best.pt'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)

def spec():return yaml.safe_load(CONFIG.read_text())

def verify():
    s=spec()
    for path,key in [(CHECK_Q,'checkpoint_quality_sha256'),(CHECK_0,'checkpoint_baseline_sha256'),
                     (MANIFEST,'manifest_sha256'),(QUALITY,'quality_config_sha256')]:
        if sha(path)!=s[key]:raise ValueError(f'{key} mismatch')
    prior=json.loads((SOURCE/'data-audit.json').read_text())
    if prior['split_hash']!='ae584ee5813be6a09f16baa4fae8271ed20c457ef40d9dd44005636a71e9234f':
        raise ValueError('split hash mismatch')
    return s,load_eat_config(QUALITY)

def complete(split,config):
    d=EATDataset(MANIFEST,split,config)
    return d,[dict(d[i],window_index=i) for i,w in enumerate(d.windows) if w['action_mask'].all()]

def ident(w):return (tuple(w['episode_key']),int(w['step']))
def map_frozen_pair(row, original_to_complete, complete_rows):
    """Gate A2 IDs address all test windows; the casebook stores complete-only IDs."""
    lo=original_to_complete[int(row['low_index'])];hi=original_to_complete[int(row['high_index'])]
    if tuple(row['low_episode'])!=tuple(complete_rows[lo]['episode_key']) or tuple(row['high_episode'])!=tuple(complete_rows[hi]['episode_key']):
        raise ValueError('frozen pair episode identity mismatch')
    if int(row['low_step'])!=int(complete_rows[lo]['step']) or int(row['high_step'])!=int(complete_rows[hi]['step']):
        raise ValueError('frozen pair step identity mismatch')
    return dict(row,low_index=lo,high_index=hi,low_original_index=int(row['low_index']),high_original_index=int(row['high_index']))
def geom(w):return np.cumsum(w['action'],axis=0)
def displacement(w):return float(np.linalg.norm(geom(w)[-1]))
def qbin(q):return min(3,int(float(q)*4))
def episode_json(e):return list(e)

def raw_metadata(needed):
    """Read small fields only, preserving original record order and episode identity."""
    result=defaultdict(list)
    m=json.loads(MANIFEST.read_text())
    for chunk in m['chunks']:
        with gzip.open(MANIFEST.parent/chunk['file'],'rt') as f:
            for line in f:
                r=json.loads(line)
                key=(r['session_id'],r['env_id'],r['episode_id'])
                if key not in needed:continue
                result[key].append({k:r[k] for k in ('step','timestamp_ns','pose','yaw','command','twist',
                    'critic_value','terminated','truncated','termination_reason','terrain_type')})
    for rows in result.values():rows.sort(key=lambda r:r['timestamp_ns'])
    return result

def source_checks(w,rows,config):
    i=int(w['local_step']);now=rows[i];dt=round(config.trajectory_interval_sec*1e9)
    ht=round(config.dem_history_interval_sec*1e9);tol=round(.05*.4*1e9)
    times=[r['timestamp_ns'] for r in rows]
    if int(now['step'])!=int(w['step']):raise ValueError('step identity mismatch')
    for offset in range(config.context_size,-1,-1):
        j=_nearest(times,now['timestamp_ns']-offset*ht,tol)
        if j is None:raise ValueError('history crosses episode')
    c,s=math.cos(now['yaw']),math.sin(now['yaw']);xy=[]
    for p in range(1,config.len_traj_pred+1):
        j=_nearest(times,now['timestamp_ns']+p*dt,tol)
        if j is None or j<=i or rows[j]['terminated'] and p<config.len_traj_pred:
            raise ValueError('target crosses episode/terminal')
        dx=np.asarray(rows[j]['pose'][:2])-np.asarray(now['pose'][:2]);xy.append((c*dx[0]+s*dx[1],-s*dx[0]+c*dx[1]))
    xy=np.asarray(xy)
    if not np.allclose(xy,geom(w),atol=1e-5):raise ValueError('teacher coordinate mismatch')
    goal=xy[-1].copy();radius=np.linalg.norm(goal)
    if radius>config.goal_max_radius_m:goal*=config.goal_max_radius_m/radius
    if not np.allclose(goal,w['goal'],atol=1e-5):raise ValueError('goal coordinate mismatch')
    return now

def audit():
    s,c=verify();train,tw=complete('train',c);test,hw=complete('test',c)
    if len(hw)!=s['complete_window_count']:raise ValueError('held-out complete count mismatch')
    if len({w['terrain_seed'] for w in tw}&{w['terrain_seed'] for w in hw}):raise ValueError('split leakage')
    epkeys={tuple(w['episode_key']) for w in hw};raw=raw_metadata(epkeys)
    if epkeys!=set(raw):raise ValueError('raw episode lookup mismatch')
    prog=[];goald=[];qs=[];angles=[];command_speeds=[];terminal=[];reasons=Counter()
    for w in hw:
        rows=raw[tuple(w['episode_key'])];now=source_checks(w,rows,c)
        g=np.asarray(w['goal']);v=np.asarray(now['command'][:2]);p=displacement(w)
        prog.append(p);goald.append(float(np.linalg.norm(g)));qs.append(float(w['quality']))
        if np.linalg.norm(v)>1e-6 and np.linalg.norm(g)>1e-6:
            angles.append(float(math.degrees(math.acos(np.clip(np.dot(v,g)/(np.linalg.norm(v)*np.linalg.norm(g)),-1,1)))))
        command_speeds.append(float(np.linalg.norm(v)))
        terminal.append(float((rows[-1]['timestamp_ns']-now['timestamp_ns'])/1e9))
        reasons.update(rows[-1]['termination_reason'])
    threshold=float(np.quantile([displacement(w) for w in tw],s['cohort_b_train_endpoint_quantile']))
    high=float(np.quantile([displacement(w) for w in tw],s['cohort_b_control_train_endpoint_quantile']))
    np.savez_compressed(ROOT/'audit-arrays.npz',progress=prog,goal=goald,q=qs,terminal=terminal,
                        command_speed=command_speeds)
    result={'heldout_complete':len(hw),'train_complete':len(tw),'cross_episode_sources':0,
            'train_no_progress_threshold_m':threshold,'train_high_progress_threshold_m':high,
            'goal_source':'future realized endpoint transformed world XY to anchor base XY, capped at 0.8 m',
            'command_source':'raw 3-component base velocity command [vx, vy, yaw_rate]; m/s and rad/s, not a commanded goal',
            'goal_normalization':'no goal normalization; radius cap only',
            'terminal_window_rule':'complete-only: all 8 future points precede same-episode end',
            'progress_goal_pearson':float(np.corrcoef(prog,goald)[0,1]),
            'q_goal_pearson':float(np.corrcoef(qs,goald)[0,1]),
            'command_goal_direction_abs_deg_median':float(np.median(angles)) if angles else None,
            'command_goal_distance_difference':'NA: velocity command [m/s] and relative goal [m] have different units',
            'terminal_time_goal_pearson':float(np.corrcoef(terminal,goald)[0,1]),
            'heldout_terminal_reason_window_counts':dict(reasons),
            'source_hashes':{'manifest':sha(MANIFEST),'quality_checkpoint':sha(CHECK_Q),'baseline_checkpoint':sha(CHECK_0)}}
    write(OUT/'goal-provenance.json',result)
    (OUT/'goal-provenance.md').write_text('# Goal provenance\n\n'+ '\n'.join(f'- **{k}**: {v}' for k,v in result.items())+'\n')
    print('AUDIT',json.dumps({'heldout':len(hw),'train_threshold':threshold,'goal_progress_r':result['progress_goal_pearson']}),flush=True)

def stratified_select(candidates,limit,seed,cap,keyfn):
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for i in candidates:groups[keyfn(i)].append(i)
    for g in groups:rng.shuffle(groups[g])
    chosen=[];uses=Counter();keys=sorted(groups,key=str)
    while len(chosen)<limit and any(groups.values()):
        progress=False
        for k in keys:
            while groups[k]:
                i=groups[k].pop();e=i[1]
                if uses[e]>=cap:continue
                chosen.append(i);uses[e]+=1;progress=True;break
            if len(chosen)>=limit:break
        if not progress:break
    return [i[0] for i in chosen]

def cohorts():
    s,c=verify();_,tw=complete('train',c);_,hw=complete('test',c)
    if len(hw)!=s['complete_window_count']:raise ValueError('heldout count mismatch')
    threshold=json.loads((OUT/'goal-provenance.json').read_text())['train_no_progress_threshold_m']
    high=json.loads((OUT/'goal-provenance.json').read_text())['train_high_progress_threshold_m']
    eligible=lambda i:(i,tuple(hw[i]['episode_key']))
    epkeys={tuple(w['episode_key']) for w in hw};raw=raw_metadata(epkeys)
    a=[i for i,w in enumerate(hw) if 'tilt_over' in raw[tuple(w['episode_key'])][-1]['termination_reason']]
    b0=[eligible(i) for i,w in enumerate(hw) if displacement(w)<=threshold]
    b=stratified_select(b0,s['cohort_b_max'],s['selection_seed'],s['cohort_episode_cap'],lambda x:hw[x[0]]['terrain_type'])
    # Match on terrain, goal distance and DEM geometry. Report gaps, never impose an impossible exact goal match.
    available=[i for i,w in enumerate(hw) if displacement(w)>=high];used=set();controls=[]
    for i in b:
        w=hw[i];mask=w['dem'][-1,1]>.5;h=w['dem'][-1,0];cands=[]
        for j in available:
            z=hw[j]
            if j in used or z['episode_key']==w['episode_key'] or z['terrain_type']!=w['terrain_type']:continue
            zm=z['dem'][-1,1]>.5;union=np.logical_or(mask,zm).sum();iou=np.logical_and(mask,zm).sum()/max(union,1)
            both=np.logical_and(mask,zm);rmse=float(np.sqrt(np.mean((h[both]-z['dem'][-1,0][both])**2))) if both.any() else 1e3
            gd=abs(np.linalg.norm(w['goal'])-np.linalg.norm(z['goal']))
            score=s['cohort_b_match_goal_weight']*gd+s['cohort_b_match_dem_rmse_weight']*rmse+s['cohort_b_match_mask_iou_weight']*(1-iou)
            cands.append((score,j,gd,rmse,iou))
        if cands:
            score,j,gd,rmse,iou=min(cands);used.add(j)
            controls.append({'low_index':i,'high_index':j,'score':float(score),'goal_distance_gap_m':float(gd),
                             'dem_rmse_m':rmse,'mask_iou':float(iou)})
    pairs=json.loads((SOURCE/'heldout-pair-controls.json').read_text())
    original_to_complete={int(w['window_index']):i for i,w in enumerate(hw)}
    selected=[];epuse=Counter();indices=set()
    for frozen in sorted(pairs['cross_q_pairs'],key=lambda r:hashlib.sha256(f"{r['low_index']}:{r['high_index']}".encode()).hexdigest()):
        if int(frozen['low_index']) not in original_to_complete or int(frozen['high_index']) not in original_to_complete:continue
        row=map_frozen_pair(frozen,original_to_complete,hw)
        lo,hi=int(row['low_index']),int(row['high_index']);le=tuple(hw[lo]['episode_key']);he=tuple(hw[hi]['episode_key'])
        if lo in indices or hi in indices or epuse[le]>=s['cohort_episode_cap'] or epuse[he]>=s['cohort_episode_cap']:continue
        if le==he or qbin(hw[lo]['quality'])!=0 or qbin(hw[hi]['quality'])!=3:continue
        selected.append(row);indices.update((lo,hi));epuse.update((le,he))
        if len(selected)>=s['cohort_c_pairs']:break
    same=[];used_same=set()
    for frozen in sorted(pairs['same_q_pairs'],key=lambda r:hashlib.sha256(f"{r['low_index']}:{r['high_index']}".encode()).hexdigest()):
        if int(frozen['low_index']) not in original_to_complete or int(frozen['high_index']) not in original_to_complete:continue
        row=map_frozen_pair(frozen,original_to_complete,hw)
        lo,hi=int(row['low_index']),int(row['high_index'])
        if lo in used_same or hi in used_same or hw[lo]['terrain_type'] not in {r['terrain_type'] for r in selected}:continue
        same.append(row);used_same.update((lo,hi))
        if len(same)>=len(selected):break
    d=stratified_select([eligible(i) for i in range(len(hw))],s['cohort_d_count'],s['selection_seed']+1,
                        s['cohort_episode_cap'],lambda x:(hw[x[0]]['terrain_type'],qbin(hw[x[0]]['quality'])))
    # All identities are from held-out windows and frozen before model access.
    allidx=sorted(set(a+b+d+[r['low_index'] for r in controls]+[r['high_index'] for r in controls]
                       +[int(r[k]) for r in selected+same for k in ('low_index','high_index')]))
    result={'A':a,'B':b,'B_controls':controls,'C':selected,'C_same_q_controls':same,'D':d,'unique_indices':allidx,
            'counts':{'A':len(a),'B':len(b),'B_matched':len(controls),'C':len(selected),'C_same_q':len(same),'D':len(d),'unique':len(allidx)},
            'rules':{k:s[k] for k in ('selection_seed','cohort_b_train_endpoint_quantile','cohort_b_control_train_endpoint_quantile',
                        'cohort_episode_cap','cohort_b_max','cohort_c_pairs','cohort_d_count')},
            'selection_uses_model_output':False,'source_pair_sha256':sha(SOURCE/'heldout-pair-controls.json')}
    write(OUT/'casebook-cohorts.json',result)
    with gzip.open(OUT/'casebook-window-identities.jsonl.gz','wt') as f:
        for i in allidx:
            w=hw[i];f.write(json.dumps({'index':i,'original_dataset_index':int(w['window_index']),'episode_key':episode_json(w['episode_key']),'step':int(w['step']),
                'terrain_seed':int(w['terrain_seed']),'terrain_type':w['terrain_type'],'q':float(w['quality']),
                'q_bin':qbin(w['quality']),'goal':w['goal'].tolist(),'teacher_endpoint':geom(w)[-1].tolist()})+'\n')
    np.savez_compressed(ROOT/'selected.npz',indices=allidx,dem=np.stack([hw[i]['dem'] for i in allidx]),
                        goal=np.stack([hw[i]['goal'] for i in allidx]),action=np.stack([hw[i]['action'] for i in allidx]),
                        quality=np.asarray([hw[i]['quality'] for i in allidx]),
                        episode=np.asarray([hw[i]['episode_key'] for i in allidx],dtype=np.int64),
                        terrain=np.asarray([hw[i]['terrain_type'] for i in allidx]))
    # Raw time series and source metadata are kept outside Git.
    rawsel=raw_metadata({tuple(hw[i]['episode_key']) for i in allidx})
    metadata={str(i):{'timestamp_ns':rawsel[tuple(hw[i]['episode_key'])][hw[i]['local_step']]['timestamp_ns'],
                      'anchor':rawsel[tuple(hw[i]['episode_key'])][hw[i]['local_step']],
                      'terminal':rawsel[tuple(hw[i]['episode_key'])][-1],
                      'series':rawsel[tuple(hw[i]['episode_key'])]} for i in allidx}
    write(ROOT/'selected-raw.json',metadata)
    print('COHORTS',json.dumps(result['counts']),flush=True)

def infer(smoke=False):
    s,c=verify();z=np.load(ROOT/'selected.npz');indices=z['indices'].tolist()
    if smoke:indices=indices[:3]
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    qc,_,_=load_checkpoint(CHECK_Q,c);bc=load_eat_config(BASELINE);bm,_,_=load_checkpoint(CHECK_0,bc)
    qc=qc.to(device).eval();bm=bm.to(device).eval()
    if qc.training or bm.training:raise ValueError('model not eval')
    mapping={int(v):j for j,v in enumerate(z['indices'])}
    # Paired streams: identical seed and explicit initial noise for every q/model call.
    pred=np.empty((len(indices),len(s['diffusion_seeds']),len(s['q_values']),c.len_traj_pred,2),dtype=np.float32)
    baseline=np.empty((len(indices),len(s['diffusion_seeds']),c.len_traj_pred,2),dtype=np.float32)
    start=time.monotonic();peak=0
    with torch.no_grad():
        for n,i in enumerate(indices):
            j=mapping[i];dem=torch.from_numpy(z['dem'][j:j+1]).to(device);goal=torch.from_numpy(z['goal'][j:j+1]).to(device)
            if not torch.isfinite(dem).all() or not torch.isfinite(goal).all():raise ValueError('nonfinite input')
            dem_bytes=dem.cpu().numpy().tobytes();goal_bytes=goal.cpu().numpy().tobytes()
            for k,seed in enumerate(s['diffusion_seeds']):
                gen=torch.Generator(device=device).manual_seed(seed);noise=torch.randn((1,c.len_traj_pred,2),generator=gen,device=device)
                for h,q in enumerate(s['q_values']):
                    if dem.cpu().numpy().tobytes()!=dem_bytes or goal.cpu().numpy().tobytes()!=goal_bytes:
                        raise ValueError('q sweep input mismatch')
                    out=qc.sample(dem,goal,torch.tensor([q],device=device),seed=seed,initial_noise=noise)
                    pred[n,k,h]=out.cpu().numpy()[0]
                baseline[n,k]=bm.sample(dem,goal,None,seed=seed,initial_noise=noise).cpu().numpy()[0]
            if not np.isfinite(pred[n]).all() or not np.isfinite(baseline[n]).all():raise ValueError('nonfinite prediction')
            if device=='cuda':peak=max(peak,torch.cuda.max_memory_allocated()/1024**3)
            if (n+1)%10==0:print('INFERENCE_PROGRESS',n+1,len(indices),flush=True)
    target=ROOT/('smoke-predictions.npz' if smoke else 'predictions.npz')
    np.savez_compressed(target,indices=indices,pred=pred,baseline=baseline)
    write(ROOT/('smoke.json' if smoke else 'inference.json'),{'windows':len(indices),'q_values':s['q_values'],
          'diffusion_seeds':s['diffusion_seeds'],'seconds':time.monotonic()-start,'peak_torch_vram_gb':peak,
          'device':device,'eval':True,'no_grad':True,'paired_initial_and_step_noise':True,
          'input_fixed_across_q':True,'prediction_sha256':sha(target)})
    print('INFER_DONE',len(indices),round(time.monotonic()-start,1),flush=True)

def trajectory_metrics(x):
    d=np.diff(np.vstack((np.zeros((1,2)),x)),axis=0)
    ang=np.arctan2(d[:,1],d[:,0]);turn=np.abs(np.arctan2(np.sin(np.diff(ang)),np.cos(np.diff(ang)))).sum()
    return {'endpoint_displacement_m':float(np.linalg.norm(x[-1])),'forward_progress_m':float(x[-1,0]),
            'path_length_m':float(np.linalg.norm(d,axis=1).sum()),'turning_rad':float(turn)}

def rms(a,b):return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
def safe_corr(a,b):return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None

def metrics():
    s,c=verify();p=np.load(ROOT/'predictions.npz');z=np.load(ROOT/'selected.npz');co=json.loads((OUT/'casebook-cohorts.json').read_text())
    ix={int(i):j for j,i in enumerate(p['indices'])};source={int(i):j for j,i in enumerate(z['indices'])}
    qvals=s['q_values'];qmid=qvals.index(.5);raw=json.loads((ROOT/'selected-raw.json').read_text())
    membership=defaultdict(list)
    for name in ('A','B','D'):
        for i in co[name]:membership[i].append(name)
    for row in co['B_controls']:
        membership[row['high_index']].append('B_control')
    for name,key in [('C','C'),('C_control','C_same_q_controls')]:
        for row in co[key]:
            membership[int(row['low_index'])].append(name);membership[int(row['high_index'])].append(name)
    rows=[];cases=[];thresholds=s['category_thresholds']
    for i in p['indices']:
        i=int(i);j=ix[i];v=source[i];teacher=geom({'action':z['action'][v]});q=float(z['quality'][v]);qix=int(np.argmin(abs(np.asarray(qvals)-q)))
        outputs=p['pred'][j];base=p['baseline'][j];effect=float(np.mean([rms(outputs[k,0],outputs[k,-1]) for k in range(len(outputs))]))
        sampling=float(np.mean([rms(outputs[k,qmid],outputs[h,qmid]) for k in range(len(outputs)) for h in range(k+1,len(outputs))]))
        mean_raw=outputs[:,qix].mean(axis=0);mean_base=base.mean(axis=0)
        low=outputs[:,:,0];high=outputs[:,:,-1]
        progress=[float(outputs[:,h,-1,0].mean()) for h in range(len(qvals))]
        turning=[float(np.mean([trajectory_metrics(x)['turning_rad'] for x in outputs[:,h]])) for h in range(len(qvals))]
        rho=safe_corr(qvals,progress);tr=trajectory_metrics(teacher);gn=trajectory_metrics(mean_raw)
        info=raw[str(i)];tm=(info['terminal']['timestamp_ns']-info['timestamp_ns'])/1e9
        cat={'A':False,'B':effect<=sampling*thresholds['q_ignored_effect_ratio_max'] and effect<=thresholds['q_ignored_rms_m_max'] and (rho is None or abs(rho)<thresholds['monotonic_spearman_min']),
             'C':sampling>=effect*thresholds['sampling_dominance_ratio_min'],
             'D':rho is not None and abs(rho)>=thresholds['monotonic_spearman_min'] and effect>thresholds['q_ignored_rms_m_max'],
             'E':False}
        row={'index':i,'episode_key':json.dumps(z['episode'][v].tolist()),'terrain':str(z['terrain'][v]),'q':q,
             'q_bin':qbin(q),'cohorts':','.join(sorted(membership[i])),'terminal_seconds':tm,
             'terminal_reason':','.join(info['terminal']['termination_reason']),
             'teacher_progress_m':tr['forward_progress_m'],'teacher_endpoint_m':tr['endpoint_displacement_m'],
             'teacher_turning_rad':tr['turning_rad'],'teacher_path_length_m':tr['path_length_m'],
             'goal_distance_m':float(np.linalg.norm(z['goal'][v])),'raw_command_speed_mps':float(np.linalg.norm(info['anchor']['command'][:2])),
             'generated_progress_m':gn['forward_progress_m'],'generated_turning_rad':gn['turning_rad'],
             'generated_path_length_m':gn['path_length_m'],'enabled_teacher_rms_m':rms(mean_raw,teacher),
             'disabled_teacher_rms_m':rms(mean_base,teacher),'q_effect_rms_m':effect,'sampling_rms_m':sampling,
             'q_sampling_ratio':effect/max(sampling,1e-9),'progress_q_correlation':rho,
             'q_low_progress_m':progress[0],'q_high_progress_m':progress[-1],
             'baseline_progress_m':mean_base[-1,0],**{f'category_{k}':int(x) for k,x in cat.items()}}
        rows.append(row);cases.append({'index':i,'q_progress_m':progress,'q_turning_rad':turning,'q_effect_rms_m':effect,
                                          'sampling_rms_m':sampling,'categories':cat})
    # Paired low/high teacher selection: one fixed seed bank for each member.
    pairrows=[]
    for pair in co['C']:
        lo,hi=int(pair['low_index']),int(pair['high_index']);lt=geom({'action':z['action'][source[lo]]});ht=geom({'action':z['action'][source[hi]]})
        for anchor in (lo,hi):
            j=ix[anchor]
            for qi,q in enumerate(qvals):
                dl=np.mean([rms(x,lt) for x in p['pred'][j,:,qi]]);dh=np.mean([rms(x,ht) for x in p['pred'][j,:,qi]])
                pairrows.append({'low_index':lo,'high_index':hi,'anchor':anchor,'q':q,'low_rms_m':float(dl),
                                 'high_rms_m':float(dh),'nearest':'low' if dl<dh else 'high'})
    for row in rows:
        i=row['index'];control=next((r for r in co['B_controls'] if r['low_index']==i),None)
        if control:
            v=source[control['high_index']];high_teacher=geom({'action':z['action'][v]});own=geom({'action':z['action'][source[i]]})
            gen=p['pred'][ix[i],:,int(np.argmin(abs(np.asarray(qvals)-row['q'])))].mean(axis=0)
            near_low=rms(gen,own)+thresholds['low_teacher_distance_margin_m']<rms(gen,high_teacher)
            consistent=np.sign(row['generated_progress_m'])==np.sign(row['teacher_progress_m'])
            row['category_A']=int(near_low and consistent and row['q_high_progress_m']-row['q_low_progress_m']<thresholds['q_ignored_rms_m_max'])
    for row in rows:
        row['category_E']=int(abs(json.loads((OUT/'goal-provenance.json').read_text())['progress_goal_pearson'])>=thresholds['goal_progress_correlation_min']
                              and row['teacher_endpoint_m']<=float(json.loads((OUT/'goal-provenance.json').read_text())['train_no_progress_threshold_m'])
                              and row['goal_distance_m']<=thresholds['goal_no_progress_max_m'])
    OUT.mkdir(parents=True,exist_ok=True)
    with open(OUT/'casebook-metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(OUT/'casebook-metrics.json',{'cases':cases,'pair_teacher_distances':pairrows,
          'category_thresholds':thresholds,'continuous_metrics_csv':'casebook-metrics.csv'})
    summary={'window_count':len(rows),'cohort_counts':co['counts'],'q_values':qvals,'seeds':s['diffusion_seeds'],
             'median_q_effect_rms_m':float(np.median([r['q_effect_rms_m'] for r in rows])),
             'median_sampling_rms_m':float(np.median([r['sampling_rms_m'] for r in rows])),
             'median_q_sampling_ratio':float(np.median([r['q_sampling_ratio'] for r in rows])),
             'enabled_mean_teacher_rms_m':float(np.mean([r['enabled_teacher_rms_m'] for r in rows])),
             'disabled_mean_teacher_rms_m':float(np.mean([r['disabled_teacher_rms_m'] for r in rows])),
             'category_counts':{k:sum(r[f'category_{k}'] for r in rows) for k in 'ABCDE'},
             'cohort_category_counts':{},'terrain_category_counts':{},'pair_count':len(co['C']),
             'pair_low_nearest_at_q_low':float(np.mean([r['nearest']=='low' for r in pairrows if r['q']==.1])),
             'pair_high_nearest_at_q_high':float(np.mean([r['nearest']=='high' for r in pairrows if r['q']==.95])),
             'interpretation':'Static XY only; no physical safety or closed-loop failure conclusion.'}
    for grouping,dest in [('cohorts','cohort_category_counts'),('terrain','terrain_category_counts')]:
        keys=sorted({g for r in rows for g in (r[grouping].split(',') if grouping=='cohorts' else [r[grouping]])})
        for key in keys:
            subset=[r for r in rows if key in (r[grouping].split(',') if grouping=='cohorts' else [r[grouping]])]
            bins=Counter(r['q_bin'] for r in subset)
            supported=grouping=='cohorts' or all(bins.get(b,0)>=s['support_min_per_terrain_q_bin'] for b in range(4))
            summary[dest][key]={'n':len(subset),'q_bins':dict(bins),'status':'supported' if supported else 'insufficient support',
                 'categories':{k:sum(r[f'category_{k}'] for r in subset) if supported else None for k in 'ABCDE'}}
            if supported and len({r['episode_key'] for r in subset})>=2:
                summary[dest][key]['q_effect_ci95_episode']=episode_cluster_ci([r['q_effect_rms_m'] for r in subset],
                    [r['episode_key'] for r in subset],seed=s['bootstrap_seed'],repeats=s['bootstrap_repeats'])[0]
    write(OUT/'casebook-summary.json',summary)
    print('METRICS',json.dumps({k:summary[k] for k in ('window_count','median_q_sampling_ratio','category_counts')}),flush=True)

def drawing(smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    p=np.load(ROOT/('smoke-predictions.npz' if smoke else 'predictions.npz'))
    z=np.load(ROOT/'selected.npz');raw=json.loads((ROOT/'selected-raw.json').read_text())
    ix={int(i):j for j,i in enumerate(z['indices'])};spec0=spec();qs=spec0['q_values'];colors=plt.cm.viridis(np.linspace(0,1,len(qs)))
    target=ROOT/'smoke' if smoke else OUT;cases=target/'cases';cases.mkdir(parents=True,exist_ok=True)
    reportrows={}
    if not smoke:
        with open(OUT/'casebook-metrics.csv') as f:reportrows={int(r['index']):r for r in csv.DictReader(f)}
    pdf=PdfPages(target/'casebook.pdf')
    for i in p['indices']:
        i=int(i);j=ix[i];n=list(p['indices']).index(i);dem=z['dem'][j,-1];goal=z['goal'][j];teacher=geom({'action':z['action'][j]});meta=raw[str(i)]
        fig,axs=plt.subplots(2,3,figsize=(16,10));axs=axs.flat
        extent=(-1,1,-1,1);axs[0].imshow(dem[0],origin='lower',extent=extent,cmap='terrain');axs[0].contour(dem[1],levels=[.5],extent=extent,colors='black',linewidths=.6)
        axs[0].plot(0,0,'ks');axs[0].arrow(0,0,.2,0,width=.005,color='black');axs[0].plot(goal[0],goal[1],'r*',label='dataset goal')
        axs[0].plot(teacher[-1,0],teacher[-1,1],'bo',label='realized endpoint');cmd=meta['anchor']['command'];axs[0].arrow(0,0,cmd[0]*.25,cmd[1]*.25,color='magenta',width=.006,label='command direction (scaled)')
        axs[0].add_patch(plt.Rectangle((-.25,-.18),.5,.36,fill=False,color='black'));axs[0].legend(fontsize=7);axs[0].set_title('A Input: base XY [m], DEM height/mask')
        axs[1].plot(teacher[:,0],teacher[:,1],'k--',label='teacher')
        for h,q in enumerate(qs):axs[1].plot(p['pred'][n,0,h,:,0],p['pred'][n,0,h,:,1],color=colors[h],label=f'q={q}')
        axs[1].legend(fontsize=7);axs[1].set_title('B q sweep, shared noise')
        for k,x in enumerate(p['pred'][n,:,2]):axs[2].plot(x[:,0],x[:,1],alpha=.5,label=str(spec0['diffusion_seeds'][k]))
        mean=p['pred'][n,:,2].mean(axis=0);axs[2].plot(mean[:,0],mean[:,1],'k-',linewidth=2);axs[2].legend(fontsize=6);axs[2].set_title('C sampling, q=0.5')
        for qidx,style in [(0,'b-'),(-1,'r-')]:
            x=p['pred'][n,0,qidx];axs[3].plot(x[:,0],x[:,1],style,label=f'enabled q={qs[qidx]}')
        x=p['baseline'][n,0];axs[3].plot(x[:,0],x[:,1],'g-',label='disabled');axs[3].plot(teacher[:,0],teacher[:,1],'k--',label='teacher')
        axs[3].legend(fontsize=7);axs[3].set_title('D enabled / disabled, shared noise')
        if i in reportrows:
            r=reportrows[i];lines=[f'{k}: {r[k]}' for k in ('teacher_progress_m','generated_progress_m','enabled_teacher_rms_m','disabled_teacher_rms_m','q_effect_rms_m','sampling_rms_m','q_sampling_ratio')]
        else:lines=['smoke: numeric metrics pending']
        axs[4].text(.02,.95,'\n'.join(lines),va='top',family='monospace',fontsize=8,transform=axs[4].transAxes);axs[4].axis('off');axs[4].set_title('E trajectory metrics')
        series=meta['series'];t0=meta['timestamp_ns'];t=np.asarray([(r['timestamp_ns']-t0)/1e9 for r in series]);v=[np.linalg.norm(r['twist'][:2]) for r in series]
        axs[5].plot(t,[r['critic_value'] for r in series],label='Critic');axs[5].plot(t,v,label='speed m/s');axs[5].plot(t,[r['twist'][5] for r in series],label='angular z rad/s');axs[5].axvline((series[-1]['timestamp_ns']-t0)/1e9,color='red',ls='--');axs[5].set_xlim(-2,4);axs[5].legend(fontsize=7);axs[5].set_title('F time relative to anchor [s]')
        for ax in (axs[0],axs[1],axs[2],axs[3]):ax.set_aspect('equal');ax.set_xlabel('forward x [m]');ax.set_ylabel('left y [m]');ax.grid(alpha=.25)
        fig.suptitle(f"held-out index {i} | terrain {z['terrain'][j]} | episode {z['episode'][j].tolist()} | q={z['quality'][j]:.3f} | terminal {meta['terminal']['termination_reason']}")
        fig.tight_layout();fig.savefig(cases/f'{i:05d}.png',dpi=110);pdf.savefig(fig);plt.close(fig)
    pdf.close()
    if smoke:return
    # Seven fixed aggregate plots; all axes and units explicit.
    with open(OUT/'casebook-metrics.csv') as f:rows=list(csv.DictReader(f))
    pr=np.asarray([float(r['teacher_progress_m']) for r in rows]);gp=np.asarray([float(r['generated_progress_m']) for r in rows]);
    def scatter(name,x,y,xlab,ylab):
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(x,y,s=10,alpha=.5);ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'plots'/name,dpi=140);plt.close(fig)
    (OUT/'plots').mkdir(exist_ok=True)
    scatter('q-vs-sampling.png',[float(r['q_effect_rms_m']) for r in rows],[float(r['sampling_rms_m']) for r in rows],'q effect RMS [m]','sampling RMS [m]')
    scatter('teacher-vs-generated-progress.png',pr,gp,'teacher forward progress [m]','generated forward progress [m]')
    scatter('teacher-vs-generated-turning.png',[float(r['teacher_turning_rad']) for r in rows],[float(r['generated_turning_rad']) for r in rows],'teacher turning [rad]','generated turning [rad]')
    pair=json.loads((OUT/'casebook-metrics.json').read_text())['pair_teacher_distances']
    scatter('nearest-low-high-target.png',[r['low_rms_m'] for r in pair],[r['high_rms_m'] for r in pair],'low teacher RMS [m]','high teacher RMS [m]')
    scatter('terminal-proximity.png',[float(r['terminal_seconds']) for r in rows],[float(r['enabled_teacher_rms_m']) for r in rows],'terminal time [s]','teacher RMS [m]')
    scatter('goal-provenance.png',[float(r['goal_distance_m']) for r in rows],pr,'dataset goal distance [m]','teacher endpoint progress [m]')
    scatter('quality-enabled-vs-disabled.png',[float(r['disabled_teacher_rms_m']) for r in rows],[float(r['enabled_teacher_rms_m']) for r in rows],'disabled teacher RMS [m]','enabled teacher RMS [m]')
    links='\n'.join(f'<li><a href="cases/{int(i):05d}.png">held-out {int(i)}</a></li>' for i in p['indices'])
    (OUT/'casebook.html').write_text('<!doctype html><html><meta charset="utf-8"><title>EAT casebook</title><body><h1>Held-out trajectory casebook</h1><p>Static base-frame XY; no physical safety conclusion.</p><ul>'+links+'</ul></body></html>')
    print('DRAW_DONE',len(p['indices']),flush=True)

def report():
    s=json.loads((OUT/'casebook-summary.json').read_text());g=json.loads((OUT/'goal-provenance.json').read_text())
    r=['# Held-out trajectory casebook','',f"Complete held-out windows: {g['heldout_complete']}; selected unique windows: {s['window_count']}.",
       f"Median q effect {s['median_q_effect_rms_m']:.4f} m; sampling variation {s['median_sampling_rms_m']:.4f} m; ratio {s['median_q_sampling_ratio']:.3f}.",
       f"Mean teacher RMS: enabled {s['enabled_mean_teacher_rms_m']:.4f} m; disabled {s['disabled_mean_teacher_rms_m']:.4f} m.",
       f"Goal is the future realized endpoint in anchor base frame, capped at 0.8 m. Goal/teacher progress Pearson r={g['progress_goal_pearson']:.3f}.",
       'Raw command is a velocity command [m/s, rad/s], not a goal [m]; distance difference is NA.',
       f"Category counts A-E: {s['category_counts']}. Category thresholds are frozen in casebook-config.yaml.",
       'Terrain with sparse q-bin support is NA; static XY cannot establish physical safety or tipping.',
       'No new training, full collection, checkpoint, model, or loss change was performed.','',
       '## Questions','',
       'See casebook-metrics.csv, casebook-summary.json, and individual case pages for teacher imitation, q response, sampling variation, alternative trajectories, baseline comparison, and goal provenance.',
       'Next step, if warranted by representative pages, is a separately approved 20–50 case closed-loop simulator replay. Full collection remains stopped.','']
    (OUT/'casebook-report.md').write_text('\n'.join(r))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','cohorts','smoke','infer','metrics','draw','report'],required=True);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    globals()[a.stage if a.stage not in ('smoke','infer','draw') else {'smoke':'infer','infer':'infer','draw':'drawing'}[a.stage]](True) if a.stage=='smoke' else globals()[{'draw':'drawing'}.get(a.stage,a.stage)]()

if __name__=='__main__':main()
