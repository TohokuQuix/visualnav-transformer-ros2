"""Causal current-frame conversion, exact endpoint labels, ancestry audits."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json
import numpy as np
from .single_pose2s_contract import INPUT_SCHEMA,PROFILE_SHA,validate_input
from .single_pose2s_collection import window_endpoints,causal_dem_context
from .single_pose2s_geometry import endpoint_target
from .bidirectional_bank import bank
from .single_pose2s_fastlio import odom_velocity_at


def trial_samples(raw,metadata,odometry=None):
    parent=metadata['parent'];stamps=np.asarray(raw['stamps_ns'],np.int64)
    out=[]
    for j,(start,end,observed) in enumerate(window_endpoints(stamps,raw['alive'])):
        t0=int(stamps[start]);cid=parent['candidates'][j]
        start_motion='unknown'
        if start>=20:
            travel=float(np.linalg.norm(np.asarray(raw['root'][start,:2])-np.asarray(raw['root'][start-20,:2])))
            requested=float(np.mean(np.abs(np.asarray(raw['command'])[start-20:start,0]))) if 'command' in raw else 0.
            spin=float(abs(np.asarray(raw['state'][start,5])))
            start_motion='stall' if requested>=.1 and travel<=.02 else 'moving' if travel>.02 else 'turning' if spin>=.1 else 'stopped'
        row=dict(start_motion=start_motion,parent_id=parent['parent_id'],role=parent['role'],seed=parent['seed'],field=parent['field'],level=parent['level'],stage=j,candidate_id=cid,terrain_hash=metadata['terrain']['geometry_sha256'],t0_ns=t0,t2_ns=int(stamps[end]),root0=np.asarray(raw['root'][start]),pose_mask=False,input_valid=False,input=None,position=np.zeros(3),rotation=np.array([1,0,0,0,1,0],float),reason=None)
        frames=np.flatnonzero(np.asarray(raw['dem_stamp_ns'])<=t0)
        if odometry is None:
            row['reason']='missing_actual_FAST_LIO';out.append(row);continue
        if not bool(raw['alive'][start]):row['reason']='start_after_terminal'
        elif not len(frames):row['reason']='no_available_DEM'
        else:
            k=int(frames[-1]);stamp=int(raw['dem_stamp_ns'][k])
            inputs=dict(schema=INPUT_SCHEMA,dem=np.asarray(raw['dem'][k]),state=np.asarray(raw['state'][start]),state_mask=np.asarray(raw['state_mask'][start]),state_age=np.asarray(raw['state_age'][start]),dem_context=causal_dem_context(raw['dem_root'][k],raw['root'][start],stamp,t0),t0_ns=t0,dem_stamp_ns=stamp,dem_received_ns=stamp,paths=np.asarray([bank()[cid].waypoints]),directions=np.array([[bank()[cid].direction]]))
            try:
                v,w,age,odom_ids=odom_velocity_at(t0,**odometry)
                inputs['state']=inputs['state'].copy();inputs['state_mask']=inputs['state_mask'].copy();inputs['state_age']=inputs['state_age'].copy()
                inputs['state'][18:21]=v;inputs['state'][21:24]=w
                inputs['state_mask'][18:24]=1;inputs['state_age'][18:24]=age
                poses=np.asarray(odometry['poses']);odom_stamps=np.asarray(odometry['stamps_ns']);received=np.asarray(odometry['received_ns'])
                frame_ids=np.flatnonzero((odom_stamps<=stamp)&(received<=stamp))
                if not len(frame_ids):raise ValueError('no causal FAST-LIO pose for DEM frame')
                frame_id=int(frame_ids[np.argmax(odom_stamps[frame_ids])])
                inputs['dem_context']=causal_dem_context(poses[frame_id],poses[odom_ids[-1]],stamp,t0)
                row['odom_source']='actual_FAST_LIO_pose_difference';row['odom_pose_indices']=odom_ids;row['odom_dem_pose_index']=frame_id
            except ValueError as exc:
                row['reason']='FAST_LIO:'+str(exc);out.append(row);continue
            row['dem_source_id']=parent['parent_id']+':'+str(stamp)
            row['dem_stamp_ns']=stamp
            row['input']=inputs
            try:validate_input(inputs);row['input_valid']=True
            except ValueError as exc:row['reason']='input:'+str(exc)
            if not observed:row['reason']='ended_before_2s'
            elif row['input_valid']:
                target=endpoint_target(raw['root'][start],raw['root'][end]);row.update(target,pose_mask=True)
        out.append(row)
    return out


def audit_split(records,allow_known_terrain=False):
    seen={};terrain=defaultdict(set);seeds=defaultdict(set);extra={k:defaultdict(set) for k in ['source_raw_sha256','initial_state_sha256']}
    for r in records:
        parent=r['parent_id'];identity=(r['role'],r['seed'],r['terrain_hash'])
        if parent in seen and seen[parent]!=identity:raise ValueError('parent ancestry mismatch')
        for key,groups in extra.items():
            if key in r:groups[r[key]].add(r['role'])
        seen[parent]=identity;terrain[r['terrain_hash']].add(r['role']);seeds[r['seed']].add(r['role'])
    for key,groups in extra.items():
        if any(len(v)>1 for v in groups.values()):raise ValueError(key+' overlap across roles')
    overlaps={k:sorted(v) for k,v in terrain.items() if len(v)>1}
    if overlaps and not allow_known_terrain:raise ValueError('terrain hash overlap across roles: '+str(overlaps))
    if any(len(v)>1 for v in seeds.values()):raise ValueError('seed overlap across roles')
    return dict(parent_count=len(seen),terrain_hash_count=len(terrain),cross_role_terrain_overlap=len(overlaps),terrain_overlap_groups=overlaps,cross_role_seed_overlap=0,cross_role_raw_overlap=0,cross_role_initial_state_overlap=0,terrain_generalization_claim='known_terrain_independent_runs' if allow_known_terrain else 'mesh_disjoint')


def build_dataset(raw_root,manifest,output):
    raw_root=Path(raw_root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    expected={p['parent_id']:p for p in manifest['parents']};rows=[];sources=[];found=set();runtime_ids=set()
    for meta_path in sorted(raw_root.rglob('metadata.json')):
        metadata=json.loads(meta_path.read_text());parent=metadata['parent'];pid=parent['parent_id']
        if pid not in expected:continue
        if pid in found:raise ValueError('duplicate raw parent '+pid)
        if parent!=expected[pid] or metadata['profile_sha256']!=PROFILE_SHA:raise ValueError('frozen manifest/profile mismatch')
        runtime_file=meta_path.parent/metadata['runtime_imports_file']
        if hashlib.sha256(runtime_file.read_bytes()).hexdigest()!=metadata['runtime_imports_sha256']:raise ValueError('runtime import manifest hash mismatch')
        identity=json.loads(runtime_file.read_text())['canonical_sha256']
        if identity!=metadata.get('runtime_identity_sha256'):raise ValueError('runtime identity mismatch')
        runtime_ids.add(identity)
        approved=manifest.get('approved_runtime_identities')
        if approved is not None and identity not in approved:raise ValueError('unapproved runtime identity')
        if len(runtime_ids)>1 and approved is None:raise ValueError('unapproved mixed runtime identities')
        path=meta_path.with_name('raw.npz');digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=metadata['raw_sha256']:raise ValueError('raw hash mismatch')
        lio_dir=meta_path.parent/'fastlio'
        completion=lio_dir/'complete.json';odom_path=lio_dir/'odometry.npz'
        if not completion.exists() or not odom_path.exists():raise ValueError('missing actual FAST-LIO sidecar '+pid)
        lio=json.loads(completion.read_text())
        from .single_pose2s_contract import PROFILE
        if lio.get('source')!='actual_FAST_LIO_process' or lio.get('binary_sha256')!=PROFILE['fastlio_binary_sha256'] or lio.get('config_sha256')!=PROFILE['fastlio_config_sha256']:raise ValueError('FAST-LIO provenance mismatch '+pid)
        if lio.get('sensor_sha256')!=metadata.get('lio_sensor_sha256'):raise ValueError('FAST-LIO sensor provenance mismatch '+pid)
        if hashlib.sha256(odom_path.read_bytes()).hexdigest()!=lio.get('odometry_npz_sha256'):raise ValueError('FAST-LIO output hash mismatch '+pid)
        with np.load(path,allow_pickle=False) as a,np.load(odom_path,allow_pickle=False) as o:
            converted=trial_samples(a,metadata,odometry={k:o[k] for k in ['stamps_ns','received_ns','poses']})
            initial=np.asarray(a['root'][0],np.float64).copy();initial[:3]-=np.asarray(metadata['env_origin_w'])
            initial_hash=hashlib.sha256(np.concatenate([initial,np.asarray(a['joint_position'][0]),np.asarray(a['joint_velocity'][0])]).astype('<f8').tobytes()).hexdigest()
        for r in converted:r['initial_state_sha256']=initial_hash
        found.add(pid);sources.append(dict(parent_id=pid,metadata_path=str(meta_path.resolve()),raw_path=str(path.resolve()),raw_sha256=digest,odometry_npz=str(odom_path.resolve()),odometry_sha256=lio['odometry_npz_sha256']))
        for r in converted:r['source_raw_sha256']=digest
        rows.extend(converted)
    if not rows:raise ValueError('no matching raw parents')
    leakage=audit_split(rows,allow_known_terrain=manifest.get('evaluation_terrain_policy')=='known_terrain_independent_runs');N=len(rows)
    arrays={k:np.zeros((N,*tail),np.float32) for k,tail in dict(dem=(3,40,40),state=(27,),state_mask=(27,),state_age=(27,),dem_context=(6,),paths=(1,8,2),directions=(1,1),position=(3,),rotation=(6,),root0=(13,)).items()}
    arrays.update(pose_mask=np.zeros(N,bool),input_valid=np.zeros(N,bool),candidate_id=np.zeros(N,np.int64),t0_ns=np.zeros(N,np.int64),dem_stamp_ns=np.zeros(N,np.int64),dem_received_ns=np.zeros(N,np.int64))
    records=[]
    for i,row in enumerate(rows):
        if row['input'] is not None:
            for k in ['dem','state','state_mask','state_age','dem_context','paths','directions','t0_ns','dem_stamp_ns','dem_received_ns']:arrays[k][i]=row['input'][k]
        for k in ['position','rotation','root0','pose_mask','input_valid','candidate_id']:arrays[k][i]=row[k]
        records.append(dict(source_row_index=i,**{k:v for k,v in row.items() if k not in ['input','position','rotation','root0']}))
    if not all(np.isfinite(v).all() for k,v in arrays.items() if k not in ['root0']):raise ValueError('nonfinite dataset')
    output.mkdir(parents=True);np.savez_compressed(output/'samples.npz',**arrays)
    splits={role:[i for i,r in enumerate(records) if r['role']==role] for role in ['train','validation','final_test']}
    for role,ids in splits.items():
        np.savez_compressed(output/(role+'-samples.npz'),**{k:v[ids] for k,v in arrays.items()})
        (output/(role+'-records.json')).write_text(json.dumps([records[i] for i in ids],indent=2)+'\n')
    coverage=Counter((r['role'],r['field'],r['level'],r['candidate_id']) for r in records if r['pose_mask'])
    audit=dict(schema='eatnav_single_dem_pose2s_dataset_audit_v1',planned_parents=len(expected),observed_parents=len(found),missing_parent_ids=sorted(set(expected)-found),planned_windows=len(expected)*5,observed_windows=N,valid_pose_windows=int(arrays['pose_mask'].sum()),invalid_reasons=dict(Counter(r['reason'] for r in records if r['reason'])),leakage=leakage,role_counts={role:dict(planned_parents=sum(p['role']==role for p in manifest['parents']),observed_parents=len({r['parent_id'] for r in records if r['role']==role}),observed_windows=len(ids),valid_windows=int(arrays['pose_mask'][ids].sum())) for role,ids in splits.items()},profile_sha256=PROFILE_SHA,complete_parent_coverage=len(found)==len(expected),runtime_identity_sha256=sorted(runtime_ids),coverage=[dict(role=k[0],field=k[1],level=k[2],candidate_id=k[3],valid_windows=v) for k,v in sorted(coverage.items())])
    for name,data in [('records.json',records),('split.json',splits),('source-index.json',sources),('audit.json',audit),('manifest.json',manifest)]:
        (output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    return audit
