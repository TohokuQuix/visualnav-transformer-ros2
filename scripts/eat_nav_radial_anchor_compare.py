"""Compare candidate-free radial fixture against immutable original records."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT=Path(os.environ.get('EATNAV_RADIAL_FIXTURE_ROOT','/tmp/eatnav-radial-anchor-contract-20261008-r1'))
HERE=Path(__file__).resolve().parent
REPORT=HERE.parent/'docs/eat-nav'/ROOT.name
R5=HERE.parent/'docs/eat-nav/local-primitive-p1-r5-20261008-attempt2'
_spec=importlib.util.spec_from_file_location('contract',HERE/'eat_nav_radial_anchor_contract.py')
contract=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(contract)


def read(path):return json.loads(Path(path).read_text())


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def trace(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def numeric(first,second,tolerance,field,step=None):
    a=np.asarray(first,dtype=np.float64);b=np.asarray(second,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'field':field,'step':step,'pass':False,'reason':'shape_or_nonfinite','old_shape':a.shape,'new_shape':b.shape}
    delta=np.abs(a-b);err=float(np.max(delta)) if delta.size else 0.
    return {'field':field,'step':step,'pass':err<=tolerance,'max_abs':err,'rms':float(np.sqrt(np.mean(delta**2))) if delta.size else 0.,
            'tolerance':tolerance}


def exact(first,second,field,step=None):
    return {'field':field,'step':step,'pass':first==second,'old':first if first!=second else None,
            'new':second if first!=second else None}


def quaternion_error(first,second):
    a=np.asarray(first,dtype=float);b=np.asarray(second,dtype=float)
    return float(2*math.acos(min(1.,abs(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))))))


def compare_one(anchor,source,fixture,tol):
    pid=anchor['pose_id'];old=source/'run';new=fixture/'run';checks=[]
    def add(item):
        checks.append(item)
        return item['pass']
    old_record=read(old/'poses'/f'{pid}.json');new_record=read(new/'poses'/f'{pid}.json')
    if not add(exact(old_record['pose_id'],new_record['pose_id'],'pose_id')):return checks
    if not add(exact(old_record['env_index'],new_record['env_index'],'env_index')):return checks
    for field in ('candidate_free','action','physics_step_count','control_step_count'):
        if not add(exact(old_record[field],new_record[field],field)):return checks
    for phase in ('reset_state','prephysics_state'):
        old_state=old_record[phase];new_state=new_record[phase]
        for field in ('joint_names','root_quaternion_order','flipper_fold_mode_latch'):
            if not add(exact(old_state[field],new_state[field],phase+'.'+field)):return checks
        if not add(exact(old_state['requested_pose'],new_state['requested_pose'],phase+'.requested_pose')):return checks
        for field,tolerance in [('joint_position',tol['joint_position_max_abs_rad']),
                                ('joint_velocity',tol['joint_velocity_max_abs_radps'])]:
            if not add(numeric(old_state[field],new_state[field],tolerance,phase+'.'+field)):return checks
        root_old=old_state['root_state_w'];root_new=new_state['root_state_w']
        if not add(numeric(root_old[:3],root_new[:3],tol['root_position_max_abs_m'],phase+'.root_position')):return checks
        qerr=quaternion_error(root_old[3:7],root_new[3:7])
        if not add({'field':phase+'.root_quaternion','pass':qerr<=tol['root_orientation_max_abs_rad'],
                    'error_rad':qerr,'tolerance_rad':tol['root_orientation_max_abs_rad']}):return checks
        if not add(numeric(root_old[7:],root_new[7:],tol['root_velocity_max_abs_mps_or_radps'],phase+'.root_velocity')):return checks
    old_trace=trace(old/'trace'/f'{pid}.jsonl');new_trace=trace(new/'trace'/f'{pid}.jsonl')
    if not add(exact(len(old_trace),len(new_trace),'trace_length')):return checks
    try:contract.require_completion(0,read(fixture/'run/sim-finished.json'),new_trace)
    except ValueError as exc:
        add({'field':'termination_reset_completeness','pass':False,'reason':str(exc)});return checks
    for i,(o,n) in enumerate(zip(old_trace,new_trace,strict=True),start=1):
        payload_old={key:value for key,value in o.items() if key not in ('wall_ns','monotonic_ns')}
        payload_new={key:value for key,value in n.items() if key not in ('wall_ns','monotonic_ns')}
        if not add({'field':'trace.physics_payload','step':i,'pass':payload_old==payload_new,
                    'excluded_timestamps':['wall_ns','monotonic_ns']}):return checks
        for field in ('physics_step','episode_id','reset_counter','post_reset_state','termination_phase'):
            if not add(exact(o[field],n[field],'trace.'+field,i)):return checks
        for key in ('bounds_valid','shape_valid','buffer_capacity','buffer_full_or_truncation_suspected','finite'):
            if not add(exact(o['contact'][key],n['contact'][key],'contact.'+key,i)):return checks
        if not n['contact']['bounds_valid'] or n['contact']['buffer_full_or_truncation_suspected'] or not n['contact']['finite']:
            add({'field':'contact_validity','step':i,'pass':False});return checks
        for field,tolerance in [('root_state_w',tol['root_position_max_abs_m']),
                                ('joint_position',tol['joint_position_max_abs_rad']),
                                ('joint_velocity',tol['joint_velocity_max_abs_radps'])]:
            if field=='root_state_w':
                old_value=o['state'][field][:3];new_value=n['state'][field][:3]
            else:
                old_value=o['state'][field];new_value=n['state'][field]
            if not add(numeric(old_value,new_value,tolerance,'trace.state.'+field,i)):return checks
    old_state=old_trace[-1]['state'];new_state=new_trace[-1]['state']
    for field in ('joint_names','flipper_fold_mode_latch'):
        if not add(exact(old_state[field],new_state[field],'settled.'+field,200)):return checks
    for field,tolerance in [('joint_position',tol['joint_position_max_abs_rad']),
                            ('joint_velocity',tol['joint_velocity_max_abs_radps'])]:
        if not add(numeric(old_state[field],new_state[field],tolerance,'settled.'+field,200)):return checks
    root_old=old_state['root_state_w'];root_new=new_state['root_state_w']
    if not add(numeric(root_old[:3],root_new[:3],tol['root_position_max_abs_m'],'settled.root_position',200)):return checks
    qerr=quaternion_error(root_old[3:7],root_new[3:7])
    if not add({'field':'settled.root_quaternion','step':200,'pass':qerr<=tol['root_orientation_max_abs_rad'],
                'error_rad':qerr,'tolerance_rad':tol['root_orientation_max_abs_rad']}):return checks
    if not add(numeric(root_old[7:],root_new[7:],tol['root_velocity_max_abs_mps_or_radps'],'settled.root_velocity',200)):return checks
    old_actor=read(old/'actor'/pid/'settled_step_200.json');new_actor=read(new/'actor'/pid/'settled_step_200.json')
    for field in ('config_identity','field_order','shape','phase','finite'):
        if not add(exact(old_actor[field],new_actor[field],'actor.'+field,200)):return checks
    if not add(numeric(old_actor['values'],new_actor['values'],tol['actor_max_abs'],'actor.values',200)):return checks
    for i in range(4):
        a=read(old/'dem'/pid/f'frame-{i:02d}.json');b=read(new/'dem'/pid/f'frame-{i:02d}.json')
        for field in ('frame_index','physics_step','control_step','frame_id','axis','resolution_m','shape','finite','logical_stamp_ns'):
            if not add(exact(a[field],b[field],f'dem[{i}].'+field)):return checks
        capture_step=50*(i+1)
        yaw_old=old_trace[capture_step-1]['state']['euler_yaw_signed_rad']
        yaw_new=new_trace[capture_step-1]['state']['euler_yaw_signed_rad']
        yaw_error=abs(math.remainder(yaw_old-yaw_new,2*math.pi))
        if not add({'field':f'dem[{i}].capture_root_yaw','step':capture_step,
                    'pass':yaw_error<=tol['dem_yaw_abs_rad'],
                    'error_rad':yaw_error,'tolerance_rad':tol['dem_yaw_abs_rad']}):return checks
        if not add(numeric(a['source_origin_xy'],b['source_origin_xy'],tol['dem_origin_max_abs_m'],f'dem[{i}].origin')):return checks
        mask_a=np.asarray(a['source_mask'],dtype=bool);mask_b=np.asarray(b['source_mask'],dtype=bool)
        if not add(exact(mask_a.tolist(),mask_b.tolist(),f'dem[{i}].source_mask')):return checks
        trans_a=np.asarray(a['transformed'],dtype=float);trans_b=np.asarray(b['transformed'],dtype=float)
        if not add(exact((trans_a[1]>.5).tolist(),(trans_b[1]>.5).tolist(),f'dem[{i}].transformed_mask')):return checks
        both=mask_a&mask_b
        for field,x,y in [('source_height',np.asarray(a['source_height']),np.asarray(b['source_height'])),
                          ('transformed_height',trans_a[0],trans_b[0])]:
            d=np.asarray(x,dtype=float)[both]-np.asarray(y,dtype=float)[both]
            rms=float(np.sqrt(np.mean(d*d))) if d.size else float('inf')
            maximum=float(np.max(np.abs(d))) if d.size else float('inf')
            if not add({'field':f'dem[{i}].'+field,'pass':rms<=tol['dem_height_rms_m'] and maximum<=tol['dem_height_max_abs_m'],
                        'rms_m':rms,'max_abs_m':maximum,'rms_tolerance_m':tol['dem_height_rms_m'],
                        'max_tolerance_m':tol['dem_height_max_abs_m'],'raw_sha_old':a['raw_sha256'],
                        'raw_sha_new':b['raw_sha256']}):return checks
    return checks


def main():
    selected=read(R5/'selected-four-anchors-r3.json')['selected']
    manifest=read(ROOT/'fixture-manifest.json');tol=read(ROOT/'reconstruction-tolerance-freeze.json')
    mapping={entry['source_shard']:Path(entry['fixture_subroot']) for entry in manifest['shards']}
    results=[]
    for anchor in selected:
        source=Path(anchor['source_shard']);fixture=mapping[anchor['source_shard']]
        if read(fixture/'status.json')['stage']!='protocol_valid':
            raise RuntimeError('fixture shard not sealed')
        checks=compare_one(anchor,source,fixture,tol)
        result={'anchor_id':anchor['anchor_id'],'pose_id':anchor['pose_id'],'source_shard':str(source),
                'fixture_shard':str(fixture),'checks':checks,'pass':bool(checks and all(x['pass'] for x in checks)),
                'first_failing_signal':next((x for x in checks if not x['pass']),None)}
        results.append(result)
        write(ROOT/'fixture-per-anchor-results.json',results)
        write(ROOT/'anchor-parity-comparison.json',results)
        compact=[{'anchor_id':r['anchor_id'],'pose_id':r['pose_id'],'source_shard':r['source_shard'],
                  'fixture_shard':r['fixture_shard'],'pass':r['pass'],'check_count':len(r['checks']),
                  'first_failing_signal':r['first_failing_signal'],
                  'max_root_position_error_m':max((x['max_abs'] for x in r['checks'] if x['field']=='settled.root_position'),default=None),
                  'max_actor_error':max((x['max_abs'] for x in r['checks'] if x['field']=='actor.values'),default=None),
                  'max_dem_height_error_m':max((x['max_abs_m'] for x in r['checks'] if x['field'].startswith('dem[') and 'max_abs_m' in x),default=None),
                  'all_dem_raw_sha_equal':all(x.get('raw_sha_old')==x.get('raw_sha_new') for x in r['checks'] if 'raw_sha_old' in x)}
                 for r in results]
        write(REPORT/'fixture-per-anchor-results.json',compact)
        write(REPORT/'anchor-parity-comparison.json',compact)
        if not result['pass']:break
    decision='radial_anchor_execution_contract_ready_for_p1' if len(results)==4 and all(x['pass'] for x in results) else 'radial_anchor_reconstruction_parity_fail'
    item={'decision':decision,'anchors_compared':len(results),'anchors_passed':sum(x['pass'] for x in results),
          'first_failing_signal':next((x['first_failing_signal'] for x in results if not x['pass']),None),
          'p1_rollout_started':False}
    write(ROOT/'final-decision.json',item);write(REPORT/'final-decision.json',item)
    print(json.dumps(item,indent=2))


if __name__=='__main__':main()
