"""Read-only, post-freeze audit of the original eight R3 settling traces."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

OLD = Path('/tmp/eatnav-r3-settling-pilot-20261008')
ROOT = Path('/tmp/eatnav-r3-transition-reserve-20261008')
sys.path.insert(0, '/tmp/eatnav-vnt-r3-transition-reserve/src')
from visualnav_transformer.eat_nav.radial_interaction import (classify_dem,
    GRADE_SLOPE, DIFFERENTIAL_M, STEP_CHANGE_M, HEIGHT_CHANGE_M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'final':float(a[-1]), 'mean':float(a.mean()), 'max':float(a.max()),
            'rms':float(np.sqrt(np.mean(a*a)))}


def analyze_one(shard, row, virtual):
    pose=row['pose_id']; base=OLD/shard/'run'
    trace_path=base/'trace'/f'{pose}.jsonl'
    trace=[json.loads(s) for s in trace_path.read_text().splitlines()]
    if len(trace)!=200 or [x['physics_step'] for x in trace]!=list(range(1,201)):
        raise ValueError('original trace incomplete: '+pose)
    f40=trace[160:]
    fields={'angular_speed_radps':[], 'linear_speed_mps':[], 'h1_abs_rad':[],
            'roll_abs_rad':[], 'pitch_abs_rad':[], 'tilt_rad':[], 'xy_drift_m':[]}
    for t in trace:
        s=t['state']
        for k in ('angular_speed_radps','linear_speed_mps','tilt_rad','xy_drift_m'):
            fields[k].append(float(s[k]))
        for k,key in [('h1_abs_rad','h1_signed_rad'),('roll_abs_rad','roll_rad'),
                      ('pitch_abs_rad','pitch_rad')]:
            fields[k].append(abs(float(s[key])))
    angular=np.asarray(fields['angular_speed_radps'])
    peak=int(np.argmax(angular)); below=angular[160:]<=.2
    runs=[];run=0
    for flag in below:
        run=run+1 if flag else 0;runs.append(run)
    contacts=[x['contact'] for x in trace]
    penetration=np.asarray([x['max_decoded_penetration_m'] for x in contacts])
    force=np.asarray([x['force_sum_n'] for x in contacts]);count=np.asarray([x['contact_count'] for x in contacts])
    separation=np.asarray([x['min_separation_m'] for x in contacts])
    force_peak=int(np.argmax(force));penetration_peak=int(np.argmax(penetration))
    frames=[]
    for path in sorted((base/'dem'/pose).glob('frame-*.json')):
        f=json.loads(path.read_text())
        a=np.asarray(f['transformed'],dtype=np.float32)
        if a.shape!=(2,20,20) or not np.isfinite(a).all() or sha(path)=='' or hashlib.sha256(a.tobytes()).hexdigest()!=f['transformed_sha256']:
            raise ValueError('original DEM invalid: '+str(path))
        interaction=classify_dem(a[0],a[1]>.5,.1)
        virtual_dem=np.asarray(virtual['dem'],dtype=np.float32)
        valid=(a[1]>.5)&(virtual_dem[1]>.5)
        rms=float(np.sqrt(np.mean((a[0][valid]-virtual_dem[0][valid])**2))) if valid.any() else None
        state_at_frame=trace[f['physics_step']-1]['state']
        req=row['requested_xy_yaw']
        frames.append({'frame_index':f['frame_index'],'step':f['physics_step'],
            'class':interaction['class'],'onset_m':interaction['interaction_start_m'],
            'slope':interaction['max_forward_slope'],'roughness_m':interaction['forward_roughness_m'],
            'step_m':interaction['max_step_m'],'lateral_m':interaction['left_center_right_difference_m'],
            'known_fraction':interaction['corridor_valid_fraction'],
            'unknown_fraction':interaction['unknown_fraction'],
            'height_rms_vs_virtual_m':rms,'height_center_delta_vs_virtual_m':float(a[0,10,10]-virtual_dem[0,10,10]),
            'raw_sha256':f['raw_sha256'],'transformed_sha256':f['transformed_sha256'],
            'root_xy_at_frame':state_at_frame['root_state_w'][:2],
            'root_xy_delta_from_requested_m':[float(state_at_frame['root_state_w'][i]-req[i]) for i in (0,1)],
            'root_z_at_frame_m':state_at_frame['root_state_w'][2],
            'root_z_delta_from_virtual_requested_m':float(state_at_frame['root_state_w'][2]-virtual['requested_root_z_m']),
            'root_yaw_error_from_requested_rad':state_at_frame['euler_yaw_signed_rad'],
            'threshold_margin':{'graded_slope':None if interaction['max_forward_slope'] is None else interaction['max_forward_slope']-GRADE_SLOPE,
                'transition_step':None if interaction['max_step_m'] is None else interaction['max_step_m']-STEP_CHANGE_M,
                'differential':None if interaction['left_center_right_difference_m'] is None else interaction['left_center_right_difference_m']-DIFFERENTIAL_M,
                'height_change':None if interaction['forward_height_change_m'] is None else abs(interaction['forward_height_change_m'])-HEIGHT_CHANGE_M}})
    if len(frames)!=4:raise ValueError('original DEM history incomplete: '+pose)
    first_switch=next((f['frame_index'] for f in frames if f['class']!=virtual['classification']),None)
    prior=json.loads((OLD/'analysis/per-pose-results.json').read_text())
    if isinstance(prior,dict): prior=prior.get('poses',prior.get('rows',[]))
    old=next(x for x in prior if x['pose_id']==pose)
    if old['rank']!=row['rank']:raise ValueError('original result rank changed')
    categories=sorted({name for c in contacts for name in c['body_categories']})
    category_timeline={name:[c['body_categories'].get(name,{}).get('contact_count',0) for c in contacts] for name in categories}
    return {'pose_id':pose,'rank':row['rank'],'source_family':row['source_family'],
        'virtual_class':virtual['classification'],'original_classification':old['classification'],
        'original_ineligible_reasons':old.get('ineligible_reasons',[]),
        'original_primary_settled_class':old.get('settled_class'),
        'original_result_sha256':sha(OLD/'analysis/per-pose-results.json'),
        'trace_sha256':sha(trace_path),
        'kinematics_final40':{k:stats(v[160:]) for k,v in fields.items()},
        'angular_speed':{'peak_step':peak+1,'peak_radps':float(angular[peak]),
            'first20_final40_mean':float(angular[160:180].mean()),
            'last20_final40_mean':float(angular[180:200].mean()),
            'final40_linear_trend_per_step':float(np.polyfit(np.arange(40),angular[160:],1)[0]),
            'peak_to_final_fraction':float(angular[-1]/angular[peak]) if angular[peak]>0 else None,
            'final40_exceedance_steps':int((~below).sum()),
            'max_consecutive_within_limit_steps':max(runs),
            'final40_limit_crossings':int(np.sum(below[1:]!=below[:-1])),
            'frozen_max_gate_margin_radps':float(angular[160:].max()-.2)},
        'contact':{'force_peak_step':force_peak+1,'force_peak_n':float(force[force_peak]),
            'force_peak_delta_from_angular_step':force_peak-peak,
            'count_at_angular_peak':int(count[peak]),'force_at_angular_peak_n':float(force[peak]),
            'separation_at_angular_peak_m':float(separation[peak]),
            'max_penetration_step':penetration_peak+1,'max_penetration_m':float(penetration[penetration_peak]),
            'occupancy_max':max(x['buffer_occupancy'] for x in contacts),
            'invalid_step_count':sum(not x['bounds_valid'] or not x['finite'] or x['buffer_full_or_truncation_suspected'] for x in contacts),
            'body_categories_at_angular_peak':contacts[peak]['body_categories'],
            'body_categories_at_force_peak':contacts[force_peak]['body_categories'],
            'body_category_contact_count_sum':{name:sum(counts) for name,counts in category_timeline.items()},
            'body_category_contact_count_initial40':{name:sum(counts[:40]) for name,counts in category_timeline.items()},
            'body_category_contact_count_final40':{name:sum(counts[160:]) for name,counts in category_timeline.items()},
            'flipper_category_observed': 'flipper' in categories},
        'virtual_metrics':{'slope':virtual['interaction']['max_forward_slope'],
            'onset_m':virtual['interaction_onset_m'],
            'known_fraction':virtual['interaction']['corridor_valid_fraction']},
        'dem_frames':frames,'first_class_switch_frame':first_switch,
        'secondary_dem_inspection_not_primary_eligibility':row['rank'] in (1,2)}


def main():
    freeze=json.loads((ROOT/'freeze-sha.json').read_text())
    if sha(ROOT/'reserve-manifest.json')!=freeze['reserve_manifest_sha256']:
        raise RuntimeError('reserve manifest not frozen before cause audit')
    primary=json.loads((OLD/'primary-manifest.json').read_text())['poses']
    virtual={}
    for radius in (.30,.40,.50,.60):
        p=Path(f'/tmp/eatnav-anchor-placement-20261008/scan-v3/radius-{radius:.2f}.json')
        virtual.update((x['pose_id'],x) for x in json.loads(p.read_text())['candidates'])
    out=[]
    for row in primary:
        shard='shard-A' if row['rank']%2 else 'shard-B'
        out.append(analyze_one(shard,row,virtual[row['pose_id']]))
    path=ROOT/'analysis';path.mkdir(exist_ok=True)
    output={'schema':'eatnav_r3_transition_eight_pose_cause_audit_v1',
        'reserve_manifest_sha256':freeze['reserve_manifest_sha256'],
        'old_result_unchanged_sha256':sha(OLD/'analysis/per-pose-results.json'),
        'classifier_source_sha256':sha('/tmp/eatnav-vnt-r3-transition-reserve/src/visualnav_transformer/eat_nav/radial_interaction.py'),
        'poses':out}
    (path/'eight-pose-cause-audit.json').write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps([{'rank':x['rank'],'angular_final40_max':x['kinematics_final40']['angular_speed_radps']['max'],
        'class_frames':[f['class'] for f in x['dem_frames']]} for x in out],indent=2))

if __name__=='__main__':main()
