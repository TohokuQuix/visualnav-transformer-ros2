"""Prepare frozen E1 runtime inputs without changing dynamics or follower behavior."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import yaml

from eat_nav_expanded_feasibility_preflight import sha,write_once
from eat_nav_expanded_feasibility_resume import RESUME,STOPPED,audit_frozen_inputs

ORIGINAL_SIM=Path(__file__).with_name('eat_nav_e1_sim.py')
EXPECTED_SIM_SHA='9e2a40f0b283d4581219302c3f2f0782b00ca825e07653bbe3924dd5c47cec07'


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('frozen simulator source contract differs: '+old[:80])
    return source.replace(old,new,1)


def patch_e1_sim_source(source):
    source=replace_once(source,"config_spec['scope']!='e1_live_candidate'",
        "config_spec['scope']!='anchor_only_then_live_candidate'")
    source=replace_once(source,"state['state_id'].startswith('g0e-train-')",
        "state['state_id'].startswith('expanded-train-')")
    old="""    fixed=np.asarray(json.loads(args.fixed_map_path.read_text()),dtype=np.float64)
    if fixed.shape!=(9,2) or hashlib.sha256(np.asarray(fixed[1:],dtype=np.float32).tobytes()).hexdigest()!=args.expected_map_path_sha256:
        raise ValueError('frozen map Path SHA differs before Path publication')"""
    new="""    if args.stage=='anchor':
        fixed=np.asarray(computed_path,dtype=np.float64)
    else:
        fixed=np.asarray(json.loads(args.fixed_map_path.read_text()),dtype=np.float64)
        if fixed.shape!=(9,2) or hashlib.sha256(np.asarray(fixed[1:],dtype=np.float32).tobytes()).hexdigest()!=args.expected_map_path_sha256:
            raise ValueError('frozen map Path SHA differs before Path publication')"""
    return replace_once(source,old,new)


def map_path_exact(base,pose):
    points=np.asarray(base,dtype=np.float64);p=np.asarray(pose,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or p.shape!=(3,) or not np.isfinite(points).all() or not np.isfinite(p).all():
        raise ValueError('finite base Path and anchor XY/yaw required')
    x,y,yaw=map(float,p);c,s=math.cos(yaw),math.sin(yaw)
    return [[float(x+c*px-s*py),float(y+s*px+c*py)] for px,py in points]


def prepare_runtime_inputs(root=RESUME):
    root=Path(root);audit_frozen_inputs()
    if sha(ORIGINAL_SIM)!=EXPECTED_SIM_SHA:raise ValueError('original E1 simulator SHA differs')
    if (root/'runtime-source-freeze.json').exists():raise FileExistsError('runtime already frozen')
    simulator=patch_e1_sim_source(ORIGINAL_SIM.read_text())
    (root/'eat_nav_expanded_sim.py').write_text(simulator)
    selected=root/'inputs/selected-states.json'
    with selected.open('wb') as stream:stream.write((root/'selected-states.json').read_bytes())
    arrays=root/'inputs/g0e-selected-states.npz'
    shutil.copyfile(root/'selected-states.npz',arrays)
    if sha(selected)!=sha(root/'selected-states.json') or sha(arrays)!=sha(root/'selected-states.npz'):
        raise ValueError('runtime selected-state copy SHA differs')
    old=Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004')
    gates=yaml.safe_load((old/'e1-gates.yaml').read_text())
    safety=yaml.safe_load((old/'e1-safety-stops.yaml').read_text())
    if gates['duplicate']['xy_rms_max_m']!=.0105 or safety['infrastructure_stop']['gpu_device_total_memory_over_mib']!=6656:
        raise ValueError('original E1 gate/resource source differs')
    safety['infrastructure_stop']['schedule_wall_clock_over_sec']=14400
    safety['infrastructure_stop']['storage_over_bytes']=1_000_000_000
    safety['scope']='expanded_pilot_resource_caps_only'
    (root/'e1-gates.yaml').write_bytes((old/'e1-gates.yaml').read_bytes())
    (root/'e1-safety-stops.yaml').write_text(yaml.safe_dump(safety,sort_keys=True))
    rows=json.loads((root/'run-manifest.json').read_text())['rows']
    for row in rows:
        folder=root/'inputs'/f"state-{row['state_index']:03d}"
        source=folder/f"{row['path_id']}-base-path.json"
        target=folder/f"{row['process_id']}-base-path.json"
        shutil.copyfile(source,target)
        if sha(source)!=sha(target):raise ValueError('per-run robot-local Path copy SHA differs')
    write_once(root/'runtime-source-freeze.json',{
        'original_simulator_sha256':sha(ORIGINAL_SIM),
        'adapted_simulator_sha256':sha(root/'eat_nav_expanded_sim.py'),
        'adaptation_scope':['expanded StateGroup ID validation','expanded config scope validation',
            'anchor-only creates direct map Path after numeric anchor gate'],
        'dynamics_reset_actor_critic_follower_changed':False,
        'selected_state_sha256':sha(selected),'selected_dem_sha256':sha(arrays),
        'frozen_e1_gates_sha256':sha(root/'e1-gates.yaml'),
        'expanded_safety_caps_sha256':sha(root/'e1-safety-stops.yaml'),
        'per_run_local_path_sha256':{row['process_id']:sha(root/'inputs'/f"state-{row['state_index']:03d}"/
            f"{row['process_id']}-base-path.json") for row in rows},
        'map_paths_status':'pending_new_anchor_only',
        'source_frozen_before_any_new_process':True})
    print('RUNTIME_INPUTS_READY',sha(root/'eat_nav_expanded_sim.py'),flush=True)


if __name__=='__main__':prepare_runtime_inputs()
