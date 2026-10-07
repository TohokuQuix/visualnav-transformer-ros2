"""Freeze first-seed DEM parity inputs in a new root; old roots are read-only."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

WORKTREE=Path('/tmp/eatnav-vnt-anchor-placement')
OLD=Path('/tmp/eatnav-radial-112-sharded-20261007')
CONT=Path('/tmp/eatnav-radial-termination-20261008')
ROOT=Path(os.environ.get('EATNAV_ANCHOR_PLACEMENT_ROOT','/tmp/eatnav-anchor-placement-20261008'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def validate_ledger(root,name):
    ledger=json.loads((root/name).read_text())
    mismatches=[x['path'] for x in ledger['files'] if
                not (root/x['path']).exists() or sha(root/x['path'])!=x['sha256']]
    if mismatches:raise ValueError('old root SHA ledger mismatch: '+repr(mismatches[:5]))
    return {'path':str(root/name),'sha256':sha(root/name),'count':len(ledger['files'])}


def remote_sha(path):
    p=subprocess.run(['docker','exec','isaac-lab-onix','sha256sum',str(path)],capture_output=True,text=True,check=True)
    return p.stdout.split()[0]


def main():
    if ROOT.exists():raise ValueError('new output root already exists')
    if subprocess.run(['git','status','--porcelain'],cwd=WORKTREE,capture_output=True,text=True,check=True).stdout:
        raise ValueError('source worktree must be clean and committed')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=WORKTREE,capture_output=True,text=True,check=True).stdout.strip()
    old_ledger=validate_ledger(OLD,'partial-output-sha-ledger.json')
    cont_ledger=validate_ledger(CONT,'final-output-sha-ledger.json')
    inventory=json.loads((OLD/'inventory-manifest.json').read_text())
    reuse=json.loads((CONT/'reuse-ledger.json').read_text())['reused_shards']
    old_freeze=json.loads((OLD/'cycle-0/shard-00/input-freeze.json').read_text())
    rows=[];origins=[]
    for si,shard in enumerate(inventory['shards']):
        sr=Path(reuse[si]['old_root']) if si<15 else CONT/'cycle-1'/shard['shard_id']
        reset=json.loads((sr/'run/reset-pre-physics.json').read_text())
        for pose in shard['poses']:
            pid=pose['pose_id'];frames=sorted((sr/'run/dem'/pid).glob('frame-*.json'))
            if not frames:raise ValueError('missing DEM: '+pid)
            frame_path=frames[-1];frame=json.loads(frame_path.read_text())
            trace_path=sr/'run/trace'/f'{pid}.jsonl'
            lines=trace_path.read_text().splitlines();step=frame['physics_step']
            if step<1 or step>len(lines):raise ValueError('frame/trace step mismatch: '+pid)
            root=json.loads(lines[step-1])['state']['root_state_w']
            w,x,y,z=root[3:7]
            yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            requested=reset['requested'][pid]
            theta=pose['theta_rad'];radius=.30
            env_origin=[requested['world_x']-radius*math.cos(theta),
                        requested['world_y']-radius*math.sin(theta),requested['world_z_reference']]
            expected={**frame,'pose_id':pid,'env_id':pose['env_index'],
                      'terrain':pose['terrain'],'terrain_seed':pose['terrain_seed'],
                      'row':pose['row'],'column':pose['column'],
                      'yaw_rad':yaw,'env_origin_xyz':env_origin,
                      'root_state_w':root,
                      'source_frame_path':str(frame_path),'source_frame_sha256':sha(frame_path),
                      'source_trace_path':str(trace_path),'source_trace_sha256':sha(trace_path),
                      'source_reset_sha256':sha(sr/'run/reset-pre-physics.json')}
            rows.append(expected)
    if len(rows)!=112 or len({r['pose_id'] for r in rows})!=112 or len({r['env_id'] for r in rows})!=112:
        raise ValueError('112 unique recorded poses required')
    selected_keys=('checkpoint_env','rl_checkpoint','terrain_generator','eat_env_cfg','rough_cfg',
                   'installed_source_4','radial_module')
    sources={key:old_freeze['source_paths'][key] for key in selected_keys}
    sources.update({'raycaster':'/workspace/isaaclab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py',
                    'raycaster_cfg':'/workspace/isaaclab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py',
                    'manager_based_env':'/workspace/isaaclab/source/isaaclab/isaaclab/envs/manager_based_env.py',
                    'onix_env':'/workspace/isaaclab_onix/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/envs.py',
                    'eat_config':old_freeze['source_paths']['eat_config']})
    source_sha={key:remote_sha(path) for key,path in sources.items()}
    for key in selected_keys:
        if source_sha[key]!=old_freeze['source_sha256'][key]:
            raise ValueError('frozen source changed: '+key)
    if source_sha['eat_config']!=old_freeze['source_sha256']['eat_config']:
        raise ValueError('EAT config changed')
    ROOT.mkdir(parents=True,exist_ok=False)
    (ROOT/'inputs').mkdir()
    (ROOT/'src/visualnav_transformer/eat_nav').mkdir(parents=True)
    for name in ('anchor_placement_parity.py','config.py','dem.py','radial_inventory.py'):
        source=WORKTREE/'src/visualnav_transformer/eat_nav'/name
        target=ROOT/'src/visualnav_transformer/eat_nav'/name
        shutil.copyfile(source,target)
        sources['placement_'+name[:-3]]=str(target)
        source_sha['placement_'+name[:-3]]=sha(target)
    script=ROOT/'inputs/eat_nav_anchor_placement_parity_sim.py'
    shutil.copyfile(WORKTREE/'scripts/eat_nav_anchor_placement_parity_sim.py',script)
    sources['placement_sim']=str(script);source_sha['placement_sim']=sha(script)
    write(ROOT/'inputs/pose-frames.json',{'schema':'eatnav_anchor_placement_parity_inputs_v1',
        'pose_frames':rows,'frozen_source_paths':sources,'frozen_source_sha256':source_sha,
        'old_ledger':old_ledger,'continuation_ledger':cont_ledger,
        'expected_pose_count':112,'batch_seed':20261007,'terrain_difficulty':.5})
    write(ROOT/'config.json',{'schema':'eatnav_anchor_placement_offline_v1',
        'batch_seed':20261007,'terrain_difficulty':.5,'base_radii_m':[.30,.40,.50,.60],
        'interpolation_radii_m':[.35,.45,.55],'direction_count':16,
        'parity_tolerance':{'height_rms_m':1e-5,'height_max_abs_m':1e-4,
                            'mask_iou':1.0,'origin_max_abs_m':1e-5,'yaw_abs_rad':1e-5},
        'interaction_onset_range_m':[.10,.50],
        'corridor_known_min':.70,'centerline_known_required':True,
        'physical_stability':'unmeasured','no_physics_control_steps':True})
    write(ROOT/'preflight.json',{'schema':'eatnav_anchor_placement_preflight_v1',
        'source_commit':commit,'branch':'exp/eatnav-anchor-placement',
        'input_sha256':sha(ROOT/'inputs/pose-frames.json'),'config_sha256':sha(ROOT/'config.json'),
        'old_ledger_sha256':old_ledger['sha256'],'continuation_ledger_sha256':cont_ledger['sha256'],
        'source_sha256':source_sha,'status':'parity_pending'})
    print(ROOT,sha(ROOT/'preflight.json'),commit)


if __name__=='__main__':main()
