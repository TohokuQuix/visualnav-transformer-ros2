"""Exact logged-action prefix reconstruction gate for frozen feasible-set anchors.

This executable only compares an existing collection anchor; it does not run a
counterfactual maneuver or alter the RL environment dynamics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--selection',type=Path,required=True)
parser.add_argument('--reference-config',type=Path,required=True)
parser.add_argument('--rl-checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--state-index',type=int,default=None)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
simulation_app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.math import euler_xyz_from_quat
import onix_isaac_lab.tasks  # noqa
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import make_collection_cfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid,robot_dem


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def prefix_records(manifest, target_step, target_env):
    steps={}
    target=None
    for chunk in manifest['chunks']:
        with gzip.open(args.manifest.parent/chunk['file'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                step=int(row['step'])
                if step>target_step:
                    continue
                steps.setdefault(step,{})[int(row['env_id'])]=row['action']
                if step==target_step and int(row['env_id'])==target_env:
                    target=row
        if target is not None and len(steps.get(target_step,{}))==manifest['metadata']['num_envs']:
            break
    if target is None or len(steps)!=target_step+1:
        raise ValueError('target record or action prefix missing')
    actions=np.asarray([[steps[step][env] for env in range(manifest['metadata']['num_envs'])]
                        for step in range(target_step+1)],dtype=np.float32)
    if actions.shape!=(target_step+1,manifest['metadata']['num_envs'],6):
        raise ValueError('recorded action prefix shape mismatch')
    return actions,target


def transformed_dem(bare,env_id,config):
    robot=bare.scene['robot']
    scanner=bare.scene['height_scanner_eat']
    pos=robot.data.root_pos_w[env_id].detach().cpu().numpy()
    _,_,yaw=euler_xyz_from_quat(robot.data.root_quat_w[env_id:env_id+1])
    hits=scanner.data.ray_hits_w[env_id,:,2].detach().cpu().numpy().reshape(config.dem_height_cells,config.dem_width_cells)
    valid=np.isfinite(hits)
    start=scanner._ray_starts_w[env_id,0,:2].detach().cpu().numpy()
    source=SourceGrid(height=np.where(valid,hits,0).astype(np.float32),valid=valid,
                      origin_xy=(float(start[0]),float(start[1])),resolution_m=config.dem_resolution_m,
                      start_index_xy=(0,0),frame_id=config.reference_frame,stamp_ns=0)
    return robot_dem(source,(float(pos[0]),float(pos[1]),float(yaw.item())),config),float(valid.mean())


def main():
    started=time.monotonic()
    selection=json.loads(args.selection.read_text())
    candidates=selection['train']
    if args.state_index is None:
        index=min(range(len(candidates)),key=lambda j:candidates[j]['step'])
    else:
        index=args.state_index
    state=candidates[index]
    meta=json.loads(args.manifest.read_text())
    spec=meta['metadata']
    if sha(args.rl_checkpoint)!=spec['checkpoint_sha256'] or spec['num_envs']!=100:
        raise ValueError('collection source mismatch')
    env_id=int(state['episode_key'][1])
    target_step=int(state['step'])
    actions,target=prefix_records(meta,target_step,env_id)
    if tuple(state['episode_key'])!=(target['session_id'],target['env_id'],target['episode_id']) or target['pre_reset']:
        raise ValueError('frozen state does not match logged post-step anchor')
    config=load_eat_config(args.reference_config)
    saved=yaml.load((args.rl_checkpoint.parent/'params'/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    toggles={key:saved[key].lower()=='true' for key in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
    if toggles!=spec['observation_toggles']:
        raise ValueError('observation toggles differ from collection')
    cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),config,spec['base_seed'],spec['num_envs'])
    env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
    bare=env.unwrapped
    runner=OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
    weights=torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    runner.alg.policy.load_state_dict(weights)
    env.reset()
    print('ANCHOR_REPLAY_START',json.dumps({'state_index':index,'step':target_step,'env':env_id,'episode':state['episode_key']}),flush=True)
    for step,batch in enumerate(actions):
        with torch.inference_mode():
            env.step(torch.as_tensor(batch,device=bare.device))
        if (step+1)%20==0 or step==target_step:
            print('ANCHOR_REPLAY_STEP',step+1,'of',target_step+1,'elapsed_sec',round(time.monotonic()-started,2),flush=True)
    robot=bare.scene['robot']
    actual_pose=robot.data.root_state_w[env_id,:7].detach().cpu().numpy().astype(float)
    expected_pose=np.asarray(target['pose'],dtype=float)
    position_error=float(np.linalg.norm(actual_pose[:3]-expected_pose[:3]))
    orientation_dot=float(abs(np.dot(actual_pose[3:7],expected_pose[3:7])))
    orientation_angle=float(2*math.acos(min(1.0,orientation_dot)))
    actual_dem,valid_fraction=transformed_dem(bare,env_id,config)
    expected_dem=np.asarray(target['dem']['transformed'],dtype=np.float32)
    if actual_dem.shape!=expected_dem.shape:
        raise ValueError('logged DEM shape mismatch')
    actual_valid=actual_dem[1]>.5
    expected_valid=expected_dem[1]>.5
    both=actual_valid&expected_valid
    dem_rms=float(np.sqrt(np.mean((actual_dem[0][both]-expected_dem[0][both])**2))) if both.any() else float('inf')
    mask_iou=float((both.sum())/max(1,(actual_valid|expected_valid).sum()))
    limits={'position_m':0.02,'orientation_rad':0.03,'dem_rms_m':0.03,'mask_iou_min':0.95}
    passed=position_error<=limits['position_m'] and orientation_angle<=limits['orientation_rad'] and dem_rms<=limits['dem_rms_m'] and mask_iou>=limits['mask_iou_min']
    result={'state_index':index,'frozen_state':state,'target_record':{'step':target['step'],'episode_id':target['episode_id']},
            'source_sha256':{'manifest':sha(args.manifest),'selection':sha(args.selection),'rl_checkpoint':sha(args.rl_checkpoint)},
            'errors':{'position_m':position_error,'orientation_rad':orientation_angle,'dem_rms_m':dem_rms,'mask_iou':mask_iou},
            'limits':limits,'actual_pose':actual_pose.tolist(),'expected_pose':expected_pose.tolist(),
            'valid_fraction':valid_fraction,'elapsed_sec':time.monotonic()-started,
            'measurement_gate':'passed' if passed else 'failed_anchor_reconstruction'}
    atomic(args.output/'anchor-preflight.json',result)
    atomic(args.output/'status.json',{'stage':'anchor_preflight','measurement_gate':result['measurement_gate'],
                                       'state_index':index,'elapsed_sec':result['elapsed_sec']})
    print('ANCHOR_REPLAY_RESULT',json.dumps(result['errors']),'passed',passed,flush=True)
    env.close()
    if not passed:
        raise SystemExit(2)

try:
    main()
finally:
    simulation_app.close()
