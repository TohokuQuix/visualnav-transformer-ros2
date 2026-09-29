"""Frozen 120-step fresh-process replay for four approved long-horizon states.

No simulator state snapshot/restore is used. One process handles one run and exits.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--stage',choices=['anchor','live','recorded'],required=True)
parser.add_argument('--repeat-id',type=int,default=0)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--prefix',type=Path,required=True)
parser.add_argument('--other-env-actions',type=Path,required=True)
parser.add_argument('--logged-anchor',type=Path,required=True)
parser.add_argument('--state',type=Path,required=True)
parser.add_argument('--base-path',type=Path,required=True)
parser.add_argument('--fixed-map-path',type=Path,required=True)
parser.add_argument('--recorded-command',type=Path,default=None)
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--reference-config',type=Path,required=True)
parser.add_argument('--rl-checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
simulation_app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat
from rsl_rl.runners import OnPolicyRunner
import onix_isaac_lab.tasks  # noqa
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import make_collection_cfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid,robot_dem
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride,ObservationCapture,capture_actor_critic


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def local_pose(env,i):
    robot=env.scene['robot']
    xy=(robot.data.root_pos_w[i,:2]-env.scene.env_origins[i,:2]).detach().cpu().numpy()
    _,_,yaw=euler_xyz_from_quat(robot.data.root_quat_w[i:i+1])
    return np.array([float(xy[0]),float(xy[1]),float(yaw.item())],dtype=np.float64)


def dem_capture(env,i,config):
    robot=env.scene['robot'];scanner=env.scene['height_scanner_eat']
    pos=robot.data.root_pos_w[i].detach().cpu().numpy()
    _,_,yaw=euler_xyz_from_quat(robot.data.root_quat_w[i:i+1])
    hits=scanner.data.ray_hits_w[i,:,2].detach().cpu().numpy().reshape(config.dem_height_cells,config.dem_width_cells)
    valid=np.isfinite(hits)
    start=scanner._ray_starts_w[i,0,:2].detach().cpu().numpy()
    source=SourceGrid(height=np.where(valid,hits,0).astype(np.float32),valid=valid,
                      origin_xy=(float(start[0]),float(start[1])),resolution_m=config.dem_resolution_m,
                      start_index_xy=(0,0),frame_id=config.reference_frame,stamp_ns=0)
    transformed=robot_dem(source,(float(pos[0]),float(pos[1]),float(yaw.item())),config)
    return {'source_height':source.height.astype(float).tolist(),
            'source_mask':valid.astype(np.uint8).tolist(),
            'source_origin_xy':list(source.origin_xy),
            'transformed':np.asarray(transformed,dtype=np.float32).astype(float).tolist(),
            'transformed_sha256':hashlib.sha256(np.asarray(transformed,dtype=np.float32).tobytes()).hexdigest()}


def tensor_data(value,i=None):
    if not isinstance(value,torch.Tensor):return None
    if i is not None and value.ndim and value.shape[0]>i:value=value[i]
    return value.detach().cpu().numpy().tolist()


def ring_state(ring,i):
    result={'pointer':getattr(ring,'_pointer',None),'num_pushes':tensor_data(getattr(ring,'_num_pushes',None),i)}
    buf=getattr(ring,'_buffer',None)
    if isinstance(buf,torch.Tensor):
        if buf.ndim>=2 and buf.shape[1]>i:result['buffer']=tensor_data(buf[:,i])
        else:result['buffer_shape']=list(buf.shape)
    return result


def detailed_anchor(env,i,policy,obs,captured,dem_history,config):
    robot=env.scene['robot'];command=env.command_manager._terms['base_velocity']
    action_manager=env.action_manager
    with torch.inference_mode():
        action,latent,value=capture_actor_critic(policy,obs['policy'][i:i+1],obs['critic'][i:i+1],latent_layer=6)
    action_terms={}
    for name,term in action_manager._terms.items():
        action_terms[name]={key:tensor_data(getattr(term,key,None),i) for key in
            ('_raw_actions','_processed_actions','_processed','_pre_clip_processed_actions','_previous_targets','_travel_direction')}
    delays={}
    for name,buffer in getattr(env,'_action_latency_buffers',{}).items():
        delays[name]={'time_lags':tensor_data(getattr(buffer,'_time_lags',None),i),
                      'ring':ring_state(buffer._circular_buffer,i)}
    history={}
    for group,terms in getattr(env.observation_manager,'_group_obs_term_history_buffer',{}).items():
        history[group]={name:ring_state(ring,i) for name,ring in terms.items()}
    contact=env.scene['contact_forces']
    scanner=env.scene['height_scanner_eat']
    def rng_digest(value):return hashlib.sha256(value).hexdigest()
    rng={'python_sha256':rng_digest(repr(random.getstate()).encode()),
         'numpy_sha256':rng_digest(repr(np.random.get_state()).encode()),
         'torch_cpu_sha256':rng_digest(torch.get_rng_state().numpy().tobytes()),
         'torch_cuda_sha256':[rng_digest(x.cpu().numpy().tobytes()) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
    return {'root_state':tensor_data(robot.data.root_state_w,i),
            'joint_position':tensor_data(robot.data.joint_pos,i),
            'joint_velocity':tensor_data(robot.data.joint_vel,i),
            'joint_target':tensor_data(getattr(robot.data,'joint_pos_target',None),i),
            'command_manager':tensor_data(command.vel_command_b,i),
            'command_time_left':tensor_data(getattr(command,'time_left',None),i),
            'episode_length':tensor_data(env.episode_length_buf,i),
            'terrain_path_length_xy':tensor_data(getattr(env,'_terrain_path_length_xy',None),i),
            'terrain_prev_root_pos_xy':tensor_data(getattr(env,'_terrain_prev_root_pos_xy',None),i),
            'policy_pre_noise':tensor_data(captured['policy_pre_noise'],i),
            'policy_post_noise_pre_delay':tensor_data(captured['policy_post_noise_pre_delay'],i),
            'policy_actor_input':tensor_data(obs['policy'],i),
            'actor_latent':tensor_data(latent,0),'actor_action':tensor_data(action,0),
            'critic_observation':tensor_data(obs['critic'],i),'critic_value':float(value.flatten()[0]),
            'action_manager_action':tensor_data(getattr(action_manager,'_action',None),i),
            'action_manager_previous_action':tensor_data(getattr(action_manager,'_prev_action',None),i),
            'action_terms':action_terms,
            'flipper_previous_target':tensor_data(getattr(env,'_onix_flipper_fold_mode',None),i),
            'latency_substeps':tensor_data(getattr(env,'_latency_substeps',None),i),
            'latency_seconds':tensor_data(getattr(env,'_onix_latency_seconds',None),i),
            'latency_policy_obs_prev':tensor_data(getattr(env,'_latency_policy_obs_prev',None),i),
            'latency_policy_obs_reset_mask':tensor_data(getattr(env,'_latency_policy_obs_reset_mask',None),i),
            'action_delay_buffers':delays,'observation_history':history,
            'contact_net_forces':tensor_data(contact.data.net_forces_w,i),
            'contact_force_history':tensor_data(getattr(contact.data,'net_forces_w_history',None),i),
            'raycast_hits':tensor_data(scanner.data.ray_hits_w,i),
            'dem_history':dem_history,'dem_history_sha256':hashlib.sha256(np.asarray(dem_history,dtype=np.float32).tobytes()).hexdigest(),
            'rng_state_sha256':rng}


def map_path(base,pose):
    x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw)
    return [[float(x+c*px-s*py),float(y+s*px+c*py)] for px,py in base]


def capture_step(env,i,obs,captured,action,latent,value,reward,terminal,config):
    robot=env.scene['robot'];contact=env.scene['contact_forces']
    terms={}
    for name,term in env.action_manager._terms.items():
        tensor=getattr(term,'_processed_actions',None)
        if tensor is None:tensor=getattr(term,'_processed',None)
        terms[name]=tensor_data(tensor,i)
    return {'command':tensor_data(env.command_manager.get_command('base_velocity'),i),
            'pre_noise':tensor_data(captured['policy_pre_noise'],i),
            'post_noise':tensor_data(captured['policy_post_noise_pre_delay'],i),
            'actor_input':tensor_data(obs['policy'],i),'critic_observation':tensor_data(obs['critic'],i),
            'actor_latent':tensor_data(latent,0),'actor_action':tensor_data(action,0),
            'critic_value':float(value.flatten()[0]),'processed_action':terms,
            'joint_target':tensor_data(getattr(robot.data,'joint_pos_target',None),i),
            'joint_position':tensor_data(robot.data.joint_pos,i),'joint_velocity':tensor_data(robot.data.joint_vel,i),
            'root_state':tensor_data(robot.data.root_state_w,i),
            'contact_force':tensor_data(contact.data.net_forces_w,i),
            'raycast_hits':tensor_data(env.scene['height_scanner_eat'].data.ray_hits_w,i),
            'dem':dem_capture(env,i,config),'reward':float(reward) if reward is not None else None,
            'terminal':terminal}


def main():
    started=time.monotonic()
    config_spec=yaml.safe_load(args.config.read_text())
    state=json.loads(args.state.read_text())['train'][config_spec['source_state']['state_index']]
    logged=json.loads(args.logged_anchor.read_text())
    base_path=json.loads(args.base_path.read_text())
    if state['episode_key']!=[logged['session_id'],logged['env_id'],logged['episode_id']] or state['step']!=logged['step']:
        raise ValueError('frozen state/logged anchor mismatch')
    if state['episode_key'][1]!=config_spec['source_state']['env_index'] or len(base_path)!=9:
        raise ValueError('frozen state/path config mismatch')
    if config_spec['control_steps']!=120 or state['state_id'] not in ('coverage-001','coverage-003','coverage-004','coverage-006'):
        raise ValueError('long-horizon frozen scope mismatch')
    if args.stage!='live':raise ValueError('only approved live long-horizon stage')
    prefix=np.load(args.prefix)
    actions=prefix['actions'];prefix_command=prefix['selected_env_commands']
    with np.load(args.other_env_actions) as data:other_env_actions=np.asarray(data['actions'],dtype=np.float32)
    if other_env_actions.shape!=(config_spec['control_steps'],100,6):raise ValueError('other-env suffix action shape mismatch')
    if actions.shape!=(state['step']+1,100,6) or prefix_command.shape!=(len(actions),3):
        raise ValueError('frozen prefix shape mismatch')
    input_paths={'manifest':args.manifest,'prefix':args.prefix,'other_env_actions':args.other_env_actions,'logged_anchor':args.logged_anchor,
                 'state':args.state,'base_path':args.base_path,'config':args.config,
                 'reference_config':args.reference_config,'rl_checkpoint':args.rl_checkpoint,
                 'runner_source':Path(__file__),
                 'bridge_client_source':Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/phase_c_bridge_client.py'),
                 'capture_source':Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/phase_c_sim.py')}
    if args.stage!='anchor':input_paths['fixed_map_path']=args.fixed_map_path
    if args.stage=='recorded':input_paths['recorded_command']=args.recorded_command
    source={name:sha(path) for name,path in input_paths.items()}
    manifest=json.loads(args.manifest.read_text())
    if source['rl_checkpoint']!=manifest['metadata']['checkpoint_sha256']:
        raise ValueError('RL checkpoint/collection mismatch')
    marker=args.output/f'{args.stage}-{args.repeat_id}.marker.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous['source']!=source or any(not Path(p).exists() or sha(p)!=digest for p,digest in previous['outputs_sha256'].items()):
            raise ValueError('invalid resume marker')
        print('RECOVERY_RESUME_SKIP',args.stage,args.repeat_id,flush=True)
        return
    cfg=load_eat_config(args.reference_config)
    saved=yaml.load((args.rl_checkpoint.parent/'params'/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    toggles={name:saved[name].lower()=='true' for name in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
    if toggles!=manifest['metadata']['observation_toggles']:raise ValueError('observation toggles mismatch')
    env_cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),cfg,manifest['metadata']['base_seed'],100)
    env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=env_cfg))
    bare=env.unwrapped
    runner=OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
    weights=torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    runner.alg.policy.load_state_dict(weights);policy=runner.alg.policy
    obs,_=env.reset();capture=ObservationCapture(bare.observation_manager)
    i=state['episode_key'][1]
    dem_frames=[];prefix_command_diffs=[];captured=None
    frame_steps={state['step']-15,state['step']-10,state['step']-5,state['step']}
    for step,batch in enumerate(actions):
        if step==state['step']:capture.begin_step()
        with torch.inference_mode():obs,reward,done,_=env.step(torch.as_tensor(batch,device=bare.device))
        if step==state['step']:captured=capture.finish_step(obs)
        # The collector retains a live command tensor and serializes it after env.step.
        # Its row command is therefore the post-step command, including resampling.
        observed_post=bare.command_manager.get_command('base_velocity')[i].detach().cpu().numpy()
        prefix_command_diffs.append(float(np.max(np.abs(observed_post-prefix_command[step]))))
        if step in frame_steps:dem_frames.append(dem_capture(bare,i,cfg)['transformed'])
        if bool(done[i]):raise ValueError('selected episode reset during prefix')
    if max(prefix_command_diffs)>1e-5:raise ValueError('natural prefix command differs from logged command')
    if len(dem_frames)!=4:raise ValueError('four-frame DEM history missing')
    anchor=detailed_anchor(bare,i,policy,obs,captured,dem_frames,cfg)
    expected_pose=np.asarray(logged['pose'],dtype=float)
    actual_pose=np.asarray(anchor['root_state'][:7],dtype=float)
    actual_dem=np.asarray(dem_frames[-1],dtype=float);expected_dem=np.asarray(logged['dem']['transformed'],dtype=float)
    valid_a=actual_dem[1]>.5;valid_b=expected_dem[1]>.5;common=valid_a&valid_b
    error={'position_m':float(np.linalg.norm(actual_pose[:3]-expected_pose[:3])),
           'orientation_rad':float(2*math.acos(min(1.,abs(np.dot(actual_pose[3:7],expected_pose[3:7]))))),
           'velocity_max_abs':float(np.max(np.abs(np.asarray(anchor['root_state'][7:13])-np.asarray(logged['twist'])))),
           'dem_rms_m':float(np.sqrt(np.mean((actual_dem[0][common]-expected_dem[0][common])**2))) if common.any() else float('inf'),
           'mask_iou':float(common.sum()/max(1,(valid_a|valid_b).sum())),
           'dem_history_sha_match':anchor['dem_history_sha256']==state['dem_sha256'],
           'prefix_command_max_abs':max(prefix_command_diffs)}
    limits=config_spec['anchor_limits']
    if error['position_m']>limits['position_m'] or error['orientation_rad']>limits['orientation_rad'] or error['dem_rms_m']>limits['dem_rms_m'] or error['mask_iou']<limits['mask_iou_min']:
        raise ValueError('logged anchor tolerance failed: '+repr(error))
    pose=local_pose(bare,i);computed_path=map_path(base_path,pose)
    if args.stage=='anchor':
        if args.fixed_map_path.exists():
            if json.loads(args.fixed_map_path.read_text())!=computed_path:raise ValueError('pre-existing fixed map Path mismatch')
        else:atomic(args.fixed_map_path,computed_path)
        trace=[];startup=None;fresh_command=[]
    else:
        fixed=json.loads(args.fixed_map_path.read_text())
        if np.max(np.abs(np.asarray(fixed)-np.asarray(computed_path)))>1e-6:
            raise ValueError('same base Path transformed to different map Path at anchor')
        trace=[];startup=None;fresh_command=[]
        bridge=None
        if args.stage=='live':
            bridge=FollowerBridgeClient(timeout=45.)
            startup=bridge.request('reset',poses=[pose.tolist()]*8,paths=[fixed]*8)
        override=CommandOverride(bare.command_manager,bare.num_envs)
        command_index=0
        recorded=None
        if args.stage=='recorded':
            with np.load(args.recorded_command) as data:recorded=np.asarray(data['command'],dtype=np.float32)
            if recorded.shape!=(config_spec['control_steps'],3):raise ValueError('recorded command shape mismatch')
        def command_provider():
            nonlocal command_index
            if recorded is not None:
                value=recorded[min(command_index,len(recorded)-1)]
            else:
                response=bridge.request('step',poses=[local_pose(bare,i).tolist()]*8,timeout_sec=2.)
                if not response['fresh']:raise RuntimeError('stale follower command')
                value=np.asarray(response['commands'][0],dtype=np.float32)
                fresh_command.append(response)
            command_index+=1
            cmd=bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
            cmd[i]=torch.as_tensor(value,device=bare.device,dtype=cmd.dtype)
            return cmd
        override.provider=command_provider
        override.set(command_provider())
        capture.begin_step()
        raw=bare.observation_manager.compute()
        obs=bare._apply_policy_observation_latency(raw)
        current=capture.finish_step(obs)
        post_follower_anchor=detailed_anchor(bare,i,policy,obs,current,dem_frames,cfg)
        original_reset=bare._reset_idx;terminal_state={}
        def capture_then_reset(env_ids):
            if i in [int(x) for x in env_ids.detach().cpu().tolist()]:
                terms=bare.termination_manager
                terminal_state.update({'root_state':tensor_data(bare.scene['robot'].data.root_state_w,i),
                    'joint_position':tensor_data(bare.scene['robot'].data.joint_pos,i),
                    'joint_velocity':tensor_data(bare.scene['robot'].data.joint_vel,i),
                    'contact_force':tensor_data(bare.scene['contact_forces'].data.net_forces_w,i),
                    'raycast_hits':tensor_data(bare.scene['height_scanner_eat'].data.ray_hits_w,i),
                    'termination_reason':[name for name in terms.active_terms if bool(terms.get_term(name)[i])]})
            return original_reset(env_ids)
        bare._reset_idx=capture_then_reset
        for control_step in range(config_spec['control_steps']):
            with torch.inference_mode():
                action,latent,value=capture_actor_critic(policy,obs['policy'][i:i+1],obs['critic'][i:i+1],latent_layer=6)
            if tuple(action.shape)!=(1,6) or tuple(obs['policy'].shape)!=(100,22) or tuple(obs['critic'].shape)!=(100,267):
                raise ValueError('Actor/Critic/action shape mismatch')
            record=capture_step(bare,i,obs,current,action,latent,value,None,None,cfg)
            batch=torch.as_tensor(other_env_actions[control_step],device=bare.device).clone()
            batch[i]=action[0]
            capture.begin_step()
            with torch.inference_mode():next_obs,reward,done,_=env.step(batch)
            current=capture.finish_step(next_obs)
            record['reward']=float(reward[i]);record['post_root_state']=tensor_data(bare.scene['robot'].data.root_state_w,i)
            if bool(done[i]):
                record['terminal']={'terminated':bool(bare.reset_terminated[i]),
                    'truncated':bool(bare.reset_time_outs[i]),**terminal_state}
                trace.append(record);break
            trace.append(record);obs=next_obs
            if control_step in (0,19,39,59,79,99,119):
                atomic(args.output/'status.json',{'stage':args.stage,'repeat_id':args.repeat_id,
                    'control_step':control_step+1,'elapsed_sec':time.monotonic()-started})
                print('RECOVERY_STEP',args.stage,args.repeat_id,control_step+1,flush=True)
        if not trace[-1]['terminal']:
            with torch.inference_mode():_,_,end_value=capture_actor_critic(policy,obs['policy'][i:i+1],obs['critic'][i:i+1],latent_layer=6)
            end_value=float(end_value.flatten()[0])
        else:end_value=None
        bare._reset_idx=original_reset
        override.close()
        if bridge is not None:bridge.close()
    args.output.mkdir(parents=True,exist_ok=True)
    summary={'stage':args.stage,'repeat_id':args.repeat_id,'state':state,'source_sha256':source,
             'anchor_error_vs_logged':error,'anchor':anchor,
             'fixed_map_path_sha256':sha(args.fixed_map_path),
             'map_path':json.loads(args.fixed_map_path.read_text()),
             'follower_startup':startup,'fresh_command_response_count':len(fresh_command),
             'suffix_steps':len(trace),'suffix_end_critic_value':end_value if args.stage!='anchor' else None,
             'elapsed_sec':time.monotonic()-started,'snapshot_restore_used':False}
    if args.stage!='anchor':summary['post_follower_anchor']=post_follower_anchor
    result_path=args.output/f'{args.stage}-{args.repeat_id}.json'
    trace_path=args.output/f'{args.stage}-{args.repeat_id}.trace.json.gz'
    if result_path.exists() or trace_path.exists():raise FileExistsError('run output exists without valid marker')
    temp=trace_path.with_name(trace_path.name+'.tmp')
    with gzip.open(temp,'wt') as stream:json.dump(trace,stream,separators=(',',':'))
    temp.replace(trace_path)
    summary['trace_sha256']=sha(trace_path)
    atomic(result_path,summary)
    atomic(marker,{'source':source,'outputs_sha256':{str(result_path):sha(result_path),str(trace_path):sha(trace_path)},
                   'fixed_map_path_sha256':sha(args.fixed_map_path)})
    atomic(args.output/'status.json',{'stage':args.stage,'repeat_id':args.repeat_id,'state':'complete',
        'elapsed_sec':time.monotonic()-started,'result_sha256':sha(result_path)})
    print('RECOVERY_RUN_COMPLETE',args.stage,args.repeat_id,'steps',len(trace),'sec',round(time.monotonic()-started,2),flush=True)
    capture.close();env.close()

try:
    main()
except BaseException as exc:
    import traceback
    print('RECOVERY_FATAL',repr(exc),flush=True)
    traceback.print_exc()
    try:atomic(args.output/'status.json',{'stage':'failed','repeat_id':args.repeat_id,'error':repr(exc),'traceback':traceback.format_exc()})
    except Exception:pass
    raise
finally:
    simulation_app.close()
