"""Same-env serial maneuver replay from exact logged-action collection anchors."""
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
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--selection',type=Path,required=True)
parser.add_argument('--maneuvers',type=Path,required=True)
parser.add_argument('--maneuver-bank',type=Path,required=True)
parser.add_argument('--precheck',type=Path,required=True)
parser.add_argument('--spec',type=Path,required=True)
parser.add_argument('--reference-config',type=Path,required=True)
parser.add_argument('--rl-checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--split',choices=['train','validation'],required=True)
parser.add_argument('--state-index',type=int,default=None)
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
from visualnav_transformer.eat_nav.feasible_set_state import SerialStateSnapshot


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


def records(manifest,ids,last_step):
    targets={(r['step'],r['episode_key'][1]):(i,r) for i,r in enumerate(ids)}
    expected={};actions={}
    for chunk in manifest['chunks']:
        with gzip.open(args.manifest.parent/chunk['file'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                step=int(row['step'])
                if step>last_step:continue
                env_id=int(row['env_id'])
                actions.setdefault(step,{})[env_id]=row['action']
                key=(step,env_id)
                if key in targets:
                    index,state=targets[key]
                    if list((row['session_id'],env_id,row['episode_id']))!=state['episode_key'] or row['pre_reset']:
                        raise ValueError('selected state is not a stable logged post-step anchor')
                    expected[index]=row
        if len(actions.get(last_step,{}))==manifest['metadata']['num_envs'] and len(expected)==len(ids):break
    if len(expected)!=len(ids):raise ValueError('missing selected anchor record')
    array=np.asarray([[actions[step][env] for env in range(manifest['metadata']['num_envs'])]
                      for step in range(last_step+1)],dtype=np.float32)
    if array.shape!=(last_step+1,manifest['metadata']['num_envs'],6):
        raise ValueError('recorded action prefix incomplete')
    return array,expected


def pose_local(bare,env_id):
    robot=bare.scene['robot']
    xy=(robot.data.root_pos_w[env_id,:2]-bare.scene.env_origins[env_id,:2]).detach().cpu().numpy()
    _,_,yaw=euler_xyz_from_quat(robot.data.root_quat_w[env_id:env_id+1])
    return np.array([float(xy[0]),float(xy[1]),float(yaw.item())])


def transformed_dem(bare,env_id,config):
    robot=bare.scene['robot'];scanner=bare.scene['height_scanner_eat']
    pos=robot.data.root_pos_w[env_id].detach().cpu().numpy()
    _,_,yaw=euler_xyz_from_quat(robot.data.root_quat_w[env_id:env_id+1])
    hits=scanner.data.ray_hits_w[env_id,:,2].detach().cpu().numpy().reshape(config.dem_height_cells,config.dem_width_cells)
    valid=np.isfinite(hits)
    start=scanner._ray_starts_w[env_id,0,:2].detach().cpu().numpy()
    source=SourceGrid(height=np.where(valid,hits,0).astype(np.float32),valid=valid,
                      origin_xy=(float(start[0]),float(start[1])),resolution_m=config.dem_resolution_m,
                      start_index_xy=(0,0),frame_id=config.reference_frame,stamp_ns=0)
    return robot_dem(source,(float(pos[0]),float(pos[1]),float(yaw.item())),config)


def anchor_errors(bare,env_id,expected,config):
    robot=bare.scene['robot']
    pose=robot.data.root_state_w[env_id,:7].detach().cpu().numpy().astype(float)
    target=np.asarray(expected['pose'],dtype=float)
    pos=float(np.linalg.norm(pose[:3]-target[:3]))
    orient=float(2*math.acos(min(1.,abs(np.dot(pose[3:],target[3:])))))
    dem=transformed_dem(bare,env_id,config)
    target_dem=np.asarray(expected['dem']['transformed'],dtype=np.float32)
    a=dem[1]>.5;b=target_dem[1]>.5;both=a&b
    rms=float(np.sqrt(np.mean((dem[0][both]-target_dem[0][both])**2))) if both.any() else float('inf')
    iou=float(both.sum()/max(1,(a|b).sum()))
    return {'position_m':pos,'orientation_rad':orient,'dem_rms_m':rms,'mask_iou':iou}


def map_path(points,pose):
    x,y,yaw=pose
    c,s=math.cos(yaw),math.sin(yaw)
    result=[]
    for px,py in np.concatenate((np.zeros((1,2)),points)):
        result.append([float(x+c*px-s*py),float(y+s*px+c*py)])
    return result


def trace_metrics(trace,path,anchor_pose,goal):
    poses=np.asarray(trace['local_pose'],dtype=float)
    commands=np.asarray(trace['command'],dtype=float)
    values=np.asarray(trace['critic_value'],dtype=float)
    rewards=np.asarray(trace['reward'],dtype=float)
    goal_xy=np.asarray(map_path(np.asarray([goal]),anchor_pose)[-1])
    start=np.asarray(anchor_pose[:2])
    final=poses[-1,:2] if len(poses) else start
    initial_goal=float(np.linalg.norm(goal_xy-start));end_goal=float(np.linalg.norm(goal_xy-final))
    progress=initial_goal-end_goal
    path_xy=np.asarray(map_path(path,anchor_pose),dtype=float)
    def distance_to_segments(point):
        a=path_xy[:-1];b=path_xy[1:];delta=b-a
        denominator=np.maximum(np.sum(delta*delta,axis=1),1e-9)
        t=np.clip(np.sum((point-a)*delta,axis=1)/denominator,0,1)
        return float(np.min(np.linalg.norm(point-(a+t[:,None]*delta),axis=1)))
    tracking=float(np.mean([distance_to_segments(p[:2]) for p in poses])) if len(poses) else float('inf')
    gamma=.99
    discounted=float(np.dot(gamma**np.arange(len(rewards)),rewards))
    terminal=trace['terminal']
    vend=trace.get('critic_value_end')
    if terminal and terminal['terminated']:
        return_h=discounted;bootstrap='true_terminal_zero';delta_v=drawdown=recovery=None
    elif terminal and terminal['truncated']:
        return_h=discounted+(gamma**len(rewards))*float(values[-1]);bootstrap='ppo_timeout_preterminal'
        delta_v=drawdown=recovery=None
    else:
        return_h=discounted+(gamma**len(rewards))*float(vend);bootstrap='endpoint_value'
        series=np.r_[values,vend]
        delta_v=float(vend-values[0]);drawdown=float(series.min()-values[0]);recovery=float(vend-series.min())
    return {'initial_goal_distance_m':initial_goal,'end_goal_distance_m':end_goal,
            'progress_m':progress,'xy_displacement_m':float(np.linalg.norm(final-start)),
            'path_tracking_error_m':tracking,'delta_v':delta_v,'advantage_h':float(return_h-values[0]),
            'drawdown':drawdown,'recovery':recovery,'return_h':return_h,'bootstrap_kind':bootstrap,
            'command_series':commands.tolist(),
            'command_first_divergence_step_from_nominal':None}


def run_state(env,bare,policy,config,rows,paths,source_actions,expected,spec,bridge,override,capture,state_folder):
    state=rows['state'];env_id=int(state['episode_key'][1]);index=rows['state_index']
    pose=pose_local(bare,env_id)
    errors=anchor_errors(bare,env_id,expected,config)
    limits=spec['replay']
    anchor_valid=(errors['position_m']<=limits['anchor_position_tolerance_m'] and
                  errors['orientation_rad']<=limits['anchor_yaw_tolerance_rad'] and
                  errors['dem_rms_m']<=limits['anchor_dem_rms_tolerance_m'] and errors['mask_iou']>=.95)
    print('SERIAL_STAGE anchor_checked',index,errors,flush=True)
    summary={'state_index':index,'split':rows['split'],'state':state,'anchor_errors':errors,
             'anchor_valid':anchor_valid,'precheck_status':rows['precheck_status']}
    if not anchor_valid:
        summary['classification']='measurement_invalid'
        atomic(state_folder/f'{index:03d}.json',summary)
        return summary
    print('SERIAL_STAGE snapshot_start',index,flush=True)
    snapshot=SerialStateSnapshot(bare)
    print('SERIAL_STAGE snapshot_ready',index,flush=True)
    anchor_obs={k:v.detach().clone() for k,v in rows['anchor_obs'].items()}
    paths=np.asarray(paths,dtype=float)
    names=list(spec['maneuvers']['names'])
    rng=random.Random(spec['replay']['candidate_order_seed']+index+(0 if rows['split']=='train' else 100000))
    shuffled=names.copy();rng.shuffle(shuffled)
    schedule=['duplicate_0','duplicate_1','duplicate_2']+shuffled
    summary['schedule']=schedule
    summary['snapshot_unreplicated']=snapshot.unreplicated
    traces=[];case_reports=[]
    target_step=int(state['step'])
    for case_index,case in enumerate(schedule):
        print('SERIAL_STAGE case_start',index,case,flush=True)
        restore=snapshot.restore()
        override.provider=None;override._command=None
        proposal='nominal' if case.startswith('duplicate_') else case
        path=paths[names.index(proposal)]
        map_xy=map_path(path,pose)
        pose_now=pose_local(bare,env_id)
        if np.max(np.abs(pose_now-pose))>1e-4:
            raise ValueError('serial replay changed local anchor pose')
        bridge.request('reset',poses=[pose_now.tolist()]*8,paths=[map_xy]*8)
        def follower_command():
            now=pose_local(bare,env_id).tolist()
            response=bridge.request('step',poses=[now]*8,timeout_sec=2.)
            if not response['fresh']:
                raise RuntimeError('stale follower command')
            command=bare.command_manager._terms['base_velocity'].vel_command_b.detach().clone()
            command[env_id]=torch.as_tensor(response['commands'][0],device=bare.device,dtype=command.dtype)
            return command
        override.provider=follower_command
        override.set(follower_command())
        capture.begin_step()
        raw=bare.observation_manager.compute()
        obs=bare._apply_policy_observation_latency(raw)
        current=capture.finish_step(obs)
        trace={'local_pose':[],'command':[],'critic_value':[],'reward':[],'contact_max':[],
               'actor_input':[],'actor_latent':[],'actor_action':[],'critic_observation':[],
               'pre_noise':[],'post_noise':[],'processed_action':[],'joint_position':[],
               'joint_velocity':[],'roll_pitch':[],'dem_height':[],'terminal':None}
        for step in range(spec['replay']['horizon_steps']):
            with torch.inference_mode():
                action,latent,value=capture_actor_critic(policy,obs['policy'][env_id:env_id+1],
                                                           obs['critic'][env_id:env_id+1],latent_layer=6)
            if tuple(action.shape)!=(1,6) or tuple(obs['policy'].shape)!=(bare.num_envs,22) or tuple(obs['critic'].shape)!=(bare.num_envs,267):
                raise ValueError('Actor/Critic/action shape mismatch')
            command=bare.command_manager.get_command('base_velocity')[env_id].detach().cpu().numpy()
            trace['command'].append(command.astype(float).tolist())
            trace['critic_value'].append(float(value.flatten()[0]))
            trace['actor_input'].append(obs['policy'][env_id].detach().cpu().numpy().astype(float).tolist())
            trace['critic_observation'].append(obs['critic'][env_id].detach().cpu().numpy().astype(float).tolist())
            trace['pre_noise'].append(current['policy_pre_noise'][env_id].detach().cpu().numpy().astype(float).tolist())
            trace['post_noise'].append(current['policy_post_noise_pre_delay'][env_id].detach().cpu().numpy().astype(float).tolist())
            trace['actor_latent'].append(latent[0].detach().cpu().numpy().astype(float).tolist())
            trace['actor_action'].append(action[0].detach().cpu().numpy().astype(float).tolist())
            batch=torch.as_tensor(source_actions[target_step+1+step],device=bare.device).clone()
            batch[env_id]=action[0]
            capture.begin_step()
            with torch.inference_mode():
                next_obs,reward,done,_=env.step(batch)
            current=capture.finish_step(next_obs)
            trace['reward'].append(float(reward[env_id]))
            pose_after=pose_local(bare,env_id)
            trace['local_pose'].append(pose_after.astype(float).tolist())
            forces=torch.linalg.vector_norm(bare.scene['contact_forces'].data.net_forces_w[env_id],dim=-1)
            trace['contact_max'].append(float(forces.max()))
            joint=bare.scene['robot'].data
            trace['joint_position'].append(joint.joint_pos[env_id].detach().cpu().numpy().astype(float).tolist())
            trace['joint_velocity'].append(joint.joint_vel[env_id].detach().cpu().numpy().astype(float).tolist())
            roll,pitch,_=euler_xyz_from_quat(joint.root_quat_w[env_id:env_id+1])
            trace['roll_pitch'].append([float(roll.item()),float(pitch.item())])
            trace['dem_height'].append(torch.nan_to_num(bare.scene['height_scanner_eat'].data.ray_hits_w[env_id,:,2]).detach().cpu().numpy().astype(float).tolist())
            processed=[]
            for term_name in ('flipper_pos','right_track_vel','left_track_vel'):
                term=bare.action_manager._terms[term_name]
                tensor=getattr(term,'_processed_actions',None)
                if tensor is None:tensor=getattr(term,'_processed',None)
                if tensor is None:raise ValueError('processed action unavailable: '+term_name)
                processed.extend(tensor[env_id].detach().cpu().flatten().numpy().astype(float).tolist())
            trace['processed_action'].append(processed)
            obs=next_obs
            if bool(done[env_id]):
                trace['terminal']={'step':step+1,'terminated':bool(bare.reset_terminated[env_id]),
                                   'truncated':bool(bare.reset_time_outs[env_id])}
                break
        if not trace['terminal']:
            with torch.inference_mode():
                _,_,end_value=capture_actor_critic(policy,obs['policy'][env_id:env_id+1],
                                                    obs['critic'][env_id:env_id+1],latent_layer=6)
            trace['critic_value_end']=float(end_value.flatten()[0])
        metrics=trace_metrics(trace,path,pose,state['goal'])
        case_reports.append({'case':case,'proposal':proposal,'restore':restore,'metrics':metrics,
                             'steps':len(trace['reward']),'terminal':trace['terminal']})
        traces.append(trace)
        override.provider=None;override._command=None
        print('SERIAL_CASE',rows['split'],index,case,'steps',len(trace['reward']),
              'progress',round(metrics['progress_m'],4),flush=True)
    snapshot.restore()
    summary['cases']=case_reports
    summary['trace_sha256']=None
    trace_path=state_folder/f'{index:03d}.traces.json.gz'
    temp=trace_path.with_name(trace_path.name+'.tmp')
    with gzip.open(temp,'wt') as stream:json.dump(traces,stream,separators=(',',':'))
    temp.replace(trace_path)
    summary['trace_sha256']=sha(trace_path)
    duplicate_xy=np.asarray([case['local_pose'] for case in traces[:3]],dtype=float)
    duplicate_rms=float(np.sqrt(np.mean((duplicate_xy-duplicate_xy[:1])**2)))
    summary['duplicate_xy_rms_m']=duplicate_rms
    summary['duplicate_valid']=duplicate_rms<=limits['duplicate_rms_max_m']
    summary['classification']='pending_analysis' if summary['duplicate_valid'] else 'measurement_invalid'
    atomic(state_folder/f'{index:03d}.json',summary)
    return summary


def main():
    started=time.monotonic()
    spec=yaml.safe_load(args.spec.read_text())
    frozen=json.loads(args.selection.read_text())
    bank=json.loads(args.maneuver_bank.read_text())
    precheck=json.loads(args.precheck.read_text())
    ids=frozen[args.split]
    indices=[args.state_index] if args.state_index is not None else list(range(len(ids)))
    if any(i<0 or i>=len(ids) for i in indices):raise ValueError('state index out of range')
    offset=0 if args.split=='train' else len(frozen['train'])
    with np.load(args.maneuvers) as data:paths=data['paths']
    if sha(args.maneuvers)!=bank['paths_sha256'] or len(precheck['rows'])!=len(bank['states']):
        raise ValueError('candidate/precheck source mismatch')
    if any(precheck['rows'][offset+i]['state']!=ids[i] for i in indices):
        raise ValueError('frozen selected state/precheck mismatch')
    source={'manifest':sha(args.manifest),'selection':sha(args.selection),'maneuvers':sha(args.maneuvers),
            'maneuver_bank':sha(args.maneuver_bank),'precheck':sha(args.precheck),'spec':sha(args.spec),
            'reference_config':sha(args.reference_config),'rl_checkpoint':sha(args.rl_checkpoint),
            'runner_source':sha(Path(__file__)),
            'state_source':sha(Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/feasible_set_state.py')),
            'phase_c_sim_source':sha(Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/phase_c_sim.py'))}
    manifest=json.loads(args.manifest.read_text())
    if source['rl_checkpoint']!=manifest['metadata']['checkpoint_sha256']:
        raise ValueError('RL checkpoint differs from collection')
    out=args.output/('rollouts-'+args.split)
    out.mkdir(parents=True,exist_ok=True)
    provenance_path=out/'execution-source-provenance.json'
    if provenance_path.exists():
        if json.loads(provenance_path.read_text())!=source:raise ValueError('execution source changed during resume')
    else:atomic(provenance_path,source)
    last_step=max(ids[i]['step'] for i in indices)+spec['replay']['horizon_steps']
    print('SERIAL_STAGE records_start',last_step,flush=True)
    actions,expected=records(manifest,[ids[i] for i in indices],last_step)
    print('SERIAL_STAGE records_loaded',actions.shape,flush=True)
    # Expected is keyed in the local selected sublist; map back to frozen index.
    expected={indices[k]:v for k,v in expected.items()}
    config=load_eat_config(args.reference_config)
    saved=yaml.load((args.rl_checkpoint.parent/'params'/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    toggles={key:saved[key].lower()=='true' for key in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
    if toggles!=manifest['metadata']['observation_toggles']:raise ValueError('collection toggles mismatch')
    print('SERIAL_STAGE config_start',flush=True)
    cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),config,manifest['metadata']['base_seed'],manifest['metadata']['num_envs'])
    print('SERIAL_STAGE config_ready',flush=True)
    env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg));bare=env.unwrapped
    print('SERIAL_STAGE env_created',flush=True)
    print('SERIAL_STAGE runner_start',flush=True)
    runner=OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
    print('SERIAL_STAGE runner_created',flush=True)
    weights=torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    runner.alg.policy.load_state_dict(weights);policy=runner.alg.policy
    obs,_=env.reset()
    print('SERIAL_STAGE reset_complete',flush=True)
    override=CommandOverride(bare.command_manager,bare.num_envs)
    capture=ObservationCapture(bare.observation_manager)
    bridge=FollowerBridgeClient(timeout=45.)
    print('SERIAL_STAGE bridge_connected',flush=True)
    by_step={}
    for index in indices:by_step.setdefault(ids[index]['step'],[]).append(index)
    try:
        for step in range(max(by_step)+1):
            batch=torch.as_tensor(actions[step],device=bare.device)
            with torch.inference_mode():obs,_,_,_=env.step(batch)
            if step%20==0:print('SERIAL_PREFIX',step,flush=True)
            if step not in by_step:continue
            for index in by_step[step]:
                marker=out/f'{index:03d}.json'
                if marker.exists():
                    previous=json.loads(marker.read_text())
                    if previous['state']!=ids[index] or previous['trace_sha256'] and previous['trace_sha256']!=sha(out/f'{index:03d}.traces.json.gz'):
                        raise ValueError('resume state marker mismatch')
                    print('SERIAL_RESUME_SKIP',args.split,index,flush=True)
                    continue
                state_row={'state':ids[index],'state_index':index,'split':args.split,
                           'precheck_status':precheck['rows'][offset+index]['status'],
                           'anchor_obs':obs}
                report=run_state(env,bare,policy,config,state_row,paths[offset+index],actions,
                                 expected[index],spec,bridge,override,capture,out)
                atomic(args.output/'status.json',{'stage':'serial_replay','split':args.split,
                    'completed_state_index':index,'last_classification':report['classification'],
                    'elapsed_sec':time.monotonic()-started,'source_sha256':sha(provenance_path)})
                print('SERIAL_STATE_COMPLETE',args.split,index,report['classification'],flush=True)
    finally:
        capture.close();override.close();bridge.close();env.close()
    print('SERIAL_SPLIT_COMPLETE',args.split,len(indices),'elapsed_sec',round(time.monotonic()-started,2),flush=True)

try:
    main()
except BaseException as exc:
    import traceback
    print('SERIAL_FATAL',repr(exc),flush=True)
    traceback.print_exc()
    try:atomic(args.output/'status.json',{'stage':'failed','error':repr(exc),'traceback':traceback.format_exc()})
    except Exception:pass
    raise
finally:
    simulation_app.close()
