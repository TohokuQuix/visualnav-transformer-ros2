"""GUI closed-loop development run using the learned fall/stall Evaluator.

No ROS, robot actuator topics, reserved test, or replay are used.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--guard-config',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--checkpoint-sha256',required=True)
parser.add_argument('--rl-checkpoint',type=Path,required=True)
parser.add_argument('--actor-env-yaml',type=Path,required=True)
parser.add_argument('--eat-config',type=Path,required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--terrain-type',default='k_rail_step_field')
parser.add_argument('--terrain-level',type=int,default=4,choices=range(5))
parser.add_argument('--control-steps',type=int,default=600)
parser.add_argument('--hold-seconds',type=float,default=60.)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.headless:parser.error('this integration run requires GUI; headless is disabled')
if not 40<=args.control_steps<=600 or not 0<=args.hold_seconds<=300:
    parser.error('bounded run requires 40..600 steps and 0..300s window hold')
simulation_app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
import omni.ui as ui
from pxr import Gf,UsdGeom
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat
from rsl_rl.runners import OnPolicyRunner
import onix_isaac_lab.tasks
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid,robot_dem
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import make_collection_cfg
from visualnav_transformer.eat_nav.demo_terrain import configure_selected_terrain
from visualnav_transformer.eat_nav.demo_reset_policy import configure_demo_reset_policy,assert_finite_sim_state
from visualnav_transformer.eat_nav.mi_fi_v1 import configure_initialization
from visualnav_transformer.eat_nav.motion_collector import body_path_to_tile_local
from visualnav_transformer.eat_nav.p1_follower_core import PureFollower,compile_core
from visualnav_transformer.eat_nav.p1_g3_rosfree_control import reconcile_command_observation
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride,ObservationCapture
from visualnav_transformer.eat_nav.fall_stall_targets import tilt
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,seal_derived
from eat_nav_risk_navigation_v2 import OnlineEvaluator,pack_inputs,body_state,validate_history,select_candidate,candidate_paths,CANDIDATE_IDS

from eat_nav_risk_guard import CONFIG,RiskGuard,guarded_command

NUM_ENVS=1
STEP_NS=50_000_000
def pose_world(robot, index):
    xyz = robot.data.root_pos_w[index].detach().cpu().tolist()
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[index:index + 1])
    return float(xyz[0]), float(xyz[1]), float(yaw.item())

def capture_dem(bare, eat_cfg, index, logical_stamp_ns):
    scanner = bare.scene['height_scanner_eat']
    hit = scanner.data.ray_hits_w[index, :, 2].detach().cpu().numpy().reshape(20, 20)
    valid = np.isfinite(hit)
    starts = (scanner.ray_starts[index] + scanner.data.pos_w[index]).detach().cpu().numpy()
    source = SourceGrid(np.where(valid, hit, 0).astype(np.float32), valid,
                        (float(starts[0, 0]), float(starts[0, 1])),
                        eat_cfg.dem_resolution_m, (0, 0), eat_cfg.reference_frame,
                        logical_stamp_ns)
    transformed = robot_dem(source, pose_world(bare.scene['robot'], index), eat_cfg)
    if transformed.shape != (2, 20, 20) or not np.isfinite(transformed).all():
        raise ValueError('nonfinite DEM')
    return transformed.astype(np.float32)

def capture_raw_dem(bare):
    """Retain the native ray geometry separately from the transformed model input."""
    scanner = bare.scene['height_scanner_eat']
    robot = bare.scene['robot']
    hits = scanner.data.ray_hits_w.detach().cpu().numpy().copy()
    starts = (scanner.ray_starts + scanner.data.pos_w.unsqueeze(1)).detach().cpu().numpy().copy()
    root = robot.data.root_state_w.detach().cpu().numpy().copy()
    origins = bare.scene.env_origins.detach().cpu().numpy().copy()
    if hits.shape != starts.shape or hits.shape[:2] != (NUM_ENVS, 400) or \
            hits.shape[-1] != 3 or not np.isfinite(starts).all() or \
            not np.isfinite(root).all() or not np.isfinite(origins).all():
        raise ValueError('raw DEM ray geometry shape/finite contract')
    return {'ray_hits_w': hits, 'ray_starts_w': starts, 'root_state_w': root,
            'env_origins_w': origins, 'ray_valid': np.isfinite(hits[:, :, 2])}

def capture_pre_reset_state(bare, index, control_step):
    """Snapshot a naturally terminated env before Isaac applies its reset."""
    robot = bare.scene['robot']
    manager = bare.termination_manager
    reasons = [name for name in manager.active_terms
               if bool(manager.get_term(name)[index])]
    return {
        'control_step': int(control_step),
        'termination_reasons': reasons,
        'episode_length_steps': int(bare.episode_length_buf[index].item()),
        'root_state': robot.data.root_state_w[index].detach().cpu().tolist(),
        'joint_position': robot.data.joint_pos[index].detach().cpu().tolist(),
        'joint_velocity': robot.data.joint_vel[index].detach().cpu().tolist(),
        'applied_torque': robot.data.applied_torque[index].detach().cpu().tolist(),
        'contact_net_forces_w': bare.scene['contact_forces'].data.net_forces_w[
            index].detach().cpu().tolist(),
    }

def draw_paths(bare,paths_world,selected,z,goal):
    stage=bare.sim.stage
    for k,points in enumerate(paths_world):
        curve=UsdGeom.BasisCurves.Define(stage,f'/World/RiskCandidate{k}')
        curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(float(x),float(y),float(z)) for x,y in points])
        curve.CreateWidthsAttr([.035 if k==selected else .012])
        curve.CreateDisplayColorAttr([Gf.Vec3f(0.,1.,.15) if k==selected else Gf.Vec3f(.5,.5,.7)])
    sphere=UsdGeom.Sphere.Define(stage,'/World/NavigationGoal');sphere.CreateRadiusAttr(.12)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(1.,.2,.1)])
    UsdGeom.Xformable(sphere).ClearXformOpOrder()
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(float(goal[0]),float(goal[1]),float(z)))


def main():
    if args.output.exists():raise FileExistsError('fresh GUI output required')
    args.output.mkdir(parents=True)
    env=override=capture=follower=stream=None
    snapshots={};active_step=-1;history=[];stamps=[];selections=[];traces=[]
    try:
        if read(args.guard_config)!=CONFIG:raise ValueError('guard configuration mismatch')
        guard=RiskGuard()
        evaluator=OnlineEvaluator(args.checkpoint,expected_sha256=args.checkpoint_sha256)
        config=load_eat_config(args.eat_config)
        saved=yaml.load(args.actor_env_yaml.read_text(),Loader=yaml.BaseLoader)
        toggles={n:saved[n].lower()=='true' for n in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
        cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),config,args.seed,1)
        cfg.episode_length_s=40.;configure_demo_reset_policy(cfg);configure_initialization(cfg,'front_folded')
        cfg.events.ignore_sprocket_self_collision=None
        terrain=configure_selected_terrain(cfg.scene.terrain.terrain_generator,cfg.scene.terrain,args.terrain_type,args.terrain_level,1)
        cfg.viewer.origin_type='asset_root';cfg.viewer.asset_name='robot';cfg.viewer.env_index=0
        cfg.viewer.eye=(3.5,-3.5,2.8);cfg.viewer.lookat=(0.,0.,.6)
        write(args.output/'input-manifest.json',dict(schema='eatnav_risk_gui_guard_v1',guard_config=CONFIG,state_dim=28,excluded_actor_indices=[6,7,8],
            seed=args.seed,terrain_type=args.terrain_type,terrain_level=args.terrain_level,
            checkpoint_sha256=sha(args.checkpoint),actor_sha256=sha(args.rl_checkpoint),
            actor_env_sha256=sha(args.actor_env_yaml),num_envs=1,headless=False,
            role='development_integration_only',reserved_test_opened=False,failure_induction=False,
            control_hz=20,replan_steps=40,max_control_steps=args.control_steps,
            continuation_caveat='trained fixed candidate continuation; online switching unvalidated',
            terrain=terrain))
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg));bare=env.unwrapped
        if abs(float(bare.step_dt)-.05)>1e-9:raise ValueError('20 Hz required')
        env.reset();robot=bare.scene['robot'];reset=bare._reset_idx
        def before_reset(ids):
            for i in (range(1) if ids is None else ids.detach().cpu().tolist()):
                snapshots[int(i)]=capture_pre_reset_state(bare,int(i),active_step)
            return reset(ids)
        bare._reset_idx=before_reset
        runner=OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
        runner.alg.policy.load_state_dict(torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict'])
        policy=runner.alg.policy
        override=CommandOverride(bare.command_manager,1);capture=ObservationCapture(bare.observation_manager)
        zeros=torch.zeros((1,3),device=bare.device);override.set(zeros)
        latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
        capture.begin_step();obs=bare._apply_policy_observation_latency(bare.observation_manager.compute());observed=capture.finish_step(obs)
        library=compile_core(args.output/'core-build')
        window=ui.Window('Learned fall / stall Evaluator',width=550,height=230)
        with window.frame:
            with ui.VStack():
                title=ui.Label('LIVE simulation | DEM + State | command-free v2 + risk stop / hysteresis',height=28)
                status=ui.Label('Settling and collecting 4 DEM frames...',word_wrap=True,height=40)
                values=ui.Label('No replay. Fixed RL Actor. New development terrain.',word_wrap=True)
        for settling in range(20):
            start=time.monotonic();override.set(zeros)
            obs,observed,_=reconcile_command_observation(obs,observed,bare,zeros,latency)
            with torch.inference_mode():action=policy.act_inference(obs)
            latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
            capture.begin_step()
            with torch.inference_mode():obs,reward,done,_=env.step(action)
            observed=capture.finish_step(obs)
            if bool(done.any()):raise ValueError('termination during settling')
            if (settling+1)%5==0:
                stamps.append((settling+1)*STEP_NS);history.append(capture_dem(bare,config,0,stamps[-1]))
            time.sleep(max(0.,.05-(time.monotonic()-start)))
        initial=np.asarray(pose_world(robot,0));c,s=math.cos(initial[2]),math.sin(initial[2])
        goal=initial[:2]+np.asarray([4*c,4*s]);origin=bare.scene.env_origins[0,:2].detach().cpu().numpy()
        write(args.output/'startup-ready.json',dict(gui=True,display=os.environ.get('DISPLAY'),goal_world_xy=goal.tolist(),
                                                   initial_pose=initial.tolist(),checkpoint_sha256=sha(args.checkpoint)))
        stream=(args.output/'trace.jsonl').open('x');stop_reason='horizon_complete'
        for step in range(args.control_steps):
            if not simulation_app.is_running():stop_reason='window_closed';break
            start=time.monotonic();active_step=step;stamp=1_000_000_000+step*STEP_NS
            assert_finite_sim_state(robot)
            pose=np.asarray(pose_world(robot,0));distance=float(np.linalg.norm(goal-pose[:2]))
            if distance<=.25:stop_reason='goal_reached';break
            measured_tilt=tilt(robot.data.root_state_w[0].detach().cpu().tolist())
            guard_before_step=guard.snapshot();was_stopped=guard.stopped
            guard.observe_tilt(measured_tilt,step)
            physical_stop_started=guard.stopped and not was_stopped
            if physical_stop_started and follower is not None:
                follower.close();follower=None
                draw_paths(bare,paths_world,-1,float(robot.data.root_pos_w[0,2])+.15,goal)
            if step%40==0:
                validate_history(stamps[-4:],stamp)
                arrays=pack_inputs(np.asarray(history[-4:]),observed['policy_pre_noise'][0].detach().cpu().numpy(),
                                   body_state(robot.data.root_state_w[0].detach().cpu().numpy()))
                probabilities,progress=evaluator.predict(arrays)
                c,s=math.cos(pose[2]),math.sin(pose[2]);delta=goal-pose[:2]
                goal_body=[c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]]
                selection=guard.decide(probabilities,progress,goal_body,step,measured_tilt);k=selection['selected_index']
                paths_world=[body_path_to_tile_local(p,tuple(pose),[0.,0.]) for p in candidate_paths()]
                if follower is not None:follower.close()
                follower=PureFollower((paths_world[k]-origin).tolist(),library=library) if k is not None else None
                np.savez_compressed(args.output/f'input-{step:04d}.npz',**arrays)
                raw=capture_raw_dem(bare);np.savez_compressed(args.output/f'raw-dem-{step:04d}.npz',**raw)
                record=dict(selection,control_step=step,stamp_ns=stamp,dem_stamps_ns=stamps[-4:],
                            pose_world=pose.tolist(),goal_body=goal_body,input_file=f'input-{step:04d}.npz',
                            raw_actor_pre_noise=arrays['actor_state'][0].tolist(),
                            paths_world=[p.tolist() for p in paths_world])
                write(args.output/f'selection-{step:04d}.json',record);selections.append(record)
                draw_paths(bare,paths_world,-1 if k is None else k,float(robot.data.root_pos_w[0,2])+.15,goal)
                # The initial Kit workspace can restore its layout after this
                # window is constructed. Re-show and dock after settling.
                window.visible=True
                if step==0:window.dock_in_window('Stage',ui.DockPosition.TOP,.45)
                window.focus()
                status.text=f"LIVE t={step/20:.1f}s | {selection['selected_candidate']} | {selection['guard_reason']} | goal: {distance:.2f}m"
                values.text='\n'.join(f'{name}: risk={probabilities[i].round(3).tolist()}  progress10={progress[i,3]:.3f}m  score={selection["scores"][i]:.3f}' for i,name in enumerate(CANDIDATE_IDS))
                print('SELECTION '+json.dumps(record),flush=True)
            if guard.stopped:
                status.text=f'LIVE STOP t={step/20:.1f}s | {guard.stop_reason} | tilt={measured_tilt:.1f}deg'
            command,diagnostic=guarded_command(guard,follower,
                (float(pose[0]-origin[0]),float(pose[1]-origin[1]),float(pose[2])),stamp)
            root_pre=robot.data.root_state_w[0].detach().cpu().tolist()
            injected=torch.tensor([command],dtype=zeros.dtype,device=bare.device);override.set(injected)
            obs,observed,command_detail=reconcile_command_observation(obs,observed,bare,injected,latency)
            with torch.inference_mode():action=policy.act_inference(obs)
            latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
            capture.begin_step()
            with torch.inference_mode():obs,reward,done,_=env.step(action)
            observed=capture.finish_step(obs)
            terminal=snapshots.pop(0,None)
            row=dict(control_step=step,logical_sim_time_ns=stamp,
                selected_candidate='STOP' if guard.stopped else CANDIDATE_IDS[k],
                planned_candidate='STOP' if k is None else CANDIDATE_IDS[k],
                guard_stopped=guard.stopped,guard_reason=guard.stop_reason if guard.stopped else selection['guard_reason'],
                guard_state_before_step=guard_before_step,guard_state_after_step=guard.snapshot(),
                measured_tilt_deg=measured_tilt,
                root_state=root_pre,post_root_state=terminal['root_state'] if terminal else robot.data.root_state_w[0].detach().cpu().tolist(),
                follower_command=list(command),follower_diagnostic=diagnostic,actor_command=command_detail['actor_seen'][0],
                terminal=terminal,goal_distance_m=distance)
            traces.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if bool(done.any()):stop_reason='natural_termination';break
            if (step+1)%5==0:
                stamps.append(stamp+STEP_NS);history.append(capture_dem(bare,config,0,stamps[-1]))
            time.sleep(max(0.,.05-(time.monotonic()-start)))
        end=np.asarray(pose_world(robot,0))
        if stop_reason=='natural_termination':end=np.asarray([*traces[-1]['post_root_state'][:2],0.])
        summary=dict(state='completed',stop_reason=stop_reason,control_steps=len(traces),
            selections=len(selections),nonzero_command_steps=sum(abs(r['follower_command'][0])>1e-3 for r in traces),
            risk_changed_goal_only_choice=sum(r['selected_index']!=r['goal_only_index'] for r in selections),
            selected_candidates=[r['selected_candidate'] for r in selections],
            start_xy_m=initial[:2].tolist(),end_xy_m=end[:2].tolist(),
            displacement_m=float(np.linalg.norm(end[:2]-initial[:2])),
            initial_goal_distance_m=4.,final_goal_distance_m=float(np.linalg.norm(goal-end[:2])),
            max_tilt_deg=max((tilt(r['post_root_state']) for r in traces),default=0.),
            guard_config=CONFIG,guard_stop_steps=sum(r['guard_stopped'] for r in traces),
            guard_stop_time_s=sum(r['guard_stopped'] for r in traces)/20,
            guard_stop_episodes=sum(r['guard_stopped'] and (i==0 or not traces[i-1]['guard_stopped']) for i,r in enumerate(traces)),
            guard_resume_episodes=sum(not r['guard_stopped'] and i>0 and traces[i-1]['guard_stopped'] for i,r in enumerate(traces)),
            gui=True,headless=False,reserved_test_opened=False,model_updated=False,
            new_rollout=True,formal_comparison=False,checkpoint_sha256=sha(args.checkpoint))
        write(args.output/'complete.json',summary);print('COMPLETE '+json.dumps(summary),flush=True)
        status.text=f'FINISHED: {stop_reason} | {len(traces)/20:.1f}s simulated; displacement {summary["displacement_m"]:.2f}m'
        # Freeze physical state while leaving the real scene visible briefly.
        override.set(zeros);bare.sim.pause()
        deadline=time.monotonic()+args.hold_seconds
        while simulation_app.is_running() and time.monotonic()<deadline:
            simulation_app.update();time.sleep(.03)
    except BaseException as exc:
        write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if stream is not None:stream.close()
        if follower is not None:follower.close()
        if capture is not None:capture.close()
        if override is not None:override.close()
        if env is not None:env.close()
        seal_derived(args.output)


if __name__=='__main__':
    try:main()
    finally:simulation_app.close()
