"""R5 six-candidate GUI evaluation: linear and turn stall, without measured-angle STOP."""
from __future__ import annotations
import argparse,json,math,os,time,traceback,random
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--phase',choices=['evaluate'],required=True)
p.add_argument('--condition',choices=['R5'],default='R5')
p.add_argument('--candidate',type=int,choices=range(6),default=0)
p.add_argument('--anchor-seconds',type=int,choices=[0,10],default=0)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--terrain-type',required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--checkpoint-sha256',required=True)
p.add_argument('--rl-checkpoint',type=Path,required=True)
p.add_argument('--actor-env-yaml',type=Path,required=True)
p.add_argument('--eat-config',type=Path,required=True)
p.add_argument('--hold-seconds',type=float,default=1.)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if args.headless:p.error('real GUI required')
simulation_app=AppLauncher(args).app
import gymnasium as gym
import numpy as np
import torch,yaml
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
from visualnav_transformer.eat_nav.fall_stall_targets import tilt,body_vector
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,seal_derived
from eat_nav_risk_navigation_v2 import pack_inputs,body_state,validate_history,candidate_paths,CANDIDATE_IDS
from eat_nav_command_free_learning import load_checkpoint,predict
from eat_nav_r5 import R5Guard as R4Guard,CONFIG
from eat_nav_risk_guard import guarded_command
from eat_nav_all_fields import *
# Override four-candidate names AFTER the legacy wildcard import.
from eat_nav_r4_star import candidate_paths,pack_inputs,CANDIDATE_IDS,ANGLES,extend_path

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



def numeric_tree(value,prefix=''):
    if isinstance(value,torch.Tensor):return {prefix:value.detach().cpu().numpy().copy()}
    if isinstance(value,np.ndarray):return {prefix:value.copy()}
    if hasattr(value,'items'):
        result={}
        for k,v in value.items():result.update(numeric_tree(v,prefix+'/'+str(k)))
        return result
    if value is None:return {prefix:np.empty(0,dtype=np.float32)}
    raise ValueError('unsupported numeric state '+prefix)

def anchor_state(robot,obs,latency,bare,history,stamps,route,paths,diag,observed,follower):
    result=dict(root=robot.data.root_state_w.detach().cpu().numpy(),joint_position=robot.data.joint_pos.detach().cpu().numpy(),
        joint_velocity=robot.data.joint_vel.detach().cpu().numpy(),torque=robot.data.applied_torque.detach().cpu().numpy(),
        dem=np.asarray(history[-4:]),dem_stamps=np.asarray(stamps[-4:],dtype=np.int64),
        route=np.asarray([route.index,route.laps,route.count]),route_start=route.start,route_completed=np.asarray(route.completed_m),
        path=np.asarray(paths) if paths is not None else np.empty((0,2)),
        torch_rng=torch.get_rng_state().cpu().numpy(),latency=np.empty(0,dtype=np.float32) if latency is None else latency.detach().cpu().numpy())
    result.update(numeric_tree(obs,'actor_observation'))
    result.update(delay_snapshot(bare))
    for key,value in observed.items():
        if isinstance(value,(torch.Tensor,np.ndarray)):result.update(numeric_tree(value,'raw_capture/'+key))
    result['prefix_follower_state']=np.asarray(follower.state() if follower is not None else [],dtype=np.float64)
    def json_bytes(value):return np.frombuffer(json.dumps(value,sort_keys=True,default=lambda a:a.tolist() if hasattr(a,'tolist') else int(a)).encode(),dtype=np.uint8)
    result['python_rng']=json_bytes(random.getstate());result['numpy_rng']=json_bytes(np.random.get_state())
    result['follower_diagnostic']=json_bytes(diag)
    for key in ('eat_nav_initial_rngs',):
        if hasattr(bare,key):result[key]=json_bytes([r.bit_generator.state for r in getattr(bare,key)])
    if torch.cuda.is_available():
        for i,v in enumerate(torch.cuda.get_rng_state_all()):result['cuda_rng_'+str(i)]=v.cpu().numpy()
    if any(v.dtype.hasobject for v in result.values()):raise ValueError('object array forbidden')
    return result

def main():
    if args.output.exists():raise FileExistsError('fresh output required')
    args.output.mkdir(parents=True)
    env=override=capture=follower=stream=prefix_stream=None
    snapshots={};active_step=-1;history=[];stamps=[];traces=[];selections=[]
    try:
        torch.set_num_threads(1)
        config=load_eat_config(args.eat_config)
        saved=yaml.load(args.actor_env_yaml.read_text(),Loader=yaml.BaseLoader)
        toggles={n:saved[n].lower()=='true' for n in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
        cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),config,args.seed,1)
        cfg.episode_length_s=120.;configure_demo_reset_policy(cfg);configure_initialization(cfg,'front_folded')
        cfg.events.ignore_sprocket_self_collision=None
        generator=cfg.scene.terrain.terrain_generator;original_size=list(generator.size);generator.size=(16.,16.)
        terrain=configure_selected_terrain(generator,cfg.scene.terrain,args.terrain_type,4,1)
        sub=generator.sub_terrains[args.terrain_type]
        pallet=(getattr(sub,'pallet_length',2.4),getattr(sub,'pallet_width',1.2))
        counts=(getattr(sub,'num_pallets_x',0),getattr(sub,'num_pallets_y',0))
        bounds=field_bounds(args.terrain_type,generator.size,4,pallet=pallet,counts=counts)
        cfg.viewer.origin_type='asset_root';cfg.viewer.asset_name='robot';cfg.viewer.env_index=0
        cfg.viewer.eye=(3.5,-3.5,2.8);cfg.viewer.lookat=(0.,0.,.6)
        env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg));bare=env.unwrapped
        if abs(bare.step_dt-.05)>1e-9:raise ValueError('20Hz required')
        env.reset();robot=bare.scene['robot'];reset=bare._reset_idx
        def before_reset(ids):
            for i in (range(1) if ids is None else ids.detach().cpu().tolist()):snapshots[int(i)]=capture_pre_reset_state(bare,int(i),active_step)
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
        window=ui.Window('All fields | fall and stall learning',width=590,height=330)
        with window.frame:
            with ui.VStack():
                title=ui.Label(f'LIVE {args.terrain_type} | {args.phase} | seed {args.seed}',height=30)
                status=ui.Label('Settling...',word_wrap=True,height=70)
                values=ui.Label('85 degree fall labels | no replay',word_wrap=True,height=170)
        for settling in range(20):
            start=time.monotonic();override.set(zeros)
            obs,observed,_=reconcile_command_observation(obs,observed,bare,zeros,latency)
            with torch.inference_mode():action=policy.act_inference(obs)
            latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
            capture.begin_step()
            with torch.inference_mode():obs,_,done,_=env.step(action)
            observed=capture.finish_step(obs)
            if bool(done.any()):raise ValueError('termination during settling')
            if (settling+1)%5==0:
                stamps.append((settling+1)*STEP_NS);history.append(capture_dem(bare,config,0,stamps[-1]))
            time.sleep(max(0.,.05-(time.monotonic()-start)))
        initial=np.asarray(pose_world(robot,0));origin=bare.scene.env_origins[0,:2].detach().cpu().numpy()
        route=Route(bounds['tile_center'],initial[:2]);bound=bounds['root_bounds']
        if not inside(initial[:2],bound):raise ValueError('spawn outside root range')
        # Confirm the geometry-derived 2m margin covers body/support extents.
        body_xy=robot.data.body_pos_w[0,:,:2].detach().cpu().numpy()
        radius=float(np.max(np.linalg.norm(body_xy-initial[:2],axis=1)))+.5
        if radius>2:raise ValueError('2m margin insufficient for body extent')
        metadata=getattr(generator,'eat_nav_tile_metadata',getattr(bare.cfg.scene.terrain.terrain_generator,'eat_nav_tile_metadata',[]))
        write(args.output/'terrain.json',dict(terrain=terrain,bounds=bounds,original_size=original_size,size=list(generator.size),
            selected_tile_metadata=metadata,subterrain_config=str(sub.to_dict()),env_origins=bare.scene.env_origins.detach().cpu().tolist(),body_margin_radius=radius))
        write(args.output/'input-manifest.json',dict(schema='eatnav_all_fields_sim_v1',phase=args.phase,condition=args.condition,
            terrain=args.terrain_type,level=4,seed=args.seed,anchor_seconds=args.anchor_seconds,candidate=args.candidate,
            actor_sha256=sha(args.rl_checkpoint),actor_env_sha256=sha(args.actor_env_yaml),checkpoint_sha256=sha(args.checkpoint),
            controller='append03-recreate-follower-v1',fall_threshold_deg=85.,headless=False,reserved_test_opened=False,r4_config=CONFIG))
        evaluator=None
        guard=R4Guard()
        if args.phase=='evaluate':
            if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('checkpoint binding')
            model,payload=load_checkpoint(args.checkpoint);evaluator=dict(payload,model=model)
            active=payload['support']['active'];needed=[2,3]
            if not all(active[i] for i in needed):raise ValueError('unsupported requested heads')
        prefix_steps=args.anchor_seconds*20 if args.phase=='collect' else 0
        limit=prefix_steps+(200 if args.phase=='collect' else 1200)
        paths=None;heading=0.;append_count=0;k=None;diagnostic={};selection={};anchored=False;stop_reason='horizon_complete'
        stream=(args.output/'trace.jsonl').open('x');prefix_stream=(args.output/'prefix-trace.jsonl').open('x')
        for step in range(limit):
            if not simulation_app.is_running():stop_reason='window_closed';break
            started=time.monotonic();active_step=step;stamp=1_000_000_000+step*STEP_NS
            assert_finite_sim_state(robot)
            pose=np.asarray(pose_world(robot,0));is_prefix=step<prefix_steps;relative=step-prefix_steps
            if step==prefix_steps:
                # Capture state BEFORE branching/follower reset or command injection.
                anchor=anchor_state(robot,obs,latency,bare,history,stamps,route,paths,diagnostic,observed,follower)
                arrays=pack_inputs(np.asarray(history[-4:]),observed['policy_pre_noise'][0].detach().cpu().numpy(),body_state(robot.data.root_state_w[0].detach().cpu().numpy()))
                np.savez_compressed(args.output/'anchor-state.npz',**anchor)
                np.savez_compressed(args.output/'initial-input.npz',**arrays)
                np.savez_compressed(args.output/'initial-raw-dem.npz',**capture_raw_dem(bare))
                write(args.output/'anchor.json',dict(root=robot.data.root_state_w[0].detach().cpu().tolist(),dem_stamps_ns=stamps[-4:],route=dict(index=route.index,count=route.count,laps=route.laps)))
                anchored=True
                if follower is not None:follower.close();follower=None
            measured=tilt(robot.data.root_state_w[0].detach().cpu().tolist())
            before=guard.snapshot()
            if args.phase=='evaluate':
                was_stopped=guard.stopped;guard.observe_tilt(measured,relative)
                if guard.stopped and not was_stopped and follower is not None:follower.close();follower=None
            if step%40==0:
                validate_history(stamps[-4:],stamp)
                arrays=pack_inputs(np.asarray(history[-4:]),observed['policy_pre_noise'][0].detach().cpu().numpy(),body_state(robot.data.root_state_w[0].detach().cpu().numpy()))
                goal_body=rotation(pose[2]).T@(route.goal-pose[:2])
                if is_prefix:k=geometry_choice(goal_body)
                elif args.phase=='collect':k=args.candidate
                else:
                    probability,progress=predict(evaluator,arrays,[0]);probability=probability[0];progress=progress[0]
                    selection=guard.decide(probability,progress,goal_body,relative,measured);k=selection['selected_index']
                previews=[np.asarray(body_path_to_tile_local(p,tuple(pose),[0.,0.])) for p in candidate_paths()]
                paths=None if k is None else previews[k].copy();heading=pose[2]+(ANGLES[k] if k is not None else 0);append_count=0
                if follower is not None:follower.close()
                follower=None if k is None else PureFollower((paths-origin).tolist(),library=library)
                filename=('prefix-' if is_prefix else '')+f'input-{step:04d}.npz'
                np.savez_compressed(args.output/filename,**arrays)
                np.savez_compressed(args.output/(('prefix-' if is_prefix else '')+f'raw-dem-{step:04d}.npz'),**capture_raw_dem(bare))
                record=dict(control_step=step,relative_step=relative,candidate=k,prefix=is_prefix,input_file=filename,
                    dem_stamps_ns=stamps[-4:],goal_body=goal_body.tolist(),selection=selection,pose=pose.tolist())
                write(args.output/(('prefix-' if is_prefix else '')+f'selection-{step:04d}.json'),record);selections.append(record)
                draw_paths(bare,previews,-1 if k is None else k,float(robot.data.root_pos_w[0,2])+.15,route.goal)
                window.visible=True
                if step==0:window.dock_in_window('Stage',ui.DockPosition.TOP,.45)
                window.focus()
                print('SELECTION '+json.dumps(record,allow_nan=False),flush=True)
            append_events=[]
            if follower is not None and not (args.phase=='evaluate' and guard.stopped):
                while needs_extension(paths,pose):
                    if append_count>=4:
                        override.set(zeros);stop_reason='control_error';break
                    paths,heading=extend_path(paths,heading,k);append_count+=1
                    follower.close();follower=PureFollower((paths-origin).tolist(),library=library)
                    append_events.append(dict(count=append_count,terminal_heading=heading,path=paths.tolist()))
            if stop_reason=='control_error':break
            if args.phase=='evaluate':command,diagnostic=guarded_command(guard,follower,(float(pose[0]-origin[0]),float(pose[1]-origin[1]),float(pose[2])),stamp)
            else:
                command,diagnostic=follower.step_diagnostic(float(pose[0]-origin[0]),float(pose[1]-origin[1]),float(pose[2]),stamp)
            root_pre=robot.data.root_state_w[0].detach().cpu().tolist()
            injected=torch.tensor([command],dtype=zeros.dtype,device=bare.device);override.set(injected)
            obs,observed,detail=reconcile_command_observation(obs,observed,bare,injected,latency)
            with torch.inference_mode():action=policy.act_inference(obs)
            latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
            capture.begin_step()
            with torch.inference_mode():obs,_,done,_=env.step(action)
            observed=capture.finish_step(obs);terminal=snapshots.pop(0,None)
            post=terminal['root_state'] if terminal else robot.data.root_state_w[0].detach().cpu().tolist()
            boundary=not inside(post[:2],bound);failed=bool(done.any()) or boundary
            wp=route.observe(post[:2],tilt(post),args.phase=='evaluate' and guard.stopped,failed)
            row=dict(control_step=step if is_prefix else relative,logical_sim_time_ns=stamp,prefix=is_prefix,
                candidate_id=CANDIDATE_IDS[k] if k is not None else 'STOP',candidate_cycle=(step if is_prefix else relative)//40,
                candidate_replan=bool((step if is_prefix else relative) and (step if is_prefix else relative)%40==0),
                root_state=root_pre,post_root_state=post,body_angular_velocity_radps=body_vector(post,post[10:13]),
                terminal=None if terminal is None else dict(pre_reset_state=terminal,termination_reasons=terminal['termination_reasons']),
                follower_command=list(command),actor_command=detail['actor_seen'][0],follower_diagnostic=diagnostic,
                append_events=append_events,path_id=f'{step//40}:{append_count}',guard_stopped=args.phase=='evaluate' and guard.stopped,
                guard_state_before_step=before,guard_state_after_step=guard.snapshot(),waypoint=wp,route_progress_m=route.progress(post[:2]),
                boundary_censor=boundary)
            (prefix_stream if is_prefix else stream).write(json.dumps(row,allow_nan=False)+'\n');(prefix_stream if is_prefix else stream).flush()
            if not is_prefix:traces.append(row)
            status.text=f'LIVE {args.phase} {args.condition if args.phase=="evaluate" else CANDIDATE_IDS[k]} | t={(step+1)/20:.2f}s | waypoint={route.count} lap={route.laps} | tilt={tilt(post):.1f}'
            values.text=f'prefix={is_prefix} | append={append_count} | STOP={row["guard_stopped"]}\ncommand={list(command)}\ngoal={route.goal.round(2).tolist()}'
            if args.phase=='evaluate' and selection:
                values.text+='\nmodel='+args.checkpoint_sha256[:12]+' | '+args.condition+' (applied risk)'
                values.text+='\n'+'\n'.join(f'{name}: risk={np.asarray(selection["probabilities"])[j].round(3).tolist()} prog10={selection["progress_m"][j][3]:.3f}' for j,name in enumerate(CANDIDATE_IDS))
            if bool(done.any()):stop_reason='anchor_unavailable' if is_prefix else 'natural_termination';break
            if boundary:stop_reason='anchor_unavailable' if is_prefix else 'boundary_censor';break
            if (step+1)%5==0:
                stamps.append(stamp+STEP_NS);history.append(capture_dem(bare,config,0,stamps[-1]))
            time.sleep(max(0.,.05-(time.monotonic()-started)))
        if not anchored and stop_reason=='window_closed':stop_reason='anchor_unavailable_window_closed'
        summary=dict(state='completed',phase=args.phase,condition=args.condition,terrain=args.terrain_type,seed=args.seed,
            anchor_seconds=args.anchor_seconds,candidate=args.candidate,anchor_available=anchored,stop_reason=stop_reason,
            control_steps=len(traces),normal_waypoints=route.count,laps=route.laps,route_progress_m=route.progress(traces[-1]['post_root_state'][:2]) if traces else 0,
            max_tilt_deg=max((tilt(r['post_root_state']) for r in traces),default=0),
            final_tilt_deg=tilt(traces[-1]['post_root_state']) if traces else None,
            stop_steps=sum(r['guard_stopped'] for r in traces),append_events=sum(len(r['append_events']) for r in traces),
            gui=True,reserved_test_opened=False,model_updated=False)
        write(args.output/'complete.json',summary);print('COMPLETE '+json.dumps(summary),flush=True)
        override.set(zeros);bare.sim.pause();end=time.monotonic()+args.hold_seconds
        while simulation_app.is_running() and time.monotonic()<end:simulation_app.update();time.sleep(.03)
    except BaseException as exc:
        if override is not None:override.set(zeros)
        write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for s in (stream,prefix_stream):
            if s is not None:s.close()
        if follower is not None:follower.close()
        if capture is not None:capture.close()
        if override is not None:override.close()
        if env is not None:env.close()
        seal_derived(args.output)

if __name__=='__main__':
    try:main()
    finally:simulation_app.close()
