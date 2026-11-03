"""Exploratory matched A/B closed-loop selector; host supplies actual-LIO K14 decisions."""
from __future__ import annotations
import argparse,json,time,traceback,hashlib,copy,sys,os
# Host owns private (0700) IPC parent; allow it to write nested container-created prefix directories.
os.umask(0)
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parents',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--rl-checkpoint',type=Path,required=True)
parser.add_argument('--actor-env-yaml',type=Path,required=True)
parser.add_argument('--eat-config',type=Path,required=True)
parser.add_argument('--record-lidar',action='store_true')
parser.add_argument('--decision-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not args.headless:parser.error('headless required')
parents=json.loads(args.parents.read_text())['parents']
if not parents or len({(p['field'],p['level']) for p in parents})!=1:parser.error('one field/level per worker')
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
application=AppLauncher(args).app
import numpy as np
import torch,yaml,gymnasium as gym
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import onix_isaac_lab.tasks
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import make_collection_cfg,EATTerrainGenerator,EATTerrainImporter,reset_root_state_eat
from visualnav_transformer.eat_nav.demo_reset_policy import configure_demo_reset_policy
from visualnav_transformer.eat_nav.mi_fi_v1 import configure_initialization
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride,ObservationCapture
from visualnav_transformer.eat_nav.p1_g3_rosfree_control import reconcile_command_observation
from visualnav_transformer.eat_nav.single_pose2s_contract import ACTOR_SHA,PROFILE,PROFILE_SHA
from visualnav_transformer.eat_nav.single_pose2s_collection import inpaint_frame,acquisition_steps,terrain_clearance_z
from visualnav_transformer.eat_nav.single_pose2s_follower import NominalFollower
from visualnav_transformer.eat_nav.bidirectional_bank import bank
from visualnav_transformer.eat_nav.moving_pose2s_geometry import quat_to_matrix
from eat_nav_r6_collection import audit_physics_properties

if not args.record_lidar:raise ValueError('actual FAST-LIO sensor prefixes required')
N=len(parents);FIELD=parents[0]['field'];LEVEL=parents[0]['level'];SPAWN_MESH={};SPAWN_LOCAL_ORIGINS={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,x):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist())+'\n');tmp.replace(p)


class ParentTerrain(EATTerrainGenerator):
    def _get_terrain_mesh(self,difficulty,cfg):
        index=self._tile_index;self._tile_index+=1;column,row=divmod(index,5);parent=parents[column]
        seed=int(np.random.SeedSequence([parent['seed'],row,813]).generate_state(1)[0]);rng=np.random.default_rng(seed)
        lo,hi=self.cfg.difficulty_range;difficulty=lo+(row+float(rng.random()))/5*(hi-lo)
        old_seed=self.cfg.seed;old_state=np.random.get_state();self.cfg.seed=seed;np.random.seed(seed)
        try:mesh,origin=TerrainGenerator._get_terrain_mesh(self,difficulty,cfg)
        finally:self.cfg.seed=old_seed;np.random.set_state(old_state)
        digest=hashlib.sha256()
        for a in [mesh.vertices,mesh.faces]:
            a=np.ascontiguousarray(a);digest.update(str((a.shape,str(a.dtype))).encode());digest.update(a.tobytes())
        if row==LEVEL:
            SPAWN_MESH[column]=(np.asarray(mesh.vertices).copy(),np.asarray(mesh.faces).copy());SPAWN_LOCAL_ORIGINS[column]=np.asarray(origin).copy()
        self.tile_metadata.append(dict(parent_id=parent['parent_id'],tile_index=index,row=row,column=column,seed=seed,difficulty=difficulty,geometry_sha256=digest.hexdigest()))
        return mesh,origin


class ParentImporter(EATTerrainImporter):
    def _compute_env_origins_curriculum(self,num_envs,origins):
        self.terrain_levels=torch.full((num_envs,),LEVEL,dtype=torch.long,device=self.device)
        self.terrain_types=torch.arange(num_envs,device=self.device);self.max_terrain_level=5
        return origins[self.terrain_levels,self.terrain_types].clone()


def reset_parent(env,env_ids,pose_range,velocity_range,base_seed):
    if not hasattr(env,'eat_nav_initial_rngs'):env.eat_nav_initial_rngs=[np.random.default_rng(p['initial_seed']) for p in parents]
    result=reset_root_state_eat(env,env_ids,pose_range,velocity_range,base_seed)
    robot=env.scene['robot'];roots=robot.data.root_state_w[env_ids].clone()
    for j,i in enumerate(env_ids.cpu().tolist()):
        origin=env.scene.env_origins[i].cpu().numpy();local=roots[j,:2].cpu().numpy()-origin[:2]
        vertices,faces=SPAWN_MESH[i];roots[j,2]=terrain_clearance_z(vertices,faces,local,float(robot.data.default_root_state[i,2]))
    robot.write_root_state_to_sim(roots,env_ids=env_ids)
    return result


def main():
    begin=time.perf_counter();torch.set_num_threads(1)
    source_hashes={}
    for name,module in sorted(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file and file.endswith('.py') and (name.startswith('visualnav_transformer.eat_nav') or name.startswith('onix_isaac_lab') or name.startswith('isaaclab.')):
            path=Path(file);source_hashes[name]=dict(path=str(path),sha256=sha(path))
    runtime_identity=dict(worker_sha256=sha(__file__),actor_sha256=sha(args.rl_checkpoint),actor_env_sha256=sha(args.actor_env_yaml),eat_config_sha256=sha(args.eat_config),module_imports=source_hashes)
    runtime_identity['late_sensor_sources']={name:sha(Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav'/name) for name in ['single_pose2s_lidar_sim.py','single_pose2s_fastlio.py']}
    canonical=dict(runtime_identity);canonical['module_imports']={name:item['sha256'] for name,item in source_hashes.items()}
    runtime_identity_sha256=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
    runtime_identity['canonical_sha256']=runtime_identity_sha256
    write(args.output/'runtime-imports.json',runtime_identity)
    runtime_imports_sha256=sha(args.output/'runtime-imports.json')

    if PROFILE.get('observation_profile')=='single_grid_actual_fastlio_pose_velocity_v2' and not args.record_lidar:raise ValueError('actual FAST-LIO profile requires LiDAR/IMU recording')
    if sha(args.rl_checkpoint)!=ACTOR_SHA:raise ValueError('Actor identity mismatch')
    cfg0=yaml.load(args.actor_env_yaml.read_text(),Loader=yaml.BaseLoader)
    toggles={name:cfg0[name].lower()=='true' for name in ['use_last_action_obs','use_joint_effort_obs','use_height_scan_obs']}
    cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),load_eat_config(args.eat_config),123,N)
    configure_demo_reset_policy(cfg);configure_initialization(cfg,'front_folded')
    cfg.episode_length_s=120.;cfg.events.push_robot=None;cfg.events.ignore_sprocket_self_collision=None
    cfg.events.reset_base.func=reset_parent
    cfg.scene.terrain.class_type=ParentImporter
    generator=cfg.scene.terrain.terrain_generator
    generator.size=(16.,16.);generator.class_type=ParentTerrain;generator.num_rows=5;generator.num_cols=N;generator.curriculum=True;generator.use_cache=False
    generator.sub_terrains={FIELD:generator.sub_terrains[FIELD]};generator.sub_terrains[FIELD].proportion=1.
    scanner=cfg.scene.height_scanner_eat
    scanner.pattern_cfg.resolution=.1;scanner.pattern_cfg.size=[3.9,3.9];scanner.pattern_cfg.ordering='xy';scanner.ray_alignment='yaw';scanner.offset.pos=(0.,0.,2.);scanner.update_period=0.
    # Sampling is explicitly scheduled below at4Hz. Internal ray refresh can be
    # requested at the acquisition tick; unused rays never become NN observations.
    env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg));bare=env.unwrapped;robot=bare.scene['robot']
    if abs(bare.step_dt-.05)>1e-9 or abs(bare.physics_dt-.005)>1e-9:raise ValueError('physics/control mismatch')
    env.reset();audit=audit_physics_properties(robot.root_physx_view.get_masses().cpu().numpy(),robot.root_physx_view.get_inertias().cpu().numpy())
    if not audit['valid']:raise ValueError('nonfinite/invalid physical properties')
    np.savez_compressed(args.output/'actual-physics-properties.npz',masses_kg=robot.root_physx_view.get_masses().cpu().numpy(),inertias_kg_m2=robot.root_physx_view.get_inertias().cpu().numpy(),material_properties=robot.root_physx_view.get_material_properties().cpu().numpy())
    runner=OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device);policy=runner.alg.policy
    policy.load_state_dict(torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict']);policy.eval()
    if getattr(policy,'is_recurrent',False):raise ValueError('Actor hidden state unsupported')
    override=CommandOverride(bare.command_manager,N);capture=ObservationCapture(bare.observation_manager)
    override.set(torch.zeros((N,3),device=bare.device));latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
    capture.begin_step();obs=bare._apply_policy_observation_latency(bare.observation_manager.compute());observed=capture.finish_step(obs)
    fnames=['rr','rl','fr','fl'];joint_names=robot.joint_names
    fids=[next(j for j,name in enumerate(joint_names) if side+'_flipper_joint' in name) for side in fnames]
    right=[j for j,name in enumerate(joint_names) if name.startswith('chassis_r') and 'sprocket' in name]
    left=[j for j,name in enumerate(joint_names) if name.startswith('chassis_l') and 'sprocket' in name]
    if not right or not left:raise ValueError('track mapping unavailable')
    root_history=[];state_history=[];joint_pos=[];joint_vel=[];effort=[];commands=[];actions=[];actor_obs=[]
    alive=np.ones(N,bool);alive_history=[alive.copy()];terminal={};active_step=-1;original_reset=bare._reset_idx
    def before_reset(ids):
        for i in ids.detach().cpu().tolist():
            if alive[i]:terminal[i]=dict(step=active_step+1,root=robot.data.root_state_w[i].cpu().numpy().copy(),root_link=robot.data.root_link_state_w[i].cpu().numpy().copy(),joint_position=robot.data.joint_pos[i].cpu().numpy().copy(),joint_velocity=robot.data.joint_vel[i].cpu().numpy().copy(),effort=robot.data.applied_torque[i].cpu().numpy().copy(),gravity=robot.data.projected_gravity_b[i].cpu().numpy().copy(),gyro=robot.data.root_ang_vel_b[i].cpu().numpy().copy(),reasons=[name for name in bare.termination_manager.active_terms if bool(bare.termination_manager.get_term(name)[i])])
        return original_reset(ids)
    bare._reset_idx=before_reset
    frames=[[] for _ in parents];frame_stamps=[[] for _ in parents];frame_roots=[[] for _ in parents];frame_steps=[set(acquisition_steps(p['dem_phase_steps'])) for p in parents]
    timings=dict(env_step=0.,actor=0.,sensor=0.,save=0.);followers=[None]*N
    decisions=[];goals=None;reached=np.zeros(N,bool);goal_step=np.full(N,-1,int);filter_stop=np.zeros(N,bool)
    def capture_state():
        roots=robot.data.root_state_w.cpu().numpy().copy();q=robot.data.joint_pos.cpu().numpy().copy();v=robot.data.joint_vel.cpu().numpy().copy();torque=robot.data.applied_torque.cpu().numpy().copy()
        state=np.zeros((N,27),np.float32)
        state[:,:3]=robot.data.projected_gravity_b.cpu().numpy();state[:,3:6]=robot.data.root_ang_vel_b.cpu().numpy()
        state[:,6:10]=q[:,fids];state[:,10:14]=v[:,fids]
        rv=v[:,right].mean(1);lv=v[:,left].mean(1);state[:,14:18]=np.stack([rv,rv,lv,lv],1)
        # Virtual wheel/IMU odometry, not future or true root linear velocity.
        state[:,18]=.0703*(rv+lv)/2;state[:,23]=state[:,5];state[:,25]=1.
        root_history.append(roots);state_history.append(state);joint_pos.append(q);joint_vel.append(v);effort.append(torque)
        return roots
    lio_sensors=None
    if args.record_lidar:
        from visualnav_transformer.eat_nav.single_pose2s_lidar_sim import SimLioSensors
        from visualnav_transformer.eat_nav.single_pose2s_fastlio import mesh_placement_translation
        mesh_translation=mesh_placement_translation(bare.scene.env_origins.cpu().numpy(),np.stack([SPAWN_LOCAL_ORIGINS[i] for i in range(N)]))
        lio_sensors=SimLioSensors(SPAWN_MESH,mesh_translation,str(bare.device),N)
        # 2s initial settling then 2s of actual sensor initialization. No reset
        # is inserted at trial t0 or any of its five two-second boundaries.
        for warm in range(80):
            override.set(torch.zeros((N,3),device=bare.device))
            obs,observed,_=reconcile_command_observation(obs,observed,bare,np.zeros((N,3),np.float32),latency)
            with torch.inference_mode():warm_action=policy.act_inference(obs)
            latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
            capture.begin_step()
            with torch.inference_mode():obs,_,_,_=env.step(warm_action)
            observed=capture.finish_step(obs)
            if warm>=39:lio_sensors.capture((warm-79)*50000000,robot.data.root_link_state_w.cpu().numpy().copy())
        terminal.clear();alive[:]=True;active_step=-1
    roots=capture_state()
    yaw0=np.array([np.arctan2(quat_to_matrix(r[3:7])[1,0],quat_to_matrix(r[3:7])[0,0]) for r in roots])
    goals=roots[:,:2]+3*np.stack([np.cos(yaw0),np.sin(yaw0)],1)
    write(args.output/'initial-selector-state.json',dict(root=roots,joint_position=joint_pos[0],joint_velocity=joint_vel[0],goals=goals,parents=parents))
    args.decision_dir.mkdir(parents=True,exist_ok=True)
    for step in range(201):
        active_step=step
        if lio_sensors is not None and step>0:
            imu_roots=robot.data.root_link_state_w.cpu().numpy().copy()
            for i,event in terminal.items():
                if event['step']==step:imu_roots[i]=event['root_link']
            lio_sensors.capture(step*50000000,imu_roots)
        started=time.perf_counter()
        need=[i for i in range(N) if alive[i] and step in frame_steps[i]]
        if need:
            sensor=bare.scene['height_scanner_eat'];hit=sensor.data.ray_hits_w.detach().cpu().numpy()[:,:,2]
            if hit.shape!=(N,1600):raise ValueError('ray cell layout mismatch')
            if lio_sensors is not None and step==0:
                alignment=lio_sensors.verify_dem_alignment(roots,sensor.data.ray_hits_w.cpu().numpy())
                write(args.output/'lidar-dem-alignment.json',alignment)
                if not alignment['passed']:raise ValueError('LiDAR mesh and physical DEM geometry disagree')
            for i in need:
                h=hit[i].reshape(40,40);valid=np.isfinite(h)
                # A reproducible coverage sensitivity profile, not a real LiDAR model.
                rng=np.random.default_rng(np.random.SeedSequence([parents[i]['seed'],step,81]))
                raw=valid.copy() if parents[i]['trial_index']%2==0 else valid&(rng.random((40,40))<.56)
                if raw.any():
                    frames[i].append(inpaint_frame(h,raw,roots[i,2]));frame_stamps[i].append(step*50000000);frame_roots[i].append(roots[i].copy())
        timings['sensor']+=time.perf_counter()-started
        if step==200:break
        newly=alive&~reached&(np.linalg.norm(roots[:,:2]-goals,axis=1)<=.2)
        reached[newly]=True;goal_step[newly]=step
        if step%40==0:
            requests=[]
            for i in range(N):
                if not alive[i] or reached[i]:continue
                parent=parents[i];directory=args.decision_dir/f"step{step:03d}"/parent['parent_id'];directory.mkdir(parents=True)
                np.savez_compressed(directory/'prefix.npz',stamps_ns=np.arange(step+1,dtype=np.int64)*50000000,root=np.asarray(root_history)[:,i],state=np.asarray(state_history)[:,i],state_mask=np.ones((step+1,27),np.float32),state_age=np.zeros((step+1,27),np.float32),dem=np.asarray(frames[i]),dem_stamp_ns=np.asarray(frame_stamps[i],np.int64),dem_root=np.asarray(frame_roots[i]))
                lio_sensors.save(directory/'lio-sensors.npz',i)
                write(directory/'metadata.json',dict(parent=dict(parent,parent_id=parent['parent_id']+f'-d{step:03d}'),terminal=None,lio_sensor_sha256=sha(directory/'lio-sensors.npz')))
                requests.append(dict(index=i,parent_id=parent['parent_id'],directory=str(directory),goal_xy=goals[i],condition=parent['condition']))
            request=args.decision_dir/f'request-step{step:03d}.json';response=request.with_name(f'response-step{step:03d}.json')
            write(request,dict(step=step,requests=requests))
            until=time.monotonic()+1200
            while not response.exists():
                if time.monotonic()>until:raise TimeoutError('host actual-LIO decision timeout')
                time.sleep(.1)
            answer=json.loads(response.read_text())
            if answer.get('error'):raise RuntimeError(answer['error'])
            filter_stop[:]=False
            for item in answer['decisions']:
                i=item['index'];cid=item['candidate_id'];decisions.append(dict(step=step,parent_id=parents[i]['parent_id'],**item))
                if cid is None:followers[i]=None;filter_stop[i]=True;parents[i]['candidates'][step//40]=-1;continue
                parents[i]['candidates'][step//40]=cid
                R=quat_to_matrix(roots[i,3:7]);yaw=np.arctan2(R[1,0],R[0,0]);c,s=np.cos(yaw),np.sin(yaw)
                path=bank()[cid].waypoints@np.array([[c,s],[-s,c]])+roots[i,:2]
                followers[i]=NominalFollower(path)
        cmd=np.zeros((N,3),np.float32)
        for i in range(N):
            if alive[i] and not reached[i] and followers[i] is not None:
                R=quat_to_matrix(roots[i,3:7]);cmd[i]=followers[i].step(roots[i,0],roots[i,1],np.arctan2(R[1,0],R[0,0]))
        started=time.perf_counter();override.set(cmd)
        obs,observed,detail=reconcile_command_observation(obs,observed,bare,cmd,latency)
        with torch.inference_mode():action=policy.act_inference(obs)
        seen=obs['policy'].detach().cpu().numpy()
        if seen.shape!=(N,22) or action.shape!=(N,6):raise ValueError('Actor schema changed')
        commands.append(cmd.copy());actions.append(action.cpu().numpy().copy());actor_obs.append(seen.copy());timings['actor']+=time.perf_counter()-started
        latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
        capture.begin_step();started=time.perf_counter()
        with torch.inference_mode():obs,_,done,_=env.step(action)
        timings['env_step']+=time.perf_counter()-started;observed=capture.finish_step(obs)
        new_roots=capture_state();done_np=done.cpu().numpy().astype(bool)
        for i in range(N):
            if alive[i] and done_np[i]:
                if i not in terminal:raise ValueError('pre-reset root not captured')
                event=terminal[i];root_history[-1][i]=event['root'];joint_pos[-1][i]=event['joint_position'];joint_vel[-1][i]=event['joint_velocity'];effort[-1][i]=event['effort']
                st=state_history[-1][i];st[:3]=event['gravity'];st[3:6]=event['gyro'];st[6:10]=event['joint_position'][fids];st[10:14]=event['joint_velocity'][fids]
                rv=event['joint_velocity'][right].mean();lv=event['joint_velocity'][left].mean();st[14:18]=[rv,rv,lv,lv];st[18]=.0703*(rv+lv)/2;st[23]=st[5]
                alive[i]=False
        alive_history.append(alive.copy());roots=new_roots
    write(args.output/'selector-decisions.json',decisions)
    write(args.output/'selector-goals.json',dict(goals=goals,goal_step=goal_step,condition=parents[0]['condition']))
    started=time.perf_counter()
    roots=np.asarray(root_history);states=np.asarray(state_history);alive_array=np.asarray(alive_history)
    metadata=getattr(generator,'eat_nav_tile_metadata',[])
    for i,parent in enumerate(parents):
        directory=args.output/parent['parent_id'];directory.mkdir()
        tile=next(x for x in metadata if x['column']==i and x['row']==LEVEL)
        np.savez_compressed(directory/'raw.npz',stamps_ns=np.arange(201,dtype=np.int64)*50000000,root=roots[:,i],state=states[:,i],state_mask=np.tile(np.array([1]*20+[0,0,0,1,1,1,1],np.float32),(201,1)),state_age=np.zeros((201,27),np.float32),alive=alive_array[:,i],joint_position=np.asarray(joint_pos)[:,i],joint_velocity=np.asarray(joint_vel)[:,i],applied_effort=np.asarray(effort)[:,i],command=np.asarray(commands)[:,i],action=np.asarray(actions)[:,i],actor_input=np.asarray(actor_obs)[:,i],dem=np.asarray(frames[i]),dem_stamp_ns=np.asarray(frame_stamps[i],np.int64),dem_root=np.asarray(frame_roots[i]))
        if lio_sensors is not None:lio_sensors.save(directory/'lio-sensors.npz',i)
        t=terminal.get(i);terminal_json=None if t is None else dict(step=t['step'],reasons=t['reasons'])
        write(directory/'metadata.json',dict(schema='eatnav_single_dem_pose2s_raw_v1',runtime_imports_file='../runtime-imports.json',runtime_imports_sha256=runtime_imports_sha256,runtime_identity_sha256=runtime_identity_sha256,parent=parent,profile=PROFILE,profile_sha256=PROFILE_SHA,terrain=tile,env_origin_w=bare.scene.env_origins[i].cpu().numpy(),initialization='seededxyyaw; root height uses max local terrain support + legacy .35m clearance; no in-trial reset',terminal=terminal_json,terminal_row_is_pre_reset=True,post_terminal_rows_excluded_from_trial=True,joint_names=joint_names,flipper_indices=fids,right_track_indices=right,left_track_indices=left,physics_properties_file='../actual-physics-properties.npz',physics_environment_index=i,observation_profile=dict(imu='ideal simulated IMU',odom='raw wheel snapshot only; NOT NN input. Actual FAST-LIO sidecar supplies all NN odom components' if args.record_lidar else 'wheel radius .0703m plus IMU yaw gyro; planar',dem='ideal geometric rays or 56% independent coverage sensitivity then nearest observed inpaint; not calibrated real sensor',root_state='teacher and frame transforms only'),actor_sha256=sha(args.rl_checkpoint),actor_env_sha256=sha(args.actor_env_yaml),raw_sha256=sha(directory/'raw.npz'),lio_sensor_file='lio-sensors.npz' if args.record_lidar else None,lio_sensor_sha256=sha(directory/'lio-sensors.npz') if args.record_lidar else None,mesh_local_origin=SPAWN_LOCAL_ORIGINS[i],mesh_world_translation=bare.scene.env_origins[i].cpu().numpy()-SPAWN_LOCAL_ORIGINS[i],lio_sensor_profile='ideal_terrain_16x512_10hz_imu20hz_collocated_root_mesh_placement_v2' if args.record_lidar else None,settling_seconds=2 if args.record_lidar else 0,lio_initialization_seconds=2 if args.record_lidar else 0))
    timings['save']=time.perf_counter()-started
    elapsed=time.perf_counter()-begin
    valid=sum(bool(alive_array[s:s+40,i].all()) for i in range(N) for s in range(0,200,40))
    write(args.output/'complete.json',dict(state='completed',parents=N,planned_windows=N*5,valid_windows=valid,elapsed_seconds=elapsed,timings=timings,window_per_wall_second=valid/elapsed,sim_seconds_per_wall_second=10*N/elapsed,physics_audit=audit,peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),headless=True,profile_sha256=PROFILE_SHA))
    capture.close();override.close();env.close()

try:main()
except Exception as exc:
    write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
finally:application.close()
