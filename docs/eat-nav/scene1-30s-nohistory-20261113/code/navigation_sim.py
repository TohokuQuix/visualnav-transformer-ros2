"""Scene1 ten isolated trials; model2000 NN selector, existingActor, actual traces."""
from __future__ import annotations
import argparse,json,time,traceback,hashlib,copy,sys
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parents',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--rl-checkpoint',type=Path,required=True)
parser.add_argument('--actor-env-yaml',type=Path,required=True)
parser.add_argument('--eat-config',type=Path,required=True)
parser.add_argument('--mesh',type=Path,required=True)
parser.add_argument('--protocol',type=Path,required=True)
parser.add_argument('--max-seconds',type=float,default=300.)
parser.add_argument('--nn-checkpoint',type=Path,required=True)
parser.add_argument('--reference-physics',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not args.headless:parser.error('headless required')
parents=json.loads(args.parents.read_text())['parents']
if not parents:parser.error('nonempty parents required')
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
application=AppLauncher(args).app
import numpy as np
import torch,yaml,gymnasium as gym,trimesh
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
from visualnav_transformer.eat_nav.model2000_policy import apply_bundle

N=len(parents);PROTOCOL=json.loads(args.protocol.read_text());M=np.load(args.mesh);OFFSETS=np.c_[np.arange(N)*12.,np.zeros(N),np.zeros(N)];TRIAL_STEPS=round(args.max_seconds/.05)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,x):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist())+'\n');tmp.replace(p)


def ground_height(xy):
    h=M['vertices'][:225*347,2].reshape(225,347);x,y=np.asarray(xy,float);u=(x+1)/.02;v=(y+1)/.02;j=int(np.floor(u));i=int(np.floor(v));u-=j;v-=i
    if not(0<=i<224 and 0<=j<346):raise ValueError('spawn footprint outside original terrain')
    return float((1-u)*h[i,j]+(u-v)*h[i,j+1]+v*h[i+1,j+1]) if u>=v else float((1-v)*h[i,j]+(v-u)*h[i+1,j]+u*h[i+1,j+1])
def spawn_z():
    yaw=PROTOCOL.get('placement_spawn_yaw_rad',PROTOCOL['start_yaw_rad']);c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s],[s,c]])
    samples=[ground_height(np.asarray(PROTOCOL.get('placement_spawn_xy',PROTOCOL['start_xy']))+R@np.array([x,y])) for x in np.linspace(-.7,.7,9) for y in np.linspace(-.35,.35,7)]
    return max(samples)+.35
SPAWN_Z=spawn_z()
class ParentTerrain:
    def __init__(self,cfg,device):
        meshes=[trimesh.Trimesh(vertices=M['vertices']+offset,faces=M['faces'],process=False) for offset in OFFSETS]
        self.terrain_mesh=trimesh.util.concatenate(meshes);self.terrain_origins=OFFSETS[None].copy();self.flat_patches={}
        cfg.eat_nav_tile_metadata=[dict(trial_id=p['trial_id'],origin=OFFSETS[i].tolist(),local_mesh_sha256=sha(args.mesh),translation_only=True) for i,p in enumerate(parents)]
class ParentImporter(EATTerrainImporter):
    def _compute_env_origins_curriculum(self,num_envs,origins):
        self.terrain_levels=torch.zeros(num_envs,dtype=torch.long,device=self.device);self.terrain_types=torch.arange(num_envs,device=self.device);self.max_terrain_level=1
        return origins[0,self.terrain_types].clone()
    def __init__(self,cfg):
        super().__init__(cfg)
        import omni.usd
        from pxr import UsdGeom,UsdPhysics
        prim=omni.usd.get_context().get_stage().GetPrimAtPath(cfg.prim_path+'/terrain/mesh')
        UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Set('none');UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
def reset_parent(env,env_ids,pose_range,velocity_range,base_seed):
    robot=env.scene['robot'];roots=robot.data.default_root_state[env_ids].clone();yaw=PROTOCOL.get('placement_spawn_yaw_rad',PROTOCOL['start_yaw_rad'])
    roots[:,:3]=env.scene.env_origins[env_ids]+torch.tensor([*PROTOCOL.get('placement_spawn_xy',PROTOCOL['start_xy']),SPAWN_Z],device=roots.device)
    roots[:,3:7]=torch.tensor([np.cos(yaw/2),0,0,np.sin(yaw/2)],device=roots.device);roots[:,7:]=0
    robot.write_root_state_to_sim(roots,env_ids=env_ids)

from visualnav_transformer.eat_nav.single_pose2s_model import SinglePose2s,prepare
from visualnav_transformer.eat_nav.moving_pose2s_geometry import sixd_to_matrix
from visualnav_transformer.eat_nav.single_pose2s_selector import select_candidate
from visualnav_transformer.eat_nav.single_pose2s_collection import causal_dem_context
from pose_contract import state_class,pose_errors
from nav_contract import inside_course,course_mask,outcome,scene_location,initialization_ok
from scipy.spatial.transform import Rotation
@torch.inference_mode()
def main():
    begin=time.perf_counter();torch.set_num_threads(1)
    cfg0=yaml.load(args.actor_env_yaml.read_text(),Loader=yaml.BaseLoader);toggles={name:cfg0[name].lower()=='true' for name in ['use_last_action_obs','use_joint_effort_obs','use_height_scan_obs']}
    cfg=make_collection_cfg(OnixRoughEnvCfg(**toggles),load_eat_config(args.eat_config),123,N);agent=apply_bundle(cfg,args.actor_env_yaml)
    if sha(args.rl_checkpoint)!=ACTOR_SHA or sha(args.nn_checkpoint)!=PROTOCOL['nn_sha256']:raise ValueError('checkpoint identity')
    configure_demo_reset_policy(cfg);configure_initialization(cfg,'front_folded')
    cfg.episode_length_s=10000.;cfg.events.push_robot=None;cfg.events.ignore_sprocket_self_collision=None;cfg.events.reset_base.func=reset_parent
    cfg.latency_randomization_enabled=False;cfg.latency_delay_actions=False;cfg.latency_delay_policy_observations=False
    cfg.scene.terrain.class_type=ParentImporter
    generator=cfg.scene.terrain.terrain_generator;generator.size=(16.,16.);generator.class_type=ParentTerrain;generator.num_rows=1;generator.num_cols=N;generator.curriculum=True;cfg.scene.terrain.max_init_terrain_level=0;generator.use_cache=False
    scanner=cfg.scene.height_scanner_eat;scanner.pattern_cfg.resolution=.1;scanner.pattern_cfg.size=[3.9,3.9];scanner.pattern_cfg.ordering='xy';scanner.ray_alignment='yaw';scanner.offset.pos=(0.,0.,2.);scanner.update_period=0.
    env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg));bare=env.unwrapped;robot=bare.scene['robot'];assert abs(bare.step_dt-.05)<1e-9 and abs(bare.physics_dt-.005)<1e-9
    env.reset();view=robot.root_physx_view
    (args.output/'resolved-env-config.txt').write_text(str(cfg.to_dict()))
    # Identical measured reference for every environment and every concurrency.
    from pose_contract import reference_properties
    ref=np.load(args.reference_physics);props=reference_properties(ref,N)
    for k,getter in [('masses',view.get_masses),('inertias',view.get_inertias),('materials',view.get_material_properties)]:
        if props[k].shape!=tuple(getter().shape):raise ValueError('reference shape mismatch '+k)
    ids=torch.arange(N,dtype=torch.long,device='cpu')
    view.set_masses(torch.as_tensor(props['masses']),ids);view.set_inertias(torch.as_tensor(props['inertias']),ids);view.set_material_properties(torch.as_tensor(props['materials']),ids)
    for k,getter in [('masses',view.get_masses),('inertias',view.get_inertias),('materials',view.get_material_properties)]:
        if not np.allclose(getter().cpu().numpy(),props[k],rtol=1e-6,atol=1e-7):raise ValueError('reference readback mismatch '+k)
    write(args.output/'terrain-tiles.json',bare.scene.terrain.cfg.terrain_generator.eat_nav_tile_metadata)
    audit=audit_physics_properties(view.get_masses().cpu().numpy(),view.get_inertias().cpu().numpy());write(args.output/'physics-audit.json',audit)
    if not audit['valid']:raise ValueError('invalid actual physics')
    np.savez_compressed(args.output/'physics.npz',masses=view.get_masses().cpu().numpy(),inertias=view.get_inertias().cpu().numpy(),materials=view.get_material_properties().cpu().numpy(),velocity_limits=view.get_dof_max_velocities().cpu().numpy(),effort_limits=view.get_dof_max_forces().cpu().numpy())
    rcfg=OnixRoughPPORunnerCfg().to_dict();rcfg.update(agent);runner=OnPolicyRunner(env,rcfg,log_dir=None,device=bare.device);policy=runner.alg.policy;policy.load_state_dict(torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict'],strict=True);policy.eval();assert not policy.is_recurrent
    payload=torch.load(args.nn_checkpoint,map_location='cpu',weights_only=True);assert payload['profile_sha256']==PROFILE_SHA
    model=SinglePose2s().to(bare.device);model.load_state_dict(payload['state_dict'],strict=True);model.eval()
    override=CommandOverride(bare.command_manager,N);capture=ObservationCapture(bare.observation_manager);override.set(np.zeros((N,3),np.float32))
    capture.begin_step();obs=bare._apply_policy_observation_latency(bare.observation_manager.compute());observed=capture.finish_step(obs)
    names=robot.joint_names;fids=[next(j for j,n in enumerate(names) if side+'_flipper_joint' in n) for side in ['rr','rl','fr','fl']];right=[j for j,n in enumerate(names) if n.startswith('chassis_r') and 'sprocket' in n];left=[j for j,n in enumerate(names) if n.startswith('chassis_l') and 'sprocket' in n]
    def snap():
        roots=robot.data.root_state_w.cpu().numpy().copy();q=robot.data.joint_pos.cpu().numpy().copy();v=robot.data.joint_vel.cpu().numpy().copy();st=np.zeros((N,27),np.float32)
        st[:,:3]=robot.data.projected_gravity_b.cpu().numpy();st[:,3:6]=robot.data.root_ang_vel_b.cpu().numpy();st[:,6:10]=q[:,fids];st[:,10:14]=v[:,fids];rv=v[:,right].mean(1);lv=v[:,left].mean(1);st[:,14:18]=np.stack([rv,rv,lv,lv],1);st[:,18:21]=robot.data.root_lin_vel_b.cpu().numpy();st[:,21:24]=robot.data.root_ang_vel_b.cpu().numpy();st[:,25]=1
        return roots,st,q,v,robot.data.applied_torque.cpu().numpy().copy()
    active=np.ones(N,bool)
    def advance(cmd,zero=False):
        nonlocal obs,observed
        override.set(cmd);obs,observed,_=reconcile_command_observation(obs,observed,bare,cmd,None);seen=obs['policy'].cpu().numpy().copy();action=policy.act_inference(obs)
        if zero:action=torch.zeros_like(action);action[:,:4]=torch.tensor([0.,0.,np.pi,np.pi],device=bare.device)
        action[torch.as_tensor(~active,device=bare.device)]=0
        capture.begin_step();obs,_,done,_=env.step(action);observed=capture.finish_step(obs)
        return done.cpu().numpy().astype(bool),action.cpu().numpy().copy(),seen
    warm_trace=[];warm_done=np.zeros(N,bool)
    for _ in range(40):
        d,_,_=advance(np.zeros((N,3),np.float32),True);warm_done|=d;warm_trace.append(snap()[0])
    np.savez_compressed(args.output/'warmup-trace.npz',root=np.asarray(warm_trace),time_s=(np.arange(40)+1)*.05)
    roots,st,q,v,eff=snap();origins=bare.scene.env_origins.cpu().numpy();goals=origins[:,:2]+np.array([PROTOCOL['goal_xy']]*N);yaw=np.array([np.arctan2(quat_to_matrix(r[3:7])[1,0],quat_to_matrix(r[3:7])[0,0]) for r in roots]);yawerr=np.abs(np.arctan2(np.sin(yaw-PROTOCOL['start_yaw_rad']),np.cos(yaw-PROTOCOL['start_yaw_rad'])));xyerr=np.linalg.norm(roots[:,:2]-origins[:,:2]-np.asarray(PROTOCOL['start_xy']),axis=1)
    active=np.array([initialization_ok(xyerr[i],yawerr[i],bool(warm_done[i]),PROTOCOL['initialization_tolerance_xy_m'],PROTOCOL['initialization_tolerance_yaw_deg']) for i in range(N)]) & np.isfinite(roots).all(1) & inside_course(roots[:,:2]-origins[:,:2]);bypass=np.zeros(N,bool);minimum_distance=np.linalg.norm(roots[:,:2]-goals,axis=1);minimum_step=np.zeros(N,int);first_bypass=np.full(N,-1,int);last_steps=np.full(N,TRIAL_STEPS,int);results=[None if active[i] else 'initialization_failure' for i in range(N)];last_steps[~active]=0
    init=[dict(trial_id=p['trial_id'],xy_deviation_m=float(xyerr[i]),yaw_deviation_deg=float(np.rad2deg(yawerr[i])),warm_termination=bool(warm_done[i]),initial_root=roots[i],origin=origins[i],goal_world=goals[i],valid=bool(active[i]),xy_tolerance_m=PROTOCOL['initialization_tolerance_xy_m'],yaw_tolerance_deg=PROTOCOL['initialization_tolerance_yaw_deg']) for i,p in enumerate(parents)]
    write(args.output/'initialization.json',init)
    trial_started=True;current_step=0;terminal={};original_reset=bare._reset_idx
    def before_reset(eids):
        for i in eids.cpu().tolist():
            if active[i]:
                a=snap();terminal[i]=dict(step=current_step+1,root=a[0][i],state=a[1][i],joint_position=a[2][i],joint_velocity=a[3][i],effort=a[4][i],reasons=[name for name in bare.termination_manager.active_terms if bool(bare.termination_manager.get_term(name)[i])])
        return original_reset(eids)
    bare._reset_idx=before_reset
    rh=[roots.copy()];sh=[st.copy()];qh=[q.copy()];vh=[v.copy()];eh=[eff.copy()];commands=[];actions=[];actor_obs=[];dems=[];demsteps=[];demroots=[];contact_history=[];decisions=[];windows=[];pending={};followers=[None]*N;chosen=[None]*N;timing=dict(step=0.,sensor=0.,inference=0.,save=0.);max_gpu=0
    paths=np.array([c.waypoints for c in bank()],np.float32);directions=np.array([[c.direction] for c in bank()],np.float32)
    def final_window(i,end):
        pend=pending.pop(i,None)
        if pend is None:return
        row={k:parents[i][k] for k in ['trial_id','field','level','goal_id']};row.update(t0_s=pend['step']*.05,t1_s=end*.05,candidate_id=pend['candidate'],start_state=pend['start_state'],initial_segment=pend['step']==0,dem_age_s=0.,speed_mps=pend['speed'],angular_speed_radps=pend['angular'],flipper_rr_rl_fr_fl_rad=pend['state'][6:10].tolist(),valid=False)
        if pend['candidate'] is None:row['reason']='input_unavailable' if pend.get('input_failure') else 'all_candidates_filtered_STOP'
        elif end-pend['step']<40:row['reason']='early_'+str(results[i])
        else:
            R0=quat_to_matrix(pend['root'][3:7]);R2=quat_to_matrix(roots[i,3:7]);truthp=R0.T@(roots[i,:3]-pend['root'][:3]);truthR=R0.T@R2;err=pose_errors(pend['p'],pend['R'],truthp,truthR,R0,paths[pend['candidate']]);row.update(err,valid=True,reason=None,predicted_position=pend['p'].tolist(),true_position=truthp.tolist(),true_rotation=truthR.tolist(),q0_wxyz=pend['root'][3:7].tolist())
        windows.append(row);print('WINDOW '+json.dumps(row,default=lambda x:x.tolist(),allow_nan=False),flush=True)
    def dump_trial(i):
        p=parents[i];d=args.output/p['trial_id']
        if (d/'run.json').exists():return
        d.mkdir(exist_ok=True);end=min(int(last_steps[i]),len(commands));local_root=np.stack([a[i] for a in rh[:end+1]]);world_root=local_root.copy();local_root[:,:3]-=origins[i];idx=[j for j,k in enumerate(demsteps) if k<=end]
        np.savez_compressed(d/'trace.npz',time_s=np.arange(end+1)*.05,root=local_root,root_world=world_root,state=np.stack([a[i] for a in sh[:end+1]]),joint_position=np.stack([a[i] for a in qh[:end+1]]),joint_velocity=np.stack([a[i] for a in vh[:end+1]]),effort=np.stack([a[i] for a in eh[:end+1]]),contact_net_forces_w=np.stack([a[i] for a in contact_history[:end]]) if end else np.zeros((0,1,3)),command=np.stack([a[i] for a in commands[:end]]) if end else np.zeros((0,3)),action=np.stack([a[i] for a in actions[:end]]) if end else np.zeros((0,6)),actor_input=np.stack([a[i] for a in actor_obs[:end]]) if end else np.zeros((0,22)),dem=np.stack([dems[j][i] for j in idx]) if idx else np.zeros((0,3,40,40)),dem_time_s=np.asarray(demsteps)[idx]*.05,dem_stamp_ns=(np.asarray(demsteps)[idx]+40)*50000000,dem_received_ns=(np.asarray(demsteps)[idx]+40)*50000000)
        write(d/'run.json',dict(parent=p,result=results[i] or 'technical_failure',duration_s=end*.05,goal_remaining_m=float(np.linalg.norm(local_root[-1,:2]-np.asarray(PROTOCOL['goal_xy']))),goal_world=goals[i],origin=origins[i],initialization=init[i],physics_environment_index=i,bypass_observed=bool(bypass[i]),first_bypass_s=None if first_bypass[i]<0 else float(first_bypass[i]*.05),minimum_course_valid_goal_distance_m=float(minimum_distance[i]),minimum_distance_time_s=float(minimum_step[i]*.05),failure_location=scene_location(local_root[-1,:2]),minimum_height_m=float(local_root[:,2].min()),below_terrain_observed=bool((local_root[:,2]<-1).any()),joint_names=names,pose_frame='Scene1 local world frame; root_world preserves translated PhysX coordinate',terminal_event=terminal.get(i),contact_source='existing contact_forces sensor; NaN at unavailable/automatic-reset terminal',reference_physics_sha256=sha(args.reference_physics),collector_sha256=sha(__file__),protocol_sha256=sha(args.protocol),mesh_sha256=sha(args.mesh),source='ideal synchronous sim; NOT FAST-LIO',overlapping_windows=False))
    for step in range(TRIAL_STEPS):
        current_step=step
        if not active.any():break
        if step%5==0:
            tick=time.perf_counter();hit=bare.scene['height_scanner_eat'].data.ray_hits_w.cpu().numpy()[:,:,2];assert hit.shape==(N,1600);frame=np.zeros((N,3,40,40),np.float32)
            for i in range(N):
                h=hit[i].reshape(40,40);mask=np.isfinite(h)
                if mask.any():frame[i]=inpaint_frame(h,mask,roots[i,2])
            dems.append(frame);demsteps.append(step);demroots.append(roots.copy());timing['sensor']+=time.perf_counter()-tick
        if step%40==0:
            for i in list(pending):final_window(i,step)
            tick=time.perf_counter();a=dict(dem=frame,state=st,state_mask=np.ones((N,27),np.float32),state_age=np.zeros((N,27),np.float32),dem_context=np.tile([0,1,0,0,0,1],(N,1)).astype(np.float32),paths=np.tile(paths,(N,1,1,1)),directions=np.tile(directions,(N,1,1)))
            pred=model(*prepare(a,payload['normalization'],bare.device));pp=pred['position_z'].cpu().numpy()*np.asarray(payload['normalization']['position_std'])+np.asarray(payload['normalization']['position_mean']);RR,valid=sixd_to_matrix(pred['rotation_6d'].cpu().numpy());timing['inference']+=time.perf_counter()-tick
            for i in np.flatnonzero(active):
                failure=not frame[i,1].any() or not valid[i].all() or not np.isfinite(pp[i]).all()
                choice=dict(candidate_id=None,all_filtered=False,input_failure=True) if failure else select_candidate(pp[i],RR[i],roots[i,:3],quat_to_matrix(roots[i,3:7]),goals[i],'B')
                original_allowed=np.asarray(choice.get('allowed',[False]*14),bool);R0=quat_to_matrix(roots[i,3:7]);yy=np.arctan2(R0[1,0],R0[0,0]);yawmat=np.array([[np.cos(yy),np.sin(yy)],[-np.sin(yy),np.cos(yy)]])
                world_paths=paths@yawmat+roots[i,:2]
                predicted_world=roots[i,:3]+pp[i]@R0.T
                course_allowed=course_mask(world_paths-origins[i,:2],predicted_world-origins[i])
                allowed=original_allowed&course_allowed;ids=np.flatnonzero(allowed)
                cid=None if not len(ids) else int(ids[np.argmax(np.asarray(choice['score_m'])[ids])]);choice.update(candidate_id=cid,original_tilt_allowed=original_allowed.tolist(),course_allowed=course_allowed.tolist(),allowed=allowed.tolist(),all_filtered=cid is None,stop_reason=('input_unavailable' if failure else 'all_tilt_or_course_filtered') if cid is None else None)
                chosen[i]=cid;followers[i]=None
                if cid is not None:followers[i]=NominalFollower(world_paths[cid])
                item=dict(trial_id=parents[i]['trial_id'],environment=int(i),step=step,t0_ns=int((40+step)*50000000),dem_stamp_ns=int((40+step)*50000000),dem_received_ns=int((40+step)*50000000),state_age=np.zeros(27),state=st[i],root=roots[i],dem_context=a['dem_context'][i],dem_valid_fraction=float(frame[i,1].mean()),raw_support_fraction=float(frame[i,2].mean()),predicted_positions=pp[i],predicted_rotations=RR[i],**choice);decisions.append(item)
                pending[i]=dict(step=step,candidate=cid,start_state=state_class(st[i,18:21],st[i,21:24]),speed=float(np.linalg.norm(st[i,18:21])),angular=float(np.linalg.norm(st[i,21:24])),root=roots[i].copy(),state=st[i].copy(),p=None if cid is None else pp[i,cid].copy(),R=None if cid is None else RR[i,cid].copy(),input_failure=failure)
        inactive=np.flatnonzero(~active)
        if len(inactive):
            ids=torch.as_tensor(inactive,device=bare.device);park=robot.data.root_state_w[ids].clone();park[:,:3]=bare.scene.env_origins[ids]+torch.tensor([0.,0.,-20.],device=bare.device);park[:,3:7]=torch.tensor([1.,0,0,0],device=bare.device);park[:,7:]=0;robot.write_root_state_to_sim(park,env_ids=ids)
        cmd=np.zeros((N,3),np.float32)
        for i in np.flatnonzero(active):
            if followers[i] is not None:
                R0=quat_to_matrix(roots[i,3:7]);cmd[i]=followers[i].step(roots[i,0],roots[i,1],np.arctan2(R0[1,0],R0[0,0]))
        tick=time.perf_counter();done,action,seen=advance(cmd);timing['step']+=time.perf_counter()-tick;commands.append(cmd);actions.append(action);actor_obs.append(seen);roots,st,q,v,eff=snap()
        for i in np.flatnonzero(active):
            if done[i]:
                event=terminal.get(i)
                if event is None:raise RuntimeError('missing pre-reset terminal')
                roots[i]=event['root'];st[i]=event['state'];q[i]=event['joint_position'];v[i]=event['joint_velocity'];eff[i]=event['effort']
            finite=np.isfinite(roots[i]).all() and np.isfinite(st[i]).all()
            tilt=float(np.rad2deg(np.arccos(np.clip(quat_to_matrix(roots[i,3:7])[2,2],-1,1)))) if finite else 180.
            local=roots[i,:3]-origins[i]
            if not bool(inside_course(local[:2])):
                bypass[i]=True
                if first_bypass[i]<0:first_bypass[i]=step+1
            distance=float(np.linalg.norm(local[:2]-np.asarray(PROTOCOL['goal_xy'])))
            if not bypass[i] and distance<minimum_distance[i]:minimum_distance[i]=distance;minimum_step[i]=step+1
            why=outcome((step+1)*.05,local,tilt,bool(bypass[i]),goal=PROTOCOL['goal_xy'],limit=args.max_seconds)
            if done[i] and why not in ['tip_over','technical_failure']:raise RuntimeError('unexpected non-tilt automatic reset')
            if why is not None:
                results[i]=why;last_steps[i]=step+1;active[i]=False;final_window(i,step+1)
        try:
            forces=bare.scene['contact_forces'].data.net_forces_w.cpu().numpy().copy()
        except (KeyError,AttributeError):forces=np.full((N,1,3),np.nan)
        forces[done]=np.nan
        contact_history.append(forces)
        rh.append(roots.copy());sh.append(st.copy());qh.append(q.copy());vh.append(v.copy());eh.append(eff.copy())
        for i in np.flatnonzero(~active):dump_trial(i)
        if step%40==39 or not active.any():
            write(args.output/'decisions.json',decisions);write(args.output/'windows.json',windows);write(args.output/'progress.json',dict(simulation_time_s=(step+1)*.05,completed_trials=int((~active).sum()),planned_trials=N,results=results))
    for i in list(pending):final_window(i,len(commands))
    measured=time.perf_counter()-begin;tick=time.perf_counter();rh=np.asarray(rh);sh=np.asarray(sh);qh=np.asarray(qh);vh=np.asarray(vh);eh=np.asarray(eh)
    for i in range(N):dump_trial(i)
    write(args.output/'decisions.json',decisions);write(args.output/'windows.json',windows)
    modules={n:dict(path=m.__file__,sha256=sha(m.__file__)) for n,m in list(sys.modules.items()) if getattr(m,'__file__',None) and m.__file__.endswith('.py') and n.startswith(('isaaclab','onix_isaac_lab','visualnav_transformer.eat_nav'))}
    write(args.output/'identity.json',dict(reference_physics_sha256=sha(args.reference_physics),runtime_versions=dict(python=sys.version,torch=torch.__version__),asset_sha256={str(p):sha(p) for p in Path('/workspace/isaaclab_onix/source/onix_isaac_lab/assets').rglob('*') if p.is_file() and p.suffix.lower() in {'.urdf','.stl','.obj','.dae','.usd','.usda','.usdc'}},worker_sha256=sha(__file__),contract_sha256=sha(Path(__file__).with_name('nav_contract.py')),pose_contract_sha256=sha(Path(__file__).with_name('pose_contract.py')),protocol_sha256=sha(args.protocol),mesh_sha256=sha(args.mesh),nominal_spawn_z=SPAWN_Z,translations=origins,checkpoint_sha256=sha(args.nn_checkpoint),actor_sha256=sha(args.rl_checkpoint),env_sha256=sha(args.actor_env_yaml),agent_sha256=sha(args.actor_env_yaml.with_name('agent.yaml')),profile=PROFILE,profile_sha256=PROFILE_SHA,input_source='ideal synchronous sim explicitly distinct from training actual_FASTLIO',module_imports=modules,joint_names=names,flipper_indices=fids,right_track_indices=right,left_track_indices=left,headless=True,physics_dt=.005,control_dt=.05,dem_hz=4,trial_seconds=args.max_seconds,goal_tolerance=.2,goal_yaw_required=False,course_bounds=PROTOCOL['course_xy_bounds'],condition='B+explicit course boundary',inactive_environment_handling='after recorded terminal park below ground and zero actions; truncated before parking'))
    timing['save']=time.perf_counter()-tick
    write(args.output/'complete.json',dict(state='completed',parents=N,planned_windows=N*(TRIAL_STEPS//40),valid_windows=sum(x['valid'] for x in windows),observed_windows=len(windows),initialization_failures=sum(x=='initialization_failure' for x in results),results={k:results.count(k) for k in set(results)},elapsed_seconds=time.perf_counter()-begin,setup_and_run_seconds=measured,simulated_environment_seconds=float(sum(last_steps)*.05),timings=timing,peak_gpu_torch_bytes=int(torch.cuda.max_memory_allocated())))
    capture.close();override.close();env.close()
try:main()
except Exception as exc:
    write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
finally:application.close()
