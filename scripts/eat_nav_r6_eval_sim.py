"""R6 fixed-duration closed-loop evaluation with explicit per-parent terrain seeds.
Isaac Python only. Default headless; --gui enables representative replay.
--max-steps below1200 marks a bounded smoke, never a full60s evaluation.
"""
from __future__ import annotations
import argparse
import copy
import json
import hashlib
import math
import random
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
for name in ('output', 'rl-checkpoint', 'actor-env-yaml', 'eat-config'):
    parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--terrain-type', required=True)
parser.add_argument('--level', type=int, choices=range(5), required=True)
parser.add_argument('--num-envs', type=int, default=10)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--terrain-cols', type=int)
parser.add_argument('--tile-indices', help='comma-separated parent column indices; retains terrain grid when chunking')
parser.add_argument('--groups', type=Path)
parser.add_argument('--scene', choices=('field','wall','narrow','one_sided','diagonal','approach'), default='field')
parser.add_argument('--max-steps',type=int,default=1200)
parser.add_argument('--mode',choices=('old_reference','event','pose','hybrid'),required=True)
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--old-checkpoint',type=Path)
parser.add_argument('--profile',type=Path)
parser.add_argument('--gui',dest='headless',action='store_false')
parser.add_argument('--parent-seeds',help='comma per-environment parent seeds; also inferred from --groups')

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
if args.num_envs<1 or not 1<=args.max_steps<=1200:parser.error('positive environments and 1..1200 fixed steps required')
if args.mode=='old_reference' and not args.old_checkpoint:parser.error('old_reference requires --old-checkpoint')
if args.mode!='old_reference' and (not args.checkpoint or not args.profile):parser.error('new policies require --checkpoint and --profile')
raw_groups=json.loads(args.groups.read_text()) if args.groups else None
if isinstance(raw_groups,dict):raw_groups=raw_groups['groups']
PARENT_SEEDS=[int(x) for x in args.parent_seeds.split(',')] if args.parent_seeds else ([int(g['seed']) for g in raw_groups] if raw_groups else [args.seed]*args.num_envs)
PER_PARENT=bool(args.parent_seeds) or len(set(PARENT_SEEDS))>1
if len(PARENT_SEEDS)!=args.num_envs:parser.error('one parent seed per environment required')
app = AppLauncher(args).app
import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat
from rsl_rl.runners import OnPolicyRunner
import onix_isaac_lab.tasks
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.agents.rsl_rl_ppo_cfg import OnixRoughPPORunnerCfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dem import SourceGrid, robot_dem
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import make_collection_cfg, EATTerrainImporter, EATTerrainGenerator, reset_root_state_eat, _seed
from visualnav_transformer.eat_nav.demo_terrain import configure_selected_terrain
from visualnav_transformer.eat_nav.demo_reset_policy import configure_demo_reset_policy
from visualnav_transformer.eat_nav.mi_fi_v1 import configure_initialization
from visualnav_transformer.eat_nav.phase_c_sim import CommandOverride, ObservationCapture
from visualnav_transformer.eat_nav.p1_g3_rosfree_control import reconcile_command_observation
from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, compile_core
from visualnav_transformer.eat_nav.fall_stall_targets import body_vector
from visualnav_transformer.eat_nav.fall_stall_audit import sha
from eat_nav_risk_navigation import body_state
from eat_nav_r4_star14 import pack_inputs, candidate_paths, CANDIDATE_IDS, ANGLES
from eat_nav_all_fields import needs_extension
from eat_nav_r6_collection import STEP_NS, fixture_boxes, validate_groups, verify_anchor, step_records, fixture_proximity, selected_tiles, fixture_contact_index, audit_physics_properties


TERRAIN_COLS = args.terrain_cols or args.num_envs
TILE_INDICES = selected_tiles(args.num_envs,TERRAIN_COLS,args.tile_indices)

class R6TerrainGenerator(EATTerrainGenerator):
    def _get_terrain_mesh(self,difficulty,cfg):
        if not PER_PARENT:
            mesh,origin=super()._get_terrain_mesh(difficulty,cfg)
        else:
            column=self._tile_index//5;row=self._tile_index%5
            parent=PARENT_SEEDS[column];tile_seed=_seed(parent,0,row)
            self._tile_index+=1;old_seed=self.cfg.seed;old_state=np.random.get_state()
            try:
                self.cfg.seed=tile_seed;np.random.seed(tile_seed)
                mesh,origin=super(EATTerrainGenerator,self)._get_terrain_mesh(difficulty,cfg)
                digest=hashlib.sha256()
                for array in (np.ascontiguousarray(mesh.vertices),np.ascontiguousarray(mesh.faces)):
                    digest.update(str((array.shape,array.dtype)).encode());digest.update(array.tobytes())
                self.tile_metadata.append(dict(tile_index=self._tile_index-1,parent_seed=parent,parent_tile_column=0,
                    seed=tile_seed,terrain_type=args.terrain_type,difficulty=float(difficulty),geometry_sha256=digest.hexdigest()))
            finally:self.cfg.seed=old_seed;np.random.set_state(old_state)
        z=np.unique(np.round(np.asarray(mesh.vertices)[:,2],6))
        self.tile_metadata[-1].update(mesh_bounds_m=np.asarray(mesh.bounds).tolist(),mesh_vertex_count=len(mesh.vertices),
            mesh_triangle_count=len(mesh.faces),surface_z_levels_m=z.tolist(),surface_z_gaps_m=np.diff(z).tolist(),
            geometry_note='mesh geometry seed is independent per parent; dynamic randomization is batch correlated')
        return mesh,origin


class R6TerrainImporter(EATTerrainImporter):
    def _compute_env_origins_curriculum(self,num_envs,origins):
        self.terrain_levels=torch.full((num_envs,),args.level,dtype=torch.long,device=self.device)
        self.terrain_types=torch.tensor(TILE_INDICES,dtype=torch.long,device=self.device)
        self.max_terrain_level=origins.shape[0]
        return origins[self.terrain_levels,self.terrain_types].clone()


def r6_reset_root(env,env_ids,pose_range,velocity_range,base_seed):
    if not hasattr(env,'eat_nav_initial_rngs'):
        env.eat_nav_initial_rngs=[np.random.default_rng(_seed(parent,2,0 if PER_PARENT else tile)) for parent,tile in zip(PARENT_SEEDS,TILE_INDICES)]
    return reset_root_state_eat(env,env_ids,pose_range,velocity_range,base_seed)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.tolist()) + '\n')


def supplemental_mesh(difficulty, cfg):
    import trimesh
    meshes = [trimesh.creation.box(extents=[*cfg.size, .1], transform=trimesh.transformations.translation_matrix([cfg.size[0]/2, cfg.size[1]/2, -.05]))]
    for box in fixture_boxes(args.scene):
        transform = trimesh.transformations.rotation_matrix(box['yaw_rad'], [0,0,1])
        transform[:3,3] = np.asarray(box['center_m']) + [cfg.size[0]/2,cfg.size[1]/2,0]
        meshes.append(trimesh.creation.box(extents=box['size_m'], transform=transform))
    return meshes, np.asarray([cfg.size[0]/2,cfg.size[1]/2,0.])


def pose(robot, i):
    root = robot.data.root_state_w[i].detach().cpu().numpy()
    yaw = euler_xyz_from_quat(robot.data.root_quat_w[i:i+1])[2].item()
    return np.asarray([root[0], root[1], yaw])


def dem(bare, config, i, stamp):
    sensor = bare.scene['height_scanner_eat']
    hit = sensor.data.ray_hits_w[i,:,2].detach().cpu().numpy().reshape(20,20)
    starts = (sensor.ray_starts[i] + sensor.data.pos_w[i]).detach().cpu().numpy()
    valid = np.isfinite(hit)
    source = SourceGrid(np.where(valid,hit,0).astype(np.float32), valid,
                        tuple(map(float, starts[0,:2])), config.dem_resolution_m,
                        (0,0), config.reference_frame, stamp)
    return robot_dem(source, tuple(pose(bare.scene['robot'],i)), config).astype(np.float32)


def physical(bare, i):
    robot = bare.scene['robot']
    return dict(root_state=robot.data.root_state_w[i].cpu().tolist(),
                joint_position=robot.data.joint_pos[i].cpu().tolist(),
                joint_velocity=robot.data.joint_vel[i].cpu().tolist(),
                applied_torque=robot.data.applied_torque[i].cpu().tolist(),
                contact_net_forces_w=bare.scene['contact_forces'].data.net_forces_w[i].cpu().tolist())


def main():
    if args.output.exists(): raise FileExistsError('fresh output required')
    args.output.mkdir(parents=True)
    env = capture = override = None
    followers = []
    try:
        torch.set_num_threads(1)
        config = load_eat_config(args.eat_config)
        saved = yaml.load(args.actor_env_yaml.read_text(), Loader=yaml.BaseLoader)
        toggles = {n:saved[n].lower()=='true' for n in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
        cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles), config, args.seed, args.num_envs)
        cfg.episode_length_s = 120.
        configure_demo_reset_policy(cfg); configure_initialization(cfg,'front_folded')
        cfg.events.ignore_sprocket_self_collision = None
        if args.scene!='field':
            cfg.scene.contact_forces.filter_prim_paths_expr=['/World/ground/.*']
            cfg.scene.contact_forces.max_contact_data_count_per_prim=64
        generator = cfg.scene.terrain.terrain_generator
        generator.size = (16.,16.)
        generator.class_type=R6TerrainGenerator
        # Frozen demo helper allows 1/4; configure its 1-env template, then extend columns.
        terrain = configure_selected_terrain(generator,cfg.scene.terrain,args.terrain_type,args.level,1)
        generator.num_cols = TERRAIN_COLS
        terrain['grid_cols'] = TERRAIN_COLS
        terrain['tile_indices'] = TILE_INDICES
        cfg.scene.terrain.class_type=R6TerrainImporter
        cfg.events.reset_base.func=r6_reset_root
        if args.scene != 'field':
            from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
            generator.sub_terrains = {args.terrain_type:MeshPlaneTerrainCfg(function=supplemental_mesh, proportion=1.)}
        env = RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
        bare = env.unwrapped; robot = bare.scene['robot']; n = args.num_envs
        if abs(bare.step_dt-.05)>1e-9: raise ValueError('20Hz physics required')
        env.reset()
        masses=robot.root_physx_view.get_masses().cpu().numpy()
        inertias=robot.root_physx_view.get_inertias().cpu().numpy()
        physics_audit=audit_physics_properties(masses,inertias)
        property_arrays=dict(masses_kg=masses,inertias_kg_m2=inertias)
        for name in ('default_mass','default_inertia'):
            value=getattr(robot.data,name,None)
            if value is not None:property_arrays[name]=value.cpu().numpy()
        np.savez_compressed(args.output/'actual-physics-properties.npz',**property_arrays)
        write(args.output/'physics-audit.json',physics_audit)
        if not physics_audit['valid']:raise ValueError('actual PhysX masses/inertias invalid; see physics-audit.json')
        runner = OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
        policy = runner.alg.policy
        policy.load_state_dict(torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict'])
        policy.eval()
        override = CommandOverride(bare.command_manager,n)
        capture = ObservationCapture(bare.observation_manager)
        zeros = torch.zeros((n,3), device=bare.device)
        override.set(zeros)
        latency = bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
        capture.begin_step(); obs=bare._apply_policy_observation_latency(bare.observation_manager.compute()); observed=capture.finish_step(obs)
        terminals={}; active_step=-1
        pending_contacts=[];contact_api_status={'available':False,'reason':'field has no analytic fixture classification'}
        if args.scene!='field':
            sensor=bare.scene['contact_forces'];view=sensor.contact_physx_view
            original_scene_update=bare.scene.update
            def contact_update(*values,**kwargs):
                original_scene_update(*values,**kwargs)
                try:
                    forces,points,normals,separations,counts,starts=view.get_contact_data(bare.physics_dt)
                    forces=forces.cpu().numpy();points=points.cpu().numpy();normals=normals.cpu().numpy();separations=separations.cpu().numpy();counts=counts.cpu().numpy();starts=starts.cpu().numpy()
                    origins_now=bare.scene.env_origins.cpu().numpy()
                    for body,filter_index in np.argwhere(counts>0):
                        env_index=int(body)//sensor.num_bodies
                        start=int(starts[body,filter_index]);count=int(counts[body,filter_index])
                        for contact_index in range(start,start+count):
                            fixture=fixture_contact_index(points[contact_index],origins_now[env_index],args.scene)
                            if fixture is not None and float(forces[contact_index,0])>1e-4:
                                pending_contacts.append(dict(env_index=env_index,body_index=int(body)%sensor.num_bodies,fixture_index=fixture,point_world_m=points[contact_index].tolist(),normal_world=normals[contact_index].tolist(),normal_force_n=float(forces[contact_index,0]),separation_m=float(separations[contact_index,0]),physics_step=int(bare._sim_step_counter),logical_sim_time_ns=int(round(bare._sim_step_counter*bare.physics_dt*1e9))))
                    contact_api_status.update(available=False,reason='PhysX GPU backend reports GPU contact filter for static triangle-mesh collider /World/ground/terrain is not supported; zero buffers are not valid negatives',tensor_api_called=True,filter_count=int(view.filter_count),definition='positive PhysX normal contact point within25mm analytic fixture surface and above30mmground',sample_period_ns=int(round(bare.physics_dt*1e9)))
                except Exception as exc:
                    contact_api_status.update(available=False,reason=type(exc).__name__+': '+str(exc))
            bare.scene.update=contact_update
        original_reset=bare._reset_idx
        def before_reset(ids):
            for i in ids.detach().cpu().tolist():
                terminals[i]=dict(physical(bare,i),control_step=active_step,
                    termination_reasons=[name for name in bare.termination_manager.active_terms if bool(bare.termination_manager.get_term(name)[i])])
            return original_reset(ids)
        bare._reset_idx=before_reset
        timings={key:0. for key in ('command_reconciliation','actor_inference','env_step','observation_capture','snapshots_and_follower','serialization','dem_capture')}
        def advance(commands):
            nonlocal obs,observed,latency
            started=time.perf_counter()
            override.set(commands)
            obs,observed,detail=reconcile_command_observation(obs,observed,bare,commands,latency)
            timings['command_reconciliation']+=time.perf_counter()-started
            started=time.perf_counter()
            with torch.inference_mode(): action=policy.act_inference(obs)
            timings['actor_inference']+=time.perf_counter()-started
            latency=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
            capture.begin_step()
            started=time.perf_counter()
            with torch.inference_mode(): obs,_,done,_=env.step(action)
            timings['env_step']+=time.perf_counter()-started
            started=time.perf_counter()
            observed=capture.finish_step(obs)
            timings['observation_capture']+=time.perf_counter()-started
            return done,detail
        history=[[] for _ in range(n)]; stamps=[]
        settling_valid=np.ones(n,dtype=bool);settling_failures={}
        for step in range(20):
            done,_=advance(zeros)
            for i in np.flatnonzero(done.cpu().numpy()):
                settling_valid[i]=False;settling_failures[int(i)]=terminals.get(int(i))
            if (step+1)%5==0:
                stamps.append((step+1)*STEP_NS)
                started=time.perf_counter()
                for i in range(n): history[i].append(dem(bare,config,i,stamps[-1]))
                timings['dem_capture']+=time.perf_counter()-started
        from visualnav_transformer.eat_nav.r6_model import load_checkpoint as load_r6,predict as predict_r6
        from visualnav_transformer.eat_nav.r6_policy import Footprint,select_candidate
        from visualnav_transformer.eat_nav.r6_evaluation import summarize_trial,aggregate_trials
        from visualnav_transformer.eat_nav.fall_stall_targets import root_yaw,tilt
        from eat_nav_command_free_learning import load_checkpoint as load_old,predict as predict_old
        from eat_nav_r5_14 import R5FourteenGuard
        from eat_nav_r4_star14 import extend_path
        from pxr import Usd,UsdGeom
        import itertools
        groups=json.loads(args.groups.read_text()) if args.groups else [dict(group_id=f'smoke-{args.seed}-{i}',role='benchmark',seed=args.seed,env_index=i) for i in range(n)]
        if isinstance(groups,dict):groups=groups['groups']
        groups=validate_groups(groups,n)
        if any(g['seed']!=PARENT_SEEDS[i] for i,g in enumerate(groups)):raise ValueError('frozen parent seed mismatch')
        if any(g.get('tile_index',0 if PER_PARENT else TILE_INDICES[i])!=(0 if PER_PARENT else TILE_INDICES[i]) for i,g in enumerate(groups)):raise ValueError('frozen parent tile index mismatch')
        initial_roots=robot.data.root_state_w.cpu().numpy().copy();origins=bare.scene.env_origins.cpu().numpy()
        goals=np.asarray([r[:2]+2*np.asarray([math.cos(root_yaw(r.tolist())),math.sin(root_yaw(r.tolist()))]) for r in initial_roots])
        active=settling_valid.copy();goal_times=[None]*n;held=np.zeros(n,bool);stopped=np.ones(n,bool)
        selected=[None]*n;paths=[None]*n;headings=[None]*n;followers=[None]*n
        traces=[[] for _ in range(n)];streams=[];decisions=[[] for _ in range(n)];guards=[R5FourteenGuard() for _ in range(n)]
        model=payload=oldfit=None
        if args.mode=='old_reference':
            model,payload=load_old(args.old_checkpoint);oldfit=dict(payload,model=model)
        else:model,payload=load_r6(args.checkpoint)
        profile=json.loads(args.profile.read_text()) if args.profile else None
        if profile and 'profile' in profile:profile=profile['profile']
        # USD world AABBs of the complete robot include actual flipper geometry.
        # Rotating all eight AABB corners into current yaw is conservative. No
        # body-frame-origin distance is substituted for occupied-body dimensions.
        def measured_footprint(i,root):
            candidates=getattr(robot.root_physx_view,'prim_paths',[])
            primpath=candidates[i] if len(candidates)>i else robot.cfg.prim_path.replace('{ENV_REGEX_NS}',f'/World/envs/env_{i}').replace('env_.*',f'env_{i}')
            prim=bare.sim.stage.GetPrimAtPath(primpath)
            if not prim.IsValid():raise ValueError('cannot resolve robot geometry '+primpath)
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
            bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange();lo=np.asarray(bounds.GetMin());hi=np.asarray(bounds.GetMax())
            corners=np.asarray(list(itertools.product(*zip(lo,hi))))
            yaw=root_yaw(root.tolist());c,s=math.cos(yaw),math.sin(yaw)
            local=(corners-root[:3])@np.asarray([[c,-s,0],[s,c,0],[0,0,1]])
            bodypositions=robot.data.body_pos_w[i].cpu().numpy()
            if not np.isfinite(local).all() or np.any(bodypositions<lo-1e-4) or np.any(bodypositions>hi+1e-4):raise ValueError('USD bounds fail body-origin containment')
            dimensions=dict(front_m=float(local[:,0].max()),rear_m=float(-local[:,0].min()),half_width_m=float(np.abs(local[:,1]).max()),stopping_margin_m=float(profile.get('stopping_margin_m',.06)) if profile else .06)
            return Footprint(**dimensions),dict(dimensions=dimensions,source_prim=str(primpath),world_aabb_min=lo.tolist(),world_aabb_max=hi.tolist(),body_origins_w=bodypositions.tolist(),method='all robot USD geometry world AABB corners yaw transformed; conservative includes visual/flipper extent')
        footprints=[]
        for i,g in enumerate(groups):
            folder=args.output/f'group-{i:03d}';folder.mkdir();streams.append((folder/'trace.jsonl').open('x'))
            fp,measurement=measured_footprint(i,initial_roots[i]);footprints.append(fp)
            write(folder/'group.json',dict(g,initial_root=initial_roots[i],goal_xy=goals[i],goal_definition='2m forward from initial yaw',initial_valid=bool(active[i]),settling_failure=settling_failures.get(i),footprint=measurement,tile_index=(0 if PER_PARENT else TILE_INDICES[i]),physical_grid_column=TILE_INDICES[i],parent_seed=PARENT_SEEDS[i]))
            np.savez_compressed(folder/'initial-state.npz',root=initial_roots[i],joints=robot.data.joint_pos[i].cpu().numpy(),dem=np.asarray(history[i]),dem_stamps_ns=np.asarray(stamps),env_origin=origins[i])
        write(args.output/'manifest.json',dict(schema='eatnav_r6_closed_loop_v1',mode=args.mode,profile=profile,
            model_sha256=sha(args.old_checkpoint if args.mode=='old_reference' else args.checkpoint),actor_sha256=sha(args.rl_checkpoint),actor_env_sha256=sha(args.actor_env_yaml),config_sha256=sha(args.eat_config),
            headless=args.headless,num_envs=n,terrain_type=args.terrain_type,level=args.level,seed=args.seed,tile_indices=TILE_INDICES,terrain_cols=TERRAIN_COLS,
            requested_duration_sec=args.max_steps*.05,full_protocol_duration_sec=60,smoke=args.max_steps!=1200,
            legacy_event_horizon_sec=10 if args.mode=='old_reference' else None,event_horizon_sec=2 if args.mode!='old_reference' else 10,
            geometry_metadata=getattr(generator,'eat_nav_tile_metadata',[]),contact_truth='net forces retained; obstacle identity unknown',
            group_seed_contract='per-environment terrain and initial parent seeds; startup dynamic randomization shares batch cfg.seed and is correlated; identical batch composition required across conditions',parent_seeds=PARENT_SEEDS,per_parent_seed_mode=PER_PARENT))
        library=compile_core(args.output/'core-build');start=time.perf_counter()
        for step in range(args.max_steps):
            active_step=step;stamp=1_000_000_000+step*STEP_NS
            roots=robot.data.root_state_w.cpu().numpy().copy()
            if step%40==0:
                for i in range(n):
                    if not active[i] or held[i]:continue
                    folder=args.output/f'group-{i:03d}'
                    if any(t>stamp for t in stamps[-4:]):raise ValueError('future DEM input')
                    arrays=pack_inputs(np.asarray(history[i][-4:]),observed['policy_pre_noise'][i].cpu().numpy(),body_state(roots[i]))
                    np.savez_compressed(folder/f'input-{step:04d}.npz',**arrays,dem_stamps_ns=np.asarray(stamps[-4:]),decision_stamp_ns=np.asarray(stamp))
                    yaw=root_yaw(roots[i].tolist());c,s=math.cos(yaw),math.sin(yaw)
                    goal_body=(goals[i]-roots[i,:2])@np.asarray([[c,-s],[s,c]])
                    inference_started=time.perf_counter()
                    if args.mode=='old_reference':
                        probabilities,progress=predict_old(oldfit,arrays,[0])
                        decision=guards[i].decide(probabilities[0],progress[0],goal_body,step,float(tilt(roots[i].tolist())))
                        decision['legacy_event_horizon_sec']=10
                    else:
                        predicted=predict_r6(model,payload['normalization'],arrays,[0])
                        # Re-measure moving flippers at every decision; dimensions
                        # remain observations, never simulator future truth.
                        footprints[i],measurement=measured_footprint(i,roots[i])
                        decision=select_candidate(predicted['trajectory'][0],predicted['events'][0],arrays['dem'][0,-1],goal_body,profile,
                            footprint=footprints[i],mode=args.mode,event_support=payload['support']['active'])
                        decision.update(predicted_trajectory=predicted['trajectory'][0],event_probabilities=predicted['events'][0],footprint=measurement)
                    decision.update(inference_and_policy_sec=time.perf_counter()-inference_started,decision_step=step,decision_stamp_ns=stamp,goal_body=goal_body,dem_stamps_ns=stamps[-4:])
                    selected[i]=decision['selected_index'];stopped[i]=selected[i] is None
                    write(folder/f'decision-{step:04d}.json',decision);decisions[i].append(decision)
                    if followers[i] is not None:followers[i].close();followers[i]=None
                    if selected[i] is not None:
                        k=selected[i];paths[i]=candidate_paths()[k]@np.asarray([[c,s],[-s,c]])+roots[i,:2]
                        headings[i]=yaw+ANGLES[k]
                        followers[i]=PureFollower((paths[i]-origins[i,:2]).tolist(),library=library)
            terminals.clear();pending_contacts.clear();before=[physical(bare,i) for i in range(n)];commands=[];diagnostics=[]
            for i in range(n):
                if not active[i] or held[i] or stopped[i]:commands.append([0.,0.,0.]);diagnostics.append({});continue
                p=pose(robot,i)
                if needs_extension(paths[i],p):
                    paths[i],headings[i]=extend_path(paths[i],headings[i],selected[i]);followers[i].close()
                    followers[i]=PureFollower((paths[i]-origins[i,:2]).tolist(),library=library)
                command,diag=followers[i].step_diagnostic(float(p[0]-origins[i,0]),float(p[1]-origins[i,1]),float(p[2]),stamp)
                commands.append(command);diagnostics.append(diag)
            done,detail=advance(torch.tensor(commands,dtype=zeros.dtype,device=bare.device))
            for i in range(n):
                if not active[i]:continue
                terminal=terminals.get(i);post=terminal['root_state'] if terminal else robot.data.root_state_w[i].cpu().tolist()
                boundary=bool(np.any(np.abs(np.asarray(post[:2])-origins[i,:2])>6.))
                decision=decisions[i][-1] if decisions[i] else {}
                row=dict(before[i],control_step=step,logical_sim_time_ns=stamp,post_timestamp_ns=stamp+STEP_NS,
                    post_root_state=post,body_angular_velocity_radps=body_vector(post,post[10:13]),
                    terminal=None if terminal is None else dict(pre_reset_state=terminal),reset_mask=bool(done[i]),boundary_censor=boundary,
                    follower_command=list(commands[i]),follower_diagnostic=diagnostics[i],actor_command=detail['actor_seen'][i],actor_observation=observed['policy_pre_noise'][i].cpu().tolist(),
                    candidate_id='STOP' if selected[i] is None else CANDIDATE_IDS[selected[i]],candidate_cycle=step//40,candidate_replan=bool(step and step%40==0),
                    candidate_path=(candidate_paths()[selected[i]].tolist() if selected[i] is not None else candidate_paths()[0].tolist()),
                    protective_stop=bool(stopped[i] and not held[i]),goal_hold=bool(held[i]),valid_mask=True,
                    contact_observation=dict(contact_api_status),actual_fixture_contacts=[x for x in pending_contacts if x['env_index']==i],
                    decision_all_excluded=bool(step%40==0 and selected[i] is None and not held[i]),group_id=groups[i]['group_id'])
                traces[i].append(row);streams[i].write(json.dumps(row,allow_nan=False,default=lambda x:x.tolist())+'\n');streams[i].flush()
                if np.linalg.norm(np.asarray(post[:2])-goals[i])<=.2 and goal_times[i] is None:
                    goal_times[i]=(step+1)*.05;held[i]=True
                if terminal or boundary:active[i]=False
            if (step+1)%5==0:
                stamps.append(stamp+STEP_NS)
                for i in range(n):history[i].append(dem(bare,config,i,stamps[-1]))
            if not active.any():break
        results=[]
        for i,g in enumerate(groups):
            streams[i].close()
            result=summarize_trial(traces[i],initial_roots[i,:2],goals[i],duration_sec=args.max_steps*.05,goal_reached_time_sec=goal_times[i])
            result.update(g,field=args.terrain_type,level=args.level,mode=args.mode,initial_valid=bool(settling_valid[i]),
                settling_failure=settling_failures.get(i),decision_count=len(decisions[i]),processing_sec=sum(d['inference_and_policy_sec'] for d in decisions[i]),
                stop_reason='settling_failure' if not settling_valid[i] else 'natural_termination_or_boundary' if not active[i] else 'fixed_horizon_complete',
                smoke=args.max_steps!=1200,full_evaluation_valid=args.max_steps==1200)
            write(args.output/f'group-{i:03d}'/'result.json',result);results.append(result)
        write(args.output/'complete.json',dict(state='completed',condition=args.mode,num_envs=n,headless=args.headless,smoke=args.max_steps!=1200,
            requested_steps=args.max_steps,elapsed_seconds=time.perf_counter()-start,recorded_rows=sum(len(r) for r in traces),
            peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),timings=timings,results=results,aggregate=aggregate_trials(results)))
        print('R6_EVAL_COMPLETE '+str(args.output/'complete.json'),flush=True)
    except BaseException as exc:
        write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for follower in followers:
            if follower is not None:follower.close()
        if capture is not None:capture.close()
        if override is not None:override.close()
        if env is not None:env.close()

if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    finally:app.close()
