"""Headless Isaac moving-state collection worker; invoked by the CPU host.

Seven causal prefix recipes, qualified visible snapshots, fixed fourteen .75 m
candidates, forty control intervals and pre-reset terminal capture. No ROS.
"""
from __future__ import annotations
import argparse
import copy
import json
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
parser.add_argument('--scene', choices=('field','wall','narrow','one_sided','diagonal','approach','wall_60x100'), default='field')
parser.add_argument('--smoke-steps', type=int, default=0)
parser.add_argument('--flipper-calibration-probe',action='store_true',help='diagnostic-only0/90/180deg geometry and joint-axis audit')
parser.add_argument('--contact-calibration-probe',action='store_true',help='diagnostic-only deliberate fixture overlap; never training data')
parser.add_argument('--horizon-seconds', type=float, default=2.)
parser.add_argument('--fresh-process',action='store_true')
parser.add_argument('--candidate-indices',type=int,nargs='+')
parser.add_argument('--prefix-steps')
parser.add_argument('--prefix-state',choices=('moving_flat','turning','uphill_before','uphill_mid','step_before','descent','slow'),required=True)
parser.add_argument('--parent-manifest',type=Path)
parser.add_argument('--duplicate-gate',action='store_true')
parser.add_argument('--restoration-method',choices=('snapshot','matched_prefix'),default='snapshot')
parser.add_argument('--anchor-seconds',type=int,default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.fresh_process and (not args.candidate_indices or len(args.candidate_indices)!=1 or args.candidate_indices[0] not in range(14) or args.duplicate_gate):parser.error('freshprocess requires one candidate index without inprocess duplicate gate')
if not args.headless:parser.error('moving-state collection requires --headless')
if args.flipper_calibration_probe and (not args.smoke_steps or args.groups):
    parser.error('flipper calibration is diagnostic-only smoke without production groups')
if args.contact_calibration_probe and (not args.smoke_steps or args.groups or args.scene=='field'):
    parser.error('contact calibration is diagnostic-only smoke without production groups')
if args.horizon_seconds != 2. or args.num_envs < 1:
    parser.error('R6 requires 2s horizon and positive num_envs')
if not args.groups and not args.smoke_steps:
    parser.error('production requires frozen --groups manifest')
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
from eat_nav_r4_star14 import pack_inputs, candidate_paths, CANDIDATE_IDS, ANGLES, extend_path
from eat_nav_moving_pose2s_collect import AnchorQualification, dem_metrics, prefix_command, compare_visible_state, duplicate_gate, mesh_height, quaternion_angle, policy_input, physical_metrics, capture_lazy_caches, restore_lazy_caches, CommonWorldAnchor, parent_visible
from eat_nav_r6_collection import STEP_NS, fixture_boxes, validate_groups, verify_anchor, step_records, fixture_proximity, selected_tiles, fixture_contact_index, audit_physics_properties, update_anchor_valid, decode_fixture_contacts


GROUPS=json.loads(args.groups.read_text())['groups'] if args.groups else [dict(group_id=f'benchmark-{args.seed}-{i}',role='benchmark',seed=args.seed,env_index=i,tile_index=i) for i in range(args.num_envs)]
PARENTS=json.loads(args.parent_manifest.read_text()) if args.parent_manifest else GROUPS
PARENT_BY_TILE={p.get('tile_index',i):p for i,p in enumerate(PARENTS)}
from eat_nav_moving_pose2s_collect import reset_recipe, anchor_inside_guard
RESET_RECIPE=reset_recipe(args.prefix_state)
SPAWN_X=RESET_RECIPE['x_m']
SPAWN_YAW=RESET_RECIPE['yaw_rad']
SPAWN_HEIGHTS={}
TERRAIN_COLS = args.terrain_cols or args.num_envs
TILE_INDICES = selected_tiles(args.num_envs,TERRAIN_COLS,args.tile_indices)

class R6TerrainGenerator(EATTerrainGenerator):
    def _get_terrain_mesh(self,difficulty,cfg):
        index=self._tile_index;parent=PARENT_BY_TILE.get(index//5,dict(seed=args.seed))
        old_seed=self.cfg.seed;self.cfg.seed=parent['seed']
        try:mesh,origin=super()._get_terrain_mesh(difficulty,cfg)
        finally:self.cfg.seed=old_seed
        spawn_height=mesh_height(mesh.vertices,mesh.faces,[SPAWN_X,0.])
        SPAWN_HEIGHTS[index]=spawn_height
        self.tile_metadata[-1].update(parent_seed=parent['seed'],spawn_surface_z_m=spawn_height,spawn_x_m=SPAWN_X,spawn_yaw_rad=SPAWN_YAW)
        z=np.unique(np.round(np.asarray(mesh.vertices)[:,2],6))
        self.tile_metadata[-1].update(mesh_bounds_m=np.asarray(mesh.bounds).tolist(),
             mesh_vertex_count=len(mesh.vertices),mesh_triangle_count=len(mesh.faces),
             surface_z_levels_m=z.tolist(),surface_z_gaps_m=np.diff(z).tolist(),
             geometry_note='mesh vertex levels/gaps; a vertical wall is not a traversable stair tread')
        return mesh,origin


class R6TerrainImporter(EATTerrainImporter):
    def import_mesh(self,name,mesh):
        super().import_mesh(name,mesh)
        if args.scene=='field':return
        # Keep identical visual/Warp mesh, use separable primitive collision bodies.
        from pxr import Usd,UsdPhysics
        import isaaclab.sim as sim_utils
        stage=sim_utils.get_current_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.cfg.prim_path+'/'+name)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        floor=sim_utils.CuboidCfg(size=(80.,TERRAIN_COLS*16.,.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=self.cfg.physics_material)
        floor.func('/World/R6PhysicsFloor',floor,translation=(0.,0.,-.05))
        for col in TILE_INDICES:
            origin=np.asarray([(args.level+.5-2.5)*16.,(col+.5-TERRAIN_COLS/2)*16.,0.])
            for i,box in enumerate(fixture_boxes(args.scene)):
                shape=sim_utils.CuboidCfg(size=tuple(box['size_m']),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                    physics_material=self.cfg.physics_material)
                yaw=box['yaw_rad'];quat=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))
                shape.func(f'/World/R6Fixtures/Tile{col}/Box{i}',shape,
                    translation=tuple(origin+box['center_m']),orientation=quat)

    def _compute_env_origins_curriculum(self,num_envs,origins):
        self.terrain_levels=torch.full((num_envs,),args.level,dtype=torch.long,device=self.device)
        self.terrain_types=torch.tensor(TILE_INDICES,dtype=torch.long,device=self.device)
        self.max_terrain_level=origins.shape[0]
        return origins[self.terrain_levels,self.terrain_types].clone()


def r6_reset_root(env,env_ids,pose_range,velocity_range,base_seed):
    if not hasattr(env,'eat_nav_initial_rngs'):
        env.eat_nav_initial_rngs=[np.random.default_rng(_seed(GROUPS[i]['seed'],2,TILE_INDICES[i])) for i in range(args.num_envs)]
    fixed_pose=dict(pose_range,x=(SPAWN_X,SPAWN_X),y=(0.,0.),yaw=(SPAWN_YAW,SPAWN_YAW))
    fixed_velocity={name:(0.,0.) for name in ('x','y','z','roll','pitch','yaw')}
    result=reset_root_state_eat(env,env_ids,fixed_pose,fixed_velocity,base_seed)
    robot=env.scene['robot'];root=robot.data.root_state_w[env_ids].clone()
    for j,i in enumerate(env_ids.cpu().tolist()):
        root[j,2]=SPAWN_HEIGHTS[TILE_INDICES[i]*5+args.level]+robot.data.default_root_state[i,2]
    robot.write_root_state_to_sim(root,env_ids=env_ids)
    return result


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
            cfg.scene.height_scanner_eat.offset.pos=(0.,0.,2.)
            cfg.scene.contact_forces.filter_prim_paths_expr=[]
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
        property_arrays=dict(masses_kg=masses,inertias_kg_m2=inertias,material_properties=robot.root_physx_view.get_material_properties().cpu().numpy())
        for property_name in ('joint_stiffness','joint_damping','joint_armature','joint_effort_limits'):
            property_value=getattr(robot.data,property_name,None)
            if property_value is not None:property_arrays[property_name]=property_value.cpu().numpy()
        for name in ('default_mass','default_inertia'):
            value=getattr(robot.data,name,None)
            if value is not None:property_arrays[name]=value.cpu().numpy()
        np.savez_compressed(args.output/'actual-physics-properties.npz',**property_arrays)
        write(args.output/'physics-audit.json',physics_audit)
        if not physics_audit['valid']:raise ValueError('actual PhysX masses/inertias invalid; see physics-audit.json')
        if args.flipper_calibration_probe:
            from pxr import Usd,UsdGeom,UsdPhysics
            from visualnav_transformer.eat_nav.r6_geometry import rotation
            names=robot.body_names;joint_names=robot.joint_names
            flippers={side:next(i for i,name in enumerate(joint_names) if side+'_flipper_joint' in name) for side in ('rr','rl','fr','fl')}
            primroot='/World/envs/env_0/onix_chassis'
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy,UsdGeom.Tokens.guide])
            lower=[];upper=[]
            for name in names:
                bound=cache.ComputeUntransformedBound(bare.sim.stage.GetPrimAtPath(primroot+'/'+name)).ComputeAlignedRange()
                lower.append(np.minimum(np.asarray(bound.GetMin()),0));upper.append(np.maximum(np.asarray(bound.GetMax()),0))
            poses={}
            for degree in (0,90,180):
                q=robot.data.default_joint_pos.clone();velocity=torch.zeros_like(q)
                for index in flippers.values():q[:,index]=math.radians(degree)
                root=robot.data.default_root_state.clone();root[:,:3]=bare.scene.env_origins;root[:,2]+=2.
                root[:,3:7]=torch.tensor([1.,0.,0.,0.],device=bare.device);root[:,7:]=0.
                robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(q,velocity)
                robot.set_joint_position_target(q);robot.set_joint_velocity_target(velocity)
                bare.scene.write_data_to_sim();bare.sim.step(render=False);bare.scene.update(bare.physics_dt)
                actual_root=robot.data.root_state_w[0].cpu().numpy();root_rotation=rotation(actual_root[3:7])
                positions=(robot.data.body_pos_w[0].cpu().numpy()-actual_root[:3])@root_rotation
                rotations=np.stack([root_rotation.T@rotation(quat) for quat in robot.data.body_quat_w[0].cpu().numpy()])
                poses[str(degree)]=dict(joint_angles_rad={side:float(robot.data.joint_pos[0,index]) for side,index in flippers.items()},body_positions_chassis=positions,body_rotations_chassis=rotations)
            reference=poses['0'];quarter=poses['90'];axes={}
            for side in flippers:
                hinge_index=names.index('chassis_'+side+'_body')
                arm_indices=[i for i,name in enumerate(names) if name.startswith('chassis_'+side+'_')]
                R0=reference['body_rotations_chassis'][hinge_index];R1=quarter['body_rotations_chassis'][hinge_index]
                A=R1@R0.T;delta=quarter['joint_angles_rad'][side]-reference['joint_angles_rad'][side]
                axis=np.array([A[2,1]-A[1,2],A[0,2]-A[2,0],A[1,0]-A[0,1]])/(2*math.sin(delta));axis/=np.linalg.norm(axis)
                points0=reference['body_positions_chassis'][arm_indices];points1=quarter['body_positions_chassis'][arm_indices]
                hinge=np.linalg.lstsq(np.eye(3)-A,(points1-points0@A.T).mean(axis=0),rcond=1e-5)[0]
                error=np.linalg.norm((points0-hinge)@A.T+hinge-points1,axis=1)
                tip=arm_indices[int(np.argmax(np.linalg.norm(points0-reference['body_positions_chassis'][hinge_index],axis=1)))]
                tip_raise=float(quarter['body_positions_chassis'][tip,2]-reference['body_positions_chassis'][tip,2])
                axes[side]=dict(axis_chassis=axis,hinge_chassis=hinge,body_indices=arm_indices,zero_angle_rad=reference['joint_angles_rad'][side],max_90degree_fit_error_m=float(error.max()),tip_body=names[tip],tip_raise_0_to90_m=tip_raise,positive90_raises_tip=tip_raise>.1)
            joint_definitions=[]
            for prim in Usd.PrimRange(bare.sim.stage.GetPrimAtPath(primroot)):
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    joint_definitions.append(dict(path=str(prim.GetPath()),attributes={attr.GetName():str(attr.Get()) for attr in prim.GetAttributes() if attr.GetName().startswith('physics:')}))
            result=dict(schema='eatnav_r6_sim_flipper_geometry_v1',diagnostic_only=True,body_names=names,local_bounds_min=lower,local_bounds_max=upper,poses=poses,axes=axes,joint_definitions=joint_definitions,
                physical_reference_source='user: outward horizontal0deg, verticalup90deg',numeric_sim_reference_verified=all(a['positive90_raises_tip'] and a['max_90degree_fit_error_m']<.005 for a in axes.values()),
                limitation='simulation CAD and FK audit; real encoder offsets and real measured body geometry are not independently verified')
            write(args.output/'flipper-geometry-calibration.json',result)
            write(args.output/'complete.json',dict(state='completed',smoke=True,diagnostic_only=True,calibration='flipper_geometry',numeric_sim_reference_verified=result['numeric_sim_reference_verified']))
            return
        runner = OnPolicyRunner(env,OnixRoughPPORunnerCfg().to_dict(),log_dir=None,device=bare.device)
        policy = runner.alg.policy
        policy.load_state_dict(torch.load(args.rl_checkpoint,map_location='cpu',weights_only=True)['model_state_dict'])
        policy.eval()
        if getattr(policy,'is_recurrent',False) or any(isinstance(m,(torch.nn.RNNBase,torch.nn.LSTMCell,torch.nn.GRUCell)) for m in policy.modules()):
            raise ValueError('recurrent low-level actor unsupported; hidden state would need preservation')
        write(args.output/'actor-audit.json',dict(feedforward=True,policy_type=type(policy).__name__,checkpoint_sha256=sha(args.rl_checkpoint)))
        override = CommandOverride(bare.command_manager,n)
        capture = ObservationCapture(bare.observation_manager)
        zeros = torch.zeros((n,3), device=bare.device)
        override.set(zeros)
        latency = bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None
        capture.begin_step(); obs=bare._apply_policy_observation_latency(bare.observation_manager.compute()); observed=capture.finish_step(obs)
        terminals={}; active_step=-1
        pending_contacts=[];contact_api_status={'available':False,'reason':'field has no analytic fixture classification'}
        if args.scene!='field':
            sensor=bare.scene['contact_forces']
            # Isaac Lab supports one-body-to-many filtering, not many-to-many.
            contact_views=[(name,sensor._physics_sim_view.create_rigid_contact_view('/World/envs/env_*/onix_chassis/'+name,filter_patterns=[f'/World/R6Fixtures/Tile*/Box{j}' for j in range(len(fixture_boxes(args.scene)))],max_contact_data_count=64*n)) for name in sensor.body_names]
            original_scene_update=bare.scene.update
            def contact_update(*values,**kwargs):
                original_scene_update(*values,**kwargs)
                try:
                    origins_now=bare.scene.env_origins.cpu().numpy()
                    raw_count=0;new_contact_count=0
                    gpu_values=[view.get_contact_data(bare.physics_dt) for _,view in contact_views]
                    # One synchronization for all empty/contact counts, rather
                    # than six CPU transfers per link on every physics tick.
                    all_counts=torch.stack([values[4] for values in gpu_values]).cpu().numpy()
                    raw_count=int(all_counts.sum())
                    for body_view in np.flatnonzero(all_counts.reshape(len(contact_views),-1).any(axis=1)):
                        name,_=contact_views[int(body_view)]
                        values=[v.cpu().numpy() for v in gpu_values[int(body_view)]]
                        new_contacts=decode_fixture_contacts(*values,origins_now,args.scene,name,bare._sim_step_counter,bare.physics_dt)
                        new_contact_count+=len(new_contacts);pending_contacts.extend(new_contacts)
                        if new_contacts and 'first_actual_contact' not in contact_api_status:contact_api_status['first_actual_contact']=new_contacts[0]
                        for point in new_contacts:contact_api_status.setdefault('first_contact_ns_by_env',{}).setdefault(str(point['env_index']),point['logical_sim_time_ns'])
                    contact_api_status.update(available=all(view.filter_count>0 for _,view in contact_views) and contact_api_status.get('read_error_count',0)==0,reason=None if contact_api_status.get('read_error_count',0)==0 else 'one_or_more_filtered_contact_reads_failed',
                        raw_contact_points_seen=contact_api_status.get('raw_contact_points_seen',0)+raw_count,classified_contact_points_seen=contact_api_status.get('classified_contact_points_seen',0)+new_contact_count,
                        tensor_api_called=True,body_view_count=len(contact_views),definition='one-body-per-env PhysX contact points from exact fixture filter columns; separate floor excluded',sample_period_ns=int(round(bare.physics_dt*1e9)))
                except Exception as exc:
                    contact_api_status.update(available=False,read_error_count=contact_api_status.get('read_error_count',0)+1,reason=type(exc).__name__+': '+str(exc))
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
            detail['executed_action']=action.detach().cpu().tolist()
            detail['actor_input']=policy_input(obs).detach().cpu().tolist()
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
        # Visible-state inventory includes manager/filter/controller/sensor clocks;
        # PhysX solver/contact warm-start caches remain explicitly unavailable.
        def runtime_objects():
            objects=[('env',bare),('robot_data',robot.data),('action_manager',bare.action_manager),('command_manager',bare.command_manager),('command_override',override),
                     ('termination_manager',bare.termination_manager),('event_manager',bare.event_manager)]
            for manager_name in ('action_manager','command_manager'):
                for key,obj in getattr(bare,manager_name)._terms.items():objects.append((manager_name+'.'+key,obj))
            for key,sensor in bare.scene.sensors.items():
                objects.extend([(key,sensor),(key+'.data',sensor._data)])
            for key,actuator in robot.actuators.items():objects.append(('actuator.'+key,actuator))
            for key,delay in getattr(bare,'_action_latency_buffers',{}).items():
                objects.extend([('delay.'+key,delay),('delay.'+key+'.ring',delay._circular_buffer)])
            for group,terms in getattr(bare.observation_manager,'_group_obs_term_history_buffer',{}).items():
                for key,ring in terms.items():objects.append(('obs_history.'+group+'.'+key,ring))
            # Delay actuators/filters can own additional circular buffers.
            for label,obj in list(objects):
                for name,value in vars(obj).items():
                    if hasattr(value,'_circular_buffer'):objects.extend([(label+'.'+name,value),(label+'.'+name+'.ring',value._circular_buffer)])
            return objects
        state_objects=runtime_objects()
        def numeric_tree(value):
            if torch.is_tensor(value) or isinstance(value,(int,float,bool,np.ndarray)):return True
            if isinstance(value,(list,tuple)):return bool(value) and all(numeric_tree(v) for v in value)
            if isinstance(value,dict):return bool(value) and all(isinstance(k,(str,int)) and numeric_tree(v) for k,v in value.items())
            return False
        def capture_runtime():
            attrs=[]
            for label,obj in state_objects:
                for name,value in vars(obj).items():
                    if torch.is_tensor(value):attrs.append((label,obj,name,value.clone()))
                    elif numeric_tree(value):attrs.append((label,obj,name,copy.deepcopy(value)))
            # Public target buffers may be properties, rather than instance attrs.
            for name in ('joint_pos_target','joint_vel_target','joint_effort_target'):
                value=getattr(robot.data,name,None)
                if torch.is_tensor(value):attrs.append(('robot_data',robot.data,name,value.clone()))
            command_sampler=bare.command_manager._terms['base_velocity'].sampler
            return dict(scene=copy.deepcopy(bare.scene.get_state()),attrs=attrs,lazy_caches={label:capture_lazy_caches(obj) for label,obj in state_objects},actual_root=robot.data.root_state_w.clone(),actual_joint=robot.data.joint_pos.clone(),actual_joint_velocity=robot.data.joint_vel.clone(),
                obs=copy.deepcopy(obs),observed=copy.deepcopy(observed),latency=copy.deepcopy(latency),
                sampler=copy.deepcopy(vars(command_sampler)),initial_rngs=copy.deepcopy(bare.eat_nav_initial_rngs),
                rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate()))
        def restore_runtime(state):
            nonlocal obs,observed,latency
            bare.scene.reset_to(state['scene'],is_relative=False)
            for label,obj,name,value in state['attrs']:
                current=getattr(obj,name)
                if torch.is_tensor(current):current.copy_(value)
                else:setattr(obj,name,copy.deepcopy(value))
            for label,obj in state_objects:restore_lazy_caches(obj,state['lazy_caches'][label])
            vars(bare.command_manager._terms['base_velocity'].sampler).update(copy.deepcopy(state['sampler']))
            bare.eat_nav_initial_rngs=copy.deepcopy(state['initial_rngs'])
            torch.set_rng_state(state['rng'][0]);torch.cuda.set_rng_state_all(state['rng'][1]);np.random.set_state(state['rng'][2]);random.setstate(state['rng'][3])
            obs=copy.deepcopy(state['obs']);observed=copy.deepcopy(state['observed']);latency=copy.deepcopy(state['latency'])
            bare.sim.forward();bare.scene.update(0.)
        def visible_state():
            result={}
            def flatten(label,value):
                if torch.is_tensor(value):result[label]=value.detach().cpu().numpy()
                elif isinstance(value,np.ndarray):result[label]=value.copy()
                elif isinstance(value,dict):
                    for key,item in value.items():flatten(label+'.'+str(key),item)
                elif isinstance(value,(list,tuple)):
                    for index,item in enumerate(value):flatten(label+'.'+str(index),item)
                elif isinstance(value,(float,int,bool)):result[label]=np.asarray(value,dtype=np.float64)
            state=capture_runtime()
            for label,obj,name,value in state['attrs']:flatten(label+'.'+name,value)
            flatten('lazy_cache',state['lazy_caches'])
            result.update(root=robot.data.root_state_w.cpu().numpy(),actor_input=policy_input(obs).cpu().numpy(),actor_pre_noise=observed['policy_pre_noise'].cpu().numpy())
            if latency is not None:result['latency']=latency.cpu().numpy()
            result['rng.torch']=torch.get_rng_state().cpu().numpy()
            for index,value in enumerate(torch.cuda.get_rng_state_all()):result['rng.cuda.'+str(index)]=value.cpu().numpy()
            import hashlib,pickle
            for label,value in [('numpy',np.random.get_state()),('python',random.getstate()),('sampler',vars(bare.command_manager._terms['base_velocity'].sampler)),('initial',bare.eat_nav_initial_rngs)]:
                result['rng.'+label]=np.frombuffer(hashlib.sha256(pickle.dumps(value)).digest(),dtype=np.uint8).copy()
            return result
        if args.fresh_process:
            candidate=args.candidate_indices[0]
            origins=bare.scene.env_origins.cpu().numpy();metadata=getattr(generator,'eat_nav_tile_metadata',[])
            library=compile_core(args.output/'core-build')
            write(args.output/'terrain.json',dict(terrain=terrain,tile_metadata=metadata,env_origins=origins,control_hz=20,scene=args.scene))
            write(args.output/'manifest.json',dict(schema='eatnav_moving_pose2s_raw_v1',groups=GROUPS,headless=True,
                seed=args.seed,actor_sha256=sha(args.rl_checkpoint),actor_env_sha256=sha(args.actor_env_yaml),eat_config_sha256=sha(args.eat_config),
                restoration_method='fresh_process_prefix',physics_cache_restorable=False,prefix_state=args.prefix_state,candidate_indices=[candidate],
                append_enabled=False,internal_replan_enabled=False,protective_stop_enabled=False))
            histories=[[] for _ in range(n)];stamps=[];prefix_rows=[[] for _ in range(n)];branch_rows=[[] for _ in range(n)]
            invalid=[None]*n;finished=np.zeros(n,dtype=bool);anchor_steps=[None]*n;anchor_times=[None]*n;qualifications=[None]*n
            follower_list=[None]*n;paths_world=[None]*n;vz_history=[[] for _ in range(n)]
            trackers=[AnchorQualification(args.prefix_state) for _ in range(n)]
            frozen_steps=[int(x) for x in args.prefix_steps.split(',')] if args.prefix_steps else None
            if frozen_steps is not None and len(frozen_steps)!=n:raise ValueError('one frozen prefix step per parent required')
            reset_roots=robot.data.root_state_w.cpu().numpy().copy()
            prefix_qualifications=[None]*n
            for step in range(20):
                terminals.clear();active_step=-1;done,_=advance(zeros)
                for i in range(n):
                    if bool(done[i]):invalid[i]='termination_during_settling';finished[i]=True
                if (step+1)%5==0:
                    stamps.append((step+1)*STEP_NS)
                    for i in range(n):histories[i].append(dem(bare,config,i,stamps[-1]))
            started_collection=time.perf_counter();total_rows=0
            for global_step in range(440):
                now=1_000_000_000+global_step*STEP_NS
                pre=robot.data.root_state_w.cpu().numpy().copy();before=[physical(bare,i) for i in range(n)]
                for i in range(n):
                    if finished[i] or anchor_steps[i] is not None:continue
                    if frozen_steps is not None and frozen_steps[i]<=0:
                        invalid[i]='unavailable_in_frozen_prefix_assignment';finished[i]=True;continue
                    if global_step>=400 and (frozen_steps is None or global_step!=frozen_steps[i]):
                        invalid[i]='qualification_not_reached_within_20s';finished[i]=True;continue
                    metrics=dem_metrics(histories[i][-1],config.dem_resolution_m)
                    metrics.update(physical_metrics(pre[i],vz_history[i] or [0.]))
                    metrics.update(dem_history_seconds=(stamps[-1]-stamps[-4])/1e9 if len(stamps)>=4 else 0.,command=prefix_command(args.prefix_state,global_step*.05))
                    result=trackers[i].observe(metrics,now);prefix_qualifications[i]=result
                    ready=result['ready'] and len(vz_history[i])>=5
                    due=ready if frozen_steps is None else global_step==frozen_steps[i]
                    if frozen_steps is not None and global_step>frozen_steps[i]:
                        invalid[i]='frozen_prefix_step_missed';finished[i]=True;continue
                    if not due:continue
                    if not ready:
                        invalid[i]='qualification_changed_at_frozen_prefix_step';finished[i]=True;continue
                    if not anchor_inside_guard(pre[i,:2],origins[i,:2]):
                        invalid[i]='qualified_anchor_outside_world_guard';finished[i]=True
                        rejected=args.output/f'group-{i:03d}';rejected.mkdir(exist_ok=True)
                        np.savez_compressed(rejected/'rejected-anchor-state.npz',root_state_w=pre[i:i+1],env_origin=origins[i],t0_ns=np.asarray(now,dtype=np.int64))
                        continue
                    anchor_steps[i]=global_step;anchor_times[i]=now;qualifications[i]=result
                    directory=args.output/f'group-{i:03d}';directory.mkdir(exist_ok=True)
                    arrays=pack_inputs(np.asarray(histories[i][-4:]),observed['policy_pre_noise'][i].cpu().numpy(),body_state(pre[i]))
                    extra=dict(dem_stamps_ns=np.asarray(stamps[-4:],dtype=np.int64),root_state_w=pre[i:i+1],t0_ns=np.asarray(now,dtype=np.int64))
                    np.savez_compressed(directory/'initial-input.npz',**arrays,**extra)
                    np.savez_compressed(directory/'reset-state.npz',root_state_w=reset_roots[i:i+1],env_origin=origins[i],reset_x_m=SPAWN_X,reset_yaw_rad=SPAWN_YAW)
                    np.savez_compressed(directory/'anchor-body-world.npz',position=robot.data.body_pos_w[i].cpu().numpy(),quaternion=robot.data.body_quat_w[i].cpu().numpy(),body_names=np.asarray(robot.body_names))
                    np.savez_compressed(directory/'anchor-state.npz',root=pre[i],joint_position=before[i]['joint_position'],joint_velocity=before[i]['joint_velocity'],dem=np.asarray(histories[i][-4:]))
                    visible=parent_visible(visible_state(),i,n)
                    visible['dem_history']=np.asarray(histories[i][-4:]);visible['dem_stamps_ns']=np.asarray(stamps[-4:],dtype=np.int64)
                    np.savez_compressed(directory/'anchor-visible-state.npz',**visible)
                    branch=directory/f'candidate-{candidate:02d}';branch.mkdir()
                    np.savez_compressed(branch/'input.npz',**arrays,**extra)
                    p=pose(robot,i);points=candidate_paths()[candidate]@np.asarray([[math.cos(p[2]),math.sin(p[2])],[-math.sin(p[2]),math.cos(p[2])]])+p[:2]-origins[i,:2]
                    paths_world[i]=points+origins[i,:2];follower_list[i]=PureFollower(points.tolist(),library=library)
                if finished.all():break
                commands=[];diagnostics=[None]*n;branch_active=np.zeros(n,dtype=bool)
                prefix_active=np.zeros(n,dtype=bool)
                for i in range(n):
                    if finished[i]:commands.append([0.,0.,0.]);continue
                    if anchor_steps[i] is None:
                        commands.append(prefix_command(args.prefix_state,global_step*.05));prefix_active[i]=True
                    else:
                        p=pose(robot,i);command,diagnostic=follower_list[i].step_diagnostic(float(p[0]-origins[i,0]),float(p[1]-origins[i,1]),float(p[2]),now)
                        if diagnostic.get('linear_recovery_started') or diagnostic.get('angular_recovery_started'):raise ValueError('fixed2s follower recovery forbidden')
                        commands.append(command);diagnostics[i]=diagnostic;branch_active[i]=True
                terminals.clear();active_step=global_step;done,detail=advance(torch.tensor(commands,device=bare.device,dtype=zeros.dtype))
                post=robot.data.root_state_w.cpu().numpy().copy()
                rows=step_records(pre,post,commands,done.cpu().tolist(),terminals,prefix_active|branch_active,global_step,candidate,1_000_000_000)
                for i,row in enumerate(rows):
                    if prefix_active[i]:
                        row.update(before[i]);row.update(phase='prefix',qualification=prefix_qualifications[i],actor_command=detail['actor_seen'][i],executed_action=detail['executed_action'][i])
                        prefix_rows[i].append(row);vz_history[i].append(float(row['post_root_state'][9]));vz_history[i]=vz_history[i][-5:]
                        if bool(done[i]):invalid[i]='termination_during_prefix';finished[i]=True
                    if not branch_active[i]:continue
                    local_step=global_step-anchor_steps[i];physical_post=np.asarray(row['post_root_state'])
                    if not np.isfinite(pre[i]).all() or not np.isfinite(physical_post).all():raise ValueError('nonfinite actual fresh-process state')
                    boundary=bool(np.any(abs(physical_post[:2]-origins[i,:2])>7.95))
                    row.update(before[i]);row.update(control_step=local_step,logical_sim_time_ns=now,post_logical_sim_time_ns=now+STEP_NS,post_timestamp_ns=now+STEP_NS,
                        candidate_id=CANDIDATE_IDS[candidate],group_id=GROUPS[i]['group_id'],role=GROUPS[i]['role'],episode_id=GROUPS[i]['group_id'],
                        actor_command=detail['actor_seen'][i],actor_input=detail['actor_input'][i],executed_action=detail['executed_action'][i],
                        follower_diagnostic=diagnostics[i],executed_path_world_xy=paths_world[i].tolist(),append_events=[],candidate_replan=local_step==0,
                        boundary_censor=boundary,phase='candidate',body_angular_velocity_radps=body_vector(row['post_root_state'],row['post_root_state'][10:13]),wall_clock_ns=time.time_ns())
                    if boundary and not row['stop_reason']:row['stop_reason']='boundary_censor'
                    if local_step==39 and not row['stop_reason']:row['stop_reason']='horizon_complete'
                    branch_rows[i].append(row);total_rows+=1
                    if row['stop_reason']:finished[i]=True
                if (global_step+1)%5==0:
                    stamps.append(now+STEP_NS)
                    for i in range(n):histories[i].append(dem(bare,config,i,stamps[-1]))
            for i,group in enumerate(GROUPS):
                directory=args.output/f'group-{i:03d}';directory.mkdir(exist_ok=True)
                group_stamps=[]
                if (directory/'initial-input.npz').exists():
                    with np.load(directory/'initial-input.npz') as data:group_stamps=data['dem_stamps_ns'].tolist()
                meta=dict(group,anchor_available=anchor_steps[i] is not None,anchor_failure_reason=invalid[i],state=args.prefix_state,
                    prefix_control_steps=anchor_steps[i],t0_ns=anchor_times[i],dem_stamps_ns=group_stamps,qualification=qualifications[i],counterfactual_complete=False,
                    reset_recipe=RESET_RECIPE,world_guard_halfwidth_m=7.95,qualification_persistence_samples=4,restoration_method='fresh_process_prefix',terrain=terrain,env_origin=origins[i],scene=args.scene,parent_terrain_seed=group['seed'],
                    parent_tile_column=TILE_INDICES[i],actual_tile_metadata=metadata[TILE_INDICES[i]*5+args.level] if metadata else None)
                write(directory/'group.json',meta)
                with (directory/'prefix-trace.jsonl').open('x') as stream:
                    for row in prefix_rows[i]:stream.write(json.dumps(row,allow_nan=False)+'\n')
                if anchor_steps[i] is not None:
                    branch=directory/f'candidate-{candidate:02d}'
                    with (branch/'trace.jsonl').open('x') as stream:
                        for row in branch_rows[i]:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    write(branch/'complete.json',dict(candidate=candidate,control_steps=len(branch_rows[i]),stop_reason=branch_rows[i][-1]['stop_reason'] if branch_rows[i] else 'trace_cutoff',horizon_seconds=2.,smoke=False,restoration_method='fresh_process_prefix'))
                if follower_list[i] is not None:follower_list[i].close()
            elapsed=time.perf_counter()-started_collection
            write(args.output/'complete.json',dict(state='completed',num_envs=n,candidate_indices=[candidate],recorded_rows=total_rows,elapsed_collection_seconds=elapsed,
                rows_per_second=total_rows/max(elapsed,1e-6),anchor_available_by_env=[x is not None for x in anchor_steps],anchor_failure_reasons=invalid,
                prefix_steps=anchor_steps,restoration_method='fresh_process_prefix',headless=True,smoke=False,physics_audit=physics_audit,
                peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),gpu_memory_free_total_bytes=list(torch.cuda.mem_get_info())))
            return
        prefix_initial=capture_runtime()
        history=[[] for _ in range(n)];stamps=[];prefix_rows=[[] for _ in range(n)]
        anchor_valid=np.ones(n,dtype=bool);anchor_failure_reasons=[None]*n
        for step in range(20):
            terminals.clear();done,_=advance(zeros)
            failed_now=anchor_valid & done.cpu().numpy().astype(bool)
            anchor_valid=update_anchor_valid(anchor_valid,done.cpu().numpy())
            for i in np.flatnonzero(failed_now):anchor_failure_reasons[int(i)]='termination_during_settling'
            if (step+1)%5==0:
                stamps.append((step+1)*STEP_NS)
                for i in range(n):history[i].append(dem(bare,config,i,stamps[-1]))
        trackers=[AnchorQualification(args.prefix_state) for _ in range(n)]
        selection=CommonWorldAnchor(n);selected_runtime=None;selected_visible=None
        selected_history=None;selected_stamps=None;selected_details=None
        vz_history=[[] for _ in range(n)]
        for step in range(400):
            active_step=step;terminals.clear();pending_contacts.clear()
            pre=robot.data.root_state_w.cpu().numpy().copy();before=[physical(bare,i) for i in range(n)]
            command=prefix_command(args.prefix_state,step*.05)
            commands=torch.tensor([command if anchor_valid[i] else [0.,0.,0.] for i in range(n)],device=bare.device,dtype=zeros.dtype)
            done,detail=advance(commands)
            now=1_000_000_000+(step+1)*STEP_NS
            post=robot.data.root_state_w.cpu().numpy().copy()
            rows=step_records(pre,post,commands.cpu().numpy(),done.cpu().tolist(),terminals,anchor_valid,step,-1,1_000_000_000)
            if (step+1)%5==0:
                stamps.append(now)
                for i in range(n):history[i].append(dem(bare,config,i,now))
            failed_now=anchor_valid & done.cpu().numpy().astype(bool)
            for i in np.flatnonzero(failed_now):anchor_failure_reasons[int(i)]='termination_during_prefix'
            anchor_valid=update_anchor_valid(anchor_valid,done.cpu().numpy())
            ready=[False]*n;details=[None]*n
            for i,row in enumerate(rows):
                if not anchor_valid[i]:continue
                root=post[i];vz_history[i].append(float(root[9]));vz_history[i]=vz_history[i][-5:]
                metrics=dem_metrics(history[i][-1],config.dem_resolution_m)
                metrics.update(physical_metrics(root,vz_history[i]))
                metrics.update(dem_history_seconds=(stamps[-1]-stamps[-4])/1e9 if len(stamps)>=4 else 0.,command=command)
                result=trackers[i].observe(metrics,now)
                row.update(before[i]);row.update(phase='prefix',qualification=result,actor_command=detail['actor_seen'][i],executed_action=detail['executed_action'][i])
                prefix_rows[i].append(row)
                ready[i]=bool(result['ready'] and len(vz_history[i])>=5);details[i]=result
            if selection.consider(ready,step+1)=='capture':
                selected_runtime=capture_runtime();selected_visible=visible_state()
                selected_history=[copy.deepcopy(frames[-4:]) for frames in history]
                selected_stamps=list(stamps[-4:]);selected_details=copy.deepcopy(details)
            if selection.complete or not anchor_valid.any():break
        if selected_runtime is None:
            selected_runtime=capture_runtime();selected_visible=visible_state()
            selected_history=[copy.deepcopy(frames[-4:]) for frames in history]
            selected_stamps=list(stamps[-4:]);selected_details=[None]*n;selection.step=step+1
        anchor_runtime=selected_runtime
        history=selected_history;stamps_by_env=[list(selected_stamps) for _ in range(n)]
        qualified_steps=[selection.step]*n;qualified_details=selected_details
        anchor_valid=np.asarray(selection.mask,dtype=bool)
        for i in range(n):
            if anchor_valid[i]:anchor_failure_reasons[i]=None
            else:anchor_failure_reasons[i]=anchor_failure_reasons[i] or 'not_qualified_at_selected_common_world_step'
        branch_starts_ns=[1_000_000_000+selection.step*STEP_NS]*n
        stamps=stamps_by_env[0];branch_start_ns=branch_starts_ns[0]
        restore_runtime(anchor_runtime)
        visible_anchor=selected_visible
        prefix_check=compare_visible_state({k:v for k,v in visible_anchor.items() if k!='root'},
            {k:v for k,v in visible_state().items() if k!='root'})
        roots_reference=anchor_runtime['actual_root'].cpu().numpy()
        root_checks=[compare_visible_state({'root':roots_reference[i]},{'root':robot.data.root_state_w[i].cpu().numpy()}) for i in range(n)]
        prefix_check.update(root_checks=root_checks,common_world_step=selection.step,qualified_by_env=selection.mask,
            strategy='one full-world snapshot; no per-env clock or RNG stitching')
        write(args.output/'prefix-snapshot-checks.json',prefix_check)
        if not prefix_check['passed'] or not all(c['passed'] for c in root_checks):
            raise ValueError('qualified full-world visible state restoration failed: '+str(prefix_check))
        np.savez_compressed(args.output/'anchor-visible-state.npz',**visible_anchor)
        torch.save(dict(scene=anchor_runtime['scene'],lazy_caches=anchor_runtime['lazy_caches'],buffers={label+'.'+name:value for label,obj,name,value in anchor_runtime['attrs']},
            observations=anchor_runtime['obs'],observed=anchor_runtime['observed'],latency=anchor_runtime['latency'],sampler=anchor_runtime['sampler'],rng=anchor_runtime['rng'],initial_rngs=anchor_runtime['initial_rngs']),args.output/'observable-runtime-snapshot.pt')
        write(args.output/'state-inventory.json',dict(saved_visible_buffers=sorted(visible_anchor),buffer_shapes_dtypes={k:dict(shape=list(np.shape(v)),dtype=str(np.asarray(v).dtype)) for k,v in visible_anchor.items()},
            saved='scene articulation rigidbody state; joint targets; action command; actuator/delay/filter tensors; sensor clocks; manager timers; observations/latency; command sampler/initial RNG; torch CUDA numpy python RNG; causal DEM history',
            unavailable=['PhysX contact warm-start and constraint solver caches'],restoration_method=args.restoration_method,qualification_strategy='full_world_common_control_step',feedforward_actor=True))
        groups = json.loads(args.groups.read_text()) if args.groups else [dict(group_id=f'benchmark-{args.seed}-{i}',role='benchmark',seed=args.seed,env_index=i) for i in range(n)]
        if isinstance(groups,dict): groups=groups['groups']
        if len(groups)!=n or len({g['group_id'] for g in groups})!=n:raise ValueError('unique frozen group per environment required')
        if any(g['role'] not in ('train','validation','audit-eval','benchmark') for g in groups):raise ValueError('unknown split role')
        origins=bare.scene.env_origins.cpu().numpy()
        if args.scene!='field':
            np.savez_compressed(args.output/'fixture-raycast-evidence.npz',ray_hits_w=bare.scene['height_scanner_eat'].data.ray_hits_w.cpu().numpy(),sensor_pos_w=bare.scene['height_scanner_eat'].data.pos_w.cpu().numpy(),origins=origins)
        metadata=getattr(generator,'eat_nav_tile_metadata',[])
        write(args.output/'terrain.json',dict(terrain=terrain,scene=args.scene,fixtures=fixture_boxes(args.scene),
              subterrain_config=str(generator.sub_terrains[args.terrain_type].to_dict()),
              tile_metadata=metadata,env_origins=origins,control_hz=20,dem_sensor_enabled=True,fixture_physics_representation='combined Warp mesh collision disabled; identical floor and kinematic rigid cuboids' if args.scene!='field' else 'original terrain mesh'))
        write(args.output/'manifest.json',dict(schema='eatnav_moving_pose2s_raw_v1',groups=groups,headless=args.headless,
              seed=args.seed,actor_sha256=sha(args.rl_checkpoint),actor_env_sha256=sha(args.actor_env_yaml),
              eat_config_sha256=sha(args.eat_config),terrain_type=args.terrain_type,level=args.level,
              scene=args.scene,diagnostic_forced_contact=args.contact_calibration_probe,prefix_state=args.prefix_state,prefix_search_limit_sec=20,restoration_method=args.restoration_method,physics_cache_restorable=False,append_enabled=False,internal_replan_enabled=False,protective_stop_enabled=False,geometry_scope='each env retains its own tile across all branches',
              dem_ray_origin_height_m=2. if args.scene!='field' else .3,contact_definition='field: obstacle identity unknown; supplemental: filtered rigidfixture normal contact points sampled every5ms'))
        initial_root=robot.data.root_state_w.clone(); initial_joint=robot.data.joint_pos.clone(); initial_vel=robot.data.joint_vel.clone()
        scene_state=anchor_runtime['scene']
        saved_tensors=[(obj,name,value) for _,obj,name,value in anchor_runtime['attrs']]
        independent_reset_check=None
        if n>1 and (args.smoke_steps or args.duplicate_gate):
            before_root=robot.data.root_state_w.clone();before_joint=robot.data.joint_pos.clone()
            bare._reset_idx(torch.tensor([0],device=bare.device))
            root_error=float(torch.max(abs(robot.data.root_state_w[1:]-before_root[1:])).item())
            joint_error=float(torch.max(abs(robot.data.joint_pos[1:]-before_joint[1:])).item())
            independent_reset_check=dict(reset_env=0,unaffected_env_count=n-1,root_max_abs_delta=root_error,joint_max_abs_delta=joint_error,terminal_hook_captured=0 in terminals,passed=root_error==0 and joint_error==0 and 0 in terminals)
            restore_runtime(anchor_runtime)
            if not independent_reset_check['passed']:raise ValueError('cross-environment reset contamination')
        rng=anchor_runtime['rng']
        anchor=dict(root=initial_root.cpu().numpy(),joint_position=initial_joint.cpu().numpy(),joint_velocity=initial_vel.cpu().numpy(),dem=np.asarray(history),live_dem=np.asarray([dem(bare,config,i,stamps_by_env[i][-1]) for i in range(n)]))
        for i,group in enumerate(groups):
            directory=args.output/f'group-{i:03d}';directory.mkdir(exist_ok=True)
            write(directory/'group.json',dict(group,anchor_available=bool(anchor_valid[i]),pre_anchor_contact_observation_scope='settling_and_prefix_5ms' if args.scene!='field' else 'unknown',anchor_failure_reason=anchor_failure_reasons[i],observed_contact_before_anchor=(str(i) in contact_api_status.get('first_contact_ns_by_env',{})) if args.scene!='field' else None,prefix_contact_observation_complete=bool(contact_api_status.get('available')) if args.scene!='field' else False,terrain=terrain,env_origin=origins[i],scene=args.scene,dem_stamps_ns=stamps_by_env[i],t0_ns=branch_starts_ns[i],state=args.prefix_state,qualification=qualified_details[i],prefix_control_steps=qualified_steps[i],counterfactual_complete=False,parent_terrain_seed=group['seed'],parent_tile_column=TILE_INDICES[i],tile_generation_index=TILE_INDICES[i]*5+args.level,actual_tile_metadata=metadata[TILE_INDICES[i]*5+args.level] if metadata else None))
            with (directory/'prefix-trace.jsonl').open('x') as prefix_stream:
                for row in prefix_rows[i]:prefix_stream.write(json.dumps(row,allow_nan=False)+'\n')
            arrays=pack_inputs(np.asarray(history[i]),observed['policy_pre_noise'][i].cpu().numpy(),body_state(initial_root[i].cpu().numpy()))
            np.savez_compressed(directory/'initial-input.npz',**arrays,dem_stamps_ns=np.asarray(stamps_by_env[i],dtype=np.int64),root_state_w=initial_root[i:i+1].cpu().numpy(),t0_ns=np.asarray(branch_starts_ns[i],dtype=np.int64))
            np.savez_compressed(directory/'anchor-state.npz',**{key:value[i] for key,value in anchor.items()})
        def replay_prefix():
            # Begin with a real environment reset, then restore the recorded
            # initial reset state/controller/RNG. The moving anchor is never
            # settled or teleported: it is reached by the original commands.
            restore_runtime(prefix_initial)
            bare._reset_idx(torch.arange(n,device=bare.device))
            restore_runtime(prefix_initial)
            replay_history=[[] for _ in range(n)];replay_stamps=[]
            replay_snaps=[None]*n;replay_frames=[None]*n;replay_times=[None]*n
            for k in range(20):
                terminals.clear();done,_=advance(zeros)
                if bool(done.any()):raise ValueError('matched-prefix replay terminated during settling')
                if (k+1)%5==0:
                    replay_stamps.append((k+1)*STEP_NS)
                    for i in range(n):replay_history[i].append(dem(bare,config,i,replay_stamps[-1]))
            for k in range(max(qualified_steps)):
                terminals.clear()
                c=prefix_command(args.prefix_state,k*.05)
                commands=torch.tensor([c for _ in range(n)],device=bare.device,dtype=zeros.dtype)
                done,_=advance(commands)
                if any(bool(done[i]) and anchor_valid[i] and k<qualified_steps[i] for i in range(n)):
                    raise ValueError('matched-prefix replay naturally terminated before recorded anchor')
                now=1_000_000_000+(k+1)*STEP_NS
                if (k+1)%5==0:
                    replay_stamps.append(now)
                    for i in range(n):replay_history[i].append(dem(bare,config,i,now))
                for i in range(n):
                    if k+1==qualified_steps[i]:
                        replay_snaps[i]=capture_runtime();replay_frames[i]=copy.deepcopy(replay_history[i][-4:]);replay_times[i]=replay_stamps[-4:]
            if len(set(qualified_steps))!=1:raise ValueError('asynchronous prefix clocks unsupported')
            result=replay_snaps[0];restore_runtime(result)
            if replay_times!=stamps_by_env:raise ValueError('matched-prefix DEM timestamps changed')
            return replay_frames
        library=compile_core(args.output/'core-build')
        total_rows=0; start=time.perf_counter(); checks=[]
        sequence=([0,1] if args.smoke_steps else list(range(14)))
        if args.duplicate_gate:sequence=list(range(14))+[0]+list(reversed(range(14)))+[0]
        run_summaries=[]
        for run_index,candidate in enumerate(sequence):
            if args.restoration_method=='matched_prefix':
                history=replay_prefix()
            else:restore_runtime(anchor_runtime)
            visible_current=visible_state()
            errors={}
            for key,expected in visible_anchor.items():
                current=visible_current.get(key)
                try:
                    expected=np.asarray(expected,dtype=np.float64);current=np.asarray(current,dtype=np.float64)
                except (TypeError,ValueError) as exc:raise ValueError('nonnumeric visible buffer '+key+': '+str(exc)) from exc
                if current is None or np.shape(expected)!=np.shape(current) or not np.allclose(expected,current,atol=1e-5,rtol=1e-5,equal_nan=False):
                    errors[key]='shape' if current is None or np.shape(expected)!=np.shape(current) else float(np.max(np.abs(expected-current)))
            visible_check=dict(passed=not errors,errors=errors)
            if not visible_check['passed']:
                write(args.output/'visible-restore-failure.json',dict(run_index=run_index,candidate=candidate,**visible_check))
                raise ValueError('visible state restoration failed: '+str(errors))
            current=dict(root=robot.data.root_state_w.cpu().numpy(),joint_position=robot.data.joint_pos.cpu().numpy(),joint_velocity=robot.data.joint_vel.cpu().numpy(),dem=np.asarray(history),live_dem=np.asarray([dem(bare,config,i,stamps_by_env[i][-1]) for i in range(n)]))
            # Historical frames are saved causal evidence, not re-rendered at anchor.
            current['dem']=np.asarray(history)
            check=verify_anchor(anchor,current);checks.append(dict(candidate=candidate,**check))
            if not check['passed']:raise ValueError('branch anchor restoration failed: '+str(check))
            paths_world=[];terminal_headings=[];append_counts=[0]*n
            followers=[]; streams=[]; active=anchor_valid.copy();counts=[0]*n; reasons=['horizon_complete' if anchor_valid[i] else anchor_failure_reasons[i] for i in range(n)]
            for i in range(n):
                directory=args.output/f'group-{i:03d}'/(f'candidate-{candidate:02d}' if run_index<14 else f'duplicate-{run_index:02d}-candidate-{candidate:02d}');directory.mkdir(exist_ok=True)
                paths=candidate_paths();np.savez_compressed(directory/'input.npz',**pack_inputs(np.asarray(history[i]),observed['policy_pre_noise'][i].cpu().numpy(),body_state(robot.data.root_state_w[i].cpu().numpy())),dem_stamps_ns=np.asarray(stamps_by_env[i],dtype=np.int64),root_state_w=robot.data.root_state_w[i:i+1].cpu().numpy(),t0_ns=np.asarray(branch_starts_ns[i],dtype=np.int64))
                streams.append((directory/'trace.jsonl').open('x'))
                p=pose(robot,i);points=candidate_paths()[candidate]@np.asarray([[math.cos(p[2]),math.sin(p[2])],[-math.sin(p[2]),math.cos(p[2])]])+p[:2]-origins[i,:2]
                paths_world.append(points+origins[i,:2]);terminal_headings.append(float(p[2]+ANGLES[candidate]))
                followers.append(PureFollower(points.tolist(),library=library))
            for step in range(args.smoke_steps or 40):
                if not active.any():break
                started=time.perf_counter()
                active_step=step;terminals.clear();pending_contacts.clear()
                pre=robot.data.root_state_w.cpu().numpy().copy();before=[physical(bare,i) for i in range(n)]
                commands=[]; diagnostics=[];append_events=[[] for _ in range(n)]
                for i in range(n):
                    p=pose(robot,i)
                    c,d=followers[i].step_diagnostic(float(p[0]-origins[i,0]),float(p[1]-origins[i,1]),float(p[2]),branch_starts_ns[i]+step*STEP_NS)
                    if d.get('linear_recovery_started') or d.get('angular_recovery_started'):raise ValueError('follower recovery intervention is forbidden within fixed 2s')
                    commands.append(c if active[i] else [0.,0.,0.]);diagnostics.append(d)
                timings['snapshots_and_follower']+=time.perf_counter()-started
                done,detail=advance(torch.tensor(commands,dtype=zeros.dtype,device=bare.device))
                rows=step_records(pre,robot.data.root_state_w.cpu().numpy(),commands,done.cpu().tolist(),terminals,active,step,candidate,branch_start_ns)
                started=time.perf_counter()
                for i,row in enumerate(rows):
                    if not active[i]:continue
                    post=row['post_root_state']
                    if not np.isfinite(pre[i]).all() or not np.isfinite(post).all():
                        write(args.output/'invalid-data-marker.json',dict(valid_mask=False,reason='nonfinite_sim_state',env_index=i,candidate=candidate,control_step=step))
                        raise ValueError('nonfinite simulation state; invalid-data-marker.json saved')
                    boundary=np.any(np.abs(np.asarray(post[:2])-origins[i,:2])>7.95)
                    row.update(before[i]);row['root_state']=pre[i].tolist()
                    if args.scene!='field':
                        contacts=[entry for entry in pending_contacts if entry['env_index']==i]
                        row['contact_observation']=dict(contact_api_status,onset=bool(contacts) if contact_api_status['available'] else None,contacts=contacts)
                    if row['contact_observation']['available']:
                        points=row['contact_observation']['contacts'];row['obstacle_contact']=bool(points);row['obstacle_contact_time_ns']=min((point['logical_sim_time_ns'] for point in points),default=None)
                    row['fixture_geometry_proxy']=fixture_proximity(robot.data.body_pos_w[i].cpu().numpy(),origins[i],args.scene)
                    row.update(candidate_id=CANDIDATE_IDS[candidate],group_id=groups[i]['group_id'],role=groups[i]['role'],
                       body_angular_velocity_radps=body_vector(post,post[10:13]),actual_linear_velocity_world_mps=post[7:10],
                       actual_angular_velocity_world_radps=post[10:13],actor_command=detail['actor_seen'][i],
                       executed_action=detail['executed_action'][i],actor_input=detail['actor_input'][i],executed_path_world_xy=paths_world[i].tolist(),candidate_replan=step==0,episode_id=groups[i]['group_id'],post_logical_sim_time_ns=branch_starts_ns[i]+(step+1)*STEP_NS,follower_diagnostic=diagnostics[i],append_events=append_events[i],boundary_censor=bool(boundary),phase='candidate',
                       terrain_metadata_file='../../terrain.json',wall_clock_ns=time.time_ns())
                    row['logical_sim_time_ns']=branch_starts_ns[i]+step*STEP_NS;row['post_timestamp_ns']=branch_starts_ns[i]+(step+1)*STEP_NS
                    if boundary and not row['stop_reason']:row['stop_reason']='boundary_censor'
                    if step+1==(args.smoke_steps or 40) and not row['stop_reason']:row['stop_reason']='smoke_limit' if args.smoke_steps else 'horizon_complete'
                    if row['stop_reason']:active[i]=False;reasons[i]=row['stop_reason']
                    counts[i]+=1;total_rows+=1;streams[i].write(json.dumps(row,allow_nan=False)+'\n')
                timings['serialization']+=time.perf_counter()-started
                if not active.any():break
            for i,stream in enumerate(streams):
                stream.close();followers[i].close()
                write(args.output/f'group-{i:03d}'/(f'candidate-{candidate:02d}' if run_index<14 else f'duplicate-{run_index:02d}-candidate-{candidate:02d}')/'complete.json',dict(control_steps=counts[i],stop_reason=reasons[i],horizon_seconds=2.,smoke=bool(args.smoke_steps),candidate=candidate))
            from visualnav_transformer.eat_nav.moving_pose2s_targets import build_window
            summaries=[]
            for i in range(n):
                name=f'candidate-{candidate:02d}' if run_index<14 else f'duplicate-{run_index:02d}-candidate-{candidate:02d}'
                rows=[json.loads(line) for line in (args.output/f'group-{i:03d}'/name/'trace.jsonl').read_text().splitlines()]
                if rows:
                    target=build_window(rows,0,dict(root_state=rows[0]['root_state'],t0_ns=rows[0]['logical_sim_time_ns'],candidate_id=CANDIDATE_IDS[candidate],path_body_xy=candidate_paths()[candidate],direction=1 if candidate<7 else -1))
                    endpoint=rows[-1]['post_root_state']
                    summaries.append(dict(root=endpoint,first_action=rows[0]['executed_action'],contact_forces=rows[0]['contact_net_forces_w'],events=np.where(target['event_mask4'],target['events4'],-1).tolist(),event_mask=np.asarray(target['event_mask4']).tolist(),count=len(rows)))
                else:summaries.append(None)
            run_summaries.append(dict(candidate=candidate,results=summaries))
            followers=[]
        duplicate_checks=[]
        if args.duplicate_gate:
            for ri,run in enumerate(run_summaries[14:],14):
                original=run_summaries[run['candidate']]
                for i,(reference,current) in enumerate(zip(original['results'],run['results'])):
                    if reference is None or current is None:duplicate_checks.append(dict(env_index=i,run=ri,passed=False,reason='anchor_unavailable'));continue
                    gate=duplicate_gate(reference['root'],current['root'],reference['events'],current['events'])
                    gate['first_action_max_abs_difference']=float(np.max(abs(np.asarray(reference['first_action'])-np.asarray(current['first_action']))))
                    gate['contact_force_max_abs_difference']=float(np.max(abs(np.asarray(reference['contact_forces'])-np.asarray(current['contact_forces']))))
                    gate['passed'] &= reference['count']==current['count'] and reference['event_mask']==current['event_mask']
                    duplicate_checks.append(dict(env_index=i,run=ri,candidate=run['candidate'],**gate))
            write(args.output/'duplicate-gate.json',dict(passed=bool(duplicate_checks) and all(x['passed'] for x in duplicate_checks),checks=duplicate_checks,order=sequence,physics_cache_restorable=False))
        for i in range(n):
            path=args.output/f'group-{i:03d}'/'group.json';meta=json.loads(path.read_text())
            meta['counterfactual_complete']=bool(anchor_valid[i] and len(run_summaries)>=14 and all(run['results'][i] is not None for run in run_summaries[:14]))
            if args.duplicate_gate:meta['restoration_gate_passed']=all(x['passed'] for x in duplicate_checks if x['env_index']==i)
            write(path,meta)
        elapsed=time.perf_counter()-start
        write(args.output/'complete.json',dict(state='completed',num_envs=n,recorded_rows=total_rows,
             elapsed_collection_seconds=elapsed,rows_per_second=total_rows/elapsed,component_host_seconds=timings,timing_note='host timings; queued CUDA inference may synchronize inside env_step; includes settling',
             simulated_env_seconds_per_second=total_rows*.05/elapsed,
             peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),gpu_memory_free_total_bytes=list(torch.cuda.mem_get_info()),anchor_checks=checks,duplicate_checks=duplicate_checks,restoration_method=args.restoration_method,independent_reset_check=independent_reset_check,
             anchor_available_by_env=anchor_valid.tolist(),anchor_failure_reasons=anchor_failure_reasons,independent_terminal_capture=True,headless=args.headless,smoke=bool(args.smoke_steps),physics_audit=physics_audit,contact_api_status=contact_api_status))
        print('MOVING_POSE2S_COMPLETE '+(args.output/'complete.json').read_text(),flush=True)
    except BaseException as exc:
        write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for follower in followers:follower.close()
        if capture is not None:capture.close()
        if override is not None:override.close()
        if env is not None:env.close()

if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    finally:app.close()
