"""Headless R6 branch collection; launch with Isaac's Python, never ROS.

Example: /isaac-sim/python.sh scripts/eat_nav_r6_collect_sim.py --headless
 --output OUT --terrain-type boxes --level 0 --num-envs 10 --seed 123
 --groups groups.json --rl-checkpoint actor.pt --actor-env-yaml env.yaml
 --eat-config config/eat_nav.yaml. Without groups only benchmark output is allowed.
 --smoke-steps 20 records one short branch; production records all fourteen 2s branches.
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
parser.add_argument('--anchor-seconds',type=int,choices=(0,2,4,6),default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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
from eat_nav_all_fields import needs_extension
from eat_nav_r6_collection import STEP_NS, fixture_boxes, validate_groups, verify_anchor, step_records, fixture_proximity, selected_tiles, fixture_contact_index, audit_physics_properties, update_anchor_valid, decode_fixture_contacts


TERRAIN_COLS = args.terrain_cols or args.num_envs
TILE_INDICES = selected_tiles(args.num_envs,TERRAIN_COLS,args.tile_indices)

class R6TerrainGenerator(EATTerrainGenerator):
    def _get_terrain_mesh(self,difficulty,cfg):
        mesh,origin=super()._get_terrain_mesh(difficulty,cfg)
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
        env.eat_nav_initial_rngs=[np.random.default_rng(_seed(args.seed,2,i)) for i in TILE_INDICES]
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
        if args.contact_calibration_probe:
            calibration_state=robot.data.root_state_w.clone()
            calibration_state[:,:3]=bare.scene.env_origins
            calibration_state[:,0]+=fixture_boxes(args.scene)[0]['center_m'][0]
            calibration_state[:,2]+=.15
            calibration_state[:,3:7]=torch.tensor([1.,0.,0.,0.],device=bare.device)
            calibration_state[:,7:]=0.
            robot.write_root_state_to_sim(calibration_state)
            bare.sim.forward();bare.scene.update(bare.step_dt)
        anchor_valid=np.ones(n,dtype=bool);anchor_failure_reasons=[None]*n
        for step in range(20):
            done,_=advance(zeros)
            failed_now=anchor_valid & done.cpu().numpy().astype(bool)
            anchor_valid=update_anchor_valid(anchor_valid,done.cpu().numpy())
            for failed in np.flatnonzero(failed_now):anchor_failure_reasons[int(failed)]='termination_during_settling'
            if (step+1)%5==0:
                stamps.append((step+1)*STEP_NS)
                started=time.perf_counter()
                for i in range(n): history[i].append(dem(bare,config,i,stamps[-1]))
                timings['dem_capture']+=time.perf_counter()-started
        prefix_rows=[[] for _ in range(n)]
        for step in range(args.anchor_seconds*20):
            active_step=step;terminals.clear();pending_contacts.clear()
            pre=robot.data.root_state_w.cpu().numpy().copy()
            before_prefix=[physical(bare,i) for i in range(n)]
            prefix_active=anchor_valid.copy()
            commands=torch.zeros_like(zeros);commands[:,0]=torch.as_tensor(anchor_valid,device=bare.device)*.3
            done,detail=advance(commands)
            rows=step_records(pre,robot.data.root_state_w.cpu().numpy(),commands.cpu().numpy(),done.cpu().tolist(),terminals,prefix_active,step,-1,1_000_000_000)
            for i,row in enumerate(rows):
                if not prefix_active[i]:continue
                row.update(before_prefix[i]);row['root_state']=pre[i].tolist()
                row.update(phase='prefix',contact_observation=dict(contact_api_status,contacts=[entry for entry in pending_contacts if entry['env_index']==i]))
                if contact_api_status['available']:
                    points=row['contact_observation']['contacts'];row['obstacle_contact']=bool(points);row['obstacle_contact_time_ns']=min((point['logical_sim_time_ns'] for point in points),default=None)
                prefix_rows[i].append(row)
            failed_now=anchor_valid & done.cpu().numpy().astype(bool)
            anchor_valid=update_anchor_valid(anchor_valid,done.cpu().numpy())
            for failed in np.flatnonzero(failed_now):anchor_failure_reasons[int(failed)]='termination_during_prefix'
            if (step+1)%5==0:
                stamps.append(1_000_000_000+(step+1)*STEP_NS)
                for i in range(n):history[i].append(dem(bare,config,i,stamps[-1]))
        history=[frames[-4:] for frames in history];stamps=stamps[-4:]
        branch_start_ns=1_000_000_000+args.anchor_seconds*1_000_000_000
        groups = json.loads(args.groups.read_text()) if args.groups else [dict(group_id=f'benchmark-{args.seed}-{i}',role='benchmark',seed=args.seed,env_index=i) for i in range(n)]
        if isinstance(groups,dict): groups=groups['groups']
        groups=validate_groups(groups,n)
        origins=bare.scene.env_origins.cpu().numpy()
        if args.scene!='field':
            np.savez_compressed(args.output/'fixture-raycast-evidence.npz',ray_hits_w=bare.scene['height_scanner_eat'].data.ray_hits_w.cpu().numpy(),sensor_pos_w=bare.scene['height_scanner_eat'].data.pos_w.cpu().numpy(),origins=origins)
        metadata=getattr(generator,'eat_nav_tile_metadata',[])
        write(args.output/'terrain.json',dict(terrain=terrain,scene=args.scene,fixtures=fixture_boxes(args.scene),
              subterrain_config=str(generator.sub_terrains[args.terrain_type].to_dict()),
              tile_metadata=metadata,env_origins=origins,control_hz=20,dem_sensor_enabled=True,fixture_physics_representation='combined Warp mesh collision disabled; identical floor and kinematic rigid cuboids' if args.scene!='field' else 'original terrain mesh'))
        write(args.output/'manifest.json',dict(schema='eatnav_r6_vector_v1',groups=groups,headless=args.headless,
              seed=args.seed,actor_sha256=sha(args.rl_checkpoint),actor_env_sha256=sha(args.actor_env_yaml),
              eat_config_sha256=sha(args.eat_config),terrain_type=args.terrain_type,level=args.level,
              scene=args.scene,diagnostic_forced_contact=args.contact_calibration_probe,anchor_seconds=args.anchor_seconds,prefix_command=[.3,0.,0.],geometry_scope='each env retains its own tile across all branches',
              dem_ray_origin_height_m=2. if args.scene!='field' else .3,contact_definition='field: obstacle identity unknown; supplemental: filtered rigidfixture normal contact points sampled every5ms'))
        initial_root=robot.data.root_state_w.clone(); initial_joint=robot.data.joint_pos.clone(); initial_vel=robot.data.joint_vel.clone()
        scene_state=copy.deepcopy(bare.scene.get_state())
        obs_anchor=copy.deepcopy(obs); observed_anchor=copy.deepcopy(observed); latency_anchor=copy.deepcopy(latency)
        # Preserve controller delay/history/action tensors, which are stateful despite frozen weights.
        saved_tensors=[]
        def save_attrs(obj,names):
            for name in names:
                value=getattr(obj,name,None)
                if torch.is_tensor(value): saved_tensors.append((obj,name,value.clone()))
                elif isinstance(value,(int,float,bool)): saved_tensors.append((obj,name,value))
        save_attrs(bare,['_latency_substeps','_onix_latency_seconds','_latency_policy_obs_prev','_latency_policy_obs_reset_mask','episode_length_buf','reset_buf','reset_terminated','reset_time_outs','common_step_counter','_sim_step_counter'])
        save_attrs(bare,[name for name,value in vars(bare).items() if torch.is_tensor(value)])
        for term in bare.action_manager._terms.values():
            save_attrs(term,[name for name,value in vars(term).items() if torch.is_tensor(value)])
        command_term=bare.command_manager._terms['base_velocity']
        save_attrs(command_term,[name for name,value in vars(command_term).items() if torch.is_tensor(value)])
        save_attrs(robot.data,['joint_pos_target','joint_vel_target','joint_effort_target'])
        save_attrs(bare.scene['contact_forces']._data,[name for name,value in vars(bare.scene['contact_forces']._data).items() if torch.is_tensor(value)])
        save_attrs(bare.action_manager,['_action','_prev_action','_prev_prev_action'])
        for delay in getattr(bare,'_action_latency_buffers',{}).values():
            save_attrs(delay,['_time_lags']); save_attrs(delay._circular_buffer,['_buffer','_num_pushes','_pointer'])
        for terms in getattr(bare.observation_manager,'_group_obs_term_history_buffer',{}).values():
            for ring in terms.values(): save_attrs(ring,['_buffer','_num_pushes','_pointer'])
        independent_reset_check = None
        if args.smoke_steps:
            before_reset_roots=robot.data.root_state_w.clone()
            before_reset_joints=robot.data.joint_pos.clone()
            bare._reset_idx(torch.tensor([0],device=bare.device))
            unaffected_root=float(torch.max(torch.abs(robot.data.root_state_w[1:]-before_reset_roots[1:])).item()) if n>1 else 0.
            unaffected_joint=float(torch.max(torch.abs(robot.data.joint_pos[1:]-before_reset_joints[1:])).item()) if n>1 else 0.
            independent_reset_check=dict(reset_env=0,unaffected_env_count=n-1,root_max_abs_delta=unaffected_root,joint_max_abs_delta=unaffected_joint,terminal_hook_captured=0 in terminals,passed=unaffected_root==0 and unaffected_joint==0 and 0 in terminals)
            if not independent_reset_check['passed']:raise ValueError('cross-environment reset contamination')
            bare.scene.reset_to(scene_state,is_relative=False)
            for obj,name,value in saved_tensors:
                current=getattr(obj,name)
                if torch.is_tensor(current):current.copy_(value)
                else:setattr(obj,name,value)
            bare.scene.update(bare.step_dt)
            terminals.clear()
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
        anchor=dict(root=initial_root.cpu().numpy(),joint_position=initial_joint.cpu().numpy(),joint_velocity=initial_vel.cpu().numpy(),dem=np.asarray(history),live_dem=np.asarray([dem(bare,config,i,stamps[-1]) for i in range(n)]))
        for i,group in enumerate(groups):
            directory=args.output/f'group-{i:03d}';directory.mkdir()
            write(directory/'group.json',dict(group,anchor_available=bool(anchor_valid[i]),pre_anchor_contact_observation_scope='settling_and_prefix_5ms' if args.scene!='field' else 'unknown',anchor_failure_reason=anchor_failure_reasons[i],observed_contact_before_anchor=(str(i) in contact_api_status.get('first_contact_ns_by_env',{})) if args.scene!='field' else None,prefix_contact_observation_complete=bool(contact_api_status.get('available')) if args.scene!='field' else False,terrain=terrain,env_origin=origins[i],scene=args.scene,dem_stamps_ns=stamps,parent_terrain_seed=args.seed,parent_tile_column=TILE_INDICES[i],tile_generation_index=TILE_INDICES[i]*5+args.level,actual_tile_metadata=metadata[TILE_INDICES[i]*5+args.level] if metadata else None))
            with (directory/'prefix-trace.jsonl').open('x') as prefix_stream:
                for row in prefix_rows[i]:prefix_stream.write(json.dumps(row,allow_nan=False)+'\n')
            arrays=pack_inputs(np.asarray(history[i]),observed['policy_pre_noise'][i].cpu().numpy(),body_state(initial_root[i].cpu().numpy()))
            np.savez_compressed(directory/'initial-input.npz',**arrays,dem_stamps_ns=np.asarray(stamps,dtype=np.int64))
            np.savez_compressed(directory/'anchor-state.npz',**{key:value[i] for key,value in anchor.items()})
        library=compile_core(args.output/'core-build')
        total_rows=0; start=time.perf_counter(); checks=[]
        for candidate in range(2 if args.smoke_steps else 14):
            if candidate:
                bare.scene.reset_to(scene_state,is_relative=False)
                for obj,name,value in saved_tensors:
                    current=getattr(obj,name)
                    if torch.is_tensor(current):current.copy_(value)
                    else:setattr(obj,name,value)
                torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
                obs=copy.deepcopy(obs_anchor);observed=copy.deepcopy(observed_anchor);latency=copy.deepcopy(latency_anchor)
                bare.scene.update(bare.step_dt)
            current=dict(root=robot.data.root_state_w.cpu().numpy(),joint_position=robot.data.joint_pos.cpu().numpy(),joint_velocity=robot.data.joint_vel.cpu().numpy(),dem=np.asarray(history),live_dem=np.asarray([dem(bare,config,i,stamps[-1]) for i in range(n)]))
            # Historical frames are saved causal evidence, not re-rendered at anchor.
            current['dem']=np.asarray(history)
            check=verify_anchor(anchor,current);checks.append(dict(candidate=candidate,**check))
            if not check['passed']:raise ValueError('branch anchor restoration failed: '+str(check))
            paths_world=[];terminal_headings=[];append_counts=[0]*n
            followers=[]; streams=[]; active=anchor_valid.copy();counts=[0]*n; reasons=['horizon_complete' if anchor_valid[i] else anchor_failure_reasons[i] for i in range(n)]
            for i in range(n):
                directory=args.output/f'group-{i:03d}'/f'candidate-{candidate:02d}';directory.mkdir()
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
                    while active[i] and needs_extension(paths_world[i],p):
                        if append_counts[i]>=4:
                            active[i]=False;reasons[i]='control_error_append_limit';break
                        paths_world[i],terminal_headings[i]=extend_path(paths_world[i],terminal_headings[i],candidate);append_counts[i]+=1
                        followers[i].close();followers[i]=PureFollower((paths_world[i]-origins[i,:2]).tolist(),library=library)
                        append_events[i].append(dict(count=append_counts[i],terminal_heading=terminal_headings[i],path=paths_world[i].tolist()))
                    c,d=followers[i].step_diagnostic(float(p[0]-origins[i,0]),float(p[1]-origins[i,1]),float(p[2]),branch_start_ns+step*STEP_NS)
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
                    boundary=np.any(np.abs(np.asarray(post[:2])-origins[i,:2])>6.)
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
                       follower_diagnostic=diagnostics[i],append_events=append_events[i],boundary_censor=bool(boundary),phase='candidate',
                       terrain_metadata_file='../../terrain.json',wall_clock_ns=time.time_ns())
                    if boundary and not row['stop_reason']:row['stop_reason']='boundary_censor'
                    if step+1==(args.smoke_steps or 40) and not row['stop_reason']:row['stop_reason']='smoke_limit' if args.smoke_steps else 'horizon_complete'
                    if row['stop_reason']:active[i]=False;reasons[i]=row['stop_reason']
                    counts[i]+=1;total_rows+=1;streams[i].write(json.dumps(row,allow_nan=False)+'\n')
                timings['serialization']+=time.perf_counter()-started
                if not active.any():break
            for i,stream in enumerate(streams):
                stream.close();followers[i].close()
                write(args.output/f'group-{i:03d}'/f'candidate-{candidate:02d}'/'complete.json',dict(control_steps=counts[i],stop_reason=reasons[i],horizon_seconds=2.,smoke=bool(args.smoke_steps),candidate=candidate))
            followers=[]
        elapsed=time.perf_counter()-start
        write(args.output/'complete.json',dict(state='completed',num_envs=n,recorded_rows=total_rows,
             elapsed_collection_seconds=elapsed,rows_per_second=total_rows/elapsed,component_host_seconds=timings,timing_note='host timings; queued CUDA inference may synchronize inside env_step; includes settling',
             simulated_env_seconds_per_second=total_rows*.05/elapsed,
             peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),gpu_memory_free_total_bytes=list(torch.cuda.mem_get_info()),anchor_checks=checks,independent_reset_check=independent_reset_check,
             anchor_available_by_env=anchor_valid.tolist(),anchor_failure_reasons=anchor_failure_reasons,independent_terminal_capture=True,headless=args.headless,smoke=bool(args.smoke_steps),physics_audit=physics_audit,contact_api_status=contact_api_status))
        print('R6_COMPLETE '+(args.output/'complete.json').read_text(),flush=True)
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
