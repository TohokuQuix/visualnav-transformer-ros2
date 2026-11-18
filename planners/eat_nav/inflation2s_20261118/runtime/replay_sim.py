"""Headless Isaac recorded-state replay; writes RGB frames, never steps physics/policy."""
from __future__ import annotations
import os,sys
# Capture the original stdout pipe BEFORE any Isaac imports/native logging. All logs use stderr.
RAW_FD=None
if '--rgb-stdout' in sys.argv:
 RAW_FD=os.dup(sys.stdout.fileno());os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
import argparse,json,time,traceback,hashlib
from pathlib import Path
from render_video import load_trace,replay_indices,joint_permutation,sha256
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trace',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True)
p.add_argument('--actor-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--identity',type=Path);p.add_argument('--run-json',type=Path);p.add_argument('--trial-id',default='selected-trial')
p.add_argument('--fps',type=int,choices=(5,10),default=5);p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
p.add_argument('--eye',type=float,nargs=3,default=(7.,-7.,6.));p.add_argument('--target',type=float,nargs=3,default=(1.56,1.56,.25))
p.add_argument('--rgb-stdout',action='store_true',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if not args.headless or not args.enable_cameras:p.error('--headless --enable_cameras are required')
if args.width%2 or args.height%2 or args.width<2 or args.height<2:p.error('even positive image dimensions required')
data=load_trace(args.trace);indices=replay_indices(data['time_s'],args.fps)
if args.output_dir.exists():raise FileExistsError(args.output_dir)
args.output_dir.mkdir(parents=True)
metadata=json.loads(args.identity.read_text()) if args.identity else {}
application=AppLauncher(args).app
import numpy as np,torch,yaml,gymnasium as gym,trimesh
from pxr import UsdGeom,UsdPhysics
import omni.usd
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import onix_isaac_lab.tasks
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import EATTerrainImporter
from visualnav_transformer.eat_nav.model2000_policy import apply_bundle

def write_json(name,data):
 (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a))+'\n')

class RecordedTerrain:
 def __init__(self,cfg,device):
  with np.load(args.mesh,allow_pickle=False) as mesh:
   self.terrain_mesh=trimesh.Trimesh(vertices=mesh['vertices'],faces=mesh['faces'],process=False)
  self.terrain_origins=np.zeros((1,1,3));self.flat_patches={}
class RecordedImporter(EATTerrainImporter):
 def __init__(self,cfg):
  super().__init__(cfg)
  prim=omni.usd.get_context().get_stage().GetPrimAtPath(cfg.prim_path+'/terrain/mesh')
  UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Set('none');UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
 def update_env_origins(self,*args,**kwargs):return

@torch.inference_mode()
def main():
 torch.set_num_threads(1)
 paired=yaml.load((args.actor_dir/'env.yaml').read_text(),Loader=yaml.BaseLoader)
 toggles={n:paired[n].lower()=='true' for n in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
 cfg=OnixRoughEnvCfg(**toggles);apply_bundle(cfg,args.actor_dir/'env.yaml')
 cfg.scene.num_envs=1;cfg.scene.terrain.class_type=RecordedImporter;cfg.scene.terrain.max_init_terrain_level=0
 gen=cfg.scene.terrain.terrain_generator;gen.class_type=RecordedTerrain;gen.num_rows=1;gen.num_cols=1;gen.curriculum=False;gen.use_cache=False
 # Reset/initial construction loads the same articulated asset. No policy runner/NN is constructed.
 cfg.events.push_robot=None;cfg.events.ignore_sprocket_self_collision=None
 cfg.scene.replay_camera=CameraCfg(prim_path='/World/RecordedReplayCamera',update_period=0.,height=args.height,width=args.width,data_types=['rgb'],spawn=PinholeCameraCfg(focal_length=24.,horizontal_aperture=20.955,clipping_range=(.01,100.)))
 env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg));bare=env.unwrapped;robot=bare.scene['robot'];env.reset();camera=bare.scene['replay_camera']
 source_names=metadata.get('joint_names',metadata.get('joints'))
 permutation=joint_permutation(source_names,robot.joint_names,data['joint_position'].shape[1]);joint_order_note='verified and reordered by recorded names' if source_names is not None else 'metadata absent: positional runtime ONIX joint order assumed; exact J matched'
 camera.set_world_poses_from_view(torch.tensor([args.eye],device=bare.device),torch.tensor([args.target],device=bare.device))
 raw=os.fdopen(RAW_FD,'wb');frame_map=[];max_state_error=0.;clock=float(bare.sim.current_time);counter=int(bare.sim.current_time_step_index)
 def state(index):
  root=torch.as_tensor(data['root'][index:index+1],dtype=torch.float32,device=bare.device)
  q=torch.as_tensor(data['joint_position'][index:index+1,permutation],dtype=torch.float32,device=bare.device)
  qd=torch.as_tensor(data['joint_velocity'][index:index+1,permutation],dtype=torch.float32,device=bare.device)
  robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(q,qd);bare.scene.write_data_to_sim();bare.sim.forward()
  return root,q,qd
 try:
  state(int(indices[0]))
  for _ in range(8):bare.sim.render()
  for video_index,source_index in enumerate(indices):
   root,q,qd=state(int(source_index))
   # IsaacLab.render explicitly disables /app/player/playSimulations around its app.update.
   # No sim.step/env.step/application.update, policy inference or path planner is used here.
   bare.sim.render();bare.sim.render();camera.update(0.,force_recompute=True)
   if float(bare.sim.current_time)!=clock or int(bare.sim.current_time_step_index)!=counter:raise RuntimeError('render advanced physics time/step count')
   actual_q=robot.root_physx_view.get_dof_positions();actual_qd=robot.root_physx_view.get_dof_velocities()
   pose=robot.root_physx_view.get_root_transforms();actual_root=torch.cat((pose[:,:3],pose[:,6:7],pose[:,3:6]),dim=1)
   position_error=float(torch.max(torch.abs(actual_root[:,:3]-root[:,:3])))
   qerror=float(torch.max(torch.abs(actual_q-q)));verror=float(torch.max(torch.abs(actual_qd-qd)))
   quaternion_error=float(torch.minimum(torch.linalg.vector_norm(actual_root[:,3:7]-root[:,3:7]),torch.linalg.vector_norm(actual_root[:,3:7]+root[:,3:7])).max())
   max_state_error=max(max_state_error,position_error,qerror,verror,quaternion_error)
   if max(position_error,qerror,verror)>2e-5 or quaternion_error>1e-3:raise RuntimeError('PhysX recorded-state write/read mismatch')
   rgb=camera.data.output['rgb'][0,:,:,:3].to(dtype=torch.uint8).cpu().numpy();rgb=np.ascontiguousarray(rgb)
   if rgb.shape!=(args.height,args.width,3):raise RuntimeError('camera RGB shape mismatch')
   raw.write(rgb.tobytes());frame_map.append(dict(video_frame=video_index,trace_index=int(source_index),recorded_time_s=float(data['time_s'][source_index])))
   if video_index%100==0:print('REPLAY_FRAME',video_index,'/',len(indices),file=sys.stderr,flush=True)
  raw.flush()
  write_json('replay-provenance.json',dict(schema='Lcorridor_recorded_state_replay_v1',status='completed',mode='recorded-state replay; not physics rerun',trace=str(args.trace),trace_sha256=sha256(args.trace),mesh=str(args.mesh),mesh_sha256=sha256(args.mesh),actor_sha256=sha256(args.actor_dir/'model_2000.pt'),env_sha256=sha256(args.actor_dir/'env.yaml'),agent_sha256=sha256(args.actor_dir/'agent.yaml'),renderer_sha256=sha256(__file__),identity_sha256=sha256(args.identity) if args.identity else None,run_json_sha256=sha256(args.run_json) if args.run_json else None,root_frame='local Lcorridor; z-up; quaternion wxyz; original velocities retained',joint_order=joint_order_note,runtime_joint_names=robot.joint_names,recorded_joint_names=source_names,joint_permutation=permutation,physics_integration_steps_during_frames=0,physics_time_initial=clock,physics_time_final=float(bare.sim.current_time),max_state_write_read_error=max_state_error,policy_inferences=0,planner_calls=0,fps=args.fps,width=args.width,height=args.height,frame_count=len(indices),frame_map=frame_map,camera_eye=args.eye,camera_target=args.target,terminal_record_preserved=bool(indices[-1]==len(data['time_s'])-1),temporal_sampling='latest recorded20Hz state at each5/10Hz display instant; no interpolated/future states; terminal included; final frame held1/fps',trial_id=args.trial_id))
 finally:raw.close();env.close()
status=0
try:main()
except Exception as exc:
 status=1;write_json('replay-failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
finally:application.close()
sys.exit(status)
