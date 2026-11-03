"""Headless matched A/B evaluation using causal actual-FAST-LIO prefixes; no push."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,signal,subprocess,time,inspect
from pathlib import Path
import numpy as np,torch
from visualnav_transformer.eat_nav.single_pose2s_selector import make_selector_manifest,validate_exploratory_authorization,decision_input,select_candidate,validate_run_identity
from visualnav_transformer.eat_nav.single_pose2s_model import predict
from visualnav_transformer.eat_nav.moving_pose2s_geometry import quat_to_matrix
from eat_nav_single_pose2s_pipeline import stage_source,serialized_gpu
REPO=Path(__file__).resolve().parents[1]
HOST=Path('/home/ubuntu/Projects/onix_isaac_lab');CONTAINER=Path('/workspace/isaaclab_onix')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,d):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def hostpath(p):return HOST/Path(p).relative_to(CONTAINER)

def process_prefix(req,payload):
 parent=hostpath(req['directory']);out=parent/'fastlio'
 cmd=['bash','-c','source /home/ubuntu/onix/install/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1; exec /usr/bin/python3 "$@"','selector-lio',str(REPO/'scripts/eat_nav_single_pose2s_fastlio_replay.py'),'--parent',str(parent),'--output',str(out)]
 proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
 try:stdout,_=proc.communicate(timeout=180)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try:stdout,_=proc.communicate(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);stdout,_=proc.communicate()
  (parent/'fastlio-host.log').write_text(stdout);raise RuntimeError('actual FAST-LIO prefix timeout')
 (parent/'fastlio-host.log').write_text(stdout)
 if proc.returncode:raise RuntimeError('actual FAST-LIO prefix failed '+str(parent))
 report=json.loads((out/'complete.json').read_text())
 from visualnav_transformer.eat_nav.single_pose2s_contract import PROFILE
 if report['binary_sha256']!=PROFILE['fastlio_binary_sha256'] or report['config_sha256']!=PROFILE['fastlio_config_sha256'] or sha(out/'odometry.npz')!=report['odometry_npz_sha256']:raise ValueError('LIO provenance mismatch')
 with np.load(parent/'prefix.npz') as f:raw={k:f[k] for k in f.files}
 with np.load(out/'odometry.npz') as f:odom={k:f[k] for k in f.files}
 t0=int(raw['stamps_ns'][-1]);sensors=np.load(parent/'lio-sensors.npz')
 if sensors['imu_stamp_ns'].max()>t0 or sensors['lidar_end_stamp_ns'].max()>t0:raise ValueError('future sensor prefix')
 inputs=decision_input(raw,odom);np.savez_compressed(parent/'model-input.npz',**inputs)
 # Torch CPU inference follows the forked replay, not in its native C++ process.
 pred=predict(payload,inputs,'cpu');valid=np.asarray(pred['rotation_valid'])
 if not valid.all():raise ValueError('invalid model prediction; do not silently exclude')
 R=quat_to_matrix(pred['relative_quaternion_wxyz'][0]);root=raw['root'][-1]
 result=select_candidate(pred['position_body_m'][0],R,root[:3],quat_to_matrix(root[3:7]),req['goal_xy'],req['condition'])
 result.update(index=req['index'],input_sha256=sha(parent/'model-input.npz'),actual_fastlio=True,odometry_sha256=report['odometry_npz_sha256'],input_t0_ns=t0,predicted_body_positions=np.asarray(pred['position_body_m'][0]).tolist(),predicted_relative_quaternion_wxyz=np.asarray(pred['relative_quaternion_wxyz'][0]).tolist())
 write(parent/'decision.json',result);return result

@serialized_gpu
def run_batch(root,stage,batch,payload,ipc):
 tag=batch['tag'];target=root/'raw'/tag
 if (target/'host-seal.json').exists():
  seal=json.loads((target/'host-seal.json').read_text())
  if seal.get('checkpoint_sha256')!=payload['_file_sha']:raise ValueError('resumed checkpoint changed')
  prior=json.loads((target/'runtime-imports.json').read_text())
  if prior['worker_sha256']!=payload['_execution_identity']['worker_sha256']:raise ValueError('resumed worker changed')
  freeze=root/'evaluation-identity.json'
  if not freeze.exists():raise ValueError('missing frozen evaluation identity')
  validate_run_identity(payload['_execution_identity'],json.loads(freeze.read_text()))
  for rel,h in seal['raw_hashes'].items():
   if sha(target/rel)!=h:raise ValueError('saved rollout hash mismatch')
  return seal
 if target.exists():raise ValueError('incomplete rollout preserved; inspect '+str(target))
 local=ipc/tag;local.mkdir(mode=0o700,parents=True,exist_ok=False);container_local=CONTAINER/local.relative_to(HOST)
 parents=local/'parents.json';write(parents,dict(parents=batch['parents']))
 cmd=['docker','exec','-e','PYTHONPATH='+stage+'/src:'+stage+'/scripts:/workspace/isaaclab_onix/source/onix_isaac_lab:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_tasks','isaac-lab-onix','timeout','--signal=INT','--kill-after=30s','2400s','/isaac-sim/python.sh',stage+'/scripts/eat_nav_single_pose2s_select_sim.py','--headless','--parents',str(container_local/'parents.json'),'--output',str(container_local/'sim-output'),'--decision-dir',str(container_local/'decisions'),'--rl-checkpoint',stage+'/actor.pt','--actor-env-yaml',stage+'/env.yaml','--eat-config',stage+'/eat_nav.yaml','--record-lidar']
 write(root/'runtime'/(tag+'.json'),dict(command=cmd,batch=batch,checkpoint_sha256=payload['_file_sha']))
 logpath=root/'logs'/(tag+'.log');logpath.parent.mkdir(parents=True,exist_ok=True);start=time.time();print('START '+tag,flush=True)
 with logpath.open('x') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);handled=set()
  while proc.poll() is None:
   for request in sorted((local/'decisions').glob('request-step*.json')):
    if request.name in handled:continue
    req=json.loads(request.read_text());step=req['step'];response=request.with_name('response-step'+f'{step:03d}'+'.json');tick=time.time()
    try:
     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:answers=list(pool.map(lambda r:process_prefix(r,payload),req['requests']))
     write(response,dict(step=step,decisions=answers,wall_seconds=time.time()-tick));handled.add(request.name)
     print('DECISION '+tag+' step='+str(step)+' n='+str(len(answers))+' wall='+str(round(time.time()-tick,2)),flush=True)
    except Exception as exc:
     write(response,dict(error=repr(exc),decisions=[]));proc.wait(timeout=60);raise
   time.sleep(.2)
 if proc.returncode or not (local/'sim-output/complete.json').exists():raise RuntimeError('selector simulator failed: '+str(logpath))
 target.parent.mkdir(parents=True,exist_ok=True)
 import shutil
 shutil.copytree(local/'sim-output',target)
 hashes={str(f.relative_to(target)):sha(f) for f in target.glob('*/raw.npz')}
 seal=dict(state='completed',runs=len(batch['parents']),condition=batch['condition'],tag=tag,checkpoint_sha256=payload['_file_sha'],raw_hashes=hashes,decision_prefix_root=str(local/'decisions'),host_wall_seconds=time.time()-start)
 write(target/'host-seal.json',seal);print('DONE '+tag+' '+json.dumps(seal),flush=True);return seal

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--prototype',action='store_true');parser.add_argument('--conditions',nargs='+',choices=['A','B'],default=['A','B']);a=parser.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 auth=json.loads((root/'execution-authorization.json').read_text());validate_exploratory_authorization(auth,sha(a.checkpoint))
 payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);payload['_file_sha']=sha(a.checkpoint)
 identity=dict(checkpoint_sha256=payload['_file_sha'],worker_sha256=sha(REPO/'scripts/eat_nav_single_pose2s_select_sim.py'),selector_behavior_sha256=hashlib.sha256((inspect.getsource(select_candidate)+inspect.getsource(decision_input)).encode()).hexdigest(),predictor_source_sha256=sha(REPO/'src/visualnav_transformer/eat_nav/single_pose2s_model.py'),follower_source_sha256=sha(REPO/'src/visualnav_transformer/eat_nav/single_pose2s_follower.py'),fastlio_replay_source_sha256=sha(REPO/'scripts/eat_nav_single_pose2s_fastlio_replay.py'))
 freeze=root/'evaluation-identity.json'
 if freeze.exists():validate_run_identity(identity,json.loads(freeze.read_text()))
 elif list((root/'raw').glob('*/host-seal.json')):raise ValueError('existing runs have no frozen identity; audit before resume')
 else:write(freeze,identity)
 payload['_execution_identity']=identity
 manifest=make_selector_manifest(conditions=tuple(a.conditions))
 if a.prototype:
  manifest['batches']=manifest['batches'][:2]
  for b in manifest['batches']:b['parents']=b['parents'][:2]
  manifest['planned_runs']=4;manifest['prototype']=True
 f=root/'manifest.json'
 if f.exists() and json.loads(f.read_text())!=manifest:raise ValueError('manifest changed')
 if not f.exists():write(f,manifest)
 # The paired seed set is separate from the already collected training/validation/final data.
 training=Path('/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v2/production/manifest.json')
 old={p['seed'] for p in json.loads(training.read_text())['parents']};new={p['seed'] for b in manifest['batches'] for p in b['parents']}
 if old&new:raise ValueError('selector seed overlaps supervised dataset')
 stage=stage_source(root);ipc=HOST/('.eatnav-single-pose2s-selector-'+root.name+'-'+str(int(time.time())))
 write(root/'selector-runtime.json',dict(stage=stage,ipc=str(ipc),pid=os.getpid(),started_unix=time.time()))
 completed=0
 try:
  for b in manifest['batches']:
   run_batch(root,stage,b,payload,ipc);completed+=len(b['parents']);write(root/'status.json',dict(phase='running',completed_runs=completed,planned_runs=manifest['planned_runs'],strict_accuracy_achieved=False))
  write(root/'status.json',dict(phase='rollouts_completed',completed_runs=completed,planned_runs=manifest['planned_runs'],strict_accuracy_achieved=False,comparison_report_pending=True))
 except BaseException as exc:
  write(root/'status.json',dict(phase='failed_preserved',completed_runs=completed,planned_runs=manifest['planned_runs'],error=repr(exc)));raise

if __name__=='__main__':main()
