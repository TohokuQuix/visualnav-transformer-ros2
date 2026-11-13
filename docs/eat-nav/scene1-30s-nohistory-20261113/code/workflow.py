"""Queued Scene1 evaluation; never overlaps Goal30 or another Isaac GPU app."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,fcntl,traceback,shutil
import numpy as np
from nav_contract import needs_walls
from analyze import analyze
from gap_walls import create_gap_wall_variant
ROOT=Path(__file__).resolve().parents[1];GOAL=Path('/home/ubuntu/eatnav-goal30-model2000-20261111');BASE=Path('/home/ubuntu/eatnav-scene1-onix-20261111');SOURCE=BASE/'sources/new';CON='isaac-lab-onix'
EXTERNAL_ROOTS=[Path('/home/ubuntu/Projects/onix_isaac_lab/source/onix_isaac_lab'),Path('/home/ubuntu/Projects/IsaacLab/source/isaaclab')]
WORKER_SHA=hashlib.sha256((ROOT/'code/navigation_sim.py').read_bytes()).hexdigest();VERSION=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'code').glob('*.py')))+(ROOT/'protocol.json').read_bytes()).hexdigest()[:10];STAGE='/tmp/eatnav-scene1-nav-'+VERSION

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def run(argv,**kwargs):return subprocess.run(list(map(str,argv)),check=True,**kwargs)
def tree_hash(root):
 root=Path(root);h=hashlib.sha256()
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix.lower() in ['.py','.urdf','.stl','.obj','.dae','.usd','.usda','.usdc']:
   h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
 return h.hexdigest()
def create_reference():
 path=ROOT/'reference-physics.npz';z=np.load(BASE/'runs/scene1-smoke-r3/physics.npz');values=dict(masses=z['masses'][0],inertias=z['inertias'][0],materials=z['material'][0])
 if path.exists():
  prior=np.load(path)
  if not all(np.array_equal(prior[k],v) for k,v in values.items()):raise RuntimeError('saved reference physics differs; preserve it')
  prior.close()
 else:np.savez_compressed(path,**values)
 z.close();return path
def run_identity(mesh):
 ref=create_reference();nn=ROOT/'checkpoint.pt'
 return dict(worker_sha256=sha(ROOT/'code/navigation_sim.py'),helpers_sha256={name:sha(ROOT/'code'/name) for name in ['nav_contract.py','pose_contract.py']},protocol_sha256=sha(ROOT/'protocol.json'),mesh_sha256=sha(mesh),reference_physics_sha256=sha(ref),nn_sha256=sha(nn),actor_bundle_sha256={name:sha(BASE/'actor'/name) for name in ['model_2000.pt','env.yaml','agent.yaml']},source_tree_sha256=tree_hash(SOURCE/'src/visualnav_transformer/eat_nav'),external_trees_sha256={str(p):tree_hash(p) for p in EXTERNAL_ROOTS})
def cleanup_process(proc,terminate=None,kill=None):
 if proc is None:return
 if proc.poll() is not None:
  if terminate is not None:terminate()
  return
 if terminate is not None:
  try:terminate()
  except Exception:proc.terminate()
 else:proc.terminate()
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:
  if kill is not None:
   try:kill()
   except Exception:pass
  if proc.poll() is None:proc.kill()
  proc.wait(timeout=20)
def gpu_idle():
 q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
 if q.returncode:raise RuntimeError('NVIDIA driver query failed: '+q.stderr.strip())
 return not q.stdout.strip()
def wait_for_predecessor():
 while True:
  phase=json.loads((GOAL/'status.json').read_text()).get('phase') if (GOAL/'status.json').exists() else None
  if phase=='completed':
   audit=json.loads((GOAL/'results/audit.json').read_text())
   if audit.get('state')!='passed' or audit.get('unique_trials')!=350 or audit.get('missing_trials'):raise RuntimeError('Goal30 completion audit invalid; retain GPU priority')
   if gpu_idle():return
  write(ROOT/'status.json',dict(phase='waiting_for_goal30_and_gpu',goal30_phase=phase,stage=STAGE,time=time.time()))
  time.sleep(30)
def stage():
 run(['docker','exec',CON,'mkdir','-p',STAGE+'/source',STAGE+'/code',STAGE+'/actor',STAGE+'/inputs',STAGE+'/runs',STAGE+'/meshes',STAGE+'/video-input'])
 for p,d in [(SOURCE,STAGE+'/source'),(ROOT/'code',STAGE+'/code'),(BASE/'actor',STAGE+'/actor')]:run(['docker','cp',str(p)+'/.',CON+':'+d])
 nn=ROOT/'checkpoint.pt';run(['docker','cp',nn,CON+':'+STAGE+'/actor/nn.pt']);run(['docker','cp',ROOT/'protocol.json',CON+':'+STAGE+'/inputs/protocol.json'])
 create_reference();run(['docker','cp',ROOT/'reference-physics.npz',CON+':'+STAGE+'/inputs/reference-physics.npz'])
 write(ROOT/'staging.json',dict(stage=STAGE,worker_sha256=WORKER_SHA,reference_physics_sha256=sha(ROOT/'reference-physics.npz'),checkpoint_sha256=sha(nn),code_sha256={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},source=str(SOURCE)))
def simpath():return STAGE+'/code:'+STAGE+'/source/src:'+STAGE+'/source/scripts:/workspace/isaaclab_onix/source/onix_isaac_lab:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_tasks'
def terminate_owned(target,sig='SIGTERM',script_name='navigation_sim.py'):
 script="import pathlib,os,signal\ntarget="+repr(target)+"\nscript_name="+repr('/'+script_name)+"\nfor p in pathlib.Path('/proc').iterdir():\n try:\n  if not p.name.isdigit():continue\n  argv=(p/'cmdline').read_text().split(chr(0))\n  if target in argv and any(x.endswith(script_name) for x in argv):os.kill(int(p.name),signal."+sig+")\n except (ProcessLookupError,FileNotFoundError,PermissionError):pass"
 run(['docker','exec',CON,'/isaac-sim/kit/python/bin/python3','-c',script])

def job(tag,parents,mesh,max_seconds):
 identity=run_identity(mesh);execution_sha=hashlib.sha256(json.dumps(dict(parents=parents,max_seconds=max_seconds),sort_keys=True).encode()).hexdigest();tag+='-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:8]
 target=ROOT/'runs'/tag
 if (target/'host-seal.json').exists():
  old=json.loads((target/'host-seal.json').read_text())
  if old.get('run_identity')!=identity or old.get('execution_sha256')!=execution_sha:raise RuntimeError('sealed job identity differs')
  return target
 if target.exists():raise RuntimeError('partial run preserved, explicit recovery required: '+str(target))
 if not gpu_idle():raise RuntimeError('GPU unexpectedly occupied; no duplicate job launched')
 run(['docker','cp',mesh,CON+':'+STAGE+'/meshes/'+mesh.name]);pfile=ROOT/'inputs'/(tag+'.json');write(pfile,dict(parents=parents));run(['docker','cp',pfile,CON+':'+STAGE+'/inputs/'+tag+'.json'])
 cmd=['docker','exec','-e','PYTHONPATH='+simpath(),CON,'timeout','--signal=TERM','--kill-after=20s','7200s','/isaac-sim/python.sh',STAGE+'/code/navigation_sim.py','--headless','--parents',STAGE+'/inputs/'+tag+'.json','--output',STAGE+'/runs/'+tag,'--rl-checkpoint',STAGE+'/actor/model_2000.pt','--actor-env-yaml',STAGE+'/actor/env.yaml','--eat-config',STAGE+'/source/config/eat_nav.yaml','--nn-checkpoint',STAGE+'/actor/nn.pt','--reference-physics',STAGE+'/inputs/reference-physics.npz','--mesh',STAGE+'/meshes/'+mesh.name,'--protocol',STAGE+'/inputs/protocol.json','--max-seconds',str(max_seconds)]
 write(ROOT/'commands'/(tag+'.json'),dict(command=cmd));log=ROOT/'logs'/(tag+'.log');log.parent.mkdir(exist_ok=True,parents=True);started=time.time();print('START',tag,flush=True)
 proc=None
 try:
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);write(ROOT/'current-process.json',dict(tag=tag,pid=proc.pid,started=started,command=cmd))
   while proc.poll() is None:
    if subprocess.run(['docker','exec',CON,'test','-f',STAGE+'/runs/'+tag+'/failure.json']).returncode==0:
     terminate_owned(STAGE+'/runs/'+tag)
     try:proc.wait(timeout=20)
     except subprocess.TimeoutExpired:terminate_owned(STAGE+'/runs/'+tag,'SIGKILL');proc.wait(timeout=20)
     break
    time.sleep(3)
 except BaseException:
  cleanup_process(proc,lambda:terminate_owned(STAGE+'/runs/'+tag),lambda:terminate_owned(STAGE+'/runs/'+tag,'SIGKILL'))
  raise
 finally:
  target.parent.mkdir(exist_ok=True,parents=True);subprocess.run(['docker','cp',CON+':'+STAGE+'/runs/'+tag,target])
 if proc.returncode or not(target/'complete.json').exists():raise RuntimeError('simulator failure: '+str(log))
 complete=json.loads((target/'complete.json').read_text());assert complete['parents']==len(parents)
 actual=json.loads((target/'identity.json').read_text());assert actual['worker_sha256']==identity['worker_sha256'] and actual['contract_sha256']==identity['helpers_sha256']['nav_contract.py'] and actual['protocol_sha256']==identity['protocol_sha256'] and actual['mesh_sha256']==identity['mesh_sha256'] and actual['reference_physics_sha256']==identity['reference_physics_sha256'] and actual['checkpoint_sha256']==identity['nn_sha256']
 write(target/'host-seal.json',dict(run_identity=identity,execution_sha256=execution_sha,worker_sha256=WORKER_SHA,protocol_sha256=sha(ROOT/'protocol.json'),mesh_sha256=sha(mesh),elapsed_wall_s=time.time()-started,exit_code=proc.returncode,complete=complete))
 print('DONE',tag,complete['results'],flush=True);return target

def parents_for(cohort):
 p=json.loads((ROOT/'protocol.json').read_text())
 return [dict(trial_id=f'{cohort}-t{i+1:02d}',parent_id=f'{cohort}-t{i+1:02d}',field='scene1',level=0,goal_id='confirmed',goal_xy=p['goal_xy'],yaw=p['start_yaw_rad'],seed=123,trial_index=i,rng_note='global simulation seed123; batched environment index separates observation draws; same exact terrain/nominal pose/physics') for i in range(10)]
def cohort(name,mesh):
 indexfile=ROOT/'indices'/(name+'.json')
 identity=run_identity(mesh)
 if indexfile.exists():
  manifest=json.loads((ROOT/'manifests'/(name+'.json')).read_text())
  if manifest.get('run_identity')!=identity:raise RuntimeError('cached cohort dependency identity differs; preserve prior cohort')
  index=json.loads(indexfile.read_text())
  for folder in {Path(v).parent for v in index.values()}:
   if json.loads((folder/'host-seal.json').read_text()).get('run_identity')!=identity:raise RuntimeError('cached batch identity mismatch')
  return analyze(name,index)
 parents=parents_for(name);write(ROOT/'manifests'/(name+'.json'),dict(run_identity=identity,parents=parents,mesh_sha256=sha(mesh),protocol_sha256=sha(ROOT/'protocol.json')));pending=parents;index={}
 for attempt in range(3):
  if not pending:break
  batch=job(name+'-formal-'+WORKER_SHA[:8]+f'-a{attempt}',pending,mesh,300.);next_pending=[]
  for p in pending:
   d=batch/p['trial_id'];r=json.loads((d/'run.json').read_text())
   if r['result'] in ['technical_failure','initialization_failure']:next_pending.append(p)
   else:index[p['trial_id']]=str(d)
  pending=next_pending
  write(ROOT/'status.json',dict(phase='formal_trials',cohort=name,completed=len(index),planned=10,pending=[p['trial_id'] for p in pending]))
 if pending:raise RuntimeError('formal trials have unresolved initialization/technical failures; nominal pose was not silently moved')
 write(indexfile,index);return analyze(name,index)
def video(summary,mesh):
 best=summary['best_trial'];cohort=summary['cohort'];out=ROOT/'video'/cohort;out.mkdir(parents=True,exist_ok=True);trace=Path(best['trace']);run_json=Path(best['run_json']);identity=run_json.parent.parent/'identity.json';render_sha=hashlib.sha256((ROOT/'code/replay_sim.py').read_bytes()+(ROOT/'code/render_video.py').read_bytes()+sha(trace).encode()+sha(mesh).encode()).hexdigest()[:10];mp4=out/(best['trial_id']+'-'+render_sha+'.mp4');status=mp4.with_suffix('.complete.json')
 if status.exists():
  old=json.loads(status.read_text())
  if old['trace_sha256']!=sha(trace) or old['mesh_sha256']!=sha(mesh) or old['sha256']!=sha(mp4):raise RuntimeError('cached video identity differs')
  return old
 if mp4.exists():raise RuntimeError('partial movie preserved; must inspect/recover: '+str(mp4))
 if not gpu_idle():raise RuntimeError('GPU occupied before replay')
 for p,name in [(trace,'trace.npz'),(run_json,'run.json'),(identity,'identity.json'),(mesh,'mesh.npz')]:run(['docker','cp',p,CON+':'+STAGE+'/video-input/'+name])
 remote_out=STAGE+'/replay-'+cohort+'-'+render_sha
 cmd=['docker','exec','-e','PYTHONPATH='+simpath(),CON,'timeout','--signal=TERM','--kill-after=20s','7200s','/isaac-sim/python.sh',STAGE+'/code/replay_sim.py','--headless','--enable_cameras','--rgb-stdout','--trace',STAGE+'/video-input/trace.npz','--identity',STAGE+'/video-input/identity.json','--run-json',STAGE+'/video-input/run.json','--mesh',STAGE+'/video-input/mesh.npz','--actor-dir',STAGE+'/actor','--output-dir',remote_out,'--trial-id',best['trial_id'],'--fps','5','--width','1280','--height','720']
 encoder=['/usr/bin/python3',str(ROOT/'code/render_video.py'),'--trace',str(trace),'--output',str(mp4),'--trial-id',best['trial_id'],'--fps','5','--width','1280','--height','720']
 write(out/'commands.json',dict(renderer=cmd,encoder=encoder));write(ROOT/'status.json',dict(phase='rendering_best_trial',trial=best['trial_id'],cohort=cohort))
 renderer=None;enc=None
 try:
  with (out/'replay.log').open('xb') as err,(out/'encoder-status.json').open('xb') as log,(out/'encoder-error.log').open('xb') as enc_err:
   renderer=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err)
   try:enc=subprocess.Popen(encoder,stdin=renderer.stdout,stdout=log,stderr=enc_err)
   finally:renderer.stdout.close()
   write(ROOT/'current-process.json',dict(tag='video-'+cohort,pid=renderer.pid,encoder_pid=enc.pid,command=cmd))
   ecode=enc.wait(timeout=7500);rcode=renderer.wait(timeout=60)
 finally:
  cleanup_process(enc)
  cleanup_process(renderer,lambda:terminate_owned(remote_out,'SIGTERM','replay_sim.py'),lambda:terminate_owned(remote_out,'SIGKILL','replay_sim.py'))
  subprocess.run(['docker','cp',CON+':'+remote_out+'/.',out])
 if ecode or rcode:raise RuntimeError('video replay/encode failed; inspect '+str(out))
 provenance=json.loads((out/'replay-provenance.json').read_text());assert provenance['status']=='completed' and provenance['physics_integration_steps_during_frames']==0 and provenance['trace_sha256']==sha(trace) and provenance['mesh_sha256']==sha(mesh)
 result=dict(video=str(mp4),sha256=sha(mp4),trial_id=best['trial_id'],trace_sha256=sha(trace),mesh_sha256=sha(mesh),kind='actual recorded-state Isaac replay, no fresh physics rollout',encoder=json.loads((out/'encoder-status.json').read_text()))
 write(status,result);write(ROOT/'video-index.json',result);return result

def main():
 wait_for_predecessor();write(ROOT/'gpu-lease.json',dict(owner='scene1-navigation',time=time.time(),predecessor_verified=True));stage();mesh=BASE/'experiment/converted/scene1_mesh.npz'
 write(ROOT/'status.json',dict(phase='smoke',planned_trials=10));smoke=job('smoke-'+WORKER_SHA[:8],parents_for('smoke')[:1],mesh,5.)
 if json.loads((smoke/'complete.json').read_text())['initialization_failures']:raise RuntimeError('Scene1 start placement smoke failed')
 summary=cohort('original',mesh)
 if needs_walls(summary['trial_rows']):
  variant=ROOT/'meshes/scene1-gap-walls.npz'
  if not variant.exists():create_gap_wall_variant(mesh,variant)
  summary=cohort('gap-walls',variant);mesh=variant
 movie=video(summary,mesh);write(ROOT/'completion.json',dict(state='completed',final_cohort=summary['cohort'],goal_count=summary['goal_count'],trials=10,summary=str(ROOT/'results'/summary['cohort']/'summary.json'),audit=str(ROOT/'results'/summary['cohort']/'audit.json'),video=movie,baseline_retained=True));write(ROOT/'status.json',dict(phase='completed',goal_count=summary['goal_count'],completed_trials=10,video=movie['video']));print('COMPLETED',summary['goal_count'],'/10',flush=True)
if __name__=='__main__':
 lock=(ROOT/'workflow.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print('Existing Scene1 workflow owns the lock; not starting a duplicate',flush=True);sys.exit(0)
 try:
  write(ROOT/'workflow-process.json',dict(pid=os.getpid(),started=time.time(),version=VERSION));main()
 except BaseException as e:
  write(ROOT/'workflow-failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()));raise
