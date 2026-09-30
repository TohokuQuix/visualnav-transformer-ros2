"""Fresh bridge/follower process for one frozen readiness replay or measurement.

The original completed runs and readiness-100 attempt-0 are never written here.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, pathlib, socket, subprocess, time
from visualnav_transformer.eat_nav.readiness_recovery import terminate_tree

REPO=pathlib.Path('/tmp/eatnav-vnt-candidate-coverage')
DOC=REPO/'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot'
ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol')
INPUT=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')
OLD=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot')
SIM=ROOT/'sim.py'
REMOTE_CODE='/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/sim.py'
REMOTE_ROOT='/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/runs'
ENV='PYTHONPATH=/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'
FOLL=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery/follower-build/simple_trajectory_follower_node')
PROD=pathlib.Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node')
PARAM=pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()

def atomic(p,d):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');t.replace(p)

def read(p):return json.loads(pathlib.Path(p).read_text())

def frozen(entry):
 freeze=read(ROOT/'r0-freeze.json')
 assert sha(DOC/'schedule.json')==freeze['schedule_sha256']
 assert sha(DOC/'input-freeze.json')==freeze['input_freeze_sha256']
 assert sha(PROD)==freeze['production_follower_binary_sha256']
 assert sha(PARAM)==freeze['production_follower_config_sha256']
 assert read(DOC/'schedule.json')['runs'][entry['order']]==entry
 state=INPUT/f"state-{entry['state_index']:03d}"
 assert sha(state/f"{entry['candidate_id']}-base-path.json")==entry['base_path_sha256']
 if entry['kind']!='anchor':
  oldfreeze=OLD/'map-path-freeze'/f"{entry['state_id']}.json"
  if oldfreeze.exists():assert sha(state/f"{entry['candidate_id']}-map-path.json")==read(oldfreeze)['map_path_sha256'][entry['candidate_id']]
  newfreeze=ROOT/'map-path-freeze'/f"{entry['state_id']}.json"
  if newfreeze.exists():assert sha(state/f"{entry['candidate_id']}-map-path.json")==read(newfreeze)['map_path_sha256'][entry['candidate_id']]
 return state

def active_processes():
 result=subprocess.check_output(['ps','-eo','pid,args'],text=True)
 return [row for row in result.splitlines() if ('eat_nav_command_delivery_bridge.py' in row or
         'simple_trajectory_follower_node' in row or REMOTE_CODE in row) and 'ps -eo' not in row]

def bridge_request(action,**kwargs):
 from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
 client=FollowerBridgeClient(timeout=5.)
 try:return client.request(action,**kwargs)
 finally:client.close()

def classify_fault_result(exit_code, failure_marker, *, complete_exists):
    error=str(failure_marker.get('error','')) if isinstance(failure_marker,dict) else ''
    return {'injected_timeout':not complete_exists and 'fresh command timeout' in error,
            'simulator_exit_code':exit_code,'isaac_native_close_exit_zero':exit_code==0}


def _docker_sim_pids(run_name):
 result=subprocess.check_output(['docker','exec','isaac-lab-onix','ps','-eo','pid,args'],text=True)
 return [int(row.split(None,1)[0]) for row in result.splitlines() if REMOTE_CODE in row and
         f'/{run_name}' in row and 'ps -eo' not in row]

def _terminate_docker_sim(run_name):
 pids=_docker_sim_pids(run_name)
 if pids:subprocess.run(['docker','exec','isaac-lab-onix','kill','-TERM',*map(str,pids)],check=False)
 deadline=time.monotonic()+3
 while time.monotonic()<deadline and _docker_sim_pids(run_name):time.sleep(.1)
 pids=_docker_sim_pids(run_name)
 if pids:subprocess.run(['docker','exec','isaac-lab-onix','kill','-KILL',*map(str,pids)],check=False)
 return pids

def archive_logs(folder):
 ledger=[]
 for p in sorted(folder.glob('follower-logs/*.log')):
  raw=sha(p);q=p.with_suffix('.log.gz')
  if q.exists():raise FileExistsError(q)
  temp=q.with_suffix('.gz.tmp')
  with p.open('rb') as inp,gzip.open(temp,'wb',compresslevel=6) as out:
   for chunk in iter(lambda:inp.read(1048576),b''):out.write(chunk)
  with gzip.open(temp,'rb') as f:
   h=hashlib.sha256()
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  if h.hexdigest()!=raw:raise ValueError('gzip content SHA mismatch')
  temp.replace(q);ledger.append({'raw_path':str(p),'raw_sha256':raw,'gzip_path':str(q),'gzip_sha256':sha(q)})
  p.unlink()
 atomic(folder/'archived-log-sha.json',ledger)
 return ledger

def run_one(entry,run_name,*,fault_drop_stamp_ns=None,expected_failure=False,timeout_sec=600):
 state=frozen(entry)
 target=ROOT/'runs'/run_name
 if target.exists():raise FileExistsError(target)
 if active_processes():raise RuntimeError('stale bridge/follower/simulator process: '+repr(active_processes()))
 target.mkdir(parents=True)
 remote=f'{REMOTE_ROOT}/{run_name}'
 subprocess.run(['docker','exec','isaac-lab-onix','mkdir','-p',remote],check=True)
 started=time.monotonic()
 atomic(target/'started.json',{'entry':entry,'run_name':run_name,'fault_drop_stamp_ns':fault_drop_stamp_ns,
        'started_unix':time.time(),'input_sha256':{name:sha(p) for name,p in {
        'prefix':state/'prefix-actions.npz','suffix':state/'suffix-actions-120.npz',
        'base_path':state/f"{entry['candidate_id']}-base-path.json",
        'map_path':state/f"{entry['candidate_id']}-map-path.json",'config':state/'config-120.yaml',
        'anchor':state/'logged-anchor.json'}.items() if p.exists()}})
 bridge=None;sim=None;error=None;peak_gpu=0;fault_seen=None;baseline=active_processes()
 try:
  if entry['kind']!='anchor':
   args=['bash','-lc','source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 '+
       str(REPO/'scripts/eat_nav_command_delivery_bridge.py')+' --follower '+str(FOLL)+
       ' --params '+str(PARAM)+' --diagnostic-dir '+str(target/'follower-logs')+
       (f' --fault-drop-stamp-ns {fault_drop_stamp_ns}' if fault_drop_stamp_ns is not None else '')]
   bridge_log=(target/'bridge.log').open('x')
   bridge=subprocess.Popen(args,stdout=bridge_log,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+20
   while True:
    if bridge.poll() is not None:raise RuntimeError('bridge exited before listen')
    try:
     with socket.create_connection(('127.0.0.1',47819),timeout=.2):break
    except OSError:
     if time.monotonic()>deadline:raise TimeoutError('bridge listen')
     time.sleep(.1)
  cmd=['docker','exec','-e',ENV,'isaac-lab-onix','/isaac-sim/python.sh',REMOTE_CODE,
      '--headless','--stage','anchor' if entry['kind']=='anchor' else 'live','--repeat-id','0',
      '--logical-t0-ns',str(entry['logical_t0_ns']),'--control-steps','120',
      '--manifest','/tmp/eatnav-extended/output/manifest.json',
      '--prefix',f'/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/state-{entry["state_index"]:03d}/prefix-actions.npz',
      '--other-env-actions',f'/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/state-{entry["state_index"]:03d}/suffix-actions-120.npz',
      '--logged-anchor',f'/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/state-{entry["state_index"]:03d}/logged-anchor.json',
      '--state','/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/selected-states.json',
      '--base-path',f'/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/state-{entry["state_index"]:03d}/{entry["candidate_id"]}-base-path.json',
      '--fixed-map-path',f'/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/state-{entry["state_index"]:03d}/{entry["candidate_id"]}-map-path.json',
      '--config',f'/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/state-{entry["state_index"]:03d}/config-120.yaml',
      '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
      '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
      '--output',remote]
  sim_log=(target/'simulator.log').open('x')
  sim=subprocess.Popen(cmd,stdout=sim_log,stderr=subprocess.STDOUT,start_new_session=True)
  while sim.poll() is None:
   elapsed=time.monotonic()-started
   if elapsed>timeout_sec:raise TimeoutError('simulator wall limit')
   if expected_failure and 'RECOVERY_FATAL' in (target/'simulator.log').read_text(errors='replace'):
    if fault_seen is None:fault_seen=time.monotonic()
    if time.monotonic()-fault_seen>60:raise TimeoutError('exception cleanup >60 sec')
   if bridge is not None and bridge.poll() is not None:raise RuntimeError('bridge exited before sim')
   try:
    gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    peak_gpu=max(peak_gpu,gpu)
   except Exception:pass
   atomic(target/'status.json',{'stage':'SIM_RUNNING','elapsed_sec':elapsed,'gpu_peak_mib':peak_gpu,
          'bridge_pid':bridge.pid if bridge else None,'sim_host_pid':sim.pid,'fault_seen':fault_seen is not None})
   time.sleep(2)
  sim_log.close()
  code=sim.returncode
  if expected_failure:
   for name in ('failure-0.marker.json','failure-0.trace.json.gz','status.json'):
    subprocess.run(['docker','cp',f'isaac-lab-onix:{remote}/{name}',str(target/name)],check=True)
   fail=read(target/'failure-0.marker.json')
   classification=classify_fault_result(code,fail,complete_exists=(target/'live-0.marker.json').exists())
   atomic(target/'fault-classification.json',classification)
   if not classification['injected_timeout']:raise RuntimeError('wrong or incomplete fault result')
   if fault_seen is None:fault_seen=time.monotonic()

  else:
   if code:raise RuntimeError(f'simulator exited {code}')
   stage='anchor' if entry['kind']=='anchor' else 'live'
   names=['anchor-0.json','anchor-0.marker.json'] if stage=='anchor' else ['live-0.json','live-0.trace.json.gz','live-0.clock.json.gz','live-0.marker.json']
   for name in names:subprocess.run(['docker','cp',f'isaac-lab-onix:{remote}/{name}',str(target/name)],check=True)
   summary=read(target/names[0])
   if summary['state']['state_id']!=entry['state_id'] or summary['requested_control_steps']!=120:raise ValueError('result identity')
   if stage=='live':
    import eat_nav_exact_stamp_run as exact
    exact.inspect_run(target,120,entry['logical_t0_ns'])
    with gzip.open(target/'live-0.trace.json.gz','rt') as f:trace=json.load(f)
    if len(trace)!=120 or any(row['terminal'] for row in trace):raise ValueError('short or reset')
  atomic(target/'sim-result.json',{'exit_code':code,'elapsed_sec':time.monotonic()-started,'peak_gpu_mib':peak_gpu,
          'expected_failure':expected_failure})
 except BaseException as exc:
  error=repr(exc)
  if sim is not None and sim.poll() is None:
   terminate_tree(sim,term_wait_sec=3,kill_wait_sec=3)
   killed=_terminate_docker_sim(run_name)
   atomic(target/'forced-sim-cleanup.json',{'error':error,'docker_pids_killed':killed})
 finally:
  if bridge is not None:
   try:
    response=bridge_request('shutdown')
    atomic(target/'bridge-shutdown.json',response)
    bridge.wait(timeout=10)
   except BaseException as exc:
    if bridge.poll() is None:terminate_tree(bridge,term_wait_sec=3,kill_wait_sec=3)
    error=error or 'bridge cleanup '+repr(exc)
   bridge_log.close()
  sim_log_open=locals().get('sim_log')
  if sim_log_open is not None and not sim_log_open.closed:sim_log_open.close()
  residual=active_processes()
  if residual:error=error or 'residual process '+repr(residual)
  if (target/'follower-logs').exists():archive_logs(target)
  atomic(target/'final-status.json',{'stage':'FAILED' if error else 'COMPLETE','error':error,
        'elapsed_sec':time.monotonic()-started,'fault_seen':fault_seen is not None,
        'residual_processes':residual,'bridge_exit':bridge.returncode if bridge else None,
        'sim_exit':sim.returncode if sim else None,'gpu_peak_mib':peak_gpu})
 if error:raise RuntimeError(error)
 if peak_gpu>12000:raise RuntimeError('GPU abnormal')
 if expected_failure:
  if time.monotonic()-fault_seen>60:raise RuntimeError('cleanup >60 sec')
  return read(target/'final-status.json')
 names=['anchor-0.json','anchor-0.marker.json'] if entry['kind']=='anchor' else ['live-0.json','live-0.trace.json.gz','live-0.clock.json.gz','live-0.marker.json']
 atomic(target/'run-complete.json',{'identity':entry,'output_sha256':{name:sha(target/name) for name in names},
       'protocol':'fresh_bridge_per_run','elapsed_sec':time.monotonic()-started,
       'gpu_peak_mib':peak_gpu})
 return read(target/'run-complete.json')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--order',type=int,required=True)
 parser.add_argument('--run-name',required=True);parser.add_argument('--fault-drop-stamp-ns',type=int)
 parser.add_argument('--expected-failure',action='store_true');args=parser.parse_args()
 schedule=read(DOC/'schedule.json')['runs'];entry=schedule[args.order]
 print(json.dumps(run_one(entry,args.run_name,fault_drop_stamp_ns=args.fault_drop_stamp_ns,
       expected_failure=args.expected_failure),indent=2),flush=True)
