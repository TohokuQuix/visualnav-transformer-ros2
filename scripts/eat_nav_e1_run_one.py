"""One fresh E1 Bezier candidate with the production follower and exact stamps."""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, pathlib, socket, subprocess, sys, time
from visualnav_transformer.eat_nav.readiness_recovery import terminate_tree
from eat_nav_e1_process_ownership import (ProcessRegistry,process_record,
    register_followers,stop_and_join_monitor)
from eat_nav_g0e_gpu_probe import proc_table
from eat_nav_g0e_gpu_run import container_identity

REPO=pathlib.Path('/tmp/eatnav-vnt-e1-feasibility')
DOC=pathlib.Path('/tmp/eatnav-e1-feasibility-20261003')
ROOT=DOC
INPUT=ROOT/'inputs'
SIM=REPO/'scripts/eat_nav_e1_sim.py'
REMOTE_CODE=str(ROOT/'eat_nav_e1_sim.py')
REMOTE_ROOT=str(ROOT/'runs')
REMOTE_INPUT_ROOT=ROOT
ENV='PYTHONPATH=/tmp/eatnav-e1-feasibility-20261003:/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'
PROD=pathlib.Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node')
FOLL=PROD
PARAM=pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')

def fixed_map_path(state_index,candidate_id):
 return f'{REMOTE_INPUT_ROOT}/inputs/state-{state_index:03d}/{candidate_id}-map-path.json'

def simulator_command(entry, remote, *, input_root=None, simulator=None, timing_telemetry=False):
 """Build the exact Docker CLI used by a formal run without starting processes."""
 input_root=pathlib.Path(input_root or REMOTE_INPUT_ROOT)
 simulator=str(simulator or REMOTE_CODE)
 state=input_root/'inputs'/f"state-{entry['state_index']:03d}"
 cmd=['docker','exec','-e',ENV,'isaac-lab-onix','/isaac-sim/python.sh',simulator,
      '--headless','--stage','anchor' if entry['kind']=='anchor' else 'live','--repeat-id','0',
      '--logical-t0-ns',str(entry['logical_t0_ns']),'--control-steps','120',
      '--manifest','/tmp/eatnav-extended/output/manifest.json',
      '--prefix',str(state/'prefix-actions.npz'),
      '--other-env-actions',str(state/'suffix-actions-120.npz'),
      '--logged-anchor',str(state/'logged-anchor.json'),
      '--state',str(input_root/'inputs/selected-states.json'),
      '--base-path',str(state/f"{entry['candidate_id']}-base-path.json"),
      '--fixed-map-path',str(state/f"{entry['candidate_id']}-map-path.json"),
      '--recorded-frames',str(state/'recorded-frames.json'),
      '--selected-dem',str(input_root/'inputs/g0e-selected-states.npz'),
      '--expected-local-path-sha256',entry['local_path_sha256'],
      '--expected-map-path-sha256',entry['map_path_sha256'],
      '--config',str(state/'config-120.yaml'),
      '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
      '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
      '--output',remote]
 if entry['kind']!='anchor':cmd.extend(['--goal-metadata',str(state/f"{entry['run_id']}-goal-metadata.json")])
 if timing_telemetry:cmd.append('--timing-telemetry')
 return cmd

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
 freeze=read(DOC/'source-freeze.json')
 continuation=read(DOC/'continuation-manifest.json')
 original=next((r for r in continuation['pending_rows'] if r['process_id']==entry['run_id']),None)
 if original is None or original['order']!=entry['order'] or original['state_group_id']!=entry['state_id'] or original['map_path_sha256']!=entry['map_path_sha256']:
  raise ValueError('E1 run identity changed')
 if sha(DOC/'continuation-manifest.json')!=freeze['E1_continuation_manifest_sha256']:raise ValueError('E1 run order SHA changed')
 if sha(INPUT/'selected-states.json')!=sha(pathlib.Path('/tmp/eatnav-candidate-bank-g0e/inputs/selected-states.json')):raise ValueError('state SHA changed')
 if sha(SIM)!=freeze['source_binary_sha256']['simulator']:raise ValueError('E1 simulator source changed')
 if sha(PROD)!=freeze['source_binary_sha256']['follower_binary']:raise ValueError('production follower binary changed')
 if sha(PARAM)!=freeze['source_binary_sha256']['follower_params']:raise ValueError('follower params changed')
 state=INPUT/f"state-{entry['state_index']:03d}"
 if sha(state/f"{entry['candidate_id']}-base-path.json")!=entry['base_path_sha256']:raise ValueError('Path SHA changed')
 if sha(state/f"{entry['candidate_id']}-map-path.json")!=entry['map_path_file_sha256']:raise ValueError('map Path file SHA changed')
 if sha(state/f"{entry['run_id']}-goal-metadata.json")!=entry['goal_metadata_sha256']:raise ValueError('goal metadata SHA changed')
 return state

def active_processes(registry=None):
 """Formal residuals come only from the run's registered measurement identities."""
 return registry.residuals() if registry is not None else []

def bridge_request(action,**kwargs):
 from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
 client=FollowerBridgeClient(timeout=5.)
 try:return client.request(action,**kwargs)
 finally:client.close()

def classify_fault_result(exit_code, failure_marker, *, complete_exists):
    error=str(failure_marker.get('error','')) if isinstance(failure_marker,dict) else ''
    return {'injected_timeout':not complete_exists and 'fresh command timeout' in error,
            'simulator_exit_code':exit_code,'isaac_native_close_exit_zero':exit_code==0}


def _docker_sim_pids(registry):
 return [entry['container_pid'] for entry in registry.by_role('simulator')
         if (observed:=process_record(entry['host_pid'])) is not None
         and observed['start_ticks']==entry['start_ticks']]

def _terminate_docker_sim(registry):
 pids=_docker_sim_pids(registry)
 if pids:subprocess.run(['docker','exec','isaac-lab-onix','kill','-TERM',*map(str,pids)],check=False)
 deadline=time.monotonic()+3
 while time.monotonic()<deadline and _docker_sim_pids(registry):time.sleep(.1)
 pids=_docker_sim_pids(registry)
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

def run_one(entry,run_name,*,fault_drop_stamp_ns=None,expected_failure=False,timeout_sec=600,
            timing_telemetry=False,timing_monitor_script=None,timing_monitor_config=None,
            timing_monitor_extra_args=(),monitor_cleanup_protocol=False):
 state=frozen(entry)
 target=ROOT/'runs'/run_name
 if target.exists():raise FileExistsError(target)
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
 bridge=None;sim=None;monitor=None;monitor_log=None;error=None;peak_gpu=0;peak_cpu_anon=0;fault_seen=None
 run_start_ns=time.monotonic_ns();registry=ProcessRegistry(run_name,run_start_ns)
 registry.register_pid('runner',os.getpid(),ownership='control')
 router=read(ROOT/'router-launch-provenance.json') if (ROOT/'router-launch-provenance.json').is_file() else None
 if router:registry.register_pid('zenoh_router',router['pid'],ownership='external')
 registry.save(target/'process-registry.json')
 try:
  if entry['kind']!='anchor':
   args=['bash','-lc','source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 '+
       str(REPO/'scripts/eat_nav_command_delivery_bridge.py')+' --follower '+str(FOLL)+
       ' --params '+str(PARAM)+' --diagnostic-dir '+str(target/'follower-logs')+
       (f' --fault-drop-stamp-ns {fault_drop_stamp_ns}' if fault_drop_stamp_ns is not None else '')]
   bridge_log=(target/'bridge.log').open('x')
   bridge=subprocess.Popen(args,stdout=bridge_log,stderr=subprocess.STDOUT,start_new_session=True)
   registry.register_pid('bridge',bridge.pid,ownership='measurement')
   registry.save(target/'process-registry.json')
   deadline=time.monotonic()+20
   while True:
    if bridge.poll() is not None:raise RuntimeError('bridge exited before listen')
    try:
     with socket.create_connection(('127.0.0.1',47819),timeout=.2):break
    except OSError:
     if time.monotonic()>deadline:raise TimeoutError('bridge listen')
     time.sleep(.1)
  cmd=simulator_command(entry,remote,timing_telemetry=timing_telemetry)
  container=container_identity();table=proc_table()
  process_ids=[[pid,p['start_ticks']] for pid,p in table.items()
               if container['container_id'] in p['cgroup']]
  atomic(target/'container-process-baseline.json',{'container_id':container['container_id'],
      'process_ids':process_ids,'sample_monotonic_ns':time.monotonic_ns()})
  sim_log=(target/'simulator.log').open('x')
  if timing_telemetry:atomic(target/'docker-exec-start.json',{'monotonic_ns':time.monotonic_ns(),'wall_ns':time.time_ns(),'command':cmd})
  sim=subprocess.Popen(cmd,stdout=sim_log,stderr=subprocess.STDOUT,start_new_session=True)
  registry.register_pid('docker_exec',sim.pid,ownership='measurement')
  registry.save(target/'process-registry.json')
  if timing_telemetry:
   monitor_log=(target/'timing-monitor.log').open('x')
   monitor_script=timing_monitor_script or REPO/'scripts/eat_nav_goal_subgoal_timing_monitor.py'
   monitor_cmd=[sys.executable,str(monitor_script),
       '--run',str(target),'--remote-progress',f'{remote}/timing-progress.json',
       '--sim-host-pid',str(sim.pid),'--run-id',run_name,'--target-step',str(entry['anchor_step']),
       '--target-episode',str(entry['episode_key'][2])]
   if timing_monitor_config is not None:monitor_cmd.extend(['--config',str(timing_monitor_config)])
   monitor_cmd.extend(map(str,timing_monitor_extra_args))
   if monitor_cleanup_protocol:monitor_cmd.extend(['--container-baseline',
       str(target/'container-process-baseline.json'),'--process-registry',
       str(target/'process-registry.json')])
   monitor=subprocess.Popen(monitor_cmd,stdout=monitor_log,stderr=subprocess.STDOUT,start_new_session=True)
   registry.register_pid('runtime_monitor',monitor.pid,ownership='control')
   registry.save(target/'process-registry.json')
  while sim.poll() is None:
   elapsed=time.monotonic()-started
   if elapsed>timeout_sec:raise TimeoutError('simulator wall limit')
   if monitor is not None:
    if monitor.poll() is not None:raise RuntimeError('timing monitor exited before simulator: '+str(monitor.returncode))
    heartbeat=target/'timing-heartbeat.json'
    if timing_monitor_config is not None:
     from eat_nav_goal_subgoal_resource_monitor import heartbeat_stale
     stale=heartbeat_stale(heartbeat,time.time(),90)
    else:stale=heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>90
    if stale:raise TimeoutError('external heartbeat older than 90 seconds')
    if (target/'timing-stall.failure.json').exists() or (target/'resource-stop.failure.json').exists():
     raise RuntimeError('external timing monitor detected resource or progress stop')
   if expected_failure and 'RECOVERY_FATAL' in (target/'simulator.log').read_text(errors='replace'):
    if fault_seen is None:fault_seen=time.monotonic()
    if time.monotonic()-fault_seen>60:raise TimeoutError('exception cleanup >60 sec')
   if bridge is not None and bridge.poll() is not None:raise RuntimeError('bridge exited before sim')
   try:
    gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    peak_gpu=max(peak_gpu,gpu)
   except Exception:pass
   try:
    stat=subprocess.check_output(['docker','exec','isaac-lab-onix','cat','/sys/fs/cgroup/memory.stat'],text=True)
    peak_cpu_anon=max(peak_cpu_anon,int(next(line.split()[1] for line in stat.splitlines() if line.startswith('anon '))))
   except Exception:pass
   atomic(target/'status.json',{'stage':'SIM_RUNNING','elapsed_sec':elapsed,'gpu_peak_mib':peak_gpu,'cpu_anon_peak_bytes':peak_cpu_anon,
          'bridge_pid':bridge.pid if bridge else None,'sim_host_pid':sim.pid,'fault_seen':fault_seen is not None})
   time.sleep(2)
  sim_log.close()
  if monitor is not None:
   if not monitor_cleanup_protocol or sim.returncode:
    atomic(target/'monitor.stop',{'simulator_exit_code':sim.returncode,'monotonic_ns':time.monotonic_ns()})
    monitor.wait(timeout=20)
    monitor_log.close()
    if monitor.returncode:raise RuntimeError('timing monitor failed: '+str(monitor.returncode))
   atomic(target/'docker-exec-return.json',{'monotonic_ns':time.monotonic_ns(),'wall_ns':time.time_ns(),
                                           'exit_code':sim.returncode})
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
   names=['anchor-0.json','anchor-0.marker.json'] if stage=='anchor' else ['anchor-gate.json','live-0.json','live-0.trace.json.gz','live-0.clock.json.gz','live-0.marker.json']
   for name in names:subprocess.run(['docker','cp',f'isaac-lab-onix:{remote}/{name}',str(target/name)],check=True)
   if timing_telemetry:
    for name in ('timing-events.jsonl','timing-progress.json','timing-step-durations.json'):
     subprocess.run(['docker','cp',f'isaac-lab-onix:{remote}/{name}',str(target/name)],check=True)
   summary=read(target/'anchor-0.json' if stage=='anchor' else target/'live-0.json')
   if summary['state']['state_id']!=entry['state_id'] or summary['requested_control_steps']!=120:raise ValueError('result identity')
   if stage=='live':
    if read(target/'anchor-gate.json')['status']!='pass' or summary['anchor_gate']['status']!='pass':
     raise ValueError('candidate anchor gate failed')
    import eat_nav_exact_stamp_run as exact
    exact.inspect_run(target,120,entry['logical_t0_ns'])
    with gzip.open(target/'live-0.trace.json.gz','rt') as f:trace=json.load(f)
    if entry['kind']=='duplicate':
     if len(trace)!=120 or any(row['terminal'] for row in trace):raise ValueError('nominal duplicate short or reset')
    elif not (0<len(trace)<=120) or (len(trace)<120 and not trace[-1]['terminal']) or any(row['terminal'] for row in trace[:-1]):
     raise ValueError('candidate trace incomplete or malformed terminal')
  atomic(target/'sim-result.json',{'exit_code':code,'elapsed_sec':time.monotonic()-started,'peak_gpu_mib':peak_gpu,
          'expected_failure':expected_failure,'cpu_anon_peak_bytes':peak_cpu_anon})
 except BaseException as exc:
  error=repr(exc)
  if sim is not None and sim.poll() is None:
   terminate_tree(sim,term_wait_sec=3,kill_wait_sec=3)
   if (target/'simulator-processes.json').is_file():
    for p in read(target/'simulator-processes.json')['processes']:
     registry.register('simulator',p,ownership='measurement')
   killed=_terminate_docker_sim(registry)
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
  if monitor is not None and monitor.poll() is None:
   try:
    stop_and_join_monitor(monitor,target/'monitor.stop',
        lambda:atomic(target/'final-resource-sample.json',read(target/'timing-heartbeat.json'))
            if (target/'timing-heartbeat.json').is_file() else None,timeout=20)
   except BaseException as exc:
    error=error or 'control_cleanup_failure '+repr(exc)
    if monitor.poll() is None:terminate_tree(monitor,term_wait_sec=3,kill_wait_sec=3)
  if monitor_log is not None and not monitor_log.closed:monitor_log.close()
  sim_log_open=locals().get('sim_log')
  if sim_log_open is not None and not sim_log_open.closed:sim_log_open.close()
  try:
   observed=target/'simulator-processes.json'
   if observed.is_file():
    processes=read(observed)
    if processes['run_id']!=run_name:raise ValueError('simulator registry run ID differs')
    for p in processes['processes']:registry.register('simulator',p,ownership='measurement')
   journal=target/'follower-logs/bridge-events.jsonl'
   if journal.is_file():register_followers(registry,journal)
   registry.register_pid('finalization_verifier',os.getpid(),ownership='control')
   if monitor_cleanup_protocol and error is None:registry.require_complete()
   registry.save(target/'process-registry.json')
   residual=active_processes(registry)
   if residual:error=error or 'residual process '+repr(residual)
  except BaseException as exc:
   residual=[];error=error or 'process registry '+repr(exc)
  if (target/'follower-logs').exists():archive_logs(target)
  atomic(target/'final-status.json',{'stage':'FAILED' if error else 'COMPLETE','error':error,
        'elapsed_sec':time.monotonic()-started,'fault_seen':fault_seen is not None,
        'residual_processes':residual,'bridge_exit':bridge.returncode if bridge else None,
        'sim_exit':sim.returncode if sim else None,'gpu_peak_mib':peak_gpu,'cpu_anon_peak_bytes':peak_cpu_anon})
 if error:raise RuntimeError(error)
 if peak_gpu>6656:raise RuntimeError('GPU above 6.5 GiB monitor threshold')
 if expected_failure:
  if time.monotonic()-fault_seen>60:raise RuntimeError('cleanup >60 sec')
  return read(target/'final-status.json')
 names=['anchor-0.json','anchor-0.marker.json'] if entry['kind']=='anchor' else ['anchor-gate.json','live-0.json','live-0.trace.json.gz','live-0.clock.json.gz','live-0.marker.json']
 atomic(target/'run-complete.json',{'identity':entry,'output_sha256':{name:sha(target/name) for name in names},
       'protocol':'fresh_bridge_per_run','elapsed_sec':time.monotonic()-started,
       'gpu_peak_mib':peak_gpu,'cpu_anon_peak_bytes':peak_cpu_anon})
 if monitor_cleanup_protocol and monitor is not None:
  done=read(target/'timing-monitor.done.json')
  runtime=read(target/'runtime-monitor-result.json')
  if (monitor.returncode or done.get('status')!='stopped' or done.get('lifecycle')!='FINALIZING'
      or runtime.get('status')!='pass'):
   raise RuntimeError('control_cleanup_failure: runtime monitor did not stop in FINALIZING')
  finalizing=next(x['monotonic_ns'] for x in runtime['transitions'] if x['to']=='FINALIZING')
  if time.monotonic_ns()-finalizing>30_000_000_000:
   raise TimeoutError('finalization_grace')
  from eat_nav_e1_monitor import verify_complete,finalization_status
  digest=verify_complete(target)
  if digest!=sha(target/'run-complete.json'):raise RuntimeError('output SHA verifier disagrees')
  final=finalization_status(target)
  if not (final['docker_exit']==0 and final['bridge_exit']==0 and final['followers_exit_ok']
          and final['output_sha_match'] and final['residual_count']==0
          and final['zombie_count']==0 and final['cleanup_marker']):
   raise RuntimeError('finalization verifier failed '+repr(final))
  if time.monotonic_ns()-finalizing>30_000_000_000:
   raise TimeoutError('finalization_grace')
  atomic(target/'finalization-verifier-result.json',{'status':'pass',
      'run_complete_sha256':digest,'finalization_grace_sec':30,
      'final':final,'process_registry_sha256':sha(target/'process-registry.json'),
      'lifecycle_transitions':runtime['transitions']+[{'from':'FINALIZING','to':'COMPLETE',
          'monotonic_ns':time.monotonic_ns(),'reason':None}]})
  atomic(target/'protocol-valid.json',{'status':'valid','identity':entry,
      'run_complete_sha256':digest,
      'monitor_done_sha256':sha(target/'timing-monitor.done.json'),
      'finalization_verifier_sha256':sha(target/'finalization-verifier-result.json')})
 return read(target/'run-complete.json')
