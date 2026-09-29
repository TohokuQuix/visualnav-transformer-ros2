"""Resume-safe fixed 20-run, 90-minute/250-MB long-horizon diagnostic."""
from __future__ import annotations
import gzip,json,os,subprocess,time,threading,socket
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120
from visualnav_transformer.eat_nav.command_preview import preview_candidate,pose_from_root_and_map_path
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from eat_nav_command_preview_stage0 import ROOT,OUT,sha,save_once,read

RUNROOT=OUT/'long-horizon'
CONTAINER='/tmp/eatnav-candidate-coverage/command-preview-long-horizon'
ENV='PYTHONPATH=/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'
SIM=CONTAINER+'/long_sim.py'
def docker(*args,**kw):return subprocess.run(['docker',*args],check=True,**kw)
def storage():
 return sum(p.stat().st_size for p in (RUNROOT/'runs').rglob('*') if p.is_file()) if (RUNROOT/'runs').exists() else 0
def status(data):
 p=RUNROOT/'status.json';p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def gpu_watch(stop,values):
 while not stop.is_set():
  try:
   s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
   values.append(int(s.splitlines()[0]))
  except Exception:pass
  stop.wait(3)
def preview_120(runs):
 seen=set();client=FollowerBridgeClient(timeout=50.)
 try:
  for run in runs:
   sid,cid=run['state_id'],run['candidate_id']
   if (sid,cid) in seen:continue
   seen.add((sid,cid));idx=int(sid[-3:])
   source=read(ROOT/'pilot/runs'/run['source_run_id']/'live-0.json')
   path_file=ROOT/'pilot-inputs'/f'state-{idx:03d}'/f'{cid}-map-path.json'
   assert sha(path_file)==run['map_path_sha256']
   path=read(path_file);pose=pose_from_root_and_map_path(source['anchor']['root_state'],path)
   out=RUNROOT/'preview-120'/f'{sid}--{cid}.json';marker=out.with_suffix('.marker.json')
   input_sha={'path':sha(path_file),'source_summary':sha(ROOT/'pilot/runs'/run['source_run_id']/'live-0.json'),'config':sha(OUT/'command-preview-config.yaml')}
   if marker.exists():
    m=read(marker);assert m['input_sha256']==input_sha and sha(out)==m['output_sha256'];continue
   result=preview_candidate(client,path,pose,steps=120,dt=.05)
   save_once(out,{'state_id':sid,'candidate_id':cid,'initial_map_pose':pose,'map_path_sha256':sha(path_file),
    'closed_command':result['closed_commands'],'predicted_pose':result['predicted_poses'],
    'preview_inputs_only':['Path','anchor_pose','production_follower_parameters','control_period','fixed_kinematics']})
   save_once(marker,{'input_sha256':input_sha,'output_sha256':sha(out)})
   print('LONG_PREVIEW_COMPLETE',sid,cid,flush=True)
 finally:client.close()

def prepare_container():
 docker('exec','isaac-lab-onix','mkdir','-p',CONTAINER)
 docker('cp','scripts/eat_nav_command_preview_long_horizon_sim.py',f'isaac-lab-onix:{SIM}')
 for i in (1,3,4,6):
  state=f'state-{i:03d}'
  docker('exec','isaac-lab-onix','mkdir','-p',f'{CONTAINER}/long-horizon-inputs/{state}')
  for name in ('config-120.yaml','suffix-actions-120.npz'):
   docker('cp',str(OUT/'long-horizon-inputs'/state/name),f'isaac-lab-onix:{CONTAINER}/long-horizon-inputs/{state}/{name}')
def verify_run(run):
 rid=f"long-{run['order']:03d}";folder=RUNROOT/'runs'/rid;marker=folder/'run-complete.json'
 if not marker.exists():return False
 m=read(marker)
 assert m['frozen_run']==run
 for name,digest in m['files_sha256'].items():assert sha(folder/name)==digest
 return True
def run_one(run,deadline):
 rid=f"long-{run['order']:03d}";folder=RUNROOT/'runs'/rid;folder.mkdir(parents=True,exist_ok=True)
 if verify_run(run):
  print('LONG_RESUME_SKIP',rid,flush=True);return read(folder/'run-complete.json')['elapsed_sec']
 review_path=folder/'startup-crash-review.json'
 retry_used=0
 log_path=folder/'runner.log'
 if log_path.exists():
  if not review_path.exists():raise ValueError(f'partial run without approved startup review: {rid}')
  review=read(review_path);original=log_path.read_text(errors='replace')
  if not (review['run_id']==rid and review['retry_attempt']==1 and review['retry_limit']==1
          and review['original_log_sha256']==sha(log_path) and review['measurement_started'] is False
          and 'Segmentation fault' in original and "appState = 'startup'" in original
          and "UptimeSeconds = '0'" in original and 'RECOVERY_STEP' not in original):
   raise ValueError(f'pre-simulation startup crash review mismatch: {rid}')
  log_path=folder/'runner-attempt-1.log'
  if log_path.exists():raise ValueError(f'allowed retry already attempted: {rid}')
  retry_used=1
 idx=int(run['state_id'][-3:]);state=f'state-{idx:03d}';cid=run['candidate_id'];old=f'/tmp/eatnav-candidate-coverage/pilot-inputs/{state}'
 target=f'{CONTAINER}/long-horizon/runs/{rid}'
 docker('exec','isaac-lab-onix','mkdir','-p',target)
 command=['docker','exec','-e',ENV,'isaac-lab-onix','/isaac-sim/python.sh',SIM,'--headless',
  '--stage','live','--repeat-id','0','--manifest','/tmp/eatnav-extended/output/manifest.json',
  '--prefix',old+'/prefix-actions.npz','--other-env-actions',f'{CONTAINER}/long-horizon-inputs/{state}/suffix-actions-120.npz',
  '--logged-anchor',old+'/logged-anchor.json','--state','/tmp/eatnav-candidate-coverage/pilot-inputs/selected-states.json',
  '--base-path',old+f'/{cid}-base-path.json','--fixed-map-path',old+f'/{cid}-map-path.json',
  '--config',f'{CONTAINER}/long-horizon-inputs/{state}/config-120.yaml',
  '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
  '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
  '--output',target]
 stop=threading.Event();values=[];thread=threading.Thread(target=gpu_watch,args=(stop,values),daemon=True);thread.start();begin=time.monotonic()
 with log_path.open('x') as log:
  log.write(f'RUN_BEGIN_UNIX {time.time()} id={rid} state={run["state_id"]} candidate={cid}\n');log.flush()
  try:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
  except subprocess.TimeoutExpired:
   log.write('WALL_CLOCK_TIMEOUT\n');log.flush()
   raise TimeoutError(f'wall-clock cap reached during {rid}')
  finally:stop.set();thread.join(4)
  log.write(f'RUN_EXIT_UNIX {time.time()} code={result.returncode}\n');log.flush()
 elapsed=time.monotonic()-begin
 if result.returncode!=0:raise RuntimeError(f'{rid} simulator failure, exit {result.returncode}; see runner.log')
 if values and max(values)>14000:raise RuntimeError(f'unexpected GPU memory {max(values)} MiB')
 for name in ('live-0.json','live-0.trace.json.gz','live-0.marker.json'):
  docker('cp',f'isaac-lab-onix:{target}/{name}',str(folder/name))
 summary=read(folder/'live-0.json');m=read(folder/'live-0.marker.json')
 assert summary['fixed_map_path_sha256']==run['map_path_sha256']
 assert summary['state']['state_id']==run['state_id'] and summary['suffix_steps']<=120
 assert summary['source_sha256']['fixed_map_path']==run['map_path_sha256']
 for p,digest in m['outputs_sha256'].items():assert sha(folder/Path(p).name)==digest
 files={n:sha(folder/n) for n in ('live-0.json','live-0.trace.json.gz','live-0.marker.json')}
 save_once(folder/'run-complete.json',{'frozen_run':run,'files_sha256':files,'elapsed_sec':elapsed,
  'gpu_peak_used_mib':max(values) if values else None,'simulation_steps':summary['suffix_steps'],
  'startup_retries_used':retry_used,'runner_log_sha256':sha(log_path)})
 print('LONG_RUN_COMPLETE',rid,run['state_id'],cid,'steps',summary['suffix_steps'],'sec',round(elapsed,1),flush=True)
 return elapsed
def duplicate(sid,runs):
 records=[]
 for run in runs:
  folder=RUNROOT/'runs'/f"long-{run['order']:03d}"
  assert verify_run(run)
  summary=read(folder/'live-0.json')
  with gzip.open(folder/'live-0.trace.json.gz','rt') as f:trace=json.load(f)
  records.append((summary,trace))
 result=duplicate_120([s for s,t in records],[t for s,t in records],.0105)
 result.update({'state_id':sid,'source_run_ids':[f"long-{r['order']:03d}" for r in runs],
                'source_run_marker_sha256':[sha(RUNROOT/'runs'/f"long-{r['order']:03d}"/'run-complete.json') for r in runs]})
 save_once(RUNROOT/'duplicate-gates'/f'{sid}.json',result)
 return result
def main():
 identity=read(OUT/'long-horizon-run-identities.json');runs=identity['runs']
 assert len(runs)==20 and read(OUT/'stage-a-score-complete.json')['status']=='complete'
 assert sha(OUT/'long-horizon-run-identities.json')==read(OUT/'long-horizon-input-freeze.json')['run_identity_sha256']
 budget=RUNROOT/'budget-start.json'
 if budget.exists():started=read(budget)['started_unix_sec']
 else:
  started=time.time();save_once(budget,{'started_unix_sec':started,'deadline_unix_sec':started+5400,
   'run_identities_sha256':sha(OUT/'long-horizon-run-identities.json'),'input_freeze_sha256':sha(OUT/'long-horizon-input-freeze.json')})
 assert read(budget)['run_identities_sha256']==sha(OUT/'long-horizon-run-identities.json')
 deadline=read(budget)['deadline_unix_sec']
 preview_120(runs)
 prepare_container()
 elapsed=[];completed=0
 try:
  for run in runs:
   if run['kind'] not in ('nominal','nomad_oracle','bank_winner'):raise ValueError('unfrozen kind')
   idx=run['order'];sid=run['state_id']
   if idx%5>=3:
    gate=RUNROOT/'duplicate-gates'/f'{sid}.json'
    if not gate.exists() or not read(gate)['pass']:raise ValueError(f'duplicate gate not passed: {sid}')
   forecast=(len(runs)-idx)*(max(135.,float(np.mean(elapsed))) if elapsed else 135.)
   if time.time()+forecast>deadline:
    status({'state':'budget_stop','completed':completed,'next_run':idx,'forecast_remaining_sec':forecast,
      'deadline_unix_sec':deadline,'storage_bytes':storage()});print('LONG_BUDGET_STOP',completed,flush=True);return
   if storage()>250_000_000:raise RuntimeError('storage cap exceeded')
   status({'state':'running','completed':completed,'next_run':idx,'deadline_unix_sec':deadline,
    'forecast_remaining_sec':forecast,'storage_bytes':storage()})
   duration=run_one(run,deadline);elapsed.append(duration);completed+=1
   if idx%5==2:
    triplet=runs[idx-2:idx+1];gate=duplicate(sid,triplet)
    print('LONG_DUPLICATE_GATE',sid,gate['xy_rms_m'],gate['step_count'],flush=True)
   if storage()>250_000_000:raise RuntimeError('storage cap exceeded')
  status({'state':'complete','completed':20,'elapsed_sec':time.time()-started,'storage_bytes':storage()})
  save_once(RUNROOT/'stage-b-runs-complete.json',{'status':'complete','run_count':20,
   'run_identity_sha256':sha(OUT/'long-horizon-run-identities.json'),'storage_bytes':storage()})
 except BaseException as exc:
  status({'state':'stopped','completed':completed,'next_run':completed,'reason':repr(exc),
   'elapsed_sec':time.time()-started,'storage_bytes':storage()})
  print('LONG_STOP',repr(exc),flush=True);raise
if __name__=='__main__':main()
