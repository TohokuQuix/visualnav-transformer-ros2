"""Resume-safe simulation-clock T4/T5 driver; freezes inputs before launching."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,subprocess,threading,time
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120

ROOT=Path('/tmp/eatnav-candidate-coverage/sim-clock-long-horizon')
BASE=Path('/tmp/eatnav-candidate-coverage')
OLD=BASE/'command-preview-long-horizon'
CONTAINER=str(ROOT)
ENV='PYTHONPATH=/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'
SIM=CONTAINER+'/sim_clock_sim.py'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(data,indent=2,sort_keys=True)+'\n'
 if p.exists():
  if p.read_text()!=s:raise ValueError(f'immutable artifact differs {p}')
  return
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(s);tmp.replace(p)
def status(stage,**kw):
 p=ROOT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'stage':stage,'time_unix':time.time(),**kw},indent=2,sort_keys=True)+'\n');tmp.replace(p)
def docker(*args,**kw):return subprocess.run(['docker',*args],check=True,**kw)
def prep():
 docker('exec','isaac-lab-onix','mkdir','-p',CONTAINER)
 docker('cp','scripts/eat_nav_sim_clock_long_horizon_sim.py',f'isaac-lab-onix:{SIM}')
 docker('cp','src/visualnav_transformer/eat_nav/sim_clock_protocol.py',f'isaac-lab-onix:{CONTAINER}/sim_clock_protocol.py')
 for i in (1,3,4,6):
  dst=f'{CONTAINER}/long-horizon-inputs/state-{i:03d}'
  docker('exec','isaac-lab-onix','mkdir','-p',dst)
  for name in ('config-120.yaml','suffix-actions-120.npz'):
   docker('cp',str(OLD/'long-horizon-inputs'/f'state-{i:03d}'/name),f'isaac-lab-onix:{dst}/{name}')

def trace(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def validate_run(folder,identity,steps,t0):
 marker=folder/'run-complete.json'
 if not marker.exists():return False
 m=read(marker)
 assert m['frozen_identity']==identity and m['steps']==steps and m['logical_t0_ns']==t0
 for name,digest in m['files_sha256'].items():assert sha(folder/name)==digest
 return True

def run_one(stage,id,identity,steps,t0,deadline=None):
 folder=ROOT/stage/'runs'/id;folder.mkdir(parents=True,exist_ok=True)
 if validate_run(folder,identity,steps,t0):return read(folder/'run-complete.json')
 if any(folder.iterdir()):raise RuntimeError('partial run; no simulation retry: '+str(folder))
 idx=int(identity['state_id'][-3:]);state=f'state-{idx:03d}';cid=identity['candidate_id']
 target=f'{CONTAINER}/{stage}/runs/{id}'
 docker('exec','isaac-lab-onix','mkdir','-p',target)
 command=['docker','exec','-e',ENV,'isaac-lab-onix','/isaac-sim/python.sh',SIM,'--headless',
  '--stage','live','--repeat-id','0','--logical-t0-ns',str(t0),'--control-steps',str(steps),
  '--manifest','/tmp/eatnav-extended/output/manifest.json',
  '--prefix',f'/tmp/eatnav-candidate-coverage/pilot-inputs/{state}/prefix-actions.npz',
  '--other-env-actions',f'{CONTAINER}/long-horizon-inputs/{state}/suffix-actions-120.npz',
  '--logged-anchor',f'/tmp/eatnav-candidate-coverage/pilot-inputs/{state}/logged-anchor.json',
  '--state','/tmp/eatnav-candidate-coverage/pilot-inputs/selected-states.json',
  '--base-path',f'/tmp/eatnav-candidate-coverage/pilot-inputs/{state}/{cid}-base-path.json',
  '--fixed-map-path',f'/tmp/eatnav-candidate-coverage/pilot-inputs/{state}/{cid}-map-path.json',
  '--config',f'{CONTAINER}/long-horizon-inputs/{state}/config-120.yaml',
  '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
  '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
  '--output',target]
 log=folder/'runner.log';start=time.monotonic();gpu=[];stop=threading.Event()
 def watch():
  while not stop.is_set():
   try:gpu.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0]))
   except Exception:pass
   stop.wait(3)
 thread=threading.Thread(target=watch,daemon=True);thread.start()
 with log.open('x') as f:
  f.write('BEGIN '+json.dumps({'stage':stage,'run_id':id,'identity':identity,'steps':steps,'logical_t0_ns':t0})+'\n');f.flush()
  try:
   timeout=min(1200,max(1,deadline-time.time())) if deadline else 1200
   result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
  finally:stop.set();thread.join(4)
  f.write(f'EXIT {result.returncode} elapsed={time.monotonic()-start}\n');f.flush()
 if result.returncode:raise RuntimeError(f'{id}: simulator exit {result.returncode}; see {log}')
 if gpu and max(gpu)>14000:raise RuntimeError(f'{id}: unexpected GPU memory {max(gpu)} MiB')
 for name in ('live-0.json','live-0.trace.json.gz','live-0.clock.json.gz','live-0.marker.json'):
  docker('cp',f'isaac-lab-onix:{target}/{name}',str(folder/name))
 summary=read(folder/'live-0.json');records=trace(folder/'live-0.trace.json.gz')
 if summary['fixed_map_path_sha256']!=identity['map_path_sha256'] or summary['state']['state_id']!=identity['state_id'] or len(records)!=steps:
  raise RuntimeError('frozen identity/step mismatch')
 if summary['logical_t0_ns']!=t0 or any(r['logical_sim_time_ns']!=t0+k*50_000_000 for k,r in enumerate(records)):
  raise RuntimeError('clock jump in simulator trace')
 for row in records:
  if row['logical_sim_time_ns']!=row['clock_stamp_ns'] or row['logical_sim_time_ns']!=row['tf_stamp_ns'] or row['logical_sim_time_ns']!=row['follower_command_stamp_ns'] or row['bridge_rejected_count'][0]:
   raise RuntimeError('clock/TF/command mismatch or rejected command')
 files={n:sha(folder/n) for n in ('live-0.json','live-0.trace.json.gz','live-0.clock.json.gz','live-0.marker.json')}
 m={'frozen_identity':identity,'steps':steps,'logical_t0_ns':t0,'files_sha256':files,'elapsed_sec':time.monotonic()-start,'gpu_peak_mib':max(gpu) if gpu else None}
 save(folder/'run-complete.json',m)
 print('SIM_CLOCK_RUN_COMPLETE',stage,id,identity['state_id'],cid,steps,round(m['elapsed_sec'],1),flush=True)
 return m

SIGNALS=['command','actor_input','actor_latent','actor_action','critic_value','reward','root_state','contact_force','raycast_hits']
def parity(identity,folder):
 old=BASE/'pilot/runs'/identity['source_run_id']
 assert sha(old/'live-0.json')==identity['source_summary_sha256']
 assert sha(old/'live-0.trace.json.gz')==identity['source_trace_sha256']
 a=trace(old/'live-0.trace.json.gz');b=trace(folder/'live-0.trace.json.gz')
 rows=[];first=None
 for signal in SIGNALS+['dem']:
  for k,(x,y) in enumerate(zip(a,b)):
   if signal=='dem':
    aa=x['dem']['transformed'];bb=y['dem']['transformed']
   else:aa=x[signal];bb=y[signal]
   if aa==bb:continue
   try:
    v=np.asarray(aa,dtype=float)-np.asarray(bb,dtype=float)
    error=float(np.max(np.abs(v)))
   except Exception:error=None
   row={'signal':signal,'step':k,'max_abs_difference':error}
   rows.append(row)
   if first is None or k<first['step']:first=row
   break
 result={'state_id':identity['state_id'],'old_source_run_id':identity['source_run_id'],
  'first_difference':first,'difference_by_signal':rows,'exact_pass':not rows,
  'clock_corrected_run_sha256':sha(folder/'run-complete.json')}
 return result

def t4(schedule):
 config=read(ROOT/'clock-corrected-run-identities.json');runs=config['runs']
 for j,entry in enumerate(schedule):
  status('T4_RUNNING',item=j+1,total=len(schedule),entry=entry)
  id=f't4-{j:02d}';identity=next(r for r in runs if r['state_id']==entry['state_id'] and r['candidate_id']==entry['candidate_id'] and r['repeat']==0)
  folder=ROOT/'t4'/'runs'/id
  run_one('t4',id,identity,entry['steps'],entry['logical_t0_ns'])
  if entry['stage']=='smoke':
   expected=read(OLD/'long-horizon/preview-120'/f"{entry['state_id']}--{entry['candidate_id']}.json")['closed_command'][0]
   got=trace(folder/'live-0.trace.json.gz')[0]['follower_command']
   diff=max(abs(x-y) for x,y in zip(expected,got))
   result={'first_command_expected':expected,'first_command_actual':got,'max_abs_error':diff,'limit':1e-5,'pass':diff<=1e-5}
   save(ROOT/'production-fixture-parity.json',result)
   if not result['pass']:raise RuntimeError('first-command production fixture mismatch')
  if entry['stage']=='parity':
   result=parity(identity,folder);save(ROOT/'forty-step-parity'/f"{entry['state_id']}.json",result)
   print('FORTY_STEP_PARITY',entry['state_id'],result['exact_pass'],result['first_difference'],flush=True)
   if not result['exact_pass']:raise RuntimeError(f'forty-step parity mismatch {entry["state_id"]}: {result["first_difference"]}')
  if entry['stage']=='duplicate' and entry['repeat']==2:
   folders=[ROOT/'t4'/'runs'/f't4-{5+k:02d}' for k in range(3)]
   summaries=[read(f/'live-0.json') for f in folders];traces=[trace(f/'live-0.trace.json.gz') for f in folders]
   dup=duplicate_120(summaries,traces,.0105)
   save(ROOT/'coverage003-clock-duplicate.json',dup)
   if not dup['pass'] or not dup['dem_sha_equal'] or any(v not in (0,None) for v in dup['signal_rms'].values()):
    raise RuntimeError('coverage003 duplicate mismatch')
 save(ROOT/'stage-t4-complete.json',{'status':'complete','run_count':len(schedule)})
 status('T4_COMPLETE',completed=len(schedule))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['t4','t5'],required=True);args=parser.parse_args()
 prep();frozen=read(ROOT/'clock-corrected-run-identities.json');schedule=frozen['t4_schedule']
 try:
  if args.stage=='t4':t4(schedule)
  else:raise NotImplementedError('T5 begins only after explicit T4 gate report')
 except BaseException as exc:
  status(args.stage.upper()+'_STOPPED',reason=repr(exc))
  print('SIM_CLOCK_STOP',repr(exc),flush=True)
  raise
if __name__=='__main__':main()
