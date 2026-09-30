"""Detached, frozen maximum-three-repeat TF/command race diagnostic."""
from __future__ import annotations
import argparse,gzip,hashlib,json,subprocess,threading,time
from pathlib import Path
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
ROOT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')
CONTAINER=str(ROOT)
ENV='PYTHONPATH=/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');t.replace(p)
def docker(*args,**kw):return subprocess.run(['docker',*args],check=True,**kw)
def prepare():
 docker('exec','isaac-lab-onix','mkdir','-p',CONTAINER)
 docker('cp','scripts/eat_nav_tf_race_sim.py',f'isaac-lab-onix:{CONTAINER}/sim.py')
 docker('cp','src/visualnav_transformer/eat_nav/sim_clock_protocol.py',f'isaac-lab-onix:{CONTAINER}/sim_clock_protocol.py')
 source='/tmp/eatnav-candidate-coverage/sim-clock-long-horizon/long-horizon-inputs/state-001'
 target=f'{CONTAINER}/long-horizon-inputs/state-001'
 docker('exec','isaac-lab-onix','mkdir','-p',target)
 for name in ('config-120.yaml','suffix-actions-120.npz'):
  docker('exec','isaac-lab-onix','cp',f'{source}/{name}',f'{target}/{name}')
def run_one(identity):
 n=identity['repeat'];folder=ROOT/f'repeat-{n:02d}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'stage-complete.json').exists():return json.loads((folder/'stage-complete.json').read_text())
 if (folder/'runner.log').exists():raise FileExistsError('partial repeat requires review; no automatic retry')
 target=f'{CONTAINER}/repeat-{n:02d}/sim';docker('exec','isaac-lab-onix','mkdir','-p',target)
 command=['docker','exec','-e',ENV,'isaac-lab-onix','/isaac-sim/python.sh',f'{CONTAINER}/sim.py','--headless',
  '--stage','live','--repeat-id',str(n),'--logical-t0-ns',str(identity['logical_t0_ns']),'--control-steps','20',
  '--manifest','/tmp/eatnav-extended/output/manifest.json',
  '--prefix','/tmp/eatnav-candidate-coverage/pilot-inputs/state-001/prefix-actions.npz',
  '--other-env-actions',f'{CONTAINER}/long-horizon-inputs/state-001/suffix-actions-120.npz',
  '--logged-anchor','/tmp/eatnav-candidate-coverage/pilot-inputs/state-001/logged-anchor.json',
  '--state','/tmp/eatnav-candidate-coverage/pilot-inputs/selected-states.json',
  '--base-path','/tmp/eatnav-candidate-coverage/pilot-inputs/state-001/bank-nominal-base-path.json',
  '--fixed-map-path','/tmp/eatnav-candidate-coverage/pilot-inputs/state-001/bank-nominal-map-path.json',
  '--config',f'{CONTAINER}/long-horizon-inputs/state-001/config-120.yaml',
  '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
  '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
  '--output',target]
 gpu=[];stop=threading.Event()
 def watch():
  while not stop.is_set():
   try:gpu.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0]))
   except Exception:pass
   stop.wait(3)
 thread=threading.Thread(target=watch,daemon=True);thread.start();start=time.monotonic()
 with (folder/'runner.log').open('x') as log:
  log.write('TF_RACE_RUN_BEGIN '+json.dumps(identity)+'\n');log.flush()
  try:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  finally:stop.set();thread.join(5)
  log.write(f'TF_RACE_RUN_EXIT {result.returncode} elapsed={time.monotonic()-start}\n');log.flush()
 conflict=(ROOT/'same-stamp-command-conflict-first.json').exists()
 if not conflict:
  client=FollowerBridgeClient(timeout=10)
  try:client.request('diagnostic_stop')
  finally:client.close()
 copied={}
 for name in ('status.json',f'live-{n}.json',f'live-{n}.trace.json.gz',f'live-{n}.clock.json.gz'):
  source=f'{target}/{name}'
  check=subprocess.run(['docker','exec','isaac-lab-onix','test','-f',source])
  if check.returncode:continue
  source_sha=subprocess.check_output(['docker','exec','isaac-lab-onix','sha256sum',source],text=True).split()[0]
  docker('cp',f'isaac-lab-onix:{source}',str(folder/name))
  if sha(folder/name)!=source_sha:raise RuntimeError('docker copy SHA mismatch '+name)
  copied[name]=source_sha
 outcome={'repeat':n,'identity':identity,'simulator_exit_code':result.returncode,
  'same_stamp_conflict':conflict,'elapsed_sec':time.monotonic()-start,
  'gpu_peak_mib':max(gpu) if gpu else None,'runner_log_sha256':sha(folder/'runner.log'),
  'copied_source_sha256':copied}
 atomic(folder/'stage-complete.json',outcome)
 if result.returncode and not conflict:raise RuntimeError('simulator failed without same-stamp conflict; stop')
 return outcome

def main():
 identities=json.loads((ROOT/'tf-race-run-identities.json').read_text())['runs'];assert len(identities)==3
 prepare();started=time.monotonic();out=[]
 try:
  for identity in identities:
   atomic(ROOT/'status.json',{'stage':'R4_RUNNING','next_repeat':identity['repeat'],'completed':len(out)})
   value=run_one(identity);out.append(value)
   print('TF_RACE_REPEAT_COMPLETE',value,flush=True)
   if value['same_stamp_conflict']:break
  atomic(ROOT/'status.json',{'stage':'R4_COMPLETE','repeats':len(out),'conflict':any(x['same_stamp_conflict'] for x in out),
   'elapsed_sec':time.monotonic()-started})
 except BaseException as exc:
  atomic(ROOT/'status.json',{'stage':'R4_STOPPED','reason':repr(exc),'completed':len(out),
   'elapsed_sec':time.monotonic()-started})
  raise
if __name__=='__main__':main()
