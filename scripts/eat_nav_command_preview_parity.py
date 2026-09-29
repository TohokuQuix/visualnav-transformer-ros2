"""Check frozen first-40-step simulator signals for every completed long run."""
from __future__ import annotations
import gzip,json
from pathlib import Path
import numpy as np
from eat_nav_command_preview_stage0 import ROOT,OUT,sha,save_once,read
KEYS=('command','actor_input','actor_latent','actor_action','critic_observation','critic_value',
      'reward','post_root_state','contact_force')

def main():
 runs=read(OUT/'long-horizon-run-identities.json')['runs'];long=OUT/'long-horizon/runs'
 rows=[]
 for run in runs:
  folder=long/f"long-{run['order']:03d}"
  if not (folder/'run-complete.json').exists():continue
  with gzip.open(folder/'live-0.trace.json.gz','rt') as f:new=json.load(f)
  oldpath=ROOT/'pilot/runs'/run['source_run_id']/'live-0.trace.json.gz'
  assert sha(oldpath)==run['source_trace_sha256']
  with gzip.open(oldpath,'rt') as f:old=json.load(f)
  n=min(len(old),len(new),40)
  deltas={}
  for key in KEYS:
   try:
    a=np.asarray([v[key] for v in new[:n]],dtype=float)
    b=np.asarray([v[key] for v in old[:n]],dtype=float)
    deltas[key]=float(np.max(np.abs(a-b))) if a.shape==b.shape else None
   except (TypeError,ValueError):deltas[key]=None
  rows.append({'order':run['order'],'state_id':run['state_id'],'candidate_id':run['candidate_id'],
               'source_run_id':run['source_run_id'],'compared_steps':n,'max_abs_by_signal':deltas,
               'long_trace_sha256':sha(folder/'live-0.trace.json.gz'),'pilot_trace_sha256':sha(oldpath)})
 out={'rows':rows,'completed_count':len(rows),'source_identity_sha256':sha(OUT/'long-horizon-run-identities.json'),
      'note':'first 40-step parity is an audit, not a candidate-selection condition'}
 path=OUT/'first40-parity.json'
 if path.exists():
  # A partial earlier audit is preserved; final audit gets a separate name.
  path=OUT/f'first40-parity-{len(rows):02d}.json'
 save_once(path,out)
 print('FIRST40_PARITY',len(rows),path,flush=True)
if __name__=='__main__':main()
