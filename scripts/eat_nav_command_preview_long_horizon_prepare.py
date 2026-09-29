"""Freeze 120-step inputs without changing any selected state, candidate or Path."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np,yaml
from eat_nav_command_preview_stage0 import ROOT,OUT,sha,save_once,read

def main():
 identity=read(OUT/'long-horizon-run-identities.json')['runs']
 assert len(identity)==20 and [r['order'] for r in identity]==list(range(20))
 result={};initial=ROOT/'pilot-inputs'
 for index in (1,3,4,6):
  sid=f'coverage-{index:03d}';src=initial/f'state-{index:03d}';dst=OUT/'long-horizon-inputs'/f'state-{index:03d}';dst.mkdir(parents=True,exist_ok=True)
  original=yaml.safe_load((src/'config.yaml').read_text())
  assert original['control_steps']==40
  extended=dict(original);extended['control_steps']=120
  path=dst/'config-120.yaml'
  data=yaml.safe_dump(extended,sort_keys=True)
  if path.exists():assert path.read_text()==data
  else:path.write_text(data)
  with np.load(src/'suffix-actions.npz') as f:actions=np.asarray(f['actions'],dtype=np.float32)
  assert actions.shape==(40,100,6)
  a=np.concatenate([actions,np.repeat(actions[-1:,:,:],80,axis=0)],axis=0)
  suffix=dst/'suffix-actions-120.npz'
  if suffix.exists():
   with np.load(suffix) as f:assert np.array_equal(f['actions'],a)
  else:np.savez_compressed(suffix,actions=a)
  result[sid]={'original_config_sha256':sha(src/'config.yaml'),'extended_config_sha256':sha(path),
               'original_other_env_actions_sha256':sha(src/'suffix-actions.npz'),
               'extended_other_env_actions_sha256':sha(suffix),
               'other_env_extension':'repeat frozen step-39 action for steps 40..119; selected env action is overwritten by live Actor output',
               'prefix_sha256':sha(src/'prefix-actions.npz'),'source_logged_anchor_sha256':sha(src/'logged-anchor.json')}
  assert all(r['prefix_steps']<=40 for r in identity if r['state_id']==sid)
 save_once(OUT/'long-horizon-input-freeze.json',{'states':result,'run_count':20,'run_identity_sha256':sha(OUT/'long-horizon-run-identities.json'),'steps':120,'source_sim_runner_sha256':sha(Path('scripts/eat_nav_duplicate_recovery_run.py')),'long_sim_runner_sha256':sha(Path('scripts/eat_nav_command_preview_long_horizon_sim.py'))})
 print('LONG_HORIZON_INPUTS_FROZEN',len(result),flush=True)
if __name__=='__main__':main()
