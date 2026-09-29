"""Freeze inputs for command-preview and approved twenty-run long-horizon diagnostic."""
from __future__ import annotations
import hashlib,json,subprocess,time
from pathlib import Path
import yaml

ROOT=Path('/tmp/eatnav-candidate-coverage')
PRIOR=ROOT/'feasible-transfer'
OUT=ROOT/'command-preview-long-horizon'
FOLLOWER=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower')
BINARY=Path('/home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node')
STATES=('coverage-001','coverage-003','coverage-004','coverage-006')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def save_once(path,obj):
 data=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_text()!=data:raise ValueError(f'frozen output differs: {path}')
  return
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(data);tmp.replace(path)

def read(p):return json.loads(Path(p).read_text())

def git(*args):return subprocess.check_output(['git',*args],text=True).strip()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 old=read(PRIOR/'source-provenance.json')
 prior_results=read(PRIOR/'evaluator-counterfactual-results.json')
 decision=read(PRIOR/'decision.json')['next_experiment_proposal']
 assert tuple(decision['state_ids'])==STATES and decision['run_count']==20 and decision['control_steps']==120
 assert prior_results['candidate_count']==91 and prior_results['state_count']==7
 for name,key in [('nomad_reference_best_ema','nomad_reference_best_ema'),('evaluator_GT_11','evaluator_GT_11'),('evaluator_GT_29','evaluator_GT_29'),('evaluator_GT_47','evaluator_GT_47'),('evaluator_train_scalers','evaluator_train_scalers')]:
  p=Path(old['container'][key]['path'])
  # The frozen GT checkpoints reside inside the running Isaac container.
  observed=subprocess.check_output(['docker','exec','isaac-lab-onix','sha256sum',str(p)],text=True).split()[0]
  assert observed==old['container'][key]['sha256'],f'{name} SHA mismatch'
 for name,expected in [('candidate-identity-table.jsonl.gz',old['candidate_identity_table_sha256']),('feasibility-by-candidate.json',None)]:
  p=PRIOR/name
  assert p.exists() and (expected is None or sha(p)==expected),p
 source_files=['candidate-identity-table.jsonl.gz','feasibility-by-candidate.json','planned-command-realized-comparison.json','evaluator-counterfactual-results.json','decision.json','source-provenance.json','stage-3-complete.json']
 source_sha={p:sha(PRIOR/p) for p in source_files}
 follower_sources=[FOLLOWER/'src/simple_trajectory_follower.cpp',FOLLOWER/'src/path_geometry.cpp',FOLLOWER/'include/simple_trajectory_follower/path_geometry.hpp',FOLLOWER/'config/onix.yaml',BINARY]
 assert all(p.exists() for p in follower_sources)
 cfg=yaml.safe_load((FOLLOWER/'config/onix.yaml').read_text())['/**']['ros__parameters']
 frozen_config={'stage_a':{'primary':'actual_production_follower_closed_loop_unicycle','secondary':'actual_production_follower_fixed_initial_pose','steps':40,'control_dt_sec':.05,'trajectory_points':8,'trajectory_interval_sec':.25,'command_cluster_threshold':{'linear_rms_mps':.03,'angular_rms_radps':.05},'gate':{'same_cluster_macro_min':.75,'closed_not_worse_states_min':5,'pairwise_min':.60,'oracle_regret_max':.10}},'stage_b':{'state_ids':list(STATES),'steps':120,'dt_sec':.05,'full_prefix_cap':40,'repeat_nominal':3,'duplicate_xy_rms_max_m':.0105,'wall_clock_cap_sec':5400,'storage_cap_bytes':250_000_000,'estimated_wall_sec':2700,'estimated_storage_bytes':100_000_000,'retry_pre_sim_infrastructure_max':1},'production_follower_parameters':cfg,'no_future_preview_inputs':['recorded_command','realized_trajectory','future_pose','future_dem','Actor_observation','Critic_value','reward','termination','delta_v']}
 path=OUT/'command-preview-config.yaml'
 if path.exists():assert yaml.safe_load(path.read_text())==frozen_config
 else:path.write_text(yaml.safe_dump(frozen_config,sort_keys=True))
 long_path=OUT/'long-horizon-config.yaml'
 if long_path.exists():assert yaml.safe_load(long_path.read_text())==frozen_config['stage_b']
 else:long_path.write_text(yaml.safe_dump(frozen_config['stage_b'],sort_keys=True))
 rows=read(PRIOR/'feasibility-by-candidate.json')['rows'];by={(r['state_id'],r['candidate_id']):r for r in rows}
 banks={r['state_id']:r['bank_candidate_id'] for r in read(PRIOR/'nomad-oracle-regret.json')['bank_winner_comparisons']}
 schedule=read(ROOT/'stage0/frozen/pilot-schedule.json')['runs'];lookup={(r['state_id'],('bank-nominal' if r['kind']=='duplicate' else ('nomad-seed-'+str(r['seed']) if r['kind']=='nomad' else 'bank-'+str(r['name'])))):r for r in schedule if r['kind'] in ('duplicate','nomad','bank')}
 identities=[]
 for sid in STATES:
  n=max((r for r in rows if r['state_id']==sid and r['generator']=='nomad'),key=lambda r:r['delta_v'])['candidate_id']
  for kind,cid,repeat in [('nominal','bank-nominal',0),('nominal','bank-nominal',1),('nominal','bank-nominal',2),('nomad_oracle',n,0),('bank_winner',banks[sid],0)]:
   src=lookup[(sid,cid)];folder=ROOT/'pilot/runs'/src['run_id']
   if kind=='nominal':src=next(r for r in schedule if r['state_id']==sid and r['kind']=='duplicate' and r['name']==repeat);folder=ROOT/'pilot/runs'/src['run_id']
   p=folder/'live-0.json';t=folder/'live-0.trace.json.gz';base=ROOT/'pilot-inputs'/f"state-{int(sid[-3:]):03d}";path_file=base/f'{cid}-map-path.json'
   assert p.exists() and t.exists() and path_file.exists()
   assert sha(p)==by[(sid,cid)]['summary_sha256'] if 'summary_sha256' in by[(sid,cid)] else True
   assert src['full_prefix_steps']<=40
   identities.append({'order':len(identities),'state_id':sid,'terrain':src['terrain_type'],'kind':kind,'repeat':repeat,'candidate_id':cid,'seed':src['seed'],'source_run_id':src['run_id'],'prefix_steps':src['full_prefix_steps'],'map_path_sha256':sha(path_file),'source_trace_sha256':sha(t),'source_summary_sha256':sha(p)})
 assert len(identities)==20
 provenance={'git':{'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),'status':git('status','--porcelain')},'source_sha256':source_sha,'prior_checkpoint_sha256':old['container'],'follower_sha256':{str(p):sha(p) for p in follower_sources},'pilot_schedule_sha256':sha(ROOT/'stage0/frozen/pilot-schedule.json'),'pilot_selection_sha256':sha(ROOT/'stage0/frozen/pilot-selection.json'),'preview_config_sha256':sha(path),'long_horizon_config_sha256':sha(long_path),'control_period_sec':.05,'trajectory_interval_sec':.25,'frames':{'planned_path':'map','bridge_pose':'map position plus body yaw','GT_input':'anchor body frame, x forward/y left'},'initial_pose_source':'source pilot live summary anchor root_state and frozen map Path origin','estimated_stage_a_execution_sec':1200,'estimated_stage_b_execution_sec':2700,'estimated_stage_b_storage_bytes':100_000_000,'stage_b_wall_cap_sec':5400,'stage_b_storage_cap_bytes':250_000_000}
 save_once(OUT/'source-provenance.json',provenance)
 save_once(OUT/'long-horizon-run-identities.json',{'runs':identities,'source_sha256':source_sha,'stage_b_config_sha256':sha(long_path)})
 marker=OUT/'stage-0-complete.json'
 fixed={'status':'complete','source_provenance_sha256':sha(OUT/'source-provenance.json'),'run_identities_sha256':sha(OUT/'long-horizon-run-identities.json')}
 if marker.exists():
  assert all(read(marker)[k]==v for k,v in fixed.items()),'Stage 0 marker differs'
 else:save_once(marker,{**fixed,'unix_sec':time.time()})
 save_once(OUT/'stage-0-checkpoint-verification.json',{'status':'complete','checkpoint_sha256':{key:old['container'][key]['sha256'] for key in ('nomad_reference_best_ema','evaluator_GT_11','evaluator_GT_29','evaluator_GT_47','evaluator_train_scalers')},'source_provenance_sha256':sha(OUT/'source-provenance.json')})
 print('STAGE0_COMPLETE',len(identities),'branch',provenance['git']['branch'])
if __name__=='__main__':main()
