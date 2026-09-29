"""Analyze only frozen completed 20-run long-horizon traces."""
from __future__ import annotations
import gzip,json,math,time
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.command_preview_long_horizon import endpoint_metrics,post_root
from eat_nav_command_preview_stage0 import ROOT,OUT,sha,save_once,read
RUNROOT=OUT/'long-horizon'

def metric_at(summary,trace,steps):
 if len(trace)<steps:return {'status':'terminated_before_horizon','available_steps':len(trace),
                                  'termination':trace[-1].get('terminal')}
 if len(trace)>steps:end=trace[steps]['critic_value']
 elif summary['suffix_end_critic_value'] is not None:end=summary['suffix_end_critic_value']
 else:
  terminal=trace[-1].get('terminal')
  end=0. if terminal and terminal.get('terminated') else trace[-1]['critic_value']
 p=summary['map_path']
 return {'status':'observed',**endpoint_metrics(trace,steps,end,p[0],p[-1],p)}

def main():
 start=time.monotonic()
 complete=read(RUNROOT/'stage-b-runs-complete.json')
 assert complete['run_count']==20
 runs=read(OUT/'long-horizon-run-identities.json')['runs']
 prior_rows={(x['state_id'],x['candidate_id']):x for x in read(ROOT/'feasible-transfer/feasibility-by-candidate.json')['rows']}
 rows=[];by_state={}
 for r in runs:
  rid=f"long-{r['order']:03d}";folder=RUNROOT/'runs'/rid
  marker=read(folder/'run-complete.json');assert marker['frozen_run']==r
  for name,digest in marker['files_sha256'].items():assert sha(folder/name)==digest
  summary=read(folder/'live-0.json')
  with gzip.open(folder/'live-0.trace.json.gz','rt') as f:trace=json.load(f)
  assert len(trace)==summary['suffix_steps'] and 0<len(trace)<=120
  preview=RUNROOT/'preview-120'/f"{r['state_id']}--{r['candidate_id']}.json"
  p=read(preview);assert sha(preview)==read(preview.with_suffix('.marker.json'))['output_sha256']
  assert len(p['closed_command'])==120
  root=np.asarray([post_root(v)[:2] for v in trace],dtype=float)
  cmd=np.asarray([v['command'] for v in trace],dtype=float)
  pc=np.asarray(p['closed_command'][:len(trace)],dtype=float)
  row={'run_id':rid,'state_id':r['state_id'],'terrain':r['terrain'],'candidate_id':r['candidate_id'],'kind':r['kind'],
       'seed':r['seed'],'repeat':r['repeat'],'source_run_id':r['source_run_id'],
       'constraint_valid':prior_rows[(r['state_id'],r['candidate_id'])]['condition']['constraint_valid'],'map_path_sha256':r['map_path_sha256'],
       'source_trace_sha256':r['source_trace_sha256'],'long_trace_sha256':sha(folder/'live-0.trace.json.gz'),
       'preview_sha256':sha(preview),'step_count':len(trace),
       'preview_command_linear_rmse_mps':float(np.sqrt(np.mean((cmd[:,0]-pc[:,0])**2))),
       'preview_command_angular_rmse_radps':float(np.sqrt(np.mean((cmd[:,2]-pc[:,2])**2))),
       '40':metric_at(summary,trace,40),'120':metric_at(summary,trace,120),
       'preview_command_120':p['closed_command'],'recorded_command_120':cmd.tolist(),
       'root_xy_120':root.tolist(),'critic_value_120':[v['critic_value'] for v in trace],
       'reward_120':[v['reward'] for v in trace],'termination_reason':trace[-1].get('terminal'),
       'actor_observation_shape':[len(trace[0]['actor_input'])],'actor_latent_shape':[len(trace[0]['actor_latent'])],
       'actor_action_shape':[len(trace[0]['actor_action'])],'critic_observation_shape':[len(trace[0]['critic_observation'])]}
  rows.append(row);by_state.setdefault(r['state_id'],[]).append(row)
 states=[];decisions=[]
 for sid,group in by_state.items():
  gate=read(RUNROOT/'duplicate-gates'/f'{sid}.json');assert gate['pass']
  nom=next(v for v in group if v['kind']=='nomad_oracle')
  bank=next(v for v in group if v['kind']=='bank_winner')
  nominal=[v for v in group if v['kind']=='nominal']
  state={'state_id':sid,'terrain':group[0]['terrain'],'duplicate':gate,'nominal_run_ids':[v['run_id'] for v in nominal],
         'nomad_run_id':nom['run_id'],'bank_run_id':bank['run_id']}
  for horizon in ('40','120'):
   nm=nom[horizon];bm=bank[horizon];base=[v[horizon] for v in nominal]
   state[horizon]={'nominal':base,'nomad':nm,'bank':bm,
                   'bank_minus_nomad_delta_v':bm['delta_v']-nm['delta_v'] if nm['status']==bm['status']=='observed' else None,
                   'bank_minus_nomad_progress_m':bm['forward_goal_progress_m']-nm['forward_goal_progress_m'] if nm['status']==bm['status']=='observed' else None}
  common=min(len(nom['root_xy_120']),len(bank['root_xy_120']))
  nr=np.asarray(nom['root_xy_120'][:common])-np.asarray(nom['root_xy_120'][0])
  br=np.asarray(bank['root_xy_120'][:common])-np.asarray(bank['root_xy_120'][0])
  realized_rms=float(np.sqrt(np.mean((nr-br)**2)))
  nc=np.asarray(nom['recorded_command_120'][:common]);bc=np.asarray(bank['recorded_command_120'][:common])
  command_linear_rms=float(np.sqrt(np.mean((nc[:,0]-bc[:,0])**2)))
  command_angular_rms=float(np.sqrt(np.mean((nc[:,2]-bc[:,2])**2)))
  distinct=realized_rms>.05 or command_linear_rms>.03 or command_angular_rms>.05
  state['bank_nomad_mode_difference_120']={'realized_xy_rms_m':realized_rms,'command_linear_rms_mps':command_linear_rms,'command_angular_rms_radps':command_angular_rms,'distinct':distinct}
  if nom['120']['status']!='observed' or bank['120']['status']!='observed':label='inconclusive'
  elif gate['xy_rms_m']>.0105:label='measurement_invalid'
  elif not distinct:label='same_mode_outcome_variation'
  elif bank['constraint_valid'] and (state['120']['bank_minus_nomad_delta_v'] or 0)>.01 and bank['120']['forward_goal_progress_m']>nom['120']['forward_goal_progress_m']+.02 and not bank['120']['stack'] and bank['120']['maximum_tilt_rad']<math.radians(85):
   label='long_horizon_undercoverage_evidence'
  elif state['120']['bank_minus_nomad_delta_v'] is not None and state['120']['bank_minus_nomad_delta_v']<-.01:
   label='long_horizon_concentration_supported'
  else:label='same_mode_outcome_variation'
  state['classification']=label;states.append(state);decisions.append(label)
 compact_rows=[{k:v for k,v in row.items() if k not in ('preview_command_120','recorded_command_120','root_xy_120','critic_value_120','reward_120')} for row in rows]
 result={'states':states,'runs':compact_rows,'state_count':4,'run_count':20,'statistical_unit':'state',
         'all_terrain_generalization_claim':False,'elapsed_analysis_sec':time.monotonic()-start,
         'long_horizon_status_counts':{x:decisions.count(x) for x in sorted(set(decisions))}}
 save_once(OUT/'long-horizon-results.json',result)
 save_once(OUT/'long-horizon-decision.json',{'state_classification':{s['state_id']:s['classification'] for s in states},
  'aggregate':'inconclusive' if len(set(decisions))>1 else decisions[0],
  'scope':'four frozen train states; 6 s only','formal_pass':False,'new_rollouts_beyond_frozen_20':False,
  'result_sha256':sha(OUT/'long-horizon-results.json')})
 save_once(RUNROOT/'stage-b-analysis-complete.json',{'status':'complete','result_sha256':sha(OUT/'long-horizon-results.json'),
  'decision_sha256':sha(OUT/'long-horizon-decision.json')})
 print('LONG_ANALYSIS_COMPLETE',decisions,flush=True)
if __name__=='__main__':main()
