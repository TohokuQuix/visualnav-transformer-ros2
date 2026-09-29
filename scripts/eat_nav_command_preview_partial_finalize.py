"""Finalize a stopped long-horizon run without treating missing cases as outcomes."""
from __future__ import annotations
import csv,gzip,json,time
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.command_preview_long_horizon import endpoint_metrics
from eat_nav_command_preview_stage0 import ROOT,OUT,sha,save_once,read
RUNROOT=OUT/'long-horizon'

def main():
 status=read(RUNROOT/'status.json')
 assert status['state']=='stopped' and status['completed']==8 and status['next_run']==8
 diagnostic=read(OUT/'duplicate-failure-diagnostic.json')
 assert diagnostic['duplicate_xy_rms_m']>.0105
 runs=read(OUT/'long-horizon-run-identities.json')['runs'];rows=[]
 for run in runs[:8]:
  p=RUNROOT/'runs'/f"long-{run['order']:03d}"
  marker=read(p/'run-complete.json');assert marker['frozen_run']==run
  for name,digest in marker['files_sha256'].items():assert sha(p/name)==digest
  s=read(p/'live-0.json')
  with gzip.open(p/'live-0.trace.json.gz','rt') as f:t=json.load(f)
  assert len(t)==120 and s['suffix_steps']==120
  mp=s['map_path'];m40=endpoint_metrics(t,40,t[40]['critic_value'],mp[0],mp[-1],mp)
  m120=endpoint_metrics(t,120,s['suffix_end_critic_value'],mp[0],mp[-1],mp)
  rows.append({'run_id':f"long-{run['order']:03d}",'state_id':run['state_id'],'terrain':run['terrain'],
   'kind':run['kind'],'candidate_id':run['candidate_id'],'seed':run['seed'],'repeat':run['repeat'],
   'source_run_id':run['source_run_id'],'map_path_sha256':run['map_path_sha256'],
   'trace_sha256':sha(p/'live-0.trace.json.gz'),'summary_sha256':sha(p/'live-0.json'),
   'preview_sha256':sha(RUNROOT/'preview-120'/f"{run['state_id']}--{run['candidate_id']}.json"),
   'actor_input_shape':[len(t[0]['actor_input'])],'actor_latent_shape':[len(t[0]['actor_latent'])],
   'actor_action_shape':[len(t[0]['actor_action'])],'critic_observation_shape':[len(t[0]['critic_observation'])],
   'metric_40':m40,'metric_120':m120,'state_duplicate_valid':run['state_id']=='coverage-001'})
 valid=rows[:5]
 state={'state_id':'coverage-001','terrain':valid[0]['terrain'],'duplicate':read(RUNROOT/'duplicate-gates/coverage-001.json'),
  'nominal_run_ids':[r['run_id'] for r in valid[:3]],'nomad_run_id':valid[3]['run_id'],'bank_run_id':valid[4]['run_id']}
 for key in ('40','120'):
  k='metric_'+key
  state[key]={'nominal':[r[k] for r in valid[:3]],'nomad':valid[3][k],'bank':valid[4][k],
   'bank_minus_nomad_delta_v':valid[4][k]['delta_v']-valid[3][k]['delta_v'],
   'bank_minus_nomad_progress_m':valid[4][k]['forward_goal_progress_m']-valid[3][k]['forward_goal_progress_m']}
 state['classification']='inconclusive'
 result={'status':'partial_stopped_by_duplicate_gate','planned_run_count':20,'completed_run_count':8,
  'valid_state_count':1,'planned_state_count':4,'states':[state],'runs':rows,
  'invalid_states':[{'state_id':'coverage-003','completed_nominal_runs':[r['run_id'] for r in rows[5:8]],
                     'reason':'duplicate XY RMS 0.035261 m > 0.0105 m','diagnostic_sha256':sha(OUT/'duplicate-failure-diagnostic.json')}],
  'not_started_run_ids':[f"long-{i:03d}" for i in range(8,20)],'all_terrain_generalization_claim':False,
  'statistical_unit':'state','no_formal_coverage_conclusion':True}
 save_once(OUT/'long-horizon-results.json',result)
 save_once(OUT/'long-horizon-decision.json',{'status':'measurement_invalid_for_planned_four_state_comparison',
  'coverage-001':'exploratory_only; bank constraint-invalid and one valid state cannot establish coverage',
  'coverage-003':'duplicate gate failed due follower stuck reversal in one nominal repeat',
  'coverage-004':'NA/not started','coverage-006':'NA/not started',
  'new_runs_started_after_failure':False,'result_sha256':sha(OUT/'long-horizon-results.json')})
 next_design={'status':'proposal_only','approval_required_before_execution':True,
  'priority_1':'restore duplicate reproducibility without changing follower control law or candidate identities',
  'root_cause_candidate':'production linear-stuck detector uses ROS elapsed time while simulator advances more slowly; one repeat reversed at step 39',
  'proposed_preflight':['audit ROS clock and simulator clock per control step','log follower motion-window sample times and travel-direction state without changing rule','synchronize simulated TF/clock delivery in a separate diagnostic runner','repeat only the frozen coverage-003 nominal triplet under a newly approved protocol, then compare with old traces'],
  'new_simulator_run_count_proposed':3,'run_horizon_steps':120,'estimated_wall_min':8,'estimated_storage_mb':10,
  'stop_rule':'duplicate XY RMS <= 0.0105 m and identical anchor/Path/command; stop on first anomaly',
  'priority_2':'Stage A failed, so separately design a small offline candidate-time state ablation using current body velocity/yaw rate, roll/pitch, controller state and Actor latent; future poses and commands are targets only',
  'offline_ablation_inputs':'reuse frozen 91 pilot traces and the 8 completed long-horizon traces; state/episode are statistical units',
  'offline_ablation_compute_hours_estimate':2,'offline_ablation_storage_mb_estimate':100,
  'model_retraining_started':False,'formal_validation_started':False,'full_collection_started':False,'real_robot_started':False}
 save_once(OUT/'next-design-decision.json',next_design)
 with (RUNROOT/'resource-samples.csv').open() as f:res=list(csv.DictReader(f))
 cpu=[float(x['sim_rss_mib']) for x in res if x['sim_rss_mib'] not in ('','ERROR')]
 gpu=[float(x['gpu_used_mib']) for x in res if x['gpu_used_mib'] not in ('','ERROR')]
 af=read(OUT/'preview-command-fidelity.json');ae=read(OUT/'preview-evaluator-results.json');gate=read(OUT/'preview-gate.json')
 prov=read(OUT/'source-provenance.json')
 lines=['# Command-preview and long-horizon coverage: stopped pilot report','',
 'All inputs were frozen before analysis. Stage A used the existing 91 candidates; Stage B started only the approved 20-run schedule and stopped after run 8 when the second state duplicate gate failed. No unapproved candidate, training, formal validation or real-robot control was started.','',
 '## Stage 0','',
 f"- Branch {prov['git']['branch']}, starting HEAD {prov['git']['head']}, starting worktree clean.",
 '- NoMaD/GT checkpoint, train-only scaler, pilot Path/trace/outcome and production follower source/binary/config SHA values are in source-provenance.json.',
 '- Production follower: lookahead 0.5 m, goal tolerance 0.2 m, turn-in-place threshold 0.6 rad, linear speed 0.3 m/s, angular limit 1.6 rad/s; control 20 Hz; GT 8 points at 0.25 s.',
 '',
 '## Stage A: online-available follower command preview','',
 '- Preview used only frozen Path, anchor pose/yaw, production follower parameters and a 0.05 s kinematic pose update. Recorded commands and outcomes were loaded only after preview markers were saved.',
 f"- All 91 candidates completed in 7.5 min. Seven production first-command fixtures matched within 1e-5. State-macro same-command-cluster rate was {af['aggregate']['same_cluster_state_macro']:.3f}, below the 0.75 gate.",
 f"- Closed-loop was no worse than open-loop in {af['aggregate']['closed_not_worse_state_count']}/7 states. First command-threshold exceedance had median step 16.",
 f"- Frozen GT(C-hat): top-1 {ae['aggregate']['top1_recall_count']}/7, top-3 {ae['aggregate']['top3_recall_count']}/7, pairwise {ae['aggregate']['state_macro_pairwise_accuracy']:.3f}, oracle regret {ae['aggregate']['state_macro_oracle_regret']:.3f}.",
 f"- Reference GT(P), GT(C), GT(R) pairwise were {ae['prior_aggregate']['P']['state_macro_pairwise_accuracy']:.3f}, {ae['prior_aggregate']['C']['state_macro_pairwise_accuracy']:.3f}, {ae['prior_aggregate']['R']['state_macro_pairwise_accuracy']:.3f}. Recorded C and realized R remain privileged diagnostics.",
 f"- Command Preview Gate: {gate['status']}. Current preview and frozen Evaluator cannot be used for online candidate selection.",
 '',
 '| State | same cluster / 13 | GT(P) pairwise | GT(C-hat) pairwise | GT(C) pairwise |',
 '|---|---:|---:|---:|---:|']
 for s in ae['states']:
  same=sum(r['closed']['same_cluster'] for r in af['candidate_rows'] if r['state_id']==s['state_id'])
  lines.append(f"| {s['state_id']} | {same}/13 | {s['old_gt']['P']['pairwise_accuracy']:.3f} | {s['gt_c_hat']['pairwise_accuracy']:.3f} | {s['old_gt']['C']['pairwise_accuracy']:.3f} |")
 lines += ['', '## Stage B: stopped at duplicate gate','',
 f"- Completed 8/20 runs in {status['elapsed_sec']/60:.1f} min; run storage {status['storage_bytes']/1e6:.1f} MB, below the 90 min / 250 MB caps. Simulator CPU RSS peak {max(cpu):.0f} MiB and GPU used peak {max(gpu):.0f} MiB.",
 '- One approved retry was used for long-005 after a Kit crash at startup with uptime 0 s and no simulation step. The original log and SHA remain in startup-crash-review.json and runner.log.',
 '- coverage-001 nominal 3 repeat XY RMS was 0 m. Its 2 s and 6 s exploratory comparisons are below; the bank winner is constraint-invalid, and one state does not establish feasible coverage.',
 '| coverage-001 | nominal Delta V | NoMaD Delta V | bank Delta V | bank minus NoMaD |',
 '|---|---:|---:|---:|---:|']
 for k in ('40','120'):
  v=state[k];lines.append(f"| step {k} | {v['nominal'][0]['delta_v']:.3f} | {v['nomad']['delta_v']:.3f} | {v['bank']['delta_v']:.3f} | {v['bank_minus_nomad_delta_v']:.3f} |")
 lines += ['',
 f"- coverage-003 duplicate XY RMS {diagnostic['duplicate_xy_rms_m']:.5f} m exceeded the frozen 0.0105 m threshold. The first difference was follower command at step 39. One nominal repeat reversed linear speed from +0.293 to -0.296 m/s after a production Linear stuck detected event; the other two repeats matched each other for 120 steps.",
 f"- The follower reported commanded distance {diagnostic['follower_event']['commanded_distance_m']:.3f} m, actual {diagnostic['follower_event']['actual_distance_m']:.3f} m, ratio {diagnostic['follower_event']['actual_to_commanded_ratio']:.3f} against its 0.25 threshold. The 20 s motion window uses ROS elapsed time; a scheduling boundary with slow simulator progress is the leading explanation, not a proven unique cause.",
 '- Candidate comparisons for coverage-003 are invalid. coverage-004 and coverage-006 were not started. The planned four-state long-horizon coverage gate is measurement-invalid; no generalization or formal pass is claimed.',
 '',
 '## Stage C: proposed work only','',
 '- First restore duplicate reproducibility by measuring ROS/simulator clock and follower motion-window state without changing the production control law. The proposed three-run diagnostic requires new approval.',
 '- Separately, since Stage A fidelity failed, design an offline state ablation using current body velocity/yaw rate, roll/pitch, controller state and Actor latent. Future simulator information remains excluded from the predictor.',
 '- No further rollout, retraining, Phase D/E, full collection or real-robot control was started.',
 '',
 '## Execution and integrity','',
 '- Stage A command: detached scripts/eat_nav_command_preview_detached.sh; frozen GT inference: scripts/eat_nav_command_preview_score.py.',
 '- Stage B command: detached scripts/eat_nav_command_preview_long_horizon_detached.sh; startup retry was source-SHA reviewed and limited to the same run.',
 '- Raw traces, checkpoints and large logs remain outside Git. Small JSON, figures, config, source, tests and this report are saved in the dedicated branch.',
 '| Artifact | SHA-256 |','|---|---|']
 for name in ('source-provenance.json','preview-command-fidelity.json','preview-evaluator-results.json','preview-gate.json','long-horizon-run-identities.json','duplicate-failure-diagnostic.json','long-horizon-results.json','long-horizon-decision.json','next-design-decision.json'):
  lines.append(f'| {name} | {sha(OUT/name)} |')
 lines.append('')
 report=OUT/'command-preview-long-horizon-report.md'
 assert not report.exists();report.write_text('\n'.join(lines))
 save_once(OUT/'stage-c-partial-complete.json',{'status':'partial_report_complete','reason':'duplicate_gate_fail','report_sha256':sha(report),
  'long_horizon_results_sha256':sha(OUT/'long-horizon-results.json')})
 print('PARTIAL_FINAL_COMPLETE',len(rows),report)
if __name__=='__main__':main()
