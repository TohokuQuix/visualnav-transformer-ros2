"""Stage C proposal and concise final report from immutable Stage A/B artifacts."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
from eat_nav_command_preview_stage0 import ROOT,OUT,sha,save_once,read
RUNROOT=OUT/'long-horizon'

def main():
 assert read(RUNROOT/'stage-b-analysis-complete.json')['status']=='complete'
 a=read(OUT/'preview-gate.json');b=read(OUT/'long-horizon-decision.json')
 af=read(OUT/'preview-command-fidelity.json');ae=read(OUT/'preview-evaluator-results.json')
 br=read(OUT/'long-horizon-results.json');provenance=read(OUT/'source-provenance.json')
 assert len(br['runs'])==20 and len(br['states'])==4
 next_design={'branch':'C-3: preview fidelity itself failed','status':'proposal_only','approval_required_before_execution':True,
  'reason':'Production follower agrees at step 0, then body-motion prediction diverges; 27.5% state-macro command-cluster agreement and GT(C-hat) pairwise 0.488.',
  'proposed_small_ablation':{
   'data':'reuse 91 frozen 40-step train pilot traces plus 20 approved 120-step traces; no new rollout in first ablation',
   'statistical_unit':'state/episode; never treat candidate runs as independent',
   'comparators':['current unicycle baseline','add current body linear/angular velocity','add roll/pitch and measured slip estimate','add online-available follower/controller internal state','Actor latent as separate explanatory input'],
   'future_information_rule':'only candidate-selection-time state may enter prediction; recorded future poses/commands are targets only',
   'criteria':['production command RMSE and cluster agreement','improvement across states, not a single terrain','GT preview ranking and oracle regret versus frozen baselines','failure cases by stack/tilt/contact'],
   'proposed_run_count':0,'estimated_compute_hours':2,'estimated_new_storage_mb':100,
   'stop_before_training_or_new_collection':True},
  'long_horizon_state_decisions':b['state_classification'],
  'formal_validation_started':False,'retraining_started':False,'full_collection_started':False,'real_robot_started':False}
 save_once(OUT/'next-design-decision.json',next_design)
 with (RUNROOT/'resource-samples.csv').open() as f:rows=list(csv.DictReader(f))
 cpu=[float(x['sim_rss_mib']) for x in rows if x['sim_rss_mib'] not in ('ERROR','')]
 gpu=[float(x['gpu_used_mib']) for x in rows if x['gpu_used_mib'] not in ('ERROR','')]
 budget=read(RUNROOT/'budget-start.json');status=read(RUNROOT/'status.json')
 elapsed=status.get('elapsed_sec',time.time()-budget['started_unix_sec'])
 lines=['# EAT-Nav command-preview / long-horizon diagnostic','',
 'Stage A used seven frozen train states and 91 existing 40-step candidates. Stage B used only four frozen train states and the approved 20 fresh-process 120-step runs. Models, follower rules, candidate identities and thresholds were not changed.','',
 '## Frozen inputs','',
 f"- Branch: {provenance['git']['branch']}; starting HEAD: {provenance['git']['head']}; starting porcelain: {provenance['git']['status'] or 'clean'}.",
 '- Production follower source, binary, parameter and checkpoint SHA values are in source-provenance.json.',
 '- Follower lookahead 0.5 m; goal tolerance 0.2 m; turn-in-place threshold 0.6 rad; linear 0.3 m/s; maximum angular 1.6 rad/s; 20 Hz control. GT uses eight future points at 0.25 s.',
 '- Path and initial pose use map coordinates; GT trajectories use anchor body-frame XY increments in metres. All frozen state, candidate, seed, order and original Path/trace SHA values are in long-horizon-run-identities.json.',
 '',
 '## Stage A','',
 f"- 91 closed-loop previews completed in 7.5 minutes. Seven first-command production fixtures passed with maximum absolute error below 1e-5.",
 f"- Same command cluster state-macro: {af['aggregate']['same_cluster_state_macro']:.3f} (gate 0.75). Closed-loop was no worse than open-loop in {af['aggregate']['closed_not_worse_state_count']}/7 states.",
 f"- GT(C-hat): top-1 {ae['aggregate']['top1_recall_count']}/7; top-3 {ae['aggregate']['top3_recall_count']}/7; state-macro pairwise {ae['aggregate']['state_macro_pairwise_accuracy']:.3f}; regret {ae['aggregate']['state_macro_oracle_regret']:.3f}.",
 f"- Frozen GT(P)/GT(C)/GT(R) pairwise: {ae['prior_aggregate']['P']['state_macro_pairwise_accuracy']:.3f} / {ae['prior_aggregate']['C']['state_macro_pairwise_accuracy']:.3f} / {ae['prior_aggregate']['R']['state_macro_pairwise_accuracy']:.3f}. Recorded C and realized R are privileged comparisons.",
 '- First command-threshold exceedance median: step 16. The worsening after the exact initial match is consistent with a kinematic response gap; this is an inference from the observed signals.',
 f"- Fixed Command Preview Gate: {a['status']}. GT(C-hat) is not suitable for online candidate ranking.",
 '',
 '| State | same-cluster / 13 | GT(P) pairwise | GT(C-hat) pairwise | GT(C) pairwise |',
 '|---|---:|---:|---:|---:|']
 for state in ae['states']:
  sid=state['state_id'];same=sum(r['closed']['same_cluster'] for r in af['candidate_rows'] if r['state_id']==sid)
  lines.append(f"| {sid} | {same}/13 | {state['old_gt']['P']['pairwise_accuracy']:.3f} | {state['gt_c_hat']['pairwise_accuracy']:.3f} | {state['old_gt']['C']['pairwise_accuracy']:.3f} |")
 lines += ['', '## Stage B','',
 f"- {status['completed']}/20 runs; wall time {elapsed/60:.1f} min (cap 90); run storage {status['storage_bytes']/1e6:.1f} MB (cap 250).",
 f"- Simulator CPU RSS peak {max(cpu) if cpu else float('nan'):.1f} MiB; GPU used peak {max(gpu) if gpu else float('nan'):.0f} MiB.",
 '',
 '| State | duplicate XY RMS m | bank minus NoMaD Delta V @40 | @120 | progress difference @120 m | classification |',
 '|---|---:|---:|---:|---:|---|']
 for state in br['states']:
  def fmt(v):return 'NA' if v is None else f'{v:.3f}'
  lines.append(f"| {state['state_id']} | {state['duplicate']['xy_rms_m']:.5f} | {fmt(state['40']['bank_minus_nomad_delta_v'])} | {fmt(state['120']['bank_minus_nomad_delta_v'])} | {fmt(state['120']['bank_minus_nomad_progress_m'])} | {state['classification']} |")
 lines += ['', '- These four train states do not establish all-terrain generalization. Raw step series, Actor input/latent/action, Critic, DEM, contact and reward remain in SHA-verified traces outside Git.',
 '',
 '## Stage C proposal','',
 '- Preview fidelity failed. next-design-decision.json proposes an offline candidate-time state ablation using current body velocity/yaw rate, then roll/pitch, controller state and Actor latent. This has not been executed.',
 '- Formal validation, retraining, additional collection, Phase D/E, full collection and real robot control remain stopped.',
 '',
 '## Reproduction','',
 '- Stage A: run stage0.py, detached command_preview_detached.sh, command_preview_run.py --evaluate, frozen GT command_preview_score.py.',
 '- Stage B: run long_horizon_prepare.py, detached long_horizon_detached.sh, then long_horizon_analyze.py.',
 '- Focused pytest and final worktree status are reported in the completion status file.',
 '',
 '| Artifact | SHA-256 |','|---|---|']
 for name in ('source-provenance.json','preview-input-audit.json','preview-command-fidelity.json','preview-evaluator-results.json','preview-gate.json','long-horizon-run-identities.json','long-horizon-results.json','long-horizon-decision.json','next-design-decision.json'):
  lines.append(f'| {name} | {sha(OUT/name)} |')
 lines.append('')
 report=OUT/'command-preview-long-horizon-report.md'
 if report.exists():raise FileExistsError('report already exists; preserve original and write an addendum')
 report.write_text('\n'.join(lines))
 print('FINAL_REPORT_COMPLETE',report,flush=True)
if __name__=='__main__':main()
