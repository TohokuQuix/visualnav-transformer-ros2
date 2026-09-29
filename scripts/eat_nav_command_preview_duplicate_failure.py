"""Read-only first-divergence audit for the stopped coverage-003 duplicates."""
from __future__ import annotations
import gzip,json,re
from pathlib import Path
import numpy as np
from eat_nav_command_preview_stage0 import ROOT,OUT,sha,save_once,read
RUNROOT=OUT/'long-horizon';IDS=(5,6,7)
def load(i):
 p=RUNROOT/'runs'/f'long-{i:03d}';s=read(p/'live-0.json')
 with gzip.open(p/'live-0.trace.json.gz','rt') as f:t=json.load(f)
 return s,t,p
def first_diff(a,b,key,eps=1e-8):
 for k,(x,y) in enumerate(zip(a,b)):
  v=np.asarray(x[key],dtype=float);w=np.asarray(y[key],dtype=float)
  if v.shape!=w.shape or float(np.max(np.abs(v-w)))>eps:return k
 return None
def main():
 summaries=[];traces=[];folders=[]
 for i in IDS:
  s,t,p=load(i);summaries.append(s);traces.append(t);folders.append(p)
 assert len({s['fixed_map_path_sha256'] for s in summaries})==1
 rows=[];base=traces[0]
 for index,t in zip(IDS[1:],traces[1:]):
  changes={k:first_diff(base,t,k) for k in ('command','root_state','actor_input','actor_latent','actor_action','critic_observation','critic_value','reward','post_root_state')}
  diffs=np.asarray([r['post_root_state'][:2] for r in t])-np.asarray(t[0]['root_state'][:2])
  norm=np.asarray([r['post_root_state'][:2] for r in base])-np.asarray(base[0]['root_state'][:2])
  rows.append({'compare_to':'long-005','run_id':f'long-{index:03d}','first_changed_step_by_signal':changes,
    'command_at_step_39':t[39]['command'],'reference_command_at_step_39':base[39]['command'],
    'relative_xy_rms_40_m':float(np.sqrt(np.mean((diffs[:40]-norm[:40])**2))),
    'relative_xy_rms_120_m':float(np.sqrt(np.mean((diffs-norm)**2))),
    'trace_sha256':sha(folders[index-5]/'live-0.trace.json.gz')})
 logpath=Path(summaries[1]['follower_startup']['readiness']['follower_logs'][0])
 text=logpath.read_text(errors='replace')
 match=re.search(r'Linear stuck detected: commanded=([0-9.]+)m actual=([0-9.]+)m ratio=([0-9]+(?:\.[0-9]+)?)',text)
 assert match and 'travel_dir=-1' in text
 warn=next(line for line in text.splitlines() if 'Linear stuck detected' in line)
 source=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src/simple_trajectory_follower.cpp')
 params=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')
 result={'status':'measurement_invalid','state_id':'coverage-003','completed_run_ids':[f'long-{i:03d}' for i in IDS],
  'duplicate_xy_rms_m':float(np.sqrt(np.mean(np.concatenate([
    (np.asarray([r['post_root_state'][:2] for r in traces[j]])-np.asarray(traces[j][0]['root_state'][:2])-
     (np.asarray([r['post_root_state'][:2] for r in base])-np.asarray(base[0]['root_state'][:2]))).ravel()**2
     for j in (1,2)])))),
  'frozen_xy_rms_limit_m':.0105,'comparisons':rows,
  'anchor_root_equal':all(s['anchor']['root_state']==summaries[0]['anchor']['root_state'] for s in summaries),
  'path_sha_equal':True,
  'follower_event':{'run_id':'long-006','log_path':str(logpath),'log_sha256':sha(logpath),'line':warn,
    'commanded_distance_m':float(match.group(1)),'actual_distance_m':float(match.group(2)),
    'actual_to_commanded_ratio':float(match.group(3)),'parameter_stuck_window_sec':20.,
    'parameter_ratio_threshold':.25},
  'production_follower_source_sha256':sha(source),'production_follower_parameter_sha256':sha(params),
  'interpretation':'One repeat entered production linear-stuck reversal at step 39 while the same anchor, Path and prior 39 steps matched. The controller uses ROS elapsed time for commanded distance; simulator advances 0.05 s per slow wall-clock step. A wall-time scheduling boundary is the leading explanation. Exact scheduler causality remains unproven.',
  'candidate_comparison_valid_for_state':False,
  'new_runs_started_after_failure':False}
 save_once(OUT/'duplicate-failure-diagnostic.json',result)
 print('DUPLICATE_FAILURE_DIAGNOSTIC',result['duplicate_xy_rms_m'],rows[0]['first_changed_step_by_signal'])
if __name__=='__main__':main()
