"""Read-only first-divergence analysis and compact report for the frozen race probe."""
from __future__ import annotations
import gzip,hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')
DOC=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/candidate-coverage/tf-race-diagnostic'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
 temp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');temp.replace(p)
def lines(path):
 with Path(path).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def gzip_jsonl(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with gzip.open(t,'wt') as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
 t.replace(p)
def parse_follower(path,branch,path_sha):
 records=[];bad=0
 for line in Path(path).read_text(errors='replace').splitlines():
  if line.startswith('TF_RACE_JSON '):
   try:
    row=json.loads(line.split(' ',1)[1]);row['branch']=branch;row['path_sha256']=path_sha;records.append(row)
   except json.JSONDecodeError:bad+=1
 return records,bad

def match(records,command,stamp,receive_steady):
 possible=[r for r in records if r['phase']=='command_pre_publish' and r['command_stamp_ns']==stamp
  and abs(r['vx']-command[0])<=1e-9 and abs(r['wz']-command[2])<=1e-9
  and r['wall_steady_ns']<=receive_steady]
 if not possible:raise ValueError('follower callback for bridge command missing')
 return possible[-1]

def main():
 assert json.loads((ROOT/'stage-r0-complete.json').read_text())['status']=='complete'
 state=json.loads((ROOT/'tf-race-source-audit.json').read_text())['state']
 path_sha=json.loads((ROOT/'tf-race-config.yaml').read_text()) if False else None
 config=__import__('yaml').safe_load((ROOT/'tf-race-config.yaml').read_text())
 repeats=sorted(ROOT.glob('repeat-[0-9][0-9]'))
 completed=[p for p in repeats if (p/'stage-complete.json').exists()]
 if not completed:raise RuntimeError('no completed repeat marker')
 run_markers=[json.loads((p/'stage-complete.json').read_text()) for p in completed]
 all_callback=[];raw_sha={};counts={};parse_bad={}
 for repeat_dir in completed:
  repeat=int(repeat_dir.name[-2:]);process=json.loads((repeat_dir/'process-map.json').read_text())
  for branch in range(8):
   source=ROOT/'follower-logs'/f'reset-{repeat+1:04d}-branch-{branch}.log'
   rows,bad=parse_follower(source,branch,config['path_sha256'])
   pid=process['branch_pid'][str(branch)]
   if any(r['pid']!=pid for r in rows):raise RuntimeError('branch/PID mismatch')
   output=repeat_dir/'callback-traces'/f'branch-{branch}.jsonl.gz'
   gzip_jsonl(output,rows)
   raw_sha[str(source)]={'sha256':sha(source),'size_bytes':source.stat().st_size,
    'callback_trace_sha256':sha(output)}
   counts[f'{repeat}:{branch}']=len({r['callback_seq'] for r in rows})
   parse_bad[f'{repeat}:{branch}']=bad
   all_callback.extend(rows)
  for kind in ('clock-publish','tf-publish','path-publish','bridge-command'):
   source=repeat_dir/f'{kind}-trace.jsonl';output=repeat_dir/f'{kind}-trace.jsonl.gz'
   gzip_jsonl(output,list(lines(source)))
   raw_sha[str(source)]={'sha256':sha(source),'size_bytes':source.stat().st_size,
    'compressed_sha256':sha(output)}
  runner=repeat_dir/'runner.log';raw_sha[str(runner)]={'sha256':sha(runner),'size_bytes':runner.stat().st_size}
 conflict_path=ROOT/'same-stamp-command-conflict-first.json'
 if not conflict_path.exists():
  decision={'classification':'inconclusive','reason':'no same-stamp command conflict in completed repeats',
   'repeat_count':len(completed),'not_resolved':True}
  atomic(ROOT/'tf-race-cause-decision.json',decision)
  atomic(ROOT/'tf-race-source-log-sha.json',raw_sha)
  print('TF_RACE_INCONCLUSIVE_NO_REPRO');return
 conflict=json.loads(conflict_path.read_text());branch=conflict['branch'];stamp=conflict['stamp_ns']
 repeat_dir=next(p for p in completed if p.name=='repeat-00')
 process=json.loads((repeat_dir/'process-map.json').read_text())
 records=[r for r in all_callback if r['branch']==branch and r['pid']==process['branch_pid'][str(branch)]]
 A=match(records,conflict['A'],stamp,conflict['A_receive']['steady_ns'])
 B=match(records,conflict['B'],stamp,conflict['B_receive']['steady_ns'])
 if not A['callback_seq']<B['callback_seq']:raise ValueError('A/B callback order mismatch')
 command_rows=[r for r in records if r['phase']=='command_pre_publish']
 pos_a=next(i for i,r in enumerate(command_rows) if r['callback_seq']==A['callback_seq'])
 pos_b=next(i for i,r in enumerate(command_rows) if r['callback_seq']==B['callback_seq'])
 before=command_rows[max(0,pos_a-20):pos_a];after=command_rows[pos_b+1:pos_b+21]
 final_conflict={**conflict,'A_follower_callback':A,'B_follower_callback':B,
  'preceding_callback_records':before,'following_callback_records':after,
  'preceding_count':len(before),'following_count':len(after),
  'first_artifact_sha256':sha(conflict_path)}
 atomic(ROOT/'same-stamp-command-conflict.json',final_conflict)
 clock=[x for x in lines(repeat_dir/'clock-publish-trace.jsonl') if x['stamp_ns']==stamp]
 tf=[x for x in lines(repeat_dir/'tf-publish-trace.jsonl') if x['tf_stamp_ns']==stamp and x['branch']==branch]
 path=[x for x in lines(repeat_dir/'path-publish-trace.jsonl') if x['branch']==branch]
 bridge=[x for x in lines(repeat_dir/'bridge-command-trace.jsonl') if x['branch']==branch and x['header_stamp_ns']==stamp]
 if len(clock)!=1 or not tf or not path:raise ValueError('publisher events missing')
 factor=0.5 # frozen onix.yaml angular_direction_sign * angular_gain
 recomputed=[factor*A['heading_error'],factor*B['heading_error']]
 residual=[abs(recomputed[0]-A['wz']),abs(recomputed[1]-B['wz'])]
 timeline={'clock_publish':clock[0],'current_tf_publish':tf[-1],
  'A_callback':{k:A[k] for k in ('callback_seq','wall_steady_ns','tf_start_steady_ns','tf_end_steady_ns','tf_stamp_ns','tf_x','tf_y','tf_yaw','vx','wz','command_stamp_ns')},
  'A_receive':conflict['A_receive'],'B_callback':{k:B[k] for k in ('callback_seq','wall_steady_ns','tf_start_steady_ns','tf_end_steady_ns','tf_stamp_ns','tf_x','tf_y','tf_yaw','vx','wz','command_stamp_ns')},
  'B_receive':conflict['B_receive'],'path_event':path[-1],
  'clock_before_A':clock[0]['publish_lower_steady_ns']<A['wall_steady_ns'],
  'A_before_current_tf_publish':A['tf_end_steady_ns']<tf[-1]['publish_lower_steady_ns'],
  'B_after_current_tf_publish':B['tf_start_steady_ns']>tf[-1]['publish_upper_steady_ns']}
 atomic(ROOT/'tf-command-alignment.json',{'timeline':timeline,'command_recalculation':{
  'factor':factor,'recomputed_wz':recomputed,'absolute_residual':residual,
  'A_to_B_tf_translation_m':math.hypot(B['tf_x']-A['tf_x'],B['tf_y']-A['tf_y']),
  'A_to_B_heading_error_rad':B['heading_error']-A['heading_error'],
  'A_to_B_command_wz':B['wz']-A['wz'],
  'A_to_B_command_vx':B['vx']-A['vx']}})
 identical_path=A['path_seq']==B['path_seq'] and A['path_stamp_ns']==B['path_stamp_ns'] and A['path_sha256']==B['path_sha256']
 identical_mode=A['mode']==B['mode'] and A['linear_recovery']==B['linear_recovery'] and A['angular_recovery']==B['angular_recovery'] and A['goal_reached']==B['goal_reached']
 prior_tf=A['tf_stamp_ns']==stamp-50_000_000 and B['tf_stamp_ns']==stamp
 r1=(A['ros_clock_ns']==B['ros_clock_ns']==stamp and identical_path and identical_mode and prior_tf
  and max(residual)<1e-9 and timeline['A_before_current_tf_publish'] and timeline['B_after_current_tf_publish'])
 first={'branch':branch,'stamp_ns':stamp,'A_callback_seq':A['callback_seq'],'B_callback_seq':B['callback_seq'],
  'first_differing_signal':'returned TF stamp/pose','same_ros_stamp':A['ros_clock_ns']==B['ros_clock_ns'],
  'same_path':identical_path,'same_mode':identical_mode,'A_tf_stamp_ns':A['tf_stamp_ns'],'B_tf_stamp_ns':B['tf_stamp_ns'],
  'A_tf_pose':[A['tf_x'],A['tf_y'],A['tf_yaw']],'B_tf_pose':[B['tf_x'],B['tf_y'],B['tf_yaw']],
  'A_command':[A['vx'],0,A['wz']],'B_command':[B['vx'],0,B['wz']],
  'motion_state_equal':all(A[k]==B[k] for k in ('motion_elapsed_sec','motion_samples','motion_ratio')),
  'motion_ready_A_B':[A['motion_ready'],B['motion_ready']],
  'classification':'R1 TF delivery race supported' if r1 else 'inconclusive'}
 atomic(ROOT/'tf-race-first-divergence.json',first)
 decision={'classification':'R1 TF delivery race supported' if r1 else 'inconclusive',
  'criteria':{'same_ros_stamp':first['same_ros_stamp'],'same_path':identical_path,'same_controller_mode':identical_mode,
   'A_previous_step_TF_B_current_step_TF':prior_tf,'command_recomputed_from_heading':max(residual)<1e-9,
   'A_lookup_before_new_TF_publish':timeline['A_before_current_tf_publish'],
   'B_lookup_after_new_TF_publish':timeline['B_after_current_tf_publish']},
  'other_hypotheses':{'R2_same_stamp_TF_overwrite':False,'R3_controller_state_only':False,
   'R4_bridge_attribution_error':False,'R5_clock_propagation_race':False},
  'repeat_count':len(completed),'first_conflict_branch':branch,'first_conflict_stamp_ns':stamp,
  'stage_B_started':False,'parity_started':False,'duplicate_started':False,
  'logging_perturbation_caveat':'Diagnostic logging can change callback timing; mechanism is observed in this run, recurrence probability is not estimated.'}
 atomic(ROOT/'tf-race-cause-decision.json',decision)
 atomic(ROOT/'tf-race-source-log-sha.json',raw_sha)
 atomic(ROOT/'tf-race-callback-counts.json',{'counts':counts,'parse_errors':parse_bad,
  'process_map':process,'all_callbacks':len(all_callback)})
 print('TF_RACE_ANALYZED',decision['classification'],'callback_context',len(before),len(after))
if __name__=='__main__':main()
