# Read-only verifier for completed actual batches, saves only new audit evidence.
from pathlib import Path
import json,gzip,hashlib,time,sys
import numpy as np
from eat_nav_r6_eval_queue import validate_completion,verify_initial_cohort
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');jid=sys.argv[1];reference=sys.argv[2] if len(sys.argv)>2 else None;job=r/'validation-evaluation'/jid;base=job/'sim-output'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ledger=read(r/'validation-evaluation/progress.json');record=ledger[jid];assert record['state']=='completed' and record['execution']=='simulated'
assignment=read(job/'assignment.json');results=validate_completion(read(base/'complete.json'),len(assignment['groups']));storage=read(base/'trace-storage.json');proof=read(job/'copied-output-sha256.json');assert proof['verified']
if reference:verify_initial_cohort(r/'validation-evaluation'/reference/'sim-output',base)
count=0;decisions=0
for folder in sorted(base.glob('group-*')):
 result=read(folder/'result.json');meta=storage[f'{folder.name}/trace.jsonl'];p=base/meta['compressed_path'];assert sha(p)==meta['compressed_sha256']
 with gzip.open(p,'rb') as f:raw=f.read()
 assert hashlib.sha256(raw).hexdigest()==meta['uncompressed_sha256']==proof['files_sha256'][f'{folder.name}/trace.jsonl']
 rows=[json.loads(line) for line in raw.splitlines() if line.strip()];assert len(rows)==round(result['observed_sec']/.05);count+=len(rows)
 if result['initial_valid']:
  with np.load(folder/'initial-state.npz',allow_pickle=False) as data:assert all(np.isfinite(data[k]).all() for k in data.files)
 if rows:
  first=rows[0]['logical_sim_time_ns']
  for i,row in enumerate(rows):assert row['control_step']==i and row['logical_sim_time_ns']==first+i*50_000_000 and row['post_timestamp_ns']==first+(i+1)*50_000_000
 for f in folder.glob('decision-*.json'):
  d=read(f);assert all(t<=d['decision_stamp_ns'] for t in d['dem_stamps_ns']);assert d['decision_stamp_ns']==rows[d['decision_step']]['logical_sim_time_ns'];decisions+=1
valid=[x for x in results if x['initial_valid']];n=len(valid);mean=lambda key:sum(x[key] for x in valid)/n if n else None
report=dict(schema='eatnav_r6_completed_batch_manual_audit_v1',job=jid,verified_unix=time.time(),protocol_trials=len(results),initial_valid_trials=n,tip_observed_trials=sum(x['tip_count']>0 for x in valid),near_fall_count=sum(x['near_fall_count'] for x in valid),linear_windows=sum(x['linear_windows'] for x in valid),turn_windows=sum(x['turn_windows'] for x in valid),observed_rows=count,observed_seconds=sum(x['observed_sec'] for x in results),causal_decision_records=decisions,mean_STOP_request_sec=mean('protective_stop_sec'),mean_ranking_STOP_upper_bound_sec=mean('ranking_STOP_upper_bound_sec'),mean_route_progress_m=mean('route_progress_m'),goal_reached_trials=sum(x['goal_reached'] for x in valid),residual_unavailable_seconds=sum(x['residual_unavailable_sec'] for x in valid),compressed_and_raw_sha_verified=True,copy_sha_verified=True,initial_arrays_finite=True,paired_initial_and_physics_verified=True if reference else None,reference_job=reference,complete_sha256=sha(base/'complete.json'),trace_storage_sha256=sha(base/'trace-storage.json'),physics_sha256=sha(base/'actual-physics-properties.npz'),verifier_sha256=sha(Path(__file__)),scope='single completed validation batch; means use initialvalidtrials, not plannedcount; no final safety/selection claim')
name='audit-'+jid+'.json'
for dest in [r/'validation-progress'/name,Path('docs/eat-nav/r6-2s-pose-20261030/validation-progress')/name]:
 assert not dest.exists(),'fresh audit output required';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
