from pathlib import Path
import gzip,hashlib,json,numpy as np,time
from eat_nav_r6_run import read,write,sha
from eat_nav_r6_eval_queue import validate_completion,verify_initial_cohort
from visualnav_transformer.eat_nav.r6_evaluation import aggregate_trials
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');q=r/'validation-evaluation';jid='pose_q3_2-k_rail_field-l0-batch00';base=q/jid/'sim-output'
ledger=read(q/'progress.json');assert ledger[jid]['state']=='completed' and ledger[jid]['execution']=='simulated'
complete=read(base/'complete.json');results=validate_completion(complete,30);assert results==ledger[jid]['results']
verify_initial_cohort(q/'old_reference-k_rail_field-l0-batch00/sim-output',base)
store=read(base/'trace-storage.json');total=inputs=decisions=0
for folder in sorted(base.glob('group-*')):
 trace=folder/'trace.jsonl.gz';meta=store[str(trace.relative_to(base)).removesuffix('.gz')];assert sha(trace)==meta['compressed_sha256']
 digest=hashlib.sha256();count=0
 with gzip.open(trace,'rb') as f:
  for line in f:
   digest.update(line);row=json.loads(line);assert row['control_step']==count
   assert row['logical_sim_time_ns']==1_000_000_000+count*50_000_000
   assert row['post_timestamp_ns']==row['logical_sim_time_ns']+50_000_000
   assert np.isfinite(row['root_state']).all() and np.isfinite(row['post_root_state']).all() and np.isfinite(row['follower_command']).all()
   count+=1
 assert digest.hexdigest()==meta['uncompressed_sha256']
 assert count==1200;total+=count
 for path in folder.glob('input-*.npz'):
  with np.load(path,allow_pickle=False) as inp:
   stamp=int(inp['decision_stamp_ns']);assert np.all(inp['dem_stamps_ns']<=stamp)
   assert all(np.isfinite(inp[k]).all() for k in inp.files)
  inputs+=1
 decisions+=len(list(folder.glob('decision-*.json')))
assert total==complete['recorded_rows']==36000
report=dict(scope='single completed K-Rail level0 pose condition; not final all-field comparison',job_id=jid,state='verified',trial_count=30,trace_rows=total,causal_finite_inputs=inputs,decision_count=decisions,compressed_and_raw_trace_hashes_verified=True,exact_paired_initial_and_physics_verified=True,completion_sha256=sha(base/'complete.json'),aggregate=aggregate_trials(results),unix=time.time())
output=r/'validation-progress/first-pose-batch-audit.json';write(output,report)
small=Path.cwd()/'docs/eat-nav/r6-2s-pose-20261030/validation-progress/first-pose-batch-audit.json';write(small,report)
print(json.dumps({k:report[k] for k in ('job_id','trial_count','trace_rows','causal_finite_inputs','decision_count')}));print(json.dumps(report['aggregate'],indent=2))
