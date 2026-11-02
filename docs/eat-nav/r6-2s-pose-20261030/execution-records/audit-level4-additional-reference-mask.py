from pathlib import Path
import json,gzip,hashlib,time,sys
import numpy as np
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030')
jid=sys.argv[1]
assert jid in ('old_reference-k_rail_field-l4-batch01','old_reference-k_rail_field-l4-batch02')
base=r/'validation-evaluation'/jid/'sim-output'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rec=read(r/'validation-evaluation/progress.json')[jid]
assert rec['state']=='completed' and rec['execution']=='simulated'
audit=read(r/'validation-progress'/('audit-'+jid+'.json'))
assert audit['copy_sha_verified'] and audit['compressed_and_raw_sha_verified']
assert audit['complete_sha256']==sha(base/'complete.json')
invalid=[];valid=[]
for i in range(30):
 folder=base/f'group-{i:03d}';g=read(folder/'group.json');result=read(folder/'result.json')
 assert g['initial_valid']==result['initial_valid']
 with np.load(folder/'initial-state.npz',allow_pickle=False) as a:
  assert all(np.isfinite(a[k]).all() for k in a.files)
 if result['initial_valid']:
  valid.append(g['group_id']);continue
 assert not list(folder.glob('decision-*.json'))
 with gzip.open(folder/'trace.jsonl.gz','rb') as stream:assert stream.read()==b''
 assert result['observed_sec']==0 and result['decision_count']==0
 assert not result['full_evaluation_valid'] and result['fall_outcome_unknown']
 assert not result['goal_reached'] and result['tip_count']==0
 assert result['linear_windows']==result['turn_windows']==0
 assert g['settling_failure']['termination_reasons']==['tilt_over']
 invalid.append(g['group_id'])
assert len(valid)==audit['initial_valid_trials'] and len(valid)+len(invalid)==30 and set(valid).isdisjoint(invalid)
report=dict(schema='eatnav_r6_initial_invalid_mask_audit_v1',verified_unix=time.time(),job=jid,protocol_trials=30,initial_valid_trials=len(valid),initial_invalid_trials=len(invalid),valid_group_ids=valid,invalid_group_ids=invalid,all_initial_arrays_finite=True,all_invalid_reasons='pre-policy settling tilt_over',invalid_trace_and_decisions_zero=True,invalid_goal_false_tip_zero_fall_unknown_true=True,invalid_not_policy_success_or_failure=True,source_complete_sha256=sha(base/'complete.json'),source_manual_audit_sha256=sha(r/'validation-progress'/('audit-'+jid+'.json')),script_sha256=sha(Path(__file__)),scope='Initial mask evidence for one completed preregistered K-Rail L4 reference batch only. No30valid claim and no cross-condition initial equivalence claimed yet; subsequent policies must match this mask.')
name='initial-invalid-'+jid+'-audit.json'
for root in [r,Path('docs/eat-nav/r6-2s-pose-20261030')]:
 out=root/'validation-progress'/name;assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['job','protocol_trials','initial_valid_trials','initial_invalid_trials','invalid_trace_and_decisions_zero','invalid_goal_false_tip_zero_fall_unknown_true']}))
