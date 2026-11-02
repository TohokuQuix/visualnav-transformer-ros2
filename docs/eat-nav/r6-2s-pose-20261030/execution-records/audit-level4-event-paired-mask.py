from pathlib import Path
import json,hashlib,gzip,sys,time
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');w=Path('docs/eat-nav/r6-2s-pose-20261030');q=r/'validation-evaluation'
batch=sys.argv[1];assert batch in ('00','01','02')
jid=f'event_2s-k_rail_field-l4-batch{batch}';ref=f'old_reference-k_rail_field-l4-batch{batch}';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ledger=read(q/'progress.json')
source=q/ref/'sim-output';target=q/jid/'sim-output';hashes={};valid=[];invalid=[]
for job,base in [(ref,source),(jid,target)]:
 assert ledger[job]['state']=='completed' and ledger[job]['execution']=='simulated'
 a=read(r/'validation-progress'/('audit-'+job+'.json'));assert a['copy_sha_verified'] and a['compressed_and_raw_sha_verified'] and a['complete_sha256']==sha(base/'complete.json')
sa=read(source.parent/'assignment.json');ta=read(target.parent/'assignment.json');assert sa['groups']==ta['groups'] and sa['physical_batch_id']==ta['physical_batch_id']
for rel in ['actual-physics-properties.npz']+[f'group-{i:03d}/initial-state.npz' for i in range(30)]:
 hashes[rel]=sha(source/rel);assert hashes[rel]==sha(target/rel)
for i in range(30):
 old=source/f'group-{i:03d}';new=target/f'group-{i:03d}';og=read(old/'group.json');ng=read(new/'group.json');o=read(old/'result.json');n=read(new/'result.json')
 assert og['group_id']==ng['group_id']==o['group_id']==n['group_id']
 assert og['initial_valid']==ng['initial_valid']==o['initial_valid']==n['initial_valid']
 if n['initial_valid']:valid.append(n['group_id']);continue
 assert n['observed_sec']==0 and n['decision_count']==0 and not n['full_evaluation_valid'] and n['fall_outcome_unknown']
 assert not n['goal_reached'] and n['tip_count']==0 and n['linear_windows']==n['turn_windows']==0
 assert not list(new.glob('decision-*.json')) and ng['settling_failure']['termination_reasons']==['tilt_over']
 with gzip.open(new/'trace.jsonl.gz','rb') as stream:assert stream.read()==b''
 invalid.append(n['group_id'])
assert len(valid)+len(invalid)==30
report=dict(schema='eatnav_r6_level4_event_paired_initial_mask_audit_v1',verified_unix=time.time(),job=jid,reference_job=ref,protocol_trials=30,common_initial_valid_trials=len(valid),common_initial_invalid_trials=len(invalid),valid_group_ids=valid,invalid_group_ids=invalid,paired_assignment_identical=True,all31_initial_and_physics_files_byte_identical=True,all_initial_masks_identical=True,invalid_trace_and_decisions_zero=True,invalid_goal_false_tip_zero_fall_unknown_true=True,invalid_not_policy_success_or_failure=True,files_sha256=hashes,source_complete_sha256=sha(source/'complete.json'),target_complete_sha256=sha(target/'complete.json'),script_sha256=sha(Path(__file__)),scope='Same batch index and fixed cohort; distinct seed batches not compared as identical. Initial invalids excluded from success/failure and negative fall outcomes; no global improvement claim.')
for root in [r,w]:
 out=root/'validation-progress'/('paired-mask-'+jid+'.json');assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['job','common_initial_valid_trials','common_initial_invalid_trials','all31_initial_and_physics_files_byte_identical','all_initial_masks_identical']}))
