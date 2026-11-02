from pathlib import Path
import json,hashlib,time,gzip
from eat_nav_r6_shadow import choice_equivalence
from visualnav_transformer.eat_nav.r6_model import load_checkpoint
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');q=r/'validation-evaluation';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checkpoint=Path('checkpoints/eat_nav/r6_2s_pose_20261030/model_seed123.pt');_,payload=load_checkpoint(checkpoint);ledger=read(q/'progress.json')
for jid in ('pose_q0_0-k_rail_field-l3-batch00',):
 directory=q/jid;assignment=read(directory/'assignment.json');proof=read(directory/'equivalence-audit.json');rec=ledger[jid];assert rec['state']=='completed' and rec['execution']=='verified_policy_equivalent_reuse'
 base=Path(proof['source']);assert str(base)==rec['source'];source_assignment=read(base.parent/'assignment.json');assert assignment['groups']==source_assignment['groups'] and assignment['physical_batch_id']==source_assignment['physical_batch_id']
 assert sha(base/'manifest.json')==proof['source_manifest_sha256'];assert sha(checkpoint)==read(base/'manifest.json')['model_sha256']==assignment['condition']['profile']['model_sha256']
 sourceaudit=r/'validation-progress'/('audit-'+base.parent.name+'.json');audit=read(sourceaudit);assert audit['copy_sha_verified'] and audit['compressed_and_raw_sha_verified'] and audit['paired_initial_and_physics_verified'] and audit['complete_sha256']==sha(base/'complete.json')
 assert len(assignment['groups'])==len(proof['checks'])==len(rec['results'])==30
 counts=[];invalid=[]
 for i in range(30):
  folder=base/f'group-{i:03d}';group=read(folder/'group.json');result=read(folder/'result.json');decisions=[read(p) for p in sorted(folder.glob('decision-*.json'))]
  assert result['initial_valid']==group['initial_valid']
  if group['initial_valid']:
   assert decisions and all('policy_features' in d for d in decisions)
   trimmed=[{key:d[key] for key in ('decision_step','selected_index','policy_features','footprint')} for d in decisions]
   check=choice_equivalence(trimmed,[None]*len(trimmed),assignment['condition']['profile'],assignment['condition']['mode'],payload['support']['active'])
   assert check['equivalent'] and check==proof['checks'][i];counts.append(check['decision_count'])
  else:
   expected=dict(equivalent=True,reason='same initialization-invalid environment; no policy action or outcome credited')
   assert proof['checks'][i]==expected and not decisions and result['observed_sec']==0 and not result['full_evaluation_valid'] and result['fall_outcome_unknown']
   with gzip.open(folder/'trace.jsonl.gz','rb') as f:assert f.read()==b''
   counts.append(0);invalid.append(group['group_id'])
  expected_result=dict(result,mode=assignment['condition']['mode'],execution='verified_policy_equivalent_reuse',timing_origin='source actual full inference; not independently timed',source=str(folder))
  assert rec['results'][i]==expected_result
 assert len(invalid)==1 and sum(n>0 for n in counts)==29 and sum(counts)==audit['causal_decision_records']
 report=dict(schema='eatnav_r6_recomputed_policy_reuse_with_invalid_mask_audit_v1',verified_unix=time.time(),job=jid,source_job=base.parent.name,protocol_environments=30,initial_valid_environments=29,initial_invalid_environments=1,invalid_group_ids=invalid,decision_counts=counts,decisions=sum(counts),all_valid_selected_indices_identical=True,all_saved_results_identical=True,initial_invalid_has_zero_trace_and_decisions=True,initial_invalid_not_credited_as_success=True,paired_group_assignment_identical=True,source_manifest_sha256=sha(base/'manifest.json'),source_audit_sha256=sha(sourceaudit),equivalence_proof_sha256=sha(directory/'equivalence-audit.json'),assignment_sha256=sha(directory/'assignment.json'),model_sha256=sha(checkpoint),script_sha256=sha(Path(__file__)),independent_simulation=False,independent_timing=False,scope='Whole30environment fixed deterministic stateless model/follower cohort;29active trajectories replayed through all actually observed decisions,1initial-invalid member explicitly excluded. Not new independent physical replications or newly measured timing.')
 for dest in [r/'validation-progress'/('masked-reuse-audit-'+jid+'.json'),Path('docs/eat-nav/r6-2s-pose-20261030/validation-progress')/('masked-reuse-audit-'+jid+'.json')]:
  assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(job=jid,valid=29,invalid=1,decisions=sum(counts),all_valid_choices_match=True,invalid_not_success=True)),flush=True)
