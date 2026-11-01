from pathlib import Path
import json, hashlib, time
from eat_nav_r6_shadow import choice_equivalence
from visualnav_transformer.eat_nav.r6_model import load_checkpoint
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030')
jid='pose_q2_0-k_rail_field-l2-batch00'; job=r/'validation-evaluation'/jid
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(job/'assignment.json'); proof=read(job/'equivalence-audit.json'); source=Path(proof['source']); manifest=read(source/'manifest.json')
assert sha(source/'manifest.json')==proof['source_manifest_sha256']
assert a['groups']==read(source.parent/'assignment.json')['groups']
checkpoint=Path('checkpoints/eat_nav/r6_2s_pose_20261030/model_seed123.pt')
assert sha(checkpoint)==manifest['model_sha256']==a['condition']['profile']['model_sha256']
_,payload=load_checkpoint(checkpoint)
ledger=read(r/'validation-evaluation/progress.json'); record=ledger[jid]
assert record['execution']=='verified_policy_equivalent_reuse' and record['state']=='completed'
assert len(record['results'])==len(a['groups'])==len(proof['checks'])==30
sourceaudit=r/'validation-progress'/('audit-'+source.parent.name+'.json')
s=read(sourceaudit); assert s['copy_sha_verified'] and s['paired_initial_and_physics_verified']
counts=[]
for i in range(30):
 folder=source/f'group-{i:03d}'; assert read(folder/'group.json')['initial_valid']
 ds=[read(p) for p in sorted(folder.glob('decision-*.json'))]
 assert all('policy_features' in d for d in ds)
 ds=[{k:d[k] for k in ('decision_step','selected_index','policy_features','footprint')} for d in ds]
 audit=choice_equivalence(ds,[None]*len(ds),a['condition']['profile'],a['condition']['mode'],payload['support']['active'])
 assert audit==proof['checks'][i] and audit['equivalent']
 result=read(folder/'result.json')
 assert record['results'][i]==dict(result,mode=a['condition']['mode'],execution='verified_policy_equivalent_reuse',timing_origin='source actual full inference; not independently timed',source=str(folder))
 counts.append(audit['decision_count'])
report=dict(schema='eatnav_r6_recomputed_policy_reuse_audit_v1',verified_unix=time.time(),job=jid,source_job=source.parent.name,source_manifest_sha256=sha(source/'manifest.json'),source_audit_sha256=sha(sourceaudit),assignment_sha256=sha(job/'assignment.json'),equivalence_proof_sha256=sha(job/'equivalence-audit.json'),model_sha256=sha(checkpoint),groups=30,decision_counts=counts,decisions=sum(counts),all_selected_indices_identical=True,all_saved_results_identical=True,paired_group_assignment_identical=True,independent_simulation=False,independent_timing=False,scope='Exact deterministic stateless model/follower command-policy reuse over the full observed source trajectories; not 30 newly simulated replications or new independent accidents.',verifier_sha256=sha(Path(__file__)))
for dest in [r/'validation-progress'/('reuse-audit-'+jid+'.json'),Path('docs/eat-nav/r6-2s-pose-20261030/validation-progress')/('reuse-audit-'+jid+'.json')]:
 assert not dest.exists(); dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
