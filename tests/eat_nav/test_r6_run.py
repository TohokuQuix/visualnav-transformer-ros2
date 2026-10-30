import json
import pytest
from eat_nav_r6_run import make_plan, pending_jobs, validate_benchmark


def test_fixed_counts_role_pure_and_disjoint_seeds():
    plan=make_plan(excluded_seeds={1,2},num_envs=10)
    assert plan['base_group_count']==700
    assert plan['base_branch_count']==9800
    assert len(plan['collection'])==70+10
    for job in plan['collection']:
        assert len({g['role'] for g in job['groups']})==1
    cohorts=[{g['seed'] for j in plan['collection'] for g in j['groups']},
             {g['seed'] for g in plan['validation_rollouts']},
             {g['seed'] for g in plan['test_rollouts']}]
    assert not cohorts[0]&cohorts[1] and not cohorts[1]&cohorts[2] and not cohorts[0]&cohorts[2]
    assert len(plan['validation_rollouts'])==35*30
    assert len(plan['test_rollouts'])==35*30
    assert plan['validation_condition_runs']==35*30*13
    assert plan['test_condition_runs']==35*30*6


def test_resume_skips_completed_and_preserves_failed():
    jobs=[dict(job_id='a'),dict(job_id='b'),dict(job_id='c')]
    assert [j['job_id'] for j in pending_jobs(jobs,{'a':'completed','b':'failed'})]==['c']


def test_missing_or_failed_benchmark_prevents_execution():
    with pytest.raises(ValueError):validate_benchmark(dict(passed=False,selected_num_envs=10),dict(production_collection_approved=True))
    with pytest.raises(ValueError):validate_benchmark(dict(passed=True,selected_num_envs=10),{})
    assert validate_benchmark(dict(passed=True,selected_num_envs=5),dict(production_collection_approved=True))==5


def test_reduced_chunks_keep_original_tile_identities_and_skip_preserved_groups():
    from eat_nav_r6_run import chunk_jobs
    plan=make_plan(num_envs=10)
    blocked={g['group_id'] for g in plan['collection'][0]['groups'][:3]}
    chunks=list(chunk_jobs(plan,2,blocked))
    seen=[g for job in chunks for g in job['groups']]
    assert len(seen)==797
    assert not blocked&{g['group_id'] for g in seen}
    original={g['group_id']:g for j in plan['collection'] for g in j['groups']}
    for g in seen:
        assert g['derived_seed_ids']==original[g['group_id']]['derived_seed_ids']
        assert g['tile_index']==original[g['group_id']]['tile_index']


def test_pack_all_failed_groups_preserves_missing_audit_with_reused_data(tmp_path):
    import numpy as np
    from eat_nav_r6_run import pack_collection,write
    from eat_nav_r6_dataset import pack_group
    old=tmp_path/'old';old.mkdir()
    initial=dict(dem=np.zeros((4,2,20,20)),actor_state=np.zeros(22),body_state=np.zeros(9))
    arrays,_=pack_group(initial,[dict(slot=0,path=np.zeros((8,2)),direction=1,rows=[])])
    np.savez_compressed(old/'samples.npz',**{k:v[None] for k,v in arrays.items()})
    write(old/'sample-index.json',[dict(field='boxes',seed=99,role='train',sample_id='old')])
    plan=dict(collection=[dict(groups=[dict(group_id='new',field='boxes',seed=100,role='train')])])
    output=tmp_path/'out'
    report=pack_collection(plan,tmp_path/'empty',output,old)
    assert report['samples']==1 and report['unavailable_fresh_groups']==1
    assert report['known_event_targets']==0
    assert json.loads((output/'collection-pack-audit.json').read_text())[0]['unknown_candidates']==14


def test_unavailable_anchor_is_not_a_failure_of_other_collected_groups(tmp_path):
    import eat_nav_r6_run as m
    import json
    expected={'group_id':'one','seed':123,'role':'train'}
    (tmp_path/'group.json').write_text(json.dumps(dict(expected,anchor_available=False,anchor_failure_reason='termination_during_settling')))
    for slot in range(14):
        p=tmp_path/f'candidate-{slot:02d}';p.mkdir();(p/'trace.jsonl').write_text('')
        (p/'complete.json').write_text(json.dumps(dict(candidate=slot,control_steps=0,smoke=False,stop_reason='termination_during_settling')))
    audit=m.validate_collected_group(tmp_path,expected)
    assert audit['structurally_complete'] and audit['available_candidates']==0
    assert audit['unavailable_candidates']==14
