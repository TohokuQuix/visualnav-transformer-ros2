import pytest
from eat_nav_r6_eval_queue import evaluation_jobs, validate_completion

def test_equal_batches_and_conditions_keep_parent_seeds():
 groups=[dict(group_id=str(i),seed=i+1,field='boxes',level=0,role='validation') for i in range(30)]
 jobs=evaluation_jobs(groups,[dict(id='a',mode='pose'),dict(id='b',mode='event')])
 assert len(jobs)==6
 assert [g['seed'] for g in jobs[0]['groups']]==[g['seed'] for g in jobs[3]['groups']]
 assert all(len(j['groups'])==10 for j in jobs)

def test_exit_zero_without_complete_or_smoke_is_rejected():
 with pytest.raises(ValueError):validate_completion(None,10)
 with pytest.raises(ValueError):validate_completion(dict(state='completed',smoke=True,num_envs=10,results=[]),10)
 good=dict(state='completed',smoke=False,num_envs=1,results=[dict(full_evaluation_valid=True,initial_valid=True)])
 assert validate_completion(good,1)==good['results']
