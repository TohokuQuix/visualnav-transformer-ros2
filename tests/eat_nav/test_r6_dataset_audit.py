import numpy as np
from eat_nav_r6_dataset_audit import branch_records, audit_arrays

def sample():
 t=np.zeros((2,14,20,9),np.float32);t[...,4]=t[...,6]=t[...,8]=1
 t[:,:, :,0]=np.arange(1,21)[None,None,:]*.01
 a=dict(dem=np.zeros((2,4,2,20,20),np.float32),actor_state=np.zeros((2,22),np.float32),body_state=np.zeros((2,9),np.float32),paths=np.zeros((2,14,8,2),np.float32),directions=np.ones((2,14,1),np.float32),candidate_mask=np.ones((2,14),bool),event_targets=np.zeros((2,14,4),np.float32),event_mask=np.ones((2,14,4),bool),trajectory=t,trajectory_mask=np.ones((2,14,20),bool))
 a['dem'][:,:,1]=1
 return a,[dict(role='train',seed=1,field='boxes',level=0,sample_id='a',episode_group='a'),dict(role='validation',seed=2,field='boxes',level=0,sample_id='b',episode_group='b')]

def test_records_only_validation_and_censor_not_success():
 a,i=sample();a['event_targets'][1,0,2]=1;a['trajectory_mask'][1,1,-1]=False
 records=branch_records(a,i)
 assert len(records)==14 and all(r['split']=='validation' for r in records)
 assert records[0]['failed'] and not records[1]['complete']
 assert records[2]['pitch_positive_duration_sec']==0
 assert records[2]['outcome_definition']=='complete_two_second_event_free; not a two_metre_climb_success'

def test_audit_checks_group_leakage_and_masks():
 a,i=sample();r=audit_arrays(a,i);assert r['passed']
 i[1]['episode_group']='a'
 assert not audit_arrays(a,i)['passed']


def test_observed_failure_before_terminal_is_not_dropped_from_threshold_statistics():
 a,i=sample();a['paths'][...,0]=np.arange(8)/10;a['trajectory_mask'][1,0,5:]=False;a['event_mask'][1,0]=[True,False,False,False];a['event_targets'][1,0,0]=1
 r=branch_records(a,i)[0]
 assert r['complete'] and r['failed'] and r['observed_failure_before_censor']
 assert not r['full_pose_horizon']


def test_strict_bank_check_does_not_replace_future_point_denominator():
 from visualnav_transformer.eat_nav.bidirectional_bank import bank
 a,i=sample();b=bank();a['paths'][:]=np.stack([p.waypoints for p in b]);a['directions'][:]=np.array([p.direction for p in b])[:,None]
 report=audit_arrays(a,i,strict_bank=True)
 assert report['passed'] and report['valid_future_points']==2*14*20
