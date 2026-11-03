import numpy as np
import pytest
from visualnav_transformer.eat_nav.single_pose2s_selector import select_candidate,validate_exploratory_authorization,make_selector_manifest

def test_only_goal_progress_and_tilt_with_deterministic_ties():
 p=np.zeros((14,3));p[:,0]=-.2;p[2,0]=1;p[3,0]=.5
 R=np.tile(np.eye(3),(14,1,1));a=np.deg2rad(61);R[2]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
 assert select_candidate(p,R,[0,0,0],np.eye(3),[3,0],'A')['candidate_id']==2
 assert select_candidate(p,R,[0,0,0],np.eye(3),[3,0],'B')['candidate_id']==3
 p[:]=[-.2,0,0];assert select_candidate(p,R,[0,0,0],np.eye(3),[3,0],'A')['candidate_id']==0
 R[:]=R[2];assert select_candidate(p,R,[0,0,0],np.eye(3),[3,0],'B')['candidate_id'] is None

def test_invalid_predictions_are_not_silently_filtered():
 p=np.zeros((14,3));p[0,0]=np.nan
 with pytest.raises(ValueError):select_candidate(p,np.tile(np.eye(3),(14,1,1)),[0,0,0],np.eye(3),[3,0],'A')

def test_explicit_user_transition_does_not_claim_strict_accuracy_pass():
 with pytest.raises(ValueError):validate_exploratory_authorization({},'abc')
 a=dict(user_explicitly_authorized=True,checkpoint_sha256='abc',scope='exploratory_selector_and_offline_bag',strict_accuracy_achieved=False,defer_github_push=True)
 validate_exploratory_authorization(a,'abc')
 with pytest.raises(ValueError):validate_exploratory_authorization(a,'other')

def test_matched_independent_700_run_manifest():
 m=make_selector_manifest();assert len(m['batches'])==70
 for i in range(0,70,2):
  A,B=m['batches'][i:i+2];assert A['condition']=='A' and B['condition']=='B'
  for a,b in zip(A['parents'],B['parents']):
   assert a['seed']==b['seed'] and a['initial_seed']==b['initial_seed'] and a['pair_id']==b['pair_id']
 assert len({p['pair_id'] for b in m['batches'] for p in b['parents']})==350

def test_decision_input_uses_only_available_prefix_and_actual_lio():
 from visualnav_transformer.eat_nav.single_pose2s_selector import decision_input
 raw=dict(stamps_ns=np.array([0,50000000,100000000]),root=np.array([[0,0,.35,1,0,0,0,0,0,0,0,0,0]]*3),state=np.zeros((3,27)),state_mask=np.ones((3,27)),state_age=np.zeros((3,27)),dem=np.ones((1,3,40,40)),dem_stamp_ns=np.array([0]),dem_root=np.array([[0,0,.35,1,0,0,0,0,0,0,0,0,0]]))
 raw['state'][:,2]=-1;raw['state'][:,25]=1
 odom=dict(stamps_ns=np.array([-100000000,0,50000000,200000000]),received_ns=np.array([-100000000,0,50000000,200000000]),poses=np.array([[x,0,0,1,0,0,0] for x in [0,.01,.02,999]]))
 a=decision_input(raw,odom)
 assert a['paths'].shape==(1,14,8,2)
 assert abs(a['state'][0,18]-.2)<1e-6
 odom['received_ns'][:]=1000000000
 with pytest.raises(ValueError):decision_input(raw,odom)

def test_b_only_preserves_future_matched_a_seeds():
 full=make_selector_manifest();b=make_selector_manifest(conditions=('B',))
 assert len(b['batches'])==35 and b['planned_runs']==350
 assert b['batches']==[x for x in full['batches'] if x['condition']=='B']
 with pytest.raises(ValueError):make_selector_manifest(conditions=('C',))

def test_resume_rejects_changed_evaluator_or_runtime():
 from visualnav_transformer.eat_nav.single_pose2s_selector import validate_run_identity
 base=dict(checkpoint_sha256='model1',worker_sha256='worker1',selector_behavior_sha256='select1')
 validate_run_identity(base,dict(base))
 for key in base:
  changed=dict(base);changed[key]='different'
  with pytest.raises(ValueError):validate_run_identity(base,changed)
