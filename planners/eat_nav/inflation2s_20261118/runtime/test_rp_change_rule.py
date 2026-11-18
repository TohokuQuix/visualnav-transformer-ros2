import numpy as np
from scipy.spatial.transform import Rotation
from rp_change_rule import score_components,C

def rot(r,p,y):return Rotation.from_euler('xyz',[r,p,y],degrees=True).as_matrix()
def test_ten_degrees_cost_fifty_cm():
 x=score_components(np.array([.3,.25]),np.array([rot(20,0,0),rot(5,0,0)]),np.eye(3),np.zeros(2))
 np.testing.assert_allclose(x['score_m'],[.3-2*C,.25-.5*C],atol=1e-12)
def test_roll_pitch_norm_and_no_yaw_penalty():
 x=score_components(np.zeros(2),np.array([rot(6,8,120),rot(0,0,90)]),np.eye(3),np.zeros(2))
 np.testing.assert_allclose(x['rp_change_deg'],[10,0],atol=1e-10)
def test_wrap_and_world_composition():
 current=rot(179,0,20);future=rot(-179,0,20)
 x=score_components(np.zeros(1),np.array([current.T@future]),current,np.zeros(1))
 np.testing.assert_allclose(x['rp_delta_deg'],[[2,0]],atol=1e-10)
def test_relative_yaw_from_tilted_pose_is_world_rp_unchanged():
 current=rot(10,20,30);future=rot(10,20,90)
 x=score_components(np.zeros(1),np.array([current.T@future]),current,np.zeros(1))
 np.testing.assert_allclose(x['rp_change_deg'],[0],atol=1e-10)
def test_nonfinite_rejected():
 import pytest
 with pytest.raises(ValueError):score_components(np.array([float('nan')]),np.array([np.eye(3)]),np.eye(3),np.zeros(1))

def test_protocol_guard_runs_after_rule_import():
 import ast,json,types
 from pathlib import Path
 path=Path(__file__).with_name('navigation_sim.py');tree=ast.parse(path.read_text());nodes=[]
 for n in tree.body:
  if isinstance(n,ast.ImportFrom) and n.module=='rp_change_rule':nodes.append(n)
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROTOCOL' for t in n.targets):nodes.append(n)
  if isinstance(n,ast.Assert) and 'PROTOCOL' in ast.unparse(n):nodes.append(n)
 scope={'args':types.SimpleNamespace(protocol=path.parents[1]/'protocol.json'),'json':json}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)

def decision_fixture():
 from nav_contract import bank_paths,course_mask
 root=np.zeros(13);root[:3]=[2,1,.3];root[3]=1
 p=np.tile([.1,0,0],(14,1));rr=np.tile(np.eye(3),(14,1,1));rr[13]=rot(180,0,0)
 goal=np.array([4.3,.5]);g=np.linalg.norm(root[:2]-goal)-np.linalg.norm(root[:2]+p[:,:2]-goal,axis=1)
 d=dict(root=root.tolist(),state=np.zeros(27).tolist(),step=0,t0_ns=2000000000,dem_stamp_ns=2000000000,dem_received_ns=2000000000,predicted_positions=p.tolist(),predicted_rotations=rr.tolist())
 d.update(score_components(g,rr,np.eye(3),np.zeros(14)))
 mask=course_mask(bank_paths()+root[:2],p+root[:3]);tilt=np.ones(14,bool);tilt[13]=False;allow=tilt&mask;ids=np.flatnonzero(allow)
 d.update(original_tilt_allowed=tilt.tolist(),course_allowed=mask.tolist(),allowed=allow.tolist(),candidate_id=int(ids[np.argmax(np.array(d['score_m'])[ids])]))
 return d,root,goal

def test_audit_wrap_exact_180_matches_approved_interval():
 from nav_contract import audit_decision
 d,root,goal=decision_fixture()
 assert audit_decision(d,root,np.zeros(27),np.zeros(3),goal)

def test_unavailable_input_stop_has_no_computed_rp_score():
 from nav_contract import audit_decision
 d,root,goal=decision_fixture()
 for k in ['score_m','goal_progress_m','rp_delta_deg','rp_change_deg','rp_penalty_m','current_world_rpy_deg','predicted_world_rpy_deg']:d.pop(k)
 d.update(input_failure=True,candidate_id=None,original_tilt_allowed=[False]*14,allowed=[False]*14,all_filtered=True,stop_reason='input_unavailable')
 assert audit_decision(d,root,np.zeros(27),np.zeros(3),goal)
