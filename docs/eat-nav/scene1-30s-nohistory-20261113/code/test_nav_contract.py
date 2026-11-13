import unittest,numpy as np,tempfile
from pathlib import Path
from gap_walls import create_gap_wall_variant
from nav_contract import inside_course,course_mask,outcome,choose_best,needs_walls,diagnose_motion,initialization_ok,audit_decision,bank_paths
class ContractTests(unittest.TestCase):
 def test_course(self):
  self.assertTrue(inside_course([[0,0],[4.9,2.48]]).all());self.assertFalse(inside_course([[-.01,1],[2,3]]).any())
 def test_mask(self):
  paths=np.array([[[.6,1.7],[.6,1]],[[.6,1.7],[-.2,1.7]]]);pred=np.array([[.6,1,0],[.5,1.5,0]])
  self.assertEqual(course_mask(paths,pred).tolist(),[True,False])
 def test_goal_and_priority(self):
  self.assertEqual(outcome(1,[4.3,.5,.5],0,False),'goal_reached')
  self.assertIsNone(outcome(1,[4.3,.5,.5],0,True))
  self.assertEqual(outcome(300,[1,1,.5],0,False),'timeout')
  self.assertEqual(outcome(1,[4.3,.5,.5],86,False),'tip_over')
  self.assertIsNone(outcome(1,[1,1,-2],0,True))
 def test_best_and_walls(self):
  a=[dict(trial_id='a',minimum_course_valid_goal_distance_m=2),dict(trial_id='b',minimum_course_valid_goal_distance_m=1)]
  self.assertEqual(choose_best(a)['trial_id'],'b');self.assertFalse(needs_walls([dict(bypass_observed=True)]*9));self.assertTrue(needs_walls([dict(bypass_observed=True)]*10));self.assertFalse(needs_walls([dict(bypass_observed=False)]*10))
 def test_diagnostics_not_goal_termination(self):
  t=np.arange(41)*.05;root=np.zeros((41,13));root[:,3]=1;cmd=np.tile([.3,0,0],(40,1))
  d=diagnose_motion(t,root,cmd);self.assertEqual(d[0]['kind'],'linear_motion_blocked');self.assertEqual(d[0]['duration_s'],2)
  root[:,0]=np.linspace(0,1,41);self.assertEqual(diagnose_motion(t,root,cmd),[])

class GeometryTests(unittest.TestCase):
 def test_gap_walls_preserve_original_and_stay_outside_course(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);v=np.zeros((225*347,3));v[:,0]=np.tile(np.arange(347)*.02-1,225);v[:,1]=np.repeat(np.arange(225)*.02-1,347);f=np.array([[0,1,348],[0,348,347]])
   np.savez(root/'original.npz',vertices=v,faces=f);report=create_gap_wall_variant(root/'original.npz',root/'closed.npz');z=np.load(root/'closed.npz')
   self.assertTrue(np.array_equal(z['vertices'][:len(v)],v));self.assertTrue(np.array_equal(z['faces'][:len(f)],f));self.assertEqual(len(report['added_walls']),2)
   extra=z['vertices'][len(v):];self.assertTrue(((extra[:,0]<=0)|(extra[:,0]>=4.9)).all());z.close()

class ReviewTests(unittest.TestCase):
 def test_initialization_tolerances(self):
  self.assertTrue(initialization_ok(.03,np.deg2rad(3),False));self.assertFalse(initialization_ok(.051,0,False));self.assertFalse(initialization_ok(0,np.deg2rad(6),False));self.assertFalse(initialization_ok(0,0,True))
 def test_recomputed_decision_rejects_false_boundary_and_wrong_root(self):
  root=np.zeros(13);root[:3]=[.6,1.7,.2];root[3]=1;positions=np.tile([.1,0,0],(14,1));rotations=np.tile(np.eye(3),(14,1,1));paths=bank_paths()+root[:2];mask=course_mask(paths,positions+root[:3]);allowed=np.flatnonzero(mask);score=np.linalg.norm(root[:2]-[4.3,.5])-np.linalg.norm(positions[:,:2]+root[:2]-[4.3,.5],axis=1);cid=int(allowed[np.argmax(score[allowed])])
  d=dict(root=root.tolist(),state=np.zeros(27).tolist(),step=0,t0_ns=2000000000,dem_stamp_ns=2000000000,dem_received_ns=2000000000,predicted_positions=positions.tolist(),predicted_rotations=rotations.tolist(),original_tilt_allowed=[True]*14,course_allowed=mask.tolist(),allowed=mask.tolist(),candidate_id=cid,score_m=score.tolist())
  self.assertTrue(audit_decision(d,root,np.zeros(27),np.zeros(3),[4.3,.5]))
  bad=dict(d,course_allowed=[True]*14)
  with self.assertRaises(ValueError):audit_decision(bad,root,np.zeros(27),np.zeros(3),[4.3,.5])
  wrong=root.copy();wrong[0]+=.1
  with self.assertRaises(ValueError):audit_decision(d,wrong,np.zeros(27),np.zeros(3),[4.3,.5])

class SettledPlacementTests(unittest.TestCase):
 def test_preserves_slope_attitude_but_uses_confirmed_xy_yaw(self):
  from nav_contract import settled_spawn_root
  from scipy.spatial.transform import Rotation
  root=np.arange(13,dtype=float);root[:3]=[.56,1.66,.38];q=Rotation.from_euler('xyz',[.4,.2,-1.65]).as_quat();root[3:7]=q[[3,0,1,2]]
  placed=settled_spawn_root(root,[.6,1.7],-np.pi/2,.02)
  np.testing.assert_allclose(placed[:3],[.6,1.7,.4]);np.testing.assert_allclose(Rotation.from_quat(placed[[4,5,6,3]]).as_euler('xyz'),[.4,.2,-np.pi/2]);self.assertTrue((placed[7:]==0).all());self.assertFalse(np.array_equal(root,placed))
