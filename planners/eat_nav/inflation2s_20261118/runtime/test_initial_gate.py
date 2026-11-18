import unittest,numpy as np
from initial_gate import verify_initial,verify_wall_samples
class Gate(unittest.TestCase):
 def test_initial(self):
  a={k:np.zeros((10,3)) for k in ['root','state','joint_position','joint_velocity','effort']};e={k:np.stack([v,v]) for k,v in a.items()};self.assertEqual(verify_initial(e,a)['state'],'passed');a['root'][0,0]=1e-8
  with self.assertRaises(ValueError):verify_initial(e,a)
 def test_both_sides(self):
  boxes=[dict(name='left',xy_bounds=[-.15,0,0,3.27]),dict(name='inner_vertical',xy_bounds=[1.12,1.27,1.12,3.27])];r=np.tile([[[-.1,2,.5],[1.2,2,.5]]],(10,1,1));self.assertEqual(verify_wall_samples(r,boxes)['state'],'passed');r[3,1,2]=0
  with self.assertRaises(ValueError):verify_wall_samples(r,boxes)
 def test_nonfinite(self):
  with self.assertRaises(ValueError):verify_wall_samples(np.full((10,2,3),np.nan),[])
