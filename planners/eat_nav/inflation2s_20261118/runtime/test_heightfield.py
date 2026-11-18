import unittest
import numpy as np
from heightfield import triangle_height
class HeightfieldTests(unittest.TestCase):
 def test_non_scene1_dimensions_and_last_row(self):
  x=np.linspace(-1,5.92,347);y=np.linspace(-1,3.5,226);xx,yy=np.meshgrid(x,y);m={'x':x,'y':y,'heights':2*xx+3*yy}
  for a,b in [(-.99,-.99),(.56,1.66),(5.91,3.49),(5.92,3.5)]:self.assertAlmostEqual(triangle_height(m,[a,b]),2*a+3*b,places=10)
 def test_triangular_not_bilinear(self):
  m={'x':np.array([0.,1.]),'y':np.array([0.,1.]),'heights':np.array([[0.,0.],[0.,1.]])};self.assertAlmostEqual(triangle_height(m,[.7,.3]),.3);self.assertAlmostEqual(triangle_height(m,[.3,.7]),.3)
 def test_outside_rejected(self):
  m={'x':np.array([0.,1.]),'y':np.array([0.,1.]),'heights':np.zeros((2,2))}
  with self.assertRaises(ValueError):triangle_height(m,[-.01,.5])
