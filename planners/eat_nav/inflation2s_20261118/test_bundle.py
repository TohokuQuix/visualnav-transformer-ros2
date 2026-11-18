from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from feasibility import DEMGrid

def grid(wall=False,unknown=False):
 d=np.zeros((3,40,40));d[1:]=1
 if wall:d[0,:,25]=.5
 if unknown:d[2]=0
 return DEMGrid(d,np.array([0,0,0,1,0,0,0]))

def test_costmap_is_known_wall_distance_with_twenty_cm_margin():
 from inflation_rule import make_costmap
 c=make_costmap(grid(True));assert c[20,25]==1;assert abs(c[20,24]-.5)<1e-12;assert c[20,23]==0

def test_unknown_only_and_no_wall_have_zero_cost():
 from inflation_rule import make_costmap
 assert not make_costmap(grid()).any();assert not make_costmap(grid(True,True)).any()

def test_all14_costs_are_penalties_not_turn_bonuses():
 from rp_change_rule import score_components
 g=np.linspace(-.2,.4,14);c=np.linspace(0,1,14);r=score_components(g,np.tile(np.eye(3),(14,1,1)),np.eye(3),c)
 np.testing.assert_allclose(r['score_m'],g-.5*c,atol=1e-14,rtol=0);np.testing.assert_allclose(r['inflation_penalty_m'],.5*c,atol=1e-14,rtol=0)

def test_footprint_max_cost_uses_vehicle_not_centre_only_and_outside_ignored():
 from inflation_rule import footprint_cost
 g=grid(True);p=np.array([[.3,-.1],[.6,-.1],[.6,.1],[.3,.1]])
 assert footprint_cost(g,p)['C']==1
 p+=20;assert footprint_cost(g,p)['C']==0

def test_nn_endpoint_orientation_changes_footprint_clearance():
 from inflation_rule import predicted_costs
 # Long body at x=.1; yaw90 rotates nose toward wall at x=.5.
 pts=np.array([[x,y,0.] for x in [-.1,.1] for y in [-.5,.5]])
 root=np.array([.1,0,0,1,0,0,0]);world=pts+root[:3];P=np.zeros((2,3));R=np.array([np.eye(3),Rotation.from_euler('z',90,degrees=True).as_matrix()])
 c=predicted_costs(grid(True),root,[world],P,R)
 assert c[0]['C']==0 and c[1]['C']==1

def test_global_yaw_translation_and_dem_acquisition_pose_are_consistent():
 from inflation_rule import predicted_costs
 g=grid(True);root=np.array([0,0,0,1,0,0,0]);pts=np.array([[x,y,z] for x in [-.1,.1] for y in [-.3,.3] for z in [-.1,.1]]);P=np.array([[.25,0,0],[.4,0,0]]);R=np.tile(np.eye(3),(2,1,1));a=predicted_costs(g,root,[pts],P,R)
 Q=Rotation.from_euler('z',73,degrees=True);t=np.array([4.,-2.,0]);q=Q.as_quat()[[3,0,1,2]];pose=np.r_[t,q];b=predicted_costs(DEMGrid(g.dem,pose),pose,[pts@Q.as_matrix().T+t],P,R)
 np.testing.assert_allclose([v['C'] for v in a],[v['C'] for v in b],rtol=0,atol=1e-14)
