import numpy as np
from visualnav_transformer.eat_nav.r6_geometry import live_footprint

def test_live_bounds_use_body_pose_not_stale_usd_world_transform():
 lo=np.array([[-.3,-.2,-.1],[-.2,-.05,-.03]]);hi=-lo
 p=np.array([[100.,50.,.1],[100.4,50.,.1]]);q=np.array([[1,0,0,0],[1,0,0,0.]])
 root=np.array([100.,50.,.1,1.,0,0,0])
 d,points=live_footprint(lo,hi,p,q,root,.06)
 assert abs(d['front_m']-.6)<1e-6 and abs(d['rear_m']-.3)<1e-6
 assert points[:,0].min()>99
 q[1]=[np.sqrt(.5),0,np.sqrt(.5),0]
 d,_=live_footprint(lo,hi,p,q,root,.06)
 assert abs(d['front_m']-.43)<1e-6


def test_flipper_current_angles_change_footprint_without_mean_imputation():
 from visualnav_transformer.eat_nav.r6_geometry import calibrated_footprint
 p=np.array([[0,0,0],[-.4,-.2,0],[-.4,.2,0],[.4,-.2,0],[.4,.2,0]])
 lo=np.tile([-.1,-.04,-.02],(5,1));lo[0]=[-.3,-.2,-.1]
 axes={side:dict(axis_chassis=[0,1 if side[0]=='r' else -1,0],hinge_chassis=[-.2 if side[0]=='r' else .2,(-.2 if side[1]=='r' else .2),0],body_indices=[i+1],zero_angle_rad=0) for i,side in enumerate(('rr','rl','fr','fl'))}
 cal=dict(schema='eatnav_r6_sim_flipper_geometry_v1',numeric_sim_reference_verified=True,local_bounds_min=lo,local_bounds_max=-lo,poses={'0':dict(body_positions_chassis=p,body_rotations_chassis=np.tile(np.eye(3),(5,1,1)))},axes=axes)
 flat=calibrated_footprint(cal,dict.fromkeys(axes,0),[0,0,-1])
 raised=calibrated_footprint(cal,dict.fromkeys(axes,np.pi/2),[0,0,-1])
 assert abs(flat['front_m']-.5)<1e-6 and abs(raised['front_m']-.3)<1e-6
 assert abs(flat['rear_m']-.5)<1e-6
