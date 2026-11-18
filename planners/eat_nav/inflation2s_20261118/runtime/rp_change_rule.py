"""User-selected c0 and +.2m*O only for forward/reverse abs60/90 candidates."""
import numpy as np
from scipy.spatial.transform import Rotation
RULE='goal_progress_minus_endpoint_inflation_v1'
C=0.0
INFLATION_LAMBDA_M=.5

def score_components(goal_progress_m,relative_rotations,current_rotation,inflation_cost):
 g=np.asarray(goal_progress_m,float);rel=np.asarray(relative_rotations,float);cur=np.asarray(current_rotation,float)
 if g.ndim!=1 or rel.shape!=(len(g),3,3) or cur.shape!=(3,3) or not all(np.isfinite(x).all() for x in [g,rel,cur]):raise ValueError('invalid RP-score input')
 current=Rotation.from_matrix(cur).as_euler('xyz',degrees=True)
 future=Rotation.from_matrix(cur[None]@rel).as_euler('xyz',degrees=True)
 delta=(future[:,:2]-current[:2]+180.)%360.-180.
 magnitude=np.linalg.norm(delta,axis=1);penalty=C*magnitude/10.
 o=np.asarray(inflation_cost,float)
 if o.shape!=g.shape or not np.isfinite(o).all() or np.any((o<0)|(o>1)):raise ValueError("inflation cost must be finite[0,1]")
 inflation_penalty=INFLATION_LAMBDA_M*o
 return dict(inflation_cost=o.tolist(),selector_rule=RULE,rp_change_c_m_per_10deg=C,goal_progress_m=g.tolist(),current_world_rpy_deg=current.tolist(),predicted_world_rpy_deg=future.tolist(),rp_delta_deg=delta.tolist(),rp_change_deg=magnitude.tolist(),rp_penalty_m=penalty.tolist(),inflation_lambda_m=INFLATION_LAMBDA_M,inflation_penalty_m=inflation_penalty.tolist(),score_m=(g-penalty-inflation_penalty).tolist())
