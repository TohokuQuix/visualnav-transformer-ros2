from pathlib import Path
import json,hashlib,csv
import numpy as np,torch
from visualnav_transformer.eat_nav.single_pose2s_contract import INPUT_SCHEMA,PROFILE_SHA
from visualnav_transformer.eat_nav.single_pose2s_model import predict
from visualnav_transformer.eat_nav.bidirectional_bank import bank
P=Path('/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v2/production');O=Path('/home/ubuntu/eatnav-moving-state-pose2s-20261103/docs/eat-nav/single-dem-pose2s-20261103/left-right-turn-audit/pose-nn-simple-comparison');O.mkdir(exist_ok=True)
torch.set_num_threads(4);cp=P/'model-seed123/checkpoint.pt';ck=torch.load(cp,map_location='cpu',weights_only=True)
state=np.zeros((1,27),np.float32);state[:,2]=-1;state[:,6:10]=[0,0,np.pi,np.pi];state[:,25]=1
# Synthetic left/right symmetric flat terrain relative to a body origin .35m above it.
dem=np.ones((1,3,40,40),np.float32);dem[:,0]=-.35
a=dict(input_schema=np.array(INPUT_SCHEMA),profile_sha256=np.array(PROFILE_SHA),dem=dem,state=state,state_mask=np.ones((1,27),np.float32),state_age=np.zeros((1,27),np.float32),dem_context=np.array([[0,1,0,0,0,1]],np.float32),t0_ns=np.array([1000000000],np.int64),dem_stamp_ns=np.array([1000000000],np.int64),dem_received_ns=np.array([1000000000],np.int64),paths=np.array([[c.waypoints for c in bank()]],np.float32),directions=np.array([[[c.direction] for c in bank()]],np.float32),q0_wxyz=np.array([[1,0,0,0]],np.float32))
np.savez_compressed(O/'symmetric-flat-input.npz',**a)
out=predict(ck,a,'cpu');assert out['rotation_valid'].all()
rows=[]
for cid,c in enumerate(bank()):
 pos=out['position_body_m'][0,cid];rpy=out['future_rpy_deg'][0,cid]
 rows.append(dict(candidate_id=cid,direction=c.direction,candidate_angle_deg=c.arc_angle_deg,pred_x_cm=float(pos[0]*100),pred_y_cm=float(pos[1]*100),pred_z_cm=float(pos[2]*100),pred_roll_deg=float(rpy[0]),pred_pitch_deg=float(rpy[1]),pred_yaw_deg=float(rpy[2])))
with (O/'predictions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pairs=[]
for direction in [1,-1]:
 for ang in [30,60,90]:
  neg=next(r for r in rows if r['direction']==direction and r['candidate_angle_deg']==-ang);pos=next(r for r in rows if r['direction']==direction and r['candidate_angle_deg']==ang)
  pairs.append(dict(direction=direction,angle_magnitude_deg=ang,minus_candidate_pred_yaw_deg=neg['pred_yaw_deg'],plus_candidate_pred_yaw_deg=pos['pred_yaw_deg'],yaw_signed_sum_deg=float(np.rad2deg(np.arctan2(np.sin(np.deg2rad(neg['pred_yaw_deg']+pos['pred_yaw_deg'])),np.cos(np.deg2rad(neg['pred_yaw_deg']+pos['pred_yaw_deg']))))),x_difference_cm=pos['pred_x_cm']-neg['pred_x_cm'],y_signed_sum_cm=pos['pred_y_cm']+neg['pred_y_cm']))
report=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),input_schema=INPUT_SCHEMA,profile_sha256=PROFILE_SHA,horizon_s=2,scope='CPU input-output probe; identical synthetic symmetric state/DEM, candidate only changes; no simulator/real robot run, no learning',input_conditions=dict(flat_dem_height_body_reference_m=-.35,dem_channels='height/inpaint_valid/raw_support',all_cells_observed=True,gravity_body=[0,0,-1],linear_angular_velocity=0,flipper_angles_rr_rl_fr_fl_rad=[0,0,float(np.pi),float(np.pi)],fold_mode=0,dem_age_s=0,state_ages_s=0),caveats=['Synthetic input is schema-valid but not proven in-distribution.','There is no ground truth for this synthetic input; this checks output asymmetry, not prediction error.','Candidate +/-90 means arc geometry, not an instruction to rotate 90deg within2s.','Asymmetric predictions may reflect asymmetric RL-driven training outcomes.'],predictions=rows,pairs=pairs)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
