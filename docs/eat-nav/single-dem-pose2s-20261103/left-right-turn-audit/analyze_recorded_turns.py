import json,csv,sys
from pathlib import Path
import numpy as np,torch
from visualnav_transformer.eat_nav.single_pose2s_model import SinglePose2s,prepare
from visualnav_transformer.eat_nav.moving_pose2s_geometry import quat_to_matrix,sixd_to_matrix,matrix_to_rpy
from visualnav_transformer.eat_nav.bidirectional_bank import bank
from visualnav_transformer.eat_nav.single_pose2s_follower import NominalFollower
P=Path('/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v2/production')
O=Path(__file__).resolve().parent;O.mkdir(exist_ok=True)
wrap=lambda x:np.arctan2(np.sin(x),np.cos(x))
torch.set_num_threads(4);ck=torch.load(P/'model-seed123/checkpoint.pt',map_location='cpu',weights_only=True);model=SinglePose2s();model.load_state_dict(ck['state_dict']);model.eval()
rows=[]
sources={s['parent_id']:s for s in json.loads((P/'dataset/source-index.json').read_text())}
for role in ['train','validation','final_test']:
 a=dict(np.load(P/f'dataset/{role}-samples.npz'));rec=json.loads((P/f'dataset/{role}-records.json').read_text());Rs=[]
 with torch.inference_mode():
  inp=prepare(a,ck['normalization'])
  for i in range(0,len(rec),128):Rs.append(model(*(x[i:i+128] for x in inp))['rotation_6d'][:,0].numpy())
 predR,_=sixd_to_matrix(np.concatenate(Rs));rawcache={}
 for i,r in enumerate(rec):
  if not a['pose_mask'][i]:continue
  pid=r['parent_id']
  if pid not in rawcache:rawcache[pid]=dict(np.load(sources[pid]['raw_path']))
  raw=rawcache[pid];meta=json.loads(Path(sources[pid]['metadata_path']).read_text());j=r['stage']*40;cid=r['candidate_id'];b=bank()[cid]
  roots=raw['root'][j:j+41];R=quat_to_matrix(roots[:,3:7]);rpy,_=matrix_to_rpy(R);dy=wrap(rpy[-1,2]-rpy[0,2]);prpy,_=matrix_to_rpy(R[0]@predR[i]);pdy=wrap(prpy[2]-rpy[0,2]);cmd=raw['command'][j:j+40];act=raw['action'][j:j+40]
  q=dict(role=role,parent_id=pid,field=r['field'],level=r['level'],candidate_id=cid,angle=b.arc_angle_deg,direction=b.direction,stage=r['stage'],gt_yaw_deg=np.rad2deg(dy),pred_yaw_deg=np.rad2deg(pdy),yaw_error_deg=np.rad2deg(wrap(pdy-dy)),cmd_yaw_deg=np.rad2deg(cmd[:,2].sum()*.05),cmd_yaw_first=float(cmd[0,2]),cmd_v_mean=float(cmd[:,0].mean()),body_wz_mean=float(raw['state'][j:j+40,5].mean()),initial_wz=float(raw['state'][j,5]),initial_roll_deg=np.rad2deg(rpy[0,0]),initial_pitch_deg=np.rad2deg(rpy[0,1]),initial_yaw_deg=np.rad2deg(rpy[0,2]),action0=float(act[:,0].mean()),action1=float(act[:,1].mean()),right_w=float(raw['joint_velocity'][j:j+40,meta['right_track_indices']].mean()),left_w=float(raw['joint_velocity'][j:j+40,meta['left_track_indices']].mean()))
  rows.append(q)
 with (O/(role+'.csv')).open('w') as f:
  rr=[r for r in rows if r['role']==role];w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
for role in ['train','final_test']:
 print('\nROLE',role)
 for cid in range(14):
  a=[r for r in rows if r['role']==role and r['candidate_id']==cid]
  print(cid,'n',len(a),{k:round(float(np.mean([r[k] for r in a])),2) for k in ['gt_yaw_deg','pred_yaw_deg','cmd_yaw_deg','initial_wz','action0','action1','right_w','left_w']},'MAE',round(np.mean([abs(r['yaw_error_deg']) for r in a]),2))
# Pure geometry/controller mirror test: mirror y/yaw/angle. Separate stateful followers.
rng=np.random.default_rng(20261103);mx=0.;bad=0
for direction in [1,-1]:
 for angle in [30,60,90]:
  from visualnav_transformer.eat_nav.bidirectional_bank import candidate
  for trial in range(50):
   pos=NominalFollower(candidate(direction,angle).waypoints);neg=NominalFollower(candidate(direction,-angle).waypoints)
   for step in range(40):
    x,y,yaw=rng.normal(size=3)*[.3,.2,.6];u=np.array(pos.step(x,y,yaw));v=np.array(neg.step(x,-y,-yaw));err=float(np.max(np.abs(u-v*np.array([1,1,-1]))));mx=max(mx,err);bad+=err>1e-8
print('FOLLOWER_MIRROR',mx,bad)
print('OUTPUT',O)
