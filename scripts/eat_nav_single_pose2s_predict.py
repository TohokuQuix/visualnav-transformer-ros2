"""Offline inference only: no ROS publishers and no planner connection."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from visualnav_transformer.eat_nav.single_pose2s_model import predict
from visualnav_transformer.eat_nav.bidirectional_bank import bank
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--all-fourteen',action='store_true');a=p.parse_args()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    with np.load(a.input,allow_pickle=False) as f:inputs={k:f[k] for k in f.files}
    if a.all_fourteen:
        b=len(inputs['dem']);inputs['paths']=np.tile(np.array([c.waypoints for c in bank()]),(b,1,1,1));inputs['directions']=np.tile(np.array([[c.direction] for c in bank()]),(b,1,1))
    result=predict(payload,inputs,a.device);a.output.write_text(json.dumps(result,default=lambda x:x.tolist(),indent=2,allow_nan=False)+'\n')
