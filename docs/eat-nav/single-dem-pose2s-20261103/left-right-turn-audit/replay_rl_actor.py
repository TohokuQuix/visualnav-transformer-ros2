import numpy as np,torch,json,csv
from pathlib import Path
P=Path('/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v2/production');N=Path('/home/ubuntu/eatnav-moving-state-pose2s-20261103')
torch.set_num_threads(4);net=torch.nn.Sequential(torch.nn.Linear(22,1024),torch.nn.ELU(),torch.nn.Linear(1024,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,6));d=torch.load(N/'checkpoints/eat_nav/rl_actor/model_10000.pt',map_location='cpu',weights_only=True)['model_state_dict'];net.load_state_dict({k[6:]:v for k,v in d.items() if k.startswith('actor.')});net.eval();rows=[];maxdiff=0
sources=json.loads((P/'dataset/source-index.json').read_text())
for src in sources:
 if not src['parent_id'].startswith('final_test'):continue
 meta=json.loads(Path(src['metadata_path']).read_text());a=np.load(src['raw_path']);obs=a['actor_input'];actions=a['action']
 with torch.inference_mode():out=net(torch.tensor(obs)).numpy()
 maxdiff=max(maxdiff,float(np.max(np.abs(out-actions))))
 for j,cid in enumerate(meta['parent']['candidates']):
  sl=slice(j*40,(j+1)*40);tgt=np.clip(actions[sl,4:6]*14.2,-14.2,14.2)
  rows.append(dict(parent_id=meta['parent']['parent_id'],field=meta['parent']['field'],level=meta['parent']['level'],cid=cid,stage=j,right_target=float(tgt[:,0].mean()),left_target=float(tgt[:,1].mean()),right_actual=float(a['joint_velocity'][sl,meta['right_track_indices']].mean()),left_actual=float(a['joint_velocity'][sl,meta['left_track_indices']].mean())))
print('ACTOR_REPLAY_MAX_ABS',maxdiff)
for cid in range(14):
 r=[x for x in rows if x['cid']==cid];print('TRACK',cid,{k:round(np.mean([x[k] for x in r]),3) for k in ['right_target','left_target','right_actual','left_actual']})
# Same observed state, only change requested yaw, to locate asymmetry before physics.
probes=[]
for src in sources:
 if not src['parent_id'].startswith('final_test'):continue
 a=np.load(src['raw_path']);probes.append(a['actor_input'][::40])
x=np.concatenate(probes);print('example_obs',x[0].tolist())
for v in [.0,.3,-.3]:
 for w in [-.5,0.,.5]:
  z=x.copy();z[:,6:9]=[v,0,w]
  with torch.inference_mode():t=np.clip(net(torch.tensor(z)).numpy()[:,4:6]*14.2,-14.2,14.2)
  print('SAME_STATE_CMD',v,w,'right',round(t[:,0].mean(),3),'left',round(t[:,1].mean(),3),'right-minus-left',round((t[:,0]-t[:,1]).mean(),3))
# Mirror all numeric observation components, using configured order.
z=x.copy();z[:,[0,2,4,7,8]]*=-1
z[:,10:14]=x[:,[11,10,13,12]];z[:,14:16]=x[:,[15,14]];z[:,16:22]=x[:,[17,16,19,18,21,20]]
with torch.inference_mode():aa=net(torch.tensor(x)).numpy();bb=net(torch.tensor(z)).numpy()
print('FULL_MIRROR_TRACK_MAE',np.mean(np.abs(aa[:,4]-bb[:,5]))*14.2,np.mean(np.abs(aa[:,5]-bb[:,4]))*14.2)
with open(Path(__file__).resolve().parent/'track_commands.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
