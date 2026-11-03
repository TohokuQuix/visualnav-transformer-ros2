from pathlib import Path
import torch,numpy as np,json
N=Path('/home/ubuntu/eatnav-moving-state-pose2s-20261103');P=Path('/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v2/production');O=N/'docs/eat-nav/single-dem-pose2s-20261103/left-right-turn-audit'
torch.set_num_threads(4);sd=torch.load(N/'checkpoints/eat_nav/rl_actor/model_10000.pt',weights_only=True,map_location='cpu')['model_state_dict'];net=torch.nn.Sequential(torch.nn.Linear(22,1024),torch.nn.ELU(),torch.nn.Linear(1024,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,6));net.load_state_dict({k[6:]:v for k,v in sd.items() if k.startswith('actor.')});net.eval()
xs=[]
for s in json.loads((P/'dataset/source-index.json').read_text()):
 if s['parent_id'].startswith('final_test'):
  with np.load(s['raw_path']) as a:xs.append(a['actor_input'][::40])
x=np.concatenate(xs);report=[]
for kind in ['recorded_state','zero_effort','zero_gyro','zero_track_velocity','symmetric_joint_position','all_symmetric_static']:
 z=x.copy();z[:,6:9]=0
 if kind=='zero_effort':z[:,16:22]=0
 if kind=='zero_gyro':z[:,:3]=0
 if kind=='zero_track_velocity':z[:,14:16]=0
 if kind=='symmetric_joint_position':
  for i,j in [(10,11),(12,13)]:z[:,i]=z[:,j]=(z[:,i]+z[:,j])/2
 if kind=='all_symmetric_static':
  z[:]=0;z[:,5]=-1;z[:,12:14]=np.pi
 with torch.inference_mode():a=net(torch.tensor(z)).numpy();t=np.clip(a[:,4:6]*14.2,-14.2,14.2);d=t[:,0]-t[:,1]
 report.append(dict(probe=kind,n=len(z),right_minus_left_mean_rad_s=float(d.mean()),median=float(np.median(d))))
bias=float((sd['actor.6.bias'][4]-sd['actor.6.bias'][5])*14.2)
result=dict(probes=report,final_output_bias_difference_scaled_rad_s=bias,neutral_hidden_contribution_scaled_rad_s=report[-1]['right_minus_left_mean_rad_s']-bias,note='Offline interventions only. Modified observations may be outside learned distribution; not a closed-loop dynamics test.')
(O/'feature-ablation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
