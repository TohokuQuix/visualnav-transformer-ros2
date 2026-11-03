from pathlib import Path
import numpy as np,torch
n=Path('/home/ubuntu/eatnav-moving-state-pose2s-20261103');torch.set_num_threads(4)
p=torch.load(n/'checkpoints/eat_nav/rl_actor/model_10000.pt',map_location='cpu',weights_only=True)['model_state_dict'];net=torch.nn.Sequential(torch.nn.Linear(22,1024),torch.nn.ELU(),torch.nn.Linear(1024,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,6));net.load_state_dict({k[6:]:v for k,v in p.items() if k.startswith('actor.')});net.eval()
for pose in [[0,0,0,0],[0,0,np.pi,np.pi],[np.pi,np.pi,0,0]]:
 for mode in [0,1,-1]:
  outs=[]
  for yaw in [-.3,0,.3]:
   x=np.zeros((1,22),np.float32);x[0,5]=-1;x[0,8]=yaw;x[0,9]=mode;x[0,10:14]=pose
   with torch.inference_mode():a=net(torch.tensor(x)).numpy()[0];t=np.clip(a[4:6]*14.2,-14.2,14.2)
   outs.append(round(float(t[0]-t[1]),3))
  print('pose',pose,'mode',mode,'RLdiff at yawcmd[-.3,0,.3]',outs)
