from pathlib import Path
exec(open(Path(__file__).resolve().with_name('replay_rl_actor.py')).read().split('rows=[];maxdiff=0')[0])
x=np.zeros((1,22),np.float32);x[0,5]=-1;x[0,12:14]=np.pi
print('NEUTRAL_STATIONARY_OBSERVATION',x.tolist())
for v in [0,.3,-.3]:
 for w in [-.5,-.3,0,.3,.5]:
  x[:,6:9]=[v,0,w]
  with torch.inference_mode():a=net(torch.tensor(x)).numpy()[0]
  t=np.clip(a[4:6]*14.2,-14.2,14.2)
  print('CMD',v,w,'track_RL',t.tolist(),'RminusL',float(t[0]-t[1]))
