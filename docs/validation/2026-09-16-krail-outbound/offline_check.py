import json,time,importlib.util,copy
from pathlib import Path
import numpy as np,torch,yaml
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import CompressedImage
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from visualnav_transformer.deployment.src.robot_interface import ImageHistory,decode_compressed,metric_path
from visualnav_transformer.deployment.src.robot_inference import NoMaDPredictor
root=Path('/tmp/krail-bag-check');out=root/'results';train=Path('/home/ubuntu/visualnav_docker/visualnav-transformer/train');bag='/media/ubuntu/Extreme SSD/20260803_experiment/2026_08_03-10_21_21'
torch.set_num_threads(1)
reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));reader.set_filter(rosbag2_py.StorageFilter(topics=['/front_camera/image_raw/compressed','/tf','/tf_static']))
buf=Buffer(cache_time=Duration(seconds=120));frames=[];tf_count=0
while reader.has_next():
 topic,data,record=reader.read_next()
 if topic.endswith('compressed'):
  msg=deserialize_message(data,CompressedImage);stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec;frames.append((stamp,msg,record))
 else:
  for tr in deserialize_message(data,TFMessage).transforms:
   (buf.set_transform_static if topic=='/tf_static' else buf.set_transform)(tr,'bag');tf_count+=1
hist=ImageHistory(4,.25);samples=[];last=-1;resets=0
for stamp,msg,record in frames:
 resets+=hist.add(decode_compressed(msg.data),stamp)
 snap=hist.snapshot()
 if snap and stamp-last>=4e9:
  try:tr=buf.lookup_transform('map','chassis/body',Time(nanoseconds=snap[1]))
  except Exception:continue
  samples.append((snap,tr));last=stamp
# Last decoded observation is the fixed goal; compare beginning/middle/end.
goal=decode_compressed(frames[-1][1].data);goal.save(out/'goal.jpg')
cfg=yaml.safe_load((root/'source/config/krail_baseline_model.yaml').read_text())
pred=NoMaDPredictor(model_config=cfg,checkpoint='/home/ubuntu/visualnav-transformer-ros2/model_weights/krail_baseline_20260904_latest.pth',goal_image=str(out/'goal.jpg'),device='cpu',num_samples=8)
base=pred.model
spec=importlib.util.spec_from_file_location('trained_nomad',train/'vint_train/models/nomad/nomad.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
latest=module.NoMaD(copy.deepcopy(base.vision_encoder),copy.deepcopy(base.noise_pred_net),copy.deepcopy(base.dist_pred_net),switchback_state_embedding=torch.nn.Embedding(2,16),switchback_state_projection=torch.nn.Linear(16,256,bias=False))
latest.load_state_dict(torch.load(train/'logs/nomad_krail/krail_directional_camera_switchback_conditioned_2026_09_06_17_30_01/latest.pth',map_location='cpu',weights_only=True),strict=True);latest.eval()
class StateAdapter(torch.nn.Module):
 def __init__(self,model,state):super().__init__();self.model=model;self.state=state;self.distance=None
 def forward(self,name,**kw):
  if name=='noise_pred_net':
   cond=kw['global_cond'];kw['global_cond']=self.model('switchback_state_condition',obsgoal_cond=cond,switchback_state=torch.full((len(cond),),self.state,dtype=torch.long,device=cond.device))
  value=self.model(name,**kw)
  if name=='dist_pred_net':self.distance=float(value.detach().flatten()[0])
  return value
rows=[]
for name,model in [('baseline',base),('latest_normal',StateAdapter(latest,0)),('latest_switchback',StateAdapter(latest,1))]:
 pred.model=model
 for index,((images,stamp),tr) in enumerate(samples):
  torch.manual_seed(42);start=time.perf_counter();result=pred(images,0);elapsed=time.perf_counter()-start
  path=metric_path(result.points,.05,tr,'map',stamp)
  rows.append(dict(model=name,index=index,time_sec=(stamp-frames[0][0])/1e9,seconds=elapsed,reached_goal=result.reached_goal,distance=getattr(model,'distance',None),local_xy_m=(result.points*.05).tolist(),map_xy=[[p.pose.position.x,p.pose.position.y] for p in path.poses]))
 print(name,'done',len(samples),flush=True)
poses=[];missing=0
for stamp,_,_ in frames[::10]:
 try:
  t=buf.lookup_transform('map','chassis/body',Time(nanoseconds=stamp)).transform.translation;poses.append([t.x,t.y])
 except Exception:missing+=1
report=dict(bag=bag,image_count=len(frames),tf_count=tf_count,duration_sec=(frames[-1][0]-frames[0][0])/1e9,history_resets=resets,sampled_observations=len(samples),sampled_tf_missing=missing,record_minus_image_stamp_ms=dict(min=min((r-s)/1e6 for s,_,r in frames),max=max((r-s)/1e6 for s,_,r in frames)),results=rows,recorded_positions=poses)
(out/'results.json').write_text(json.dumps(report,indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,5));xy=np.asarray(poses)
for ax,name in zip(axes,['baseline','latest_normal','latest_switchback']):
 ax.plot(xy[:,0],xy[:,1],color='gray',label='Recorded robot trajectory')
 for row in rows:
  if row['model']==name:
   p=np.asarray(row['map_xy']);ax.plot(p[:,0],p[:,1],marker='.',lw=1);ax.text(p[0,0],p[0,1],str(row['index']),fontsize=7)
 ax.set_title(name);ax.set_aspect('equal');ax.set_xlabel('map x [m]');ax.set_ylabel('map y [m]');ax.grid();ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(out/'paths.png',dpi=170)
print(json.dumps({k:v for k,v in report.items() if k not in ['results','recorded_positions']},indent=2))
for name in ['baseline','latest_normal','latest_switchback']:
 rr=[r for r in rows if r['model']==name];print(name,'latency',min(r['seconds'] for r in rr),max(r['seconds'] for r in rr),'goal flags',[r['index'] for r in rr if r['reached_goal']])
